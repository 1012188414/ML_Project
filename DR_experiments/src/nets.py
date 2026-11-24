"""Network, datasets and scoring shared by training, feature extraction and end-to-end prediction."""
import json
import math

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import common as C
import degrade


# ---------------------------------------------------------------- photo caches
class _Cache:
    def __init__(self, stem, key):
        meta = json.loads((C.CACHE / f'{stem}_{C.IMAGE_SIZE}.json').read_text())
        self.row = {key(k): r for r, k in enumerate(meta['ids'])}
        self.path = C.CACHE / f'{stem}_{C.IMAGE_SIZE}.npy'
        self.array = None

    def __getstate__(self):  # never pickle the memory map into DataLoader workers
        state = self.__dict__.copy()
        state['array'] = None
        return state

    def fetch(self, key):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode='r')
        return np.array(self.array[self.row[key]])


class BaseCache(_Cache):
    def __init__(self):
        super().__init__('base', lambda k: k)


class TestCache(_Cache):
    def __init__(self):
        super().__init__('test', lambda k: (k[0], k[1]))

    def fetch(self, key):
        return super().fetch(tuple(key))


def quality_table():
    q = pd.read_csv(C.CACHE / f'quality_{C.IMAGE_SIZE}.csv', dtype={'image_id': str})
    return q.set_index(['image_id', 'condition'])


def training_augmentation():
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15, interpolation=v2.InterpolationMode.BILINEAR, fill=0.0),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])


class PhotoSet(Dataset):
    """Items are photos with an optional degradation.
      source 'base': photo from the base cache; degradation = (kind, value), or a fresh random one per call
                     when `family` is set (robust training)
      source 'test': pre-degraded test photo from the test cache; `kind` holds the condition name
    Returns (image, clinical, q, label, index); q is the raw quality signal (0 when quality='none')."""

    def __init__(self, image_ids, labels, mean, std, sources=None, kinds=None, values=None, family=None,
                 basic=False, clinical=None, quality='none', q_fixed=None):
        n = len(image_ids)
        self.ids = list(image_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sources = list(sources) if sources is not None else ['base'] * n
        self.kinds = list(kinds) if kinds is not None else ['clean'] * n
        self.values = np.asarray(values if values is not None else np.zeros(n), dtype=float)
        self.family = family
        self.basic = training_augmentation() if basic else None
        self.clinical = clinical
        self.quality = quality
        self.q_fixed = np.asarray(q_fixed, dtype=float) if q_fixed is not None else np.full(n, np.nan)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.base = BaseCache()
        self.test = TestCache() if 'test' in self.sources else None
        self.rng = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.rng is None:  # one generator per DataLoader worker, seeded from torch
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        if self.sources[i] == 'test':
            pixels = self.test.fetch((self.ids[i], self.kinds[i]))
            kind, value = degrade.parse(self.kinds[i])
        else:
            pixels = self.base.fetch(self.ids[i])
            if self.family:
                kind, value = degrade.sample_spec(self.rng, self.family)
            else:
                kind, value = self.kinds[i], self.values[i]
            pixels = degrade.apply(pixels, kind, value, seed=C.stable_seed(self.ids[i], kind, value))
        q = 0.0
        if self.quality != 'none':
            q = self.q_fixed[i]
            if not np.isfinite(q):
                q = degrade.q_lap(pixels) if self.quality == 'lap' else degrade.severity(kind, value)
        image = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float() / 255.0
        if self.basic is not None:
            image = self.basic(image).clamp_(0.0, 1.0)
        image = (image - self.mean) / self.std
        clinical = torch.from_numpy(self.clinical[i]) if self.clinical is not None else torch.zeros(0)
        return image, clinical, torch.tensor(q, dtype=torch.float32), self.labels[i], i


def make_loader(dataset, batch_size, workers, shuffle, seed=C.SEED, device=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=device is not None and device.type == 'cuda',
                      generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- network
class DRNet(nn.Module):
    """backbone -> 256-d embedding -> linear classifier. With n_clinical > 0 the embedding is concatenated with
    the encoded clinical features before the classifier (end-to-end fusion)."""

    def __init__(self, backbone, pretrained, n_clinical=0):
        super().__init__()
        self.encoder = timm.create_model(C.BACKBONES[backbone], pretrained=pretrained, **C.BACKBONE_KWARGS.get(backbone, {}))
        # width of the pooled features; self-supervised weights (DINOv3) come without a classifier to read it from
        pooled = getattr(self.encoder.get_classifier(), 'in_features', None) or self.encoder.num_features
        self.encoder.reset_classifier(0)
        self.projection = nn.Sequential(nn.Linear(pooled, C.EMBEDDING_SIZE), nn.LayerNorm(C.EMBEDDING_SIZE))
        self.n_clinical = n_clinical
        self.classifier = nn.Linear(C.EMBEDDING_SIZE + n_clinical, 1)

    def forward(self, images, clinical=None):
        features = self.projection(self.encoder(images))
        if self.n_clinical:
            features = torch.cat([features, clinical.to(features.dtype)], dim=1)
        return self.classifier(features).squeeze(-1)

    def head_parameters(self):
        return [*self.projection.parameters(), *self.classifier.parameters()]


def set_encoder_trainable(model, prefixes):
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad = any(name == p or name.startswith(p + '.') for p in prefixes)
    return [p for p in model.encoder.parameters() if p.requires_grad]


def set_train_mode(model):
    """BatchNorm running statistics stay at their pretrained values (affine weights still train)."""
    model.train()
    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def use_amp(device):
    return device.type == 'cuda' and torch.cuda.is_bf16_supported()


@torch.inference_mode()
def score(model, loader, device):
    """Logits (N,) and the raw quality signal (N,) in dataset order."""
    model.eval()
    amp = use_amp(device)
    n = len(loader.dataset)
    logits, qs = np.full(n, np.nan), np.full(n, np.nan)
    for images, clinical, q, _, idx in loader:
        idx = idx.numpy()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(images.to(device, non_blocking=True), clinical.to(device))
        logits[idx] = out.float().cpu().numpy()
        qs[idx] = q.numpy()
    if not np.isfinite(logits).all():
        raise FloatingPointError('Some photos received a non-finite score.')
    return logits, qs


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = DRNet(config['backbone'], pretrained=False, n_clinical=len(config['clinical_features'] or []))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), config
