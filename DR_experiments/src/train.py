"""Train the networks of one model set (see common.MODEL_SETS) for the given backbones and folds.

Cross-fitting: the fold-k network trains on the training patients outside fold k. Finished networks
(metrics.json present) are skipped, so the run can be resumed.

Protocol (identical for every network, set in common.py): last block unfrozen, BatchNorm statistics frozen,
head-only warm-up, then 1-epoch linear warm-up + cosine decay. For every network, including the robust ones,
the checkpoint with the highest clean validation AP over all epochs run is kept; training stops when the clean
validation AP has not improved by at least 0.005 for 6 epochs (the 0.005 only decides when the patience
counter resets). Experiments therefore differ only in their training data / inputs.

Monitoring only (never used for stopping or selection): each epoch also records the AP on one fixed degraded
copy of the validation photos (val_degraded_ap). It shows whether robustness was still improving after the
selected epoch, i.e. how much the clean-validation rule may understate robust training.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

import common as C
import degrade
import nets


def evaluate(model, loader, device):
    logits, _ = nets.score(model, loader, device)
    y = loader.dataset.labels
    return C.average_precision(y, logits), float(roc_auc_score(y, logits))


def train_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    nets.set_train_mode(model)
    amp = nets.use_amp(device)
    total, count = torch.zeros((), device=device), 0
    for images, clinical, _, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(model(images, clinical.to(device)).float(), labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.detach() * len(labels)
        count += len(labels)
    mean = total.item() / count
    if not math.isfinite(mean):
        raise FloatingPointError('Training loss became non-finite.')
    return mean


def run_one(model_set, backbone, fold, args, device):
    spec = C.MODEL_SETS[model_set]
    out = C.model_dir(model_set, backbone, fold)
    if (out / 'metrics.json').is_file() and not args.force:
        print(f'[skip] {model_set} {backbone} fold {fold}: already trained', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    table = C.load_table(include_inadequate=False)
    if spec['clinical'] == 'shuffled':
        table = C.shuffled_clinical(table)
    train = table[(table.split == 'train') & (table.fold != fold)].reset_index(drop=True)
    val = table[table.split == 'val'].reset_index(drop=True)
    # clinical preprocessing is fitted on this fold's training patients only
    encoder = C.ClinicalEncoder().fit(train) if spec['clinical'] != 'none' else None
    val_clinical = encoder.transform(val) if encoder else None

    model = nets.DRNet(backbone, pretrained=not C.NO_PRETRAINED, n_clinical=len(encoder.names) if encoder else 0).to(device)
    data_config = timm.data.resolve_model_data_config(model.encoder)
    mean, std = data_config['mean'], data_config['std']
    train_set = nets.PhotoSet(train.image_id, train[C.TARGET], mean, std, family=C.degradation_family(spec['aug']),
                              basic=C.uses_basic_aug(spec['aug']), clinical=encoder.transform(train) if encoder else None)
    val_set = nets.PhotoSet(val.image_id, val[C.TARGET], mean, std, clinical=val_clinical)
    rng = np.random.default_rng(C.SEED + 1)   # the same degraded validation copy for every network
    specs = [degrade.sample_spec(rng, 'both', force=True) for _ in range(len(val))]
    monitor_set = nets.PhotoSet(val.image_id, val[C.TARGET], mean, std, kinds=[k for k, _ in specs],
                                values=[v for _, v in specs], clinical=val_clinical)
    train_loader = nets.make_loader(train_set, C.BATCH_SIZE, args.workers, shuffle=True, device=device)
    val_loader = nets.make_loader(val_set, C.BATCH_SIZE, args.workers, shuffle=False, device=device)
    monitor_loader = nets.make_loader(monitor_set, C.BATCH_SIZE, args.workers, shuffle=False, device=device)

    positives = int(train[C.TARGET].sum())
    pos_weight = (len(train) - positives) / positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    prefixes = C.UNFREEZE[backbone]
    nets.set_encoder_trainable(model, [])
    total = sum(p.numel() for p in model.encoder.parameters())
    trainable = sum(p.numel() for n, p in model.encoder.named_parameters()
                    if any(n == q or n.startswith(q + '.') for q in prefixes))
    config = {
        'model_set': model_set, **spec, 'backbone': backbone, 'fold': fold,
        'pretrained_weights': None if C.NO_PRETRAINED else C.BACKBONES[backbone],
        'image_size': C.IMAGE_SIZE, 'normalization': {'mean': list(mean), 'std': list(std)},
        'unfrozen_modules': prefixes, 'encoder_parameters': total, 'encoder_trainable_parameters': trainable,
        'clinical_features': encoder.names if encoder else None,
        'clinical_preprocessing': encoder.state if encoder else None,
        'train_images': len(train), 'train_dr_positive': positives, 'val_images': len(val_set),
        'epochs_max': C.EPOCHS, 'warmup_epochs': C.WARMUP_EPOCHS, 'patience': C.PATIENCE,
        'min_ap_gain': C.MIN_AP_GAIN, 'early_stopping': 'clean validation AP', 'batch_size': C.BATCH_SIZE,
        'encoder_lr': C.ENCODER_LR, 'head_lr': C.HEAD_LR, 'weight_decay': C.WEIGHT_DECAY, 'pos_weight': pos_weight,
        'seed': C.SEED, 'device': str(device), 'torch_version': torch.__version__, 'timm_version': timm.__version__,
    }
    print(f'{model_set} {backbone} fold {fold}: {len(train)} training / {len(val_set)} validation photos, '
          f'unfrozen {trainable:,}/{total:,} = {100 * trainable / total:.1f}% of encoder', flush=True)

    optimizer = torch.optim.AdamW(model.head_parameters(), lr=C.HEAD_LR, weight_decay=C.WEIGHT_DECAY)
    scheduler, history, best, patience_ap, stale = None, [], None, -math.inf, 0
    started_all = time.perf_counter()
    for epoch in range(1, C.EPOCHS + 1):
        phase = 'warm-up' if epoch <= C.WARMUP_EPOCHS else 'fine-tune'
        if epoch == C.WARMUP_EPOCHS + 1:
            encoder_params = nets.set_encoder_trainable(model, prefixes)
            optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': C.ENCODER_LR},
                                           {'params': model.head_parameters(), 'lr': C.HEAD_LR}],
                                          weight_decay=C.WEIGHT_DECAY)
            scheduler = nets.cosine_with_warmup(optimizer, len(train_loader),
                                                len(train_loader) * (C.EPOCHS - C.WARMUP_EPOCHS))
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
        ap, auc = evaluate(model, val_loader, device)
        degraded_ap, _ = evaluate(model, monitor_loader, device)
        metrics = {'epoch': epoch, 'phase': phase, 'train_loss': loss, 'ap': ap, 'roc_auc': auc,
                   'val_degraded_ap': degraded_ap, 'seconds': round(time.perf_counter() - started, 1)}
        history.append(metrics)
        if best is None or ap > best['ap']:
            best = metrics
            torch.save({'config': config, 'metrics': best, 'model_state': model.state_dict()}, out / '.best.pt.tmp')
            (out / '.best.pt.tmp').replace(out / 'best.pt')
        print(f'  epoch {epoch:2d} ({phase}): train loss {loss:.4f}, val AP {ap:.4f}, ROC-AUC {auc:.4f}, '
              f'degraded-val AP {degraded_ap:.4f}', flush=True)
        if phase == 'warm-up':
            continue
        if ap > patience_ap + C.MIN_AP_GAIN:
            patience_ap, stale = ap, 0
        else:
            stale += 1
        if stale >= C.PATIENCE:
            break
    report = {'config': config, 'best_validation': best, 'epochs': history,
              'total_seconds': round(time.perf_counter() - started_all, 1)}
    (out / '.metrics.json.tmp').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out / '.metrics.json.tmp').replace(out / 'metrics.json')
    print(f'  best: epoch {best["epoch"]}, val AP {best["ap"]:.4f} ({report["total_seconds"]:.0f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-set', choices=list(C.MODEL_SETS), required=True)
    parser.add_argument('--backbones', nargs='+', choices=list(C.BACKBONES), default=list(C.BACKBONES))
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(C.FOLDS)))
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Retrain networks that already finished.')
    args = parser.parse_args()
    device = nets.resolve_device(args.device)
    for backbone in args.backbones:
        for fold in args.folds:
            run_one(args.model_set, backbone, fold, args, device)


if __name__ == '__main__':
    main()
