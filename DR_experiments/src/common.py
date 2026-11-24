"""Shared settings and helpers.

Every script imports its controls from this one file (folds, training protocol, degradations, experiment
definitions), so the experiments cannot drift apart. Change a setting here and it changes everywhere.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
# Folder locations live in dr_paths.py, which is not part of the training signature (moving folders is not an
# experimental change). The names are re-exported here, so every script keeps using C.RESULTS, C.CACHE, ...
from dr_paths import (ROOT, DATA, IMAGES, DIABETIC_CSV, INADEQUATE_CSV, GRADE_CSV, WEIGHTS, HF_CACHE,  # noqa: F401
                      SMOKE, NO_PRETRAINED, DEFAULT_RUN, RESULTS_NAME, RESULTS, CACHE, MODELS_DIR, FEATURES_DIR,
                      PREDICTIONS_DIR, SUMMARY_DIR, FIGURES_DIR, LOG_DIR)

# ---------------------------------------------------------------- data
SEED = 42
FOLDS = 4                                   # cross-fitting folds (column `fold` of brset_diabetic.csv)
TARGET = 'diabetic_retinopathy'
CLINICAL_CONTINUOUS = ['age', 'diabetes_duration_y']
CLINICAL_BINARY = ['sex', 'insulin', 'hypertension_record']
CLINICAL = ['age', 'sex', 'diabetes_duration_y', 'insulin', 'hypertension_record']

# ---------------------------------------------------------------- backbones (fine-tune the last block only)
BACKBONES = {
    'convnextv2_nano': 'convnextv2_nano.fcmae_ft_in1k',
    'densenet169': 'densenet169.tv_in1k',
    'efficientnet_b3': 'efficientnet_b3.ra2_in1k',
    'mobilenetv3_large_150d': 'mobilenetv3_large_150d.ra4_e3600_r256_in1k',
}
DEFAULT_BACKBONES = list(BACKBONES)   # the four backbones of the main run; run_all.py uses these unless --backbones is given
# Extra backbones (added after the main run). They are only trained when named with --backbones, and run_all.py
# requires --results-name for them. Same protocol as the others: last block + final normalisation layer unfrozen.
BACKBONES.update({
    'pvt_v2_b1': 'pvt_v2_b1.in1k',                                            # pyramid vision transformer, 13.5M, supervised
    'vit_small_patch16_dinov3': 'vit_small_patch16_dinov3.lvd1689m',          # ViT-S/16, DINOv3 self-supervised
    'vit_small_patch16_384': 'vit_small_patch16_384.augreg_in21k_ft_in1k',    # ViT-S/16, supervised, native 384 px
    'resnet50': 'resnet50.a1_in1k',                                           # classic CNN reference
})
UNFREEZE = {   # last repeated block of each network + the final normalisation layer
    'convnextv2_nano': ['stages.3.blocks.1', 'head.norm'],
    'densenet169': ['features.denseblock4.denselayer32', 'features.norm5'],
    'efficientnet_b3': ['blocks.6.1', 'bn2'],
    'mobilenetv3_large_150d': ['blocks.5.3'],
    'pvt_v2_b1': ['stages.3.blocks.1', 'stages.3.norm'],   # 23.5% of the encoder
    'vit_small_patch16_dinov3': ['blocks.11', 'norm'],   # 8.2% of the encoder
    'vit_small_patch16_384': ['blocks.11', 'norm'],      # 8.1% of the encoder
    'resnet50': ['layer4.2'],                            # 19.0% of the encoder (its normalisation layers are inside the block)
}

# ---------------------------------------------------------------- training protocol (identical for every network)
IMAGE_SIZE = 64 if SMOKE else 384
# Extra arguments for timm.create_model. Vision transformers are built for one input size, so they are told ours.
BACKBONE_KWARGS = {
    'vit_small_patch16_dinov3': {'img_size': IMAGE_SIZE},
    'vit_small_patch16_384': {'img_size': IMAGE_SIZE},
}
EMBEDDING_SIZE = 256
BATCH_SIZE = 8 if SMOKE else 32
EPOCHS = 2 if SMOKE else 30            # upper limit; early stopping normally ends training sooner
WARMUP_EPOCHS = 1 if SMOKE else 2      # head-only epochs before the last block is unfrozen
PATIENCE = 1 if SMOKE else 6
MIN_AP_GAIN = 0.005                    # a validation-AP gain smaller than this does not reset patience
ENCODER_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

# ---------------------------------------------------------------- degradations
AUG_PROB = 0.5                         # robust training: probability that a training photo is degraded
BLUR_SIGMA = (0.5, 5.0)                # Gaussian blur sigma (px at 384 px), sampled uniformly
DOWN_FACTOR = (1.5, 6.0)               # down- then up-sampling factor, sampled uniformly
TRAIN_COPIES = 2                       # two-stage fusion with robust models: degraded copies per training photo
CLEAN_WEIGHT = 0.5                     # weights per training photo: clean 0.5 + 2 copies x 0.25 = 1, so every photo
COPY_WEIGHT = 0.25                     # counts once (same LR regularisation as B / C1) and clean : degraded = 1 : 1

SEEN_CONDITIONS = ['blur_s1', 'blur_s2', 'blur_s3', 'blur_s5', 'down_x2', 'down_x3', 'down_x4', 'down_x6']
UNSEEN_CONDITIONS = ['jpeg_q30', 'jpeg_q15', 'jpeg_q8', 'noise_s10', 'noise_s20']   # degradation types never trained on
CONTRAST_CONDITIONS = ['contrast_c0.5', 'contrast_c0.3']   # far stronger than the +/-10% colour jitter of C1 / C2
TEST_CONDITIONS = ['clean'] + SEEN_CONDITIONS + UNSEEN_CONDITIONS + CONTRAST_CONDITIONS
VAL_CONDITIONS = ['clean'] + SEEN_CONDITIONS   # validation photos are also scored under blur / downsampling
STRONGEST_CONDITIONS = ['blur_s5', 'down_x6']
MILDEST_CONDITIONS = ['blur_s1', 'down_x2']
DOSE_MIN_DROP = 0.05   # step 0: the strongest levels must lower the image-only validation AP by at least this much
BLUR_CONDITIONS = [c for c in SEEN_CONDITIONS if c.startswith('blur')]
DOWN_CONDITIONS = [c for c in SEEN_CONDITIONS if c.startswith('down')]

# ---------------------------------------------------------------- analysis
BOOTSTRAP = 30 if SMOKE else 5000      # patient-level bootstrap resamples (same draws for every model)
PRIMARY_LEVEL = 100 * (1 - 0.05 / 3)   # 98.33% Bonferroni intervals for the three primary comparisons
LR_C = 1.0                             # fusion LR: L2, C = 1, balanced class weights, no tuning
LR_C_SENSITIVITY = [0.1, 10.0]         # sensitivity analysis only (experiment B, image and image+all)
SHUFFLE_REPEATS = 10                   # two-stage negative control: number of clinical permutations
PERM_REPEATS = 10                      # permutation importance: permutations per clinical feature
PERM_CONDITIONS = ['clean'] + SEEN_CONDITIONS   # test conditions scored with permuted features
SHARPNESS_GROUPS = 3                   # natural-quality analysis: clean test photos split into sharpness thirds

# ---------------------------------------------------------------- experiments
# Each model set = one kind of network, trained for every backbone x fold (4 x 4 = 16 networks).
#   clinical: none | real | shuffled (negative control: clinical vectors exchanged between patients)
#   aug:      none | basic (flip, rotation, colour jitter) | both (basic + random blur / downsampling)
MODEL_SETS = {
    'B_image':  dict(experiment='B',  setting='image',          clinical='none',     aug='none'),
    'B_clin':   dict(experiment='B',  setting='image+all',      clinical='real',     aug='none'),
    'B_shuf':   dict(experiment='B',  setting='image+shuffled', clinical='shuffled', aug='none'),
    'C1_image': dict(experiment='C1', setting='image',          clinical='none',     aug='basic'),
    'C1_clin':  dict(experiment='C1', setting='image+all',      clinical='real',     aug='basic'),
    'C2_image': dict(experiment='C2', setting='image',          clinical='none',     aug='both'),
    'C2_clin':  dict(experiment='C2', setting='image+all',      clinical='real',     aug='both'),
}
STAGE_MODEL_SETS = {'B': ['B_image', 'B_clin', 'B_shuf'], 'C': ['C1_image', 'C1_clin', 'C2_image', 'C2_clin']}
# Two-stage fusion uses the image-only networks of each experiment.
FEATURE_VARIANTS = {'B': 'B_image', 'C1': 'C1_image', 'C2': 'C2_image',
                    'Bj': 'B_clin'}   # Bj: image part of the end-to-end networks (P2 decomposition)


TRAINING_FILES = ['common.py', 'degrade.py', 'nets.py', 'train.py', 'build_cache.py']
DOWNSTREAM_FILES = ['extract_features.py', 'predict_endtoend.py', 'fit_fusion.py']


def _digest(files, with_data):
    digest = hashlib.sha256()
    for name in files:
        digest.update((ROOT / 'src' / name).read_bytes())
    if with_data:
        for path in (DIABETIC_CSV, INADEQUATE_CSV):
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def training_signature():
    """Fingerprint of everything the networks and caches depend on: settings, degradations, network and
    training code, cache code, dataset CSVs. A change requires retraining (run_all.py --force)."""
    return _digest(TRAINING_FILES, with_data=True)


def downstream_signature():
    """Fingerprint of the code that turns trained networks into predictions (features, fusion, end-to-end
    scoring). A change makes run_all.py recompute those steps while keeping the trained networks.
    evaluate_all.py is not included: the evaluation is recomputed at every run anyway."""
    return _digest(DOWNSTREAM_FILES, with_data=False)


def permuted_feature(frame, feature, repeat):
    """Permutation importance: within each split, every patient receives another patient's value of one clinical
    feature (raw value, so its missing indicator moves with it); the other features stay in place."""
    out = frame.copy()
    for split, part in frame.groupby('split'):
        rng = np.random.default_rng(stable_seed('perm', feature, split, repeat))
        first = part.drop_duplicates('patient_id').set_index('patient_id')[feature].sort_index()
        patients = first.index.to_numpy()
        donor = dict(zip(patients, rng.permutation(patients)))
        out.loc[part.index, feature] = first.loc[part.patient_id.map(donor).to_numpy()].to_numpy()
    return out


def eval_conditions(split):
    """Conditions under which the photos of a split are scored."""
    return TEST_CONDITIONS if split == 'test' else VAL_CONDITIONS if split == 'val' else ['clean']


def degradation_family(aug):
    return aug if aug in ('both', 'blur', 'down') else None


def uses_basic_aug(aug):
    return aug != 'none'


def model_dir(model_set, backbone, fold):
    return MODELS_DIR / model_set / backbone / f'fold{fold}'


def stable_seed(*parts):
    return int(hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- data loading
def load_table(include_inadequate=True):
    """All photos with split, fold, label and clinical features (one row per photo, sorted by image_id)."""
    dtype = {'image_id': str, 'patient_id': str, 'split': str}
    frame = pd.read_csv(DIABETIC_CSV, dtype=dtype)
    if include_inadequate and INADEQUATE_CSV.is_file():
        frame = pd.concat([frame, pd.read_csv(INADEQUATE_CSV, dtype=dtype)], ignore_index=True)
    frame['fold'] = frame['fold'].astype(int)
    frame[TARGET] = frame[TARGET].astype(int)
    for c in CLINICAL:
        frame[c] = pd.to_numeric(frame[c], errors='raise').astype(float)
    if frame.image_id.duplicated().any():
        raise ValueError('Repeated image_id in the dataset CSVs.')
    frame = resolve_conflicts(frame)
    if SMOKE:  # three photos per split x fold x class
        frame = pd.concat([g.sample(min(len(g), 3), random_state=SEED)
                           for _, g in frame.groupby(['split', 'fold', TARGET])])
    return frame.sort_values('image_id').reset_index(drop=True)


AGE_TOLERANCE = 3   # years; photos of one patient whose ages differ by more cannot all be right


def clinical_conflicts(frame):
    """Patients whose records contradict each other in BRSET itself (checked against label_brset.csv): sex that
    differs between photos, or ages more than AGE_TOLERANCE years apart. Returns {field: [patient_id, ...]}."""
    grouped = frame.groupby('patient_id')
    sex = grouped.sex.nunique()
    age = grouped.age.agg(lambda a: a.max() - a.min())
    return {'sex': sorted(sex[sex > 1].index), 'age': sorted(age[age > AGE_TOLERANCE].index)}


def resolve_conflicts(frame):
    """Pre-specified rule: a field that contradicts itself within a patient is set to missing for all of that
    patient's photos (it is then imputed and flagged by the missing indicator, like any other missing value)."""
    frame = frame.copy()
    for field, patients in clinical_conflicts(frame).items():
        frame.loc[frame.patient_id.isin(patients), field] = np.nan
    return frame


def shuffled_clinical(frame, seed=SEED, repeat=0):
    """Negative control: every patient receives another patient's whole clinical vector (permutation number `repeat` of
    patients within each split), so the features keep their distribution but carry no information about
    this patient."""
    out = frame.copy()
    for split, part in frame.groupby('split'):
        rng = np.random.default_rng(stable_seed('shuffle', seed, split, repeat))   # one fixed permutation per split
        first = part.drop_duplicates('patient_id').set_index('patient_id')[CLINICAL].sort_index()
        patients = first.index.to_numpy()
        donor = dict(zip(patients, rng.permutation(patients)))
        out.loc[part.index, CLINICAL] = first.loc[part.patient_id.map(donor).to_numpy()].to_numpy()
    return out


class ClinicalEncoder:
    """Clinical preprocessing shared by both fusion pipelines, fitted on training photos only: median /
    most-frequent imputation, standardisation of continuous features, and a missing-value indicator for every
    feature with missing values in the training data. The fitted state is saved with each model."""

    def __init__(self, state=None):
        self.state = state

    def fit(self, frame):
        cols = {c: frame[c].to_numpy(dtype=float) for c in CLINICAL}
        median = {c: float(np.nanmedian(cols[c])) if np.isfinite(cols[c]).any() else 0.0 for c in CLINICAL_CONTINUOUS}
        filled = {c: np.where(np.isnan(cols[c]), median[c], cols[c]) for c in CLINICAL_CONTINUOUS}
        mode = {}
        for c in CLINICAL_BINARY:
            values = cols[c][~np.isnan(cols[c])].astype(int)
            mode[c] = float(np.bincount(values).argmax()) if len(values) else 0.0
        self.state = {
            'median': median, 'mode': mode,
            'mean': {c: float(filled[c].mean()) for c in CLINICAL_CONTINUOUS},
            'std': {c: float(filled[c].std() or 1.0) for c in CLINICAL_CONTINUOUS},
            'indicators': [c for c in CLINICAL if np.isnan(cols[c]).any()],
        }
        return self

    @property
    def names(self):
        return CLINICAL_CONTINUOUS + CLINICAL_BINARY + [f'{c}_missing' for c in self.state['indicators']]

    def transform(self, frame):
        st, parts = self.state, []
        for c in CLINICAL_CONTINUOUS:
            v = frame[c].to_numpy(dtype=float)
            parts.append((np.where(np.isnan(v), st['median'][c], v) - st['mean'][c]) / st['std'][c])
        for c in CLINICAL_BINARY:
            v = frame[c].to_numpy(dtype=float)
            parts.append(np.where(np.isnan(v), st['mode'][c], v))
        for c in st['indicators']:
            parts.append(np.isnan(frame[c].to_numpy(dtype=float)).astype(float))
        return np.stack(parts, axis=1).astype(np.float32)

    def columns_for(self, features):
        """Positions of the encoded columns that belong to the given raw clinical features."""
        return [i for i, n in enumerate(self.names) if n in features or n.replace('_missing', '') in features]


# ---------------------------------------------------------------- metrics
def average_precision(y, s):
    """Step-wise AP with ties grouped (same definition as sklearn.metrics.average_precision_score)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if y.min() == y.max():
        return float('nan')
    order = np.argsort(-s, kind='mergesort')
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -np.inf   # sklearn adds an infinite first threshold
    return float(thr[np.argmax(j)])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
