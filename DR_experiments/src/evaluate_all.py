"""Evaluate every prediction file with one shared patient-level bootstrap.

The results are organised as a ladder; key_results.csv carries the layer of every summary.
  Step 0  checks on VALIDATION photos, before any test performance is looked at (not results):
          dose check - do the strongest blur / downsampling levels lower the image-only AP enough for the
          experiment to be informative? - and a rehearsal of the main analysis on validation photos
  Layer 1 each modality alone: clinical only, image only, and how much the image loses under degradation
  Layer 2 does clinical information compensate? (standard training; the fusion weights are fixed, so any
          compensation is passive): gain, P1, trend, negative control, fusion vs clinical only
  Layer 3 how to fuse and which features: P2 and its decomposition, feature analyses, sensitivity to C
  Layer 4 after robust training (C1, C2): is clinical information still needed? (2 x 2 table)
  Layer 5 quality-aware fusion (D): can the fusion shift weight actively?

For each model (pipeline x experiment x backbone x setting x classifier), evaluation set and condition:
AP (primary metric), ROC-AUC, sensitivity / specificity at the Youden threshold chosen on clean validation
photos, raw Brier score and ECE (appendix: class-weighted training can move the raw probabilities away from
the actual risk, so calibration has to be assessed rather than assumed), and the spread of AP over the four
fold networks. A resampled patient brings all of his or her photos, conditions and model predictions, so every
difference has a paired CI. Summaries are computed per backbone and then averaged over the backbones.

Three pre-specified primary comparisons, judged with Bonferroni 98.33% intervals:
  P1 (two-stage), P1 (end-to-end): mean gain over the 8 blur / downsampling conditions minus the gain on
      clean photos, where gain = AP(image + clinical) - AP(image only)
  P2: mean over the 8 conditions of AP(two-stage image+all) - AP(end-to-end image+all)
Everything else is exploratory and reported with 95% intervals, including
  * natural quality variation: clean test photos split into sharpness thirds (q_lap measured on the photo)
  * DR severity (table_severity.csv): by ICDR grade, the net change in detection rate caused by degradation
    and by adding clinical features, split into photos newly flagged and photos no longer flagged (net =
    newly flagged - no longer flagged); read every detection gain together with the change in false
    positives (no-DR row), because image-only and fusion models have different thresholds
  * permutation importance of each clinical feature in both pipelines (feature_results.csv): the model's
    reliance on that feature together with its missingness pattern; features are correlated, so a small
    value does not mean the feature carries no information. The spread over the PERM_REPEATS permutations
    is reported separately (the patient bootstrap does not cover it).
  * performance decomposition of P2 with the hybrid pipeline as reference (two-stage LR on the image part
    of the end-to-end networks' logit): P2 = [two-stage - hybrid] + [hybrid - end-to-end]. The two terms
    point to the training steps where the difference may arise; they do not isolate causes.

Before anything is computed the predictions are checked for completeness: every expected model (XGBoost
included), all four backbones, every evaluation photo and condition exactly once, labels as in the dataset,
finite probabilities, four finite fold probabilities whose mean is the reported probability (all models
except the clinical-only ones), and the sharpness table needed for the natural-quality analysis.
The final evaluation stops if anything is missing or duplicated (--allow-incomplete turns this into a
warning). A backbone average is only reported when all requested backbones are present.

--val-only evaluates validation photos only (step 0); it checks only the experiments that have predictions
so far. --dose-gate makes a failed dose check end the program with exit code 3. Outputs: results/summary/.
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C
import degrade

KEY = ['pipeline', 'experiment', 'backbone', 'setting', 'classifier']
MAIN_CLF = {'two_stage': 'lr', 'end_to_end': 'joint', 'clinical_only': 'lr', 'late_average': 'mean', 'hybrid': 'lr'}
PIPELINES = ['two_stage', 'end_to_end']
PIPE_LABEL = {'two_stage': 'two-stage', 'end_to_end': 'end-to-end'}
SHUFFLED = {'two_stage': ['image+shuffled'] + [f'image+shuffled_r{r}' for r in range(1, C.SHUFFLE_REPEATS)],
            'end_to_end': ['image+shuffled']}
SEEN, UNSEEN, CONTRAST = C.SEEN_CONDITIONS, C.UNSEEN_CONDITIONS, C.CONTRAST_CONDITIONS
SETS = {'clean': ('test', ['clean'], 'clean'), 'seen': ('test', SEEN, '8 blur/downsampling'),
        'unseen': ('test', UNSEEN, '5 JPEG/noise'), 'contrast': ('test', CONTRAST, '2 low contrast'),
        'inadequate': ('inadequate', ['clean'], 'inadequate set'), 'val': ('val', ['clean'], 'validation (clean)'),
        'val_seen': ('val', SEEN, 'validation, 8 blur/downsampling'),
        'sharp_low': ('test_sharp_low', ['clean'], 'clean test, least sharp third'),
        'sharp_high': ('test_sharp_high', ['clean'], 'clean test, sharpest third')}
PERM_SETTINGS = {f: [f'perm:{f}:r{r}' for r in range(C.PERM_REPEATS)] for f in C.CLINICAL}
LAYERS = {0: 'Step 0 - checks on validation photos (not results)',
          1: 'Layer 1 - each modality alone',
          2: 'Layer 2 - does clinical information compensate? (standard training, fixed fusion weights)',
          3: 'Layer 3 - how to fuse, and which features',
          4: 'Layer 4 - after robust training: is clinical information still needed?',
          5: 'Layer 5 - quality-aware fusion: can the fusion shift weight actively?'}
PERM_REFERENCE = 'perm:none'   # unpermuted predictions computed exactly like the permuted ones
BOOT_SEED = {'test': 0, 'inadequate': 1, 'val': 2}


def load_predictions():
    files = [f for f in sorted(C.PREDICTIONS_DIR.rglob('*.csv.gz')) if not f.name.startswith('.')]
    if not files:
        raise SystemExit(f'No prediction files under {C.PREDICTIONS_DIR}.')
    return pd.concat([pd.read_csv(f, dtype={'image_id': str, 'patient_id': str}, keep_default_na=False,
                                  na_values=['', 'nan', 'NaN']) for f in files], ignore_index=True)


STANDARD_SETTINGS = ['image', 'image+all'] + [f'image+{c}' for c in C.CLINICAL] + [f'image+all-{c}' for c in C.CLINICAL]
D_SETTINGS = ['image+q', 'image+all+q', 'image+all+qL', 'image+all+qint', 'image+all+qoracle']


def expected_models(backbones):
    """{experiment: [model key, ...]} for the full design."""
    exp = {'A': [('clinical_only', 'A', 'no_image', 'clinical', clf) for clf in ['lr', 'rf', 'xgb']]}
    for b in backbones:
        exp.setdefault('B', []).extend(
            [('two_stage', 'B', b, st, 'lr') for st in STANDARD_SETTINGS + ['image+missing'] + SHUFFLED['two_stage']]
            + [('two_stage', 'B', b, st, f'lr_c{c:g}') for c in C.LR_C_SENSITIVITY for st in ['image', 'image+all']]
            + [('end_to_end', 'B', b, st, 'joint') for st in ['image', 'image+all', 'image+shuffled']]
            + [('late_average', 'B', b, 'late_average', 'mean')]
            + [(pipe, 'B', b, st, MAIN_CLF[pipe]) for pipe in PIPELINES for st in [PERM_REFERENCE] + sum(PERM_SETTINGS.values(), [])]
            + [('hybrid', 'B', b, st, 'lr') for st in ['image', 'image+all']])
        for v in ['C1', 'C2']:
            exp.setdefault(v, []).extend([('two_stage', v, b, st, 'lr') for st in STANDARD_SETTINGS]
                                         + [('end_to_end', v, b, st, 'joint') for st in ['image', 'image+all']])
        exp.setdefault('D', []).extend([('two_stage', 'D', b, st, 'lr') for st in D_SETTINGS])
    return exp


def expected_cells(key, eval_sets):
    """(eval_set, condition) pairs a model must cover. The oracle model exists only where the true severity is known."""
    cells = []
    if key[3].startswith('perm:'):
        return [('test', c) for c in C.PERM_CONDITIONS] if 'test' in eval_sets else []
    for es in eval_sets:
        conds = C.eval_conditions(es)
        if key[3] == 'image+all+qoracle':
            if es == 'inadequate':
                continue
            conds = [c for c in conds if c == 'clean' or c in SEEN]
        cells += [(es, c) for c in conds]
    return cells


def check_complete(pred, table, backbones, eval_sets, final):
    """Returns a list of problems (empty when the predictions are complete and consistent)."""
    problems = []
    expected = expected_models(backbones)
    present_experiments = set(pred.experiment)
    photos = {es: set(table.loc[table.split == es, 'image_id']) for es in eval_sets}
    labels = table.set_index('image_id')[C.TARGET]
    sub = pred[pred.eval_set.isin(eval_sets)]
    groups = {k: g for k, g in sub.groupby(KEY + ['eval_set', 'condition'])}
    for exp, keys in expected.items():
        if not final and exp not in present_experiments:
            continue
        for key in keys:
            for es, cond in expected_cells(key, eval_sets):
                g = groups.get((*key, es, cond))
                name = f'{"/".join(key)} [{es}, {cond}]'
                if g is None:
                    problems.append(f'missing: {name}')
                    continue
                if g.image_id.duplicated().any():
                    problems.append(f'duplicated photos: {name}')
                if set(g.image_id) != photos[es]:
                    problems.append(f'wrong photo set ({g.image_id.nunique()} of {len(photos[es])}): {name}')
                if not np.isfinite(g.prob.to_numpy(dtype=float)).all():
                    problems.append(f'non-finite probabilities: {name}')
                if (g.y.to_numpy() != labels.reindex(g.image_id).to_numpy()).any():
                    problems.append(f'labels differ from the dataset CSV: {name}')
                if key[0] != 'clinical_only':
                    folds = g[[f'prob_f{k}' for k in range(C.FOLDS)]].to_numpy(dtype=float)
                    if not np.isfinite(folds).all():
                        problems.append(f'missing or non-finite fold probabilities: {name}')
                    elif np.abs(folds.mean(axis=1) - g.prob.to_numpy(dtype=float)).max() > 1e-6:
                        problems.append(f'probability is not the mean of the four fold probabilities: {name}')
    if final and 'test' in eval_sets:
        quality = C.CACHE / f'quality_{C.IMAGE_SIZE}.csv'
        if not quality.is_file():
            problems.append(f'missing sharpness table {quality} (rebuild the cache) - needed for the natural-quality analysis')
        else:
            q = pd.read_csv(quality, dtype={'image_id': str})
            q = q[q.condition == 'clean'].set_index('image_id').q_lap
            if q.reindex(sorted(photos['test'])).isna().any():
                problems.append('sharpness table does not cover every clean test photo')
    return problems


class Bootstrap:
    """Patient-level resampling expressed as photo weights, shared by all models of one evaluation set."""

    def __init__(self, photos, seed):
        self.y = photos.y.to_numpy().astype(float)
        codes, patients = pd.factorize(photos.patient_id)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(patients), size=(C.BOOTSTRAP, len(patients)))
        counts = np.stack([np.bincount(d, minlength=len(patients)) for d in draws])
        weights = counts[:, codes].astype(np.float32)
        positives = weights @ self.y
        self.W = weights[(positives > 0) & (positives < weights.sum(axis=1))]

    def ap(self, scores, mask=None):
        """AP for every bootstrap draw; with a mask, on that subset of photos (NaN where a draw has no positive)."""
        y_all, W = self.y, self.W
        if mask is not None:
            scores, y_all, W = scores[mask], y_all[mask], W[:, mask]
        order = np.argsort(-scores, kind='mergesort')
        s, y = scores[order], y_all[order]
        last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
        Wo = W[:, order].astype(float)
        tp = np.cumsum(Wo * y, axis=1)[:, last]
        nn = np.cumsum(Wo, axis=1)[:, last]
        precision = np.divide(tp, nn, out=np.zeros_like(tp), where=nn > 0)
        dtp = np.diff(np.concatenate([np.zeros((len(tp), 1)), tp], axis=1), axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return (dtp * precision).sum(axis=1) / tp[:, -1]


def ece(y, p, bins=10):
    which = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def ci(values, level=95.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    lo, hi = np.percentile(values, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def sharpness_groups(photos):
    """Masks over the test photos: least sharp and sharpest third by q_lap of the clean photo."""
    path = C.CACHE / f'quality_{C.IMAGE_SIZE}.csv'
    if not path.is_file():
        return {}
    q = pd.read_csv(path, dtype={'image_id': str})
    q = q[q.condition == 'clean'].set_index('image_id').q_lap.reindex(photos.image_id).to_numpy()
    if np.isnan(q).any():
        return {}
    low, high = np.quantile(q, [1 / C.SHARPNESS_GROUPS, 1 - 1 / C.SHARPNESS_GROUPS])
    return {'test_sharp_low': q <= low, 'test_sharp_high': q >= high}


def metric_row(key, es, cond, y, p, draws, thr, fold_aps):
    positive = p >= thr
    lo, hi = ci(draws)
    has_both = 0 < y.sum() < len(y)
    return {**dict(zip(KEY, key)), 'eval_set': es, 'condition': cond, 'n_photos': len(y), 'n_positive': int(y.sum()),
            'ap': C.average_precision(y, p) if has_both else np.nan, 'ap_lo': lo, 'ap_hi': hi,
            'roc_auc': float(roc_auc_score(y, p)) if has_both else np.nan, 'threshold': thr,
            'sensitivity': float((positive & (y == 1)).sum() / max(1, (y == 1).sum())) if np.isfinite(thr) else np.nan,
            'specificity': float((~positive & (y == 0)).sum() / max(1, (y == 0).sum())) if np.isfinite(thr) else np.nan,
            'brier_raw': float(np.mean((p - y) ** 2)), 'ece_raw': ece(y, p),
            'fold_ap_sd': float(np.std(fold_aps)) if len(fold_aps) > 1 else np.nan}


def evaluate(pred, table, eval_sets):
    photo_sets = {}
    for es in eval_sets:
        photos = table[table.split == es][['image_id', 'patient_id', C.TARGET]].rename(columns={C.TARGET: 'y'})
        if len(photos):
            photo_sets[es] = (photos.reset_index(drop=True), Bootstrap(photos, C.SEED + BOOT_SEED[es]))
    groups = sharpness_groups(photo_sets['test'][0]) if 'test' in photo_sets else {}
    thresholds = {key: C.youden_threshold(g.y.to_numpy(), g.prob.to_numpy())
                  for key, g in pred[(pred.eval_set == 'val') & (pred.condition == 'clean')].groupby(KEY)}
    rows, boot = [], {}
    fold_cols = [f'prob_f{k}' for k in range(C.FOLDS)]
    for full_key, g in pred[pred.eval_set.isin(list(photo_sets))].groupby(KEY + ['eval_set', 'condition']):
        key, es, cond = tuple(full_key[:5]), full_key[5], full_key[6]
        photos, bs = photo_sets[es]
        if g.image_id.duplicated().any():
            raise ValueError(f'Duplicated predictions for {full_key}.')
        g = g.set_index('image_id').reindex(photos.image_id)
        if g.prob.isna().any():
            continue
        y, p = photos.y.to_numpy(), g.prob.to_numpy(dtype=float)
        thr = thresholds.get(key, np.nan)
        folds = [g[c].to_numpy(float) for c in fold_cols if g[c].notna().all()]
        draws = bs.ap(p).astype(np.float32)
        boot[(key, es, cond)] = draws
        rows.append(metric_row(key, es, cond, y, p, draws, thr, [C.average_precision(y, f) for f in folds]))
        if es == 'test' and cond == 'clean':
            for group, mask in groups.items():
                sub_draws = bs.ap(p, mask).astype(np.float32)
                boot[(key, group, cond)] = sub_draws
                rows.append(metric_row(key, group, cond, y[mask], p[mask], sub_draws, thr,
                                       [C.average_precision(y[mask], f[mask]) for f in folds]))
    return pd.DataFrame(rows), boot, {'photo_sets': photo_sets, 'thresholds': thresholds}


class Summary:
    """Point estimates and bootstrap draws of AP-based quantities. Every method returns (point, draws) or None."""

    def __init__(self, metrics, boot, backbones):
        self.metrics = metrics.set_index(KEY + ['eval_set', 'condition'])
        self.boot = boot
        self.backbones = list(backbones)

    def get(self, key, es, cond):
        if (key, es, cond) not in self.boot:
            return None
        return float(self.metrics.loc[(*key, es, cond), 'ap']), self.boot[(key, es, cond)].astype(float)

    @staticmethod
    def mean(items):
        if not items or any(i is None for i in items):
            return None
        return float(np.mean([i[0] for i in items])), np.mean([i[1] for i in items], axis=0)

    @staticmethod
    def sub(a, b):
        if a is None or b is None:
            return None
        return a[0] - b[0], a[1] - b[1]

    def ap(self, pipe, exp, b, setting, es, conds, clf=None):
        """Mean AP over the conditions (and over the settings when a list is given, e.g. shuffle repeats)."""
        backbone = 'no_image' if pipe == 'clinical_only' else b
        settings = [setting] if isinstance(setting, str) else setting
        return self.mean([self.get((pipe, exp, backbone, s, clf or MAIN_CLF[pipe]), es, c) for s in settings for c in conds])

    def over_backbones(self, fn):
        per = {b: fn(b) for b in self.backbones}
        per = {b: v for b, v in per.items() if v is not None}
        if len(per) < len(self.backbones):   # never average over an incomplete set of backbones
            return None
        return {'point': float(np.mean([v[0] for v in per.values()])),
                'boot': np.nanmean([v[1] for v in per.values()], axis=0),
                'per_backbone': {b: round(v[0], 4) for b, v in per.items()}}

    def gain(self, pipe, exp, b, es, conds, setting='image+all', clf=None):
        return self.sub(self.ap(pipe, exp, b, setting, es, conds, clf), self.ap(pipe, exp, b, 'image', es, conds, clf))

    def delta(self, pipe, exp, b, conds, setting='image+all', clf=None, es='test'):
        return self.sub(self.gain(pipe, exp, b, es, conds, setting, clf), self.gain(pipe, exp, b, es, ['clean'], setting, clf))

    def trend(self, pipe, exp, b, conds, setting='image+all'):
        """Least-squares slope of the gain against severity (0 = clean ... 1 = strongest level)."""
        conds = ['clean'] + list(conds)
        x = np.array([degrade.severity(*degrade.parse(c)) for c in conds])
        w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        gains = [self.gain(pipe, exp, b, 'test', [c], setting) for c in conds]
        if any(g is None for g in gains):
            return None
        return float(sum(wi * g[0] for wi, g in zip(w, gains))), sum(wi * g[1] for wi, g in zip(w, gains))

    def trend_both(self, pipe, exp, b, setting='image+all'):
        return self.mean([self.trend(pipe, exp, b, C.BLUR_CONDITIONS, setting), self.trend(pipe, exp, b, C.DOWN_CONDITIONS, setting)])

    def headroom(self, pipe, exp, b, conds):
        """sum(gain) / sum(1 - AP image) over the conditions (ratio of sums, stable when AP image is near 1)."""
        gain, base = self.gain(pipe, exp, b, 'test', conds), self.ap(pipe, exp, b, 'image', 'test', conds)
        if gain is None or base is None or 1 - base[0] < 0.01:
            return None
        room = 1 - base[1]
        return gain[0] / (1 - base[0]), np.where(room >= 0.01, gain[1] / np.maximum(room, 1e-6), np.nan)

    def compare(self, a, b, es, conds):
        """a, b = (pipeline, experiment, setting[, classifier]); AP(a) - AP(b), averaged over backbones."""
        def one(bb):
            return self.sub(self.ap(a[0], a[1], bb, a[2], es, conds, a[3] if len(a) > 3 else None),
                            self.ap(b[0], b[1], bb, b[2], es, conds, b[3] if len(b) > 3 else None))
        return self.over_backbones(one)


def interpret(lo, hi):
    if not np.isfinite(lo):
        return 'no CI'
    return 'CI above 0' if lo > 0 else 'CI below 0' if hi < 0 else 'CI includes 0'


def layer_of(rid, group):
    if group == 'check':
        return 0
    if rid.startswith('L1_'):
        return 1
    if group == 'C':
        return 4
    if group == 'D':
        return 5
    if group == 'Sensitivity' or rid.startswith(('P2', 'B_P1_hybrid', 'B_missing', 'B_values')):
        return 3
    return 2


def key_results(s):
    out = []

    def add(rid, group, label, res, conds, primary=False, note='', level_only=False):
        if res is None:
            return
        level = C.PRIMARY_LEVEL if primary else 95.0
        lo, hi = ci(res['boot'], level)
        out.append({'layer': layer_of(rid, group), 'id': rid, 'group': group, 'primary': primary, 'label': label, 'conditions': conds,
                    'estimate': round(res['point'], 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                    'ci_level': round(level, 2), 'reading': 'level' if level_only else interpret(lo, hi), 'n_backbones': len(res['per_backbone']),
                    'per_backbone': json.dumps(res['per_backbone']), 'note': note})

    def repeat_range(fn):
        points = [r['point'] for r in map(fn, SHUFFLED['two_stage']) if r is not None]
        return f'{len(points)} permutations: {min(points):+.4f} to {max(points):+.4f}' if len(points) > 1 else ''

    image, clinical_only = ('end_to_end', 'B', 'image'), ('clinical_only', 'A', 'clinical')

    def level(model, es, conds):
        return s.over_backbones(lambda b: s.ap(model[0], model[1], b, model[2], es, conds))

    # ---- step 0: checks on validation photos, before any test performance is looked at
    add('VAL_image_ap_clean', 'check', 'Image-only AP, clean validation photos', level(image, 'val', ['clean']), 'validation (clean)', level_only=True)
    add('VAL_image_ap_mildest', 'check', 'Image-only AP, mildest levels (blur sigma 1, downsampling x2)', level(image, 'val', C.MILDEST_CONDITIONS), 'validation', level_only=True)
    add('VAL_image_ap_strongest', 'check', 'Image-only AP, strongest levels (blur sigma 5, downsampling x6)', level(image, 'val', C.STRONGEST_CONDITIONS), 'validation', level_only=True)
    add('VAL_dose_drop', 'check', f'Dose check (development check, mean of backbones and of blur sigma 5 / downsampling x6): image-only AP, clean minus strongest levels (rule of thumb: at least {C.DOSE_MIN_DROP})',
        s.over_backbones(lambda b: s.sub(s.ap('end_to_end', 'B', b, 'image', 'val', ['clean']),
                                         s.ap('end_to_end', 'B', b, 'image', 'val', C.STRONGEST_CONDITIONS))), 'validation')
    add('VAL_clinical_ap', 'check', 'Clinical-only AP (LR), validation photos', level(clinical_only, 'val', ['clean']), 'validation (clean)', level_only=True)
    for pipe in PIPELINES:
        P = PIPE_LABEL[pipe]
        add(f'VAL_gain_clean_{pipe}', 'check', f'Rehearsal on validation: gain from clinical features ({P})',
            s.over_backbones(lambda b: s.gain(pipe, 'B', b, 'val', ['clean'])), 'validation (clean)')
        add(f'VAL_gain_seen_{pipe}', 'check', f'Rehearsal on validation: gain from clinical features ({P})',
            s.over_backbones(lambda b: s.gain(pipe, 'B', b, 'val', SEEN)), 'validation, 8 blur/downsampling')
        add(f'VAL_P1_{pipe}', 'check', f'Rehearsal on validation: P1 ({P})',
            s.over_backbones(lambda b: s.delta(pipe, 'B', b, SEEN, es='val')), 'validation, 8 blur/downsampling vs clean')
        add(f'VAL_NC_gain_seen_{pipe}', 'check', f'Rehearsal on validation: shuffled-clinical gain ({P}); expected about 0',
            s.over_backbones(lambda b: s.gain(pipe, 'B', b, 'val', SEEN, SHUFFLED[pipe])), 'validation, 8 blur/downsampling')

    # ---- layer 1: each modality alone (test)
    add('L1_clinical_ap', 'A', 'Clinical-only AP (LR)', level(clinical_only, 'test', ['clean']), 'test', level_only=True)
    add('L1_image_ap_clean', 'A', 'Image-only AP', level(image, 'test', ['clean']), 'clean', level_only=True)
    add('L1_image_ap_seen', 'A', 'Image-only AP', level(image, 'test', SEEN), '8 blur/downsampling', level_only=True)
    add('L1_image_ap_strongest', 'A', 'Image-only AP', level(image, 'test', C.STRONGEST_CONDITIONS), 'strongest blur/downsampling', level_only=True)
    add('L1_image_loss_seen', 'A', 'What degradation costs the image model: AP clean minus AP under blur/downsampling',
        s.over_backbones(lambda b: s.sub(s.ap('end_to_end', 'B', b, 'image', 'test', ['clean']), s.ap('end_to_end', 'B', b, 'image', 'test', SEEN))), '8 blur/downsampling vs clean')
    add('L1_image_minus_clinical_clean', 'A', 'AP image only minus AP clinical only', s.compare(image, clinical_only, 'test', ['clean']), 'clean')
    add('L1_image_minus_clinical_strongest', 'A', 'AP image only minus AP clinical only', s.compare(image, clinical_only, 'test', C.STRONGEST_CONDITIONS), 'strongest blur/downsampling')

    # ---- B: primary comparisons and their supporting summaries
    for pipe in PIPELINES:
        P = PIPE_LABEL[pipe]
        add(f'P1_{pipe}', 'B', f'P1 ({P}): gain under blur/downsampling minus gain on clean photos',
            s.over_backbones(lambda b: s.delta(pipe, 'B', b, SEEN)), '8 blur/downsampling vs clean', True)
    add('P2', 'B', 'P2: AP two-stage minus AP end-to-end (image + clinical)',
        s.compare(('two_stage', 'B', 'image+all'), ('end_to_end', 'B', 'image+all'), 'test', SEEN), '8 blur/downsampling', True)
    for pipe in PIPELINES:
        P = PIPE_LABEL[pipe]
        add(f'B_trend_{pipe}', 'B', f'Trend ({P}): change in gain from clean to the strongest blur/downsampling (slope vs severity)',
            s.over_backbones(lambda b: s.trend_both(pipe, 'B', b)), 'clean + 8 blur/downsampling')
        for name, (es, conds, label) in SETS.items():
            if es == 'val':
                continue
            add(f'B_gain_{name}_{pipe}', 'B', f'Gain from clinical features ({P})',
                s.over_backbones(lambda b: s.gain(pipe, 'B', b, es, conds)), label)
        add(f'B_headroom_{pipe}', 'B', f'Headroom-normalised gain, sum(gain) / sum(1 - AP image) ({P})',
            s.over_backbones(lambda b: s.headroom(pipe, 'B', b, SEEN)), '8 blur/downsampling')
        for name in ['unseen', 'contrast']:
            add(f'B_delta_{name}_{pipe}', 'B', f'Gain under these conditions minus gain on clean ({P})',
                s.over_backbones(lambda b: s.delta(pipe, 'B', b, SETS[name][1])), SETS[name][2] + ' vs clean')
    # natural quality variation within the adequate test photos
    for pipe in PIPELINES:
        P = PIPE_LABEL[pipe]
        add(f'B_gain_sharp_low_minus_high_{pipe}', 'B', f'Natural quality: gain in the least sharp third minus gain in the sharpest third ({P})',
            s.over_backbones(lambda b: s.sub(s.gain(pipe, 'B', b, 'test_sharp_low', ['clean']), s.gain(pipe, 'B', b, 'test_sharp_high', ['clean']))),
            'clean test photos by measured sharpness')

    # P2 decomposition: P2 = (two-stage - hybrid) + (hybrid - end-to-end)
    for name in ['clean', 'seen']:
        es, conds, label = SETS[name]
        add(f'P2_part1_{name}', 'B', 'P2 decomposition, part 1: AP two-stage minus AP hybrid (same LR fitting; image network trained alone vs jointly, plus its head and checkpoint)',
            s.compare(('two_stage', 'B', 'image+all'), ('hybrid', 'B', 'image+all'), es, conds), label)
        add(f'P2_part2_{name}', 'B', 'P2 decomposition, part 2: AP hybrid minus AP end-to-end (same image network; out-of-fold LR refit vs joint head, regularisation, preprocessing)',
            s.compare(('hybrid', 'B', 'image+all'), ('end_to_end', 'B', 'image+all'), es, conds), label)
    add('B_P1_hybrid', 'B', 'P1 (hybrid): gain under blur/downsampling minus gain on clean photos',
        s.over_backbones(lambda b: s.delta('hybrid', 'B', b, SEEN)), '8 blur/downsampling vs clean')

    clinical = ('clinical_only', 'A', 'clinical')
    for pipe in PIPELINES:
        for name in ['clean', 'seen', 'unseen', 'contrast', 'inadequate']:
            es, conds, label = SETS[name]
            add(f'B_fusion_vs_clinical_{name}_{pipe}', 'B',
                f'AP(image + clinical) - AP(clinical only) ({PIPE_LABEL[pipe]}): additional performance over the clinical model',
                s.compare((pipe, 'B', 'image+all'), clinical, es, conds), label)
        add(f'B_image_vs_clinical_seen_{pipe}', 'B', f'AP(image only) - AP(clinical only) ({PIPE_LABEL[pipe]})',
            s.compare((pipe, 'B', 'image'), clinical, 'test', SEEN), '8 blur/downsampling')
    for name in ['clean', 'seen']:
        es, conds, label = SETS[name]
        add(f'B_missing_only_{name}', 'B', 'Missing indicators only: AP(image + missing indicators) - AP(image) (two-stage)',
            s.compare(('two_stage', 'B', 'image+missing'), ('two_stage', 'B', 'image'), es, conds), label)
        add(f'B_values_beyond_missing_{name}', 'B', 'Clinical values beyond missingness: AP(image + all) - AP(image + missing indicators) (two-stage)',
            s.compare(('two_stage', 'B', 'image+all'), ('two_stage', 'B', 'image+missing'), es, conds), label)
    for name in ['clean', 'unseen', 'contrast', 'inadequate']:
        es, conds, label = SETS[name]
        add(f'P2_{name}', 'B', 'AP two-stage minus AP end-to-end (image + clinical)',
            s.compare(('two_stage', 'B', 'image+all'), ('end_to_end', 'B', 'image+all'), es, conds), label)

    # ---- negative control: clinical vectors exchanged between patients
    for pipe in PIPELINES:
        P = PIPE_LABEL[pipe]
        for name in ['val', 'clean', 'seen']:
            es, conds, label = SETS[name]
            note = repeat_range(lambda sh: s.over_backbones(lambda b: s.gain(pipe, 'B', b, es, conds, sh))) if pipe == 'two_stage' else ''
            add(f'NC_gain_{name}_{pipe}', 'Negative control', f'Shuffled clinical: AP(image + shuffled) - AP(image) ({P}); expected about 0',
                s.over_backbones(lambda b: s.gain(pipe, 'B', b, es, conds, SHUFFLED[pipe])), label, note=note)
            add(f'NC_real_minus_shuffled_{name}_{pipe}', 'Negative control', f'AP(image + real clinical) - AP(image + shuffled clinical) ({P})',
                s.over_backbones(lambda b: s.sub(s.ap(pipe, 'B', b, 'image+all', es, conds), s.ap(pipe, 'B', b, SHUFFLED[pipe], es, conds))), label)

    # ---- A: baseline without learning; sensitivity of the fixed LR regularisation
    for name in ['clean', 'seen']:
        es, conds, label = SETS[name]
        add(f'A_late_average_{name}', 'A', 'AP late average minus AP two-stage LR (image + clinical)',
            s.compare(('late_average', 'B', 'late_average'), ('two_stage', 'B', 'image+all'), es, conds), label)
    for c in C.LR_C_SENSITIVITY:
        clf = f'lr_c{c:g}'
        add(f'SENS_P1_two_stage_{clf}', 'Sensitivity', f'P1 (two-stage) with LR C = {c:g} instead of {C.LR_C:g}',
            s.over_backbones(lambda b: s.delta('two_stage', 'B', b, SEEN, clf=clf)), '8 blur/downsampling vs clean')
        add(f'SENS_P2_{clf}', 'Sensitivity', f'P2 with LR C = {c:g} instead of {C.LR_C:g}',
            s.compare(('two_stage', 'B', 'image+all', clf), ('end_to_end', 'B', 'image+all'), 'test', SEEN), '8 blur/downsampling')

    # ---- C: robust training
    for v in ['C1', 'C2']:
        for pipe in PIPELINES:
            P = PIPE_LABEL[pipe]
            add(f'{v}_P1_{pipe}', 'C', f'{v}: gain under blur/downsampling minus gain on clean ({P})',
                s.over_backbones(lambda b: s.delta(pipe, v, b, SEEN)), '8 blur/downsampling vs clean')
            add(f'{v}_P1_change_{pipe}', 'C', f'{v}: change of that difference relative to standard training B ({P})',
                s.over_backbones(lambda b: s.sub(s.delta(pipe, v, b, SEEN), s.delta(pipe, 'B', b, SEEN))), '8 blur/downsampling vs clean')
            for name in ['clean', 'seen']:
                es, conds, label = SETS[name]
                add(f'{v}_gain_{name}_{pipe}', 'C', f'{v}: gain from clinical features ({P})',
                    s.over_backbones(lambda b: s.gain(pipe, v, b, es, conds)), label)
        add(f'{v}_P2', 'C', f'{v}: AP two-stage minus AP end-to-end (image + clinical)',
            s.compare(('two_stage', v, 'image+all'), ('end_to_end', v, 'image+all'), 'test', SEEN), '8 blur/downsampling')
        for name in ['clean', 'seen', 'unseen', 'contrast', 'inadequate']:
            es, conds, label = SETS[name]
            add(f'{v}_image_vs_B_{name}', 'C', f'{v}: image-only AP minus standard (B) image-only AP',
                s.compare(('end_to_end', v, 'image'), ('end_to_end', 'B', 'image'), es, conds), label)
    for name in ['clean', 'seen', 'unseen', 'contrast', 'inadequate']:
        es, conds, label = SETS[name]
        add(f'C2_vs_C1_image_{name}', 'C', 'Degradation augmentation beyond basic augmentation: image-only AP, C2 minus C1',
            s.compare(('end_to_end', 'C2', 'image'), ('end_to_end', 'C1', 'image'), es, conds), label)

    # ---- D: quality-aware two-stage fusion on the C2 logits
    plain, qint = ('two_stage', 'C2', 'image+all'), ('two_stage', 'D', 'image+all+qint')
    for name in ['clean', 'seen', 'unseen', 'contrast', 'inadequate']:
        es, conds, label = SETS[name]
        add(f'D_qint_vs_plain_{name}', 'D', 'Quality-interaction fusion (L, x, q, qL, qx) minus plain fusion (L, x)', s.compare(qint, plain, es, conds), label)
    for name in ['clean', 'seen']:
        es, conds, label = SETS[name]
        add(f'D_q_main_vs_plain_{name}', 'D', 'Quality main effect only (L, x, q) minus plain fusion (L, x)',
            s.compare(('two_stage', 'D', 'image+all+q'), plain, es, conds), label)
        add(f'D_qint_vs_q_main_{name}', 'D', 'Interactions beyond the main effect: (L, x, q, qL, qx) minus (L, x, q)',
            s.compare(qint, ('two_stage', 'D', 'image+all+q'), es, conds), label)
        add(f'D_qL_vs_q_main_{name}', 'D', 'Quality x image interaction: (L, x, q, qL) minus (L, x, q)',
            s.compare(('two_stage', 'D', 'image+all+qL'), ('two_stage', 'D', 'image+all+q'), es, conds), label)
        add(f'D_qx_beyond_qL_{name}', 'D', 'Quality x clinical interactions: (L, x, q, qL, qx) minus (L, x, q, qL)',
            s.compare(qint, ('two_stage', 'D', 'image+all+qL'), es, conds), label)
        add(f'D_image_q_vs_image_{name}', 'D', 'Quality without clinical features: (L, q, qL) minus image only (L)',
            s.compare(('two_stage', 'D', 'image+q'), ('two_stage', 'C2', 'image'), es, conds), label)
        add(f'D_qint_vs_image_q_{name}', 'D', 'Clinical features given quality: (L, x, q, qL, qx) minus (L, q, qL)',
            s.compare(qint, ('two_stage', 'D', 'image+q'), es, conds), label)
        add(f'D_oracle_vs_plain_{name}', 'D', 'Reference with known synthetic severity (not guaranteed to be an upper bound): oracle interactions minus plain fusion',
            s.compare(('two_stage', 'D', 'image+all+qoracle'), plain, es, conds), label)
    out = pd.DataFrame(out)
    return out.sort_values('layer', kind='stable').reset_index(drop=True) if len(out) else out


def feature_results(s):
    """Which clinical feature helps.
    two-stage: add-one gain AP(image + feature) - AP(image); drop-one loss AP(image + all) - AP(image + all but one)
    both pipelines: permutation importance AP(image + all) - AP(image + all with the feature permuted), mean of
    PERM_REPEATS permutations; and its change from clean photos to blur / downsampling."""
    rows = []

    def record(pipeline, exp, feature, quantity, label, res, spread=None):
        if res is None:
            return
        lo, hi = ci(res['boot'])
        rows.append({'pipeline': pipeline, 'experiment': exp, 'feature': feature, 'quantity': quantity, 'conditions': label,
                     'estimate': round(res['point'], 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4), 'reading': interpret(lo, hi),
                     'sd_over_permutations': spread})

    for exp in ['B', 'C1', 'C2']:
        for f in C.CLINICAL:
            for name in ['clean', 'seen']:
                es, conds, label = SETS[name]
                record('two-stage', exp, f, 'add-one gain: AP(image + feature) - AP(image)', label,
                       s.compare(('two_stage', exp, f'image+{f}'), ('two_stage', exp, 'image'), es, conds))
                record('two-stage', exp, f, 'drop-one loss: AP(image + all) - AP(image + all but feature)', label,
                       s.compare(('two_stage', exp, 'image+all'), ('two_stage', exp, f'image+all-{f}'), es, conds))
            record('two-stage', exp, f, 'add-one gain under blur/downsampling minus on clean', '8 blur/downsampling vs clean',
                   s.over_backbones(lambda b: s.delta('two_stage', exp, b, SEEN, f'image+{f}')))
    for pipe in PIPELINES:
        for f in C.CLINICAL:
            def importance(b, conds, pipe=pipe, f=f, settings=None):
                return s.sub(s.ap(pipe, 'B', b, PERM_REFERENCE, 'test', conds),
                             s.ap(pipe, 'B', b, settings or PERM_SETTINGS[f], 'test', conds))
            for name in ['clean', 'seen']:
                conds = SETS[name][1]
                per_perm = [s.over_backbones(lambda b, st=st: importance(b, conds, settings=[st])) for st in PERM_SETTINGS[f]]
                spread = round(float(np.std([r['point'] for r in per_perm])), 4) if all(r is not None for r in per_perm) else None
                record(PIPE_LABEL[pipe], 'B', f, 'permutation importance: AP(image + all) - AP(feature permuted), mean of permutations',
                       SETS[name][2], s.over_backbones(lambda b: importance(b, conds)), spread)
            record(PIPE_LABEL[pipe], 'B', f, 'permutation importance under blur/downsampling minus on clean', '8 blur/downsampling vs clean',
                   s.over_backbones(lambda b: s.sub(importance(b, SEEN), importance(b, ['clean']))))
    return pd.DataFrame(rows)


def condition_tables(s):
    """Mean AP over backbones for every condition (points only; CIs are in metrics.csv / the figures)."""
    index = C.TEST_CONDITIONS + ['inadequate (real)']

    def col(model):
        cells = [(('test', [c])) for c in C.TEST_CONDITIONS] + [('inadequate', ['clean'])]
        values = []
        for es, conds in cells:
            r = s.over_backbones(lambda b: s.ap(model[0], model[1], b, model[2], es, conds))
            values.append(np.nan if r is None else round(r['point'], 4))
        return values

    tables = {}
    b = {'clinical only': col(('clinical_only', 'A', 'clinical'))}
    for pipe in PIPELINES:
        for setting, label in [('image', 'image'), ('image+all', 'image+clinical'), ('image+shuffled', 'image+shuffled')]:
            b[f'{PIPE_LABEL[pipe]}: {label}'] = col((pipe, 'B', setting))
    tables['table_B_conditions'] = pd.DataFrame(b, index=index)
    tables['table_C_image'] = pd.DataFrame({f'{v} image only': col(('end_to_end', v, 'image')) for v in ['B', 'C1', 'C2']}, index=index)
    gains = {}
    for pipe in PIPELINES:
        for v in ['B', 'C1', 'C2']:
            gains[f'{PIPE_LABEL[pipe]} {v} gain'] = [round(x - y, 4) for x, y in zip(col((pipe, v, 'image+all')), col((pipe, v, 'image')))]
    tables['table_C_gain'] = pd.DataFrame(gains, index=index)
    d = {'C2 image (L)': col(('two_stage', 'C2', 'image')), 'image+q (L,q,qL)': col(('two_stage', 'D', 'image+q')),
         'C2 plain (L,x)': col(('two_stage', 'C2', 'image+all')), 'main effect (L,x,q)': col(('two_stage', 'D', 'image+all+q')),
         'image interaction (L,x,q,qL)': col(('two_stage', 'D', 'image+all+qL')),
         'interactions (L,x,q,qL,qx)': col(('two_stage', 'D', 'image+all+qint')), 'oracle interactions': col(('two_stage', 'D', 'image+all+qoracle'))}
    tables['table_D'] = pd.DataFrame(d, index=index)
    rows = []
    for pipe in PIPELINES:
        for bb in s.backbones:
            def pt(r):
                return np.nan if r is None else round(r[0], 4)
            rows.append({'pipeline': PIPE_LABEL[pipe], 'backbone': bb,
                         'AP image (clean)': pt(s.ap(pipe, 'B', bb, 'image', 'test', ['clean'])),
                         'AP image+clinical (clean)': pt(s.ap(pipe, 'B', bb, 'image+all', 'test', ['clean'])),
                         'AP image (8 blur/down)': pt(s.ap(pipe, 'B', bb, 'image', 'test', SEEN)),
                         'AP image+clinical (8 blur/down)': pt(s.ap(pipe, 'B', bb, 'image+all', 'test', SEEN)),
                         'P1': pt(s.delta(pipe, 'B', bb, SEEN))})
    tables['table_B_backbones'] = pd.DataFrame(rows).set_index('pipeline')
    # layer 4: training regime x clinical information (which remedy helps a poor photo more?)
    rows = []
    for pipe in PIPELINES:
        for v, regime in [('B', 'standard'), ('C1', 'basic augmentation'), ('C2', 'basic + blur / downsampling')]:
            def pt(setting, conds):
                r = s.over_backbones(lambda b: s.ap(pipe, v, b, setting, 'test', conds))
                return np.nan if r is None else round(r['point'], 4)
            rows.append({'pipeline': PIPE_LABEL[pipe], 'training': f'{v} ({regime})',
                         'clean: image': pt('image', ['clean']), 'clean: image+clinical': pt('image+all', ['clean']),
                         'degraded: image': pt('image', SEEN), 'degraded: image+clinical': pt('image+all', SEEN)})
    tables['table_robust_x_clinical'] = pd.DataFrame(rows).set_index('pipeline')
    return tables


def severity_table(pred, ctx, backbones):
    """By ICDR grade, at each model's fixed threshold (experiment B, test photos, mean of backbones):
      flagged rates of image-only and fusion models on clean and degraded photos;
      net change in detection rate = rate(model a) - rate(model b), with a paired 95% CI;
      and its two parts, per photo: recovered = missed by the image model but flagged by the fusion, newly
      missed = flagged by the image model but missed by the fusion (net = recovered - newly missed).
    In the no-DR row "flagged" means false positive, so its net change is the change in the false-positive
    rate (= minus the change in specificity): read every detection gain together with it."""
    if not C.GRADE_CSV.is_file() or 'test' not in ctx['photo_sets']:
        return None
    photos, bs = ctx['photo_sets']['test']
    grade = pd.read_csv(C.GRADE_CSV, dtype={'image_id': str}).set_index('image_id').dr_icdr.reindex(photos.image_id).to_numpy()
    if np.isnan(grade).any():
        return None
    groups = {'no DR (flagged = false positive)': grade == 0, 'mild NPDR (ICDR 1)': grade == 1,
              'moderate NPDR (ICDR 2)': grade == 2, 'severe NPDR / PDR (ICDR 3-4)': grade >= 3,
              'all DR (ICDR 1-4)': grade >= 1}
    conds = ['clean'] + SEEN
    sub = pred[(pred.eval_set == 'test') & (pred.experiment == 'B') & pred.setting.isin(['image', 'image+all'])
               & pred.condition.isin(conds) & pred.pipeline.isin(PIPELINES)]
    flagged = {}
    for key, g in sub.groupby(KEY + ['condition']):
        model, cond = tuple(key[:5]), key[5]
        if model[4] != MAIN_CLF[model[0]] or model not in ctx['thresholds']:
            continue
        p = g.set_index('image_id').prob.reindex(photos.image_id).to_numpy(dtype=float)
        flagged[(model[0], model[2], model[3], cond)] = (p >= ctx['thresholds'][model]).astype(float)
    W = bs.W.astype(float)

    def average(pipe, cs, mask, fn):
        """fn(backbone, condition) -> per-photo 0/1 vector or None; mean over photos, conditions, backbones."""
        items = []
        for b in backbones:
            for c in cs:
                v = fn(b, c)
                if v is None:
                    return None
                with np.errstate(invalid='ignore', divide='ignore'):
                    items.append((float(v[mask].mean()), (W[:, mask] @ v[mask]) / W[:, mask].sum(axis=1)))
        return float(np.mean([i[0] for i in items])), np.nanmean([i[1] for i in items], axis=0)

    def f(pipe, setting):
        return lambda b, c: flagged.get((pipe, b, setting, c))

    def both(pipe, rule):
        def fn(b, c):
            img, fus = flagged.get((pipe, b, 'image', c)), flagged.get((pipe, b, 'image+all', c))
            return None if img is None or fus is None else rule(img, fus)
        return fn

    rows = []
    for pipe in PIPELINES:
        for name, mask in groups.items():
            if not mask.any():
                continue
            cells = {'image clean': average(pipe, ['clean'], mask, f(pipe, 'image')),
                     'fusion clean': average(pipe, ['clean'], mask, f(pipe, 'image+all')),
                     'image degraded': average(pipe, SEEN, mask, f(pipe, 'image')),
                     'fusion degraded': average(pipe, SEEN, mask, f(pipe, 'image+all'))}
            if any(v is None for v in cells.values()):
                continue
            row = {'pipeline': PIPE_LABEL[pipe], 'group': name, 'n_photos': int(mask.sum()),
                   'n_patients': int(photos.patient_id[mask].nunique()), **{f'flagged, {k}': round(v[0], 4) for k, v in cells.items()}}
            for label, (a, b) in {'net change by degradation, image (degraded - clean)': ('image degraded', 'image clean'),
                                  'net change from clinical, degraded (fusion - image)': ('fusion degraded', 'image degraded'),
                                  'net change from clinical, clean (fusion - image)': ('fusion clean', 'image clean')}.items():
                lo, hi = ci(cells[a][1] - cells[b][1])
                row[label] = round(cells[a][0] - cells[b][0], 4)
                row[label.split(' (')[0] + ', 95% CI'] = f'[{lo:+.3f}, {hi:+.3f}]'
            for label, rule in {'degraded: newly flagged by fusion (image no, fusion yes)': lambda i, u: (1 - i) * u,
                                'degraded: no longer flagged by fusion (image yes, fusion no)': lambda i, u: i * (1 - u)}.items():
                r = average(pipe, SEEN, mask, both(pipe, rule))
                row[label] = None if r is None else round(r[0], 4)
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def training_table():
    rows = []
    for path in sorted(C.MODELS_DIR.glob('*/*/fold*/metrics.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        best, epochs = report['best_validation'], report['epochs']
        rows.append({'model_set': report['config']['model_set'], 'backbone': report['config']['backbone'],
                     'best_epoch': best['epoch'], 'epochs_run': len(epochs), 'hit_epoch_limit': len(epochs) >= C.EPOCHS,
                     'val_ap': best['ap'], 'degraded_val_ap_at_best': best.get('val_degraded_ap', np.nan),
                     'degraded_val_ap_max': max(e.get('val_degraded_ap', np.nan) for e in epochs),
                     'minutes': report.get('total_seconds', np.nan) / 60})
    if not rows:
        return None
    return (pd.DataFrame(rows).groupby(['model_set', 'backbone'])
            .agg(networks=('val_ap', 'size'), best_epoch=('best_epoch', 'mean'), epochs_run=('epochs_run', 'mean'),
                 hit_epoch_limit=('hit_epoch_limit', 'sum'), val_ap=('val_ap', 'mean'), val_ap_sd=('val_ap', 'std'),
                 degraded_val_ap_at_best=('degraded_val_ap_at_best', 'mean'), degraded_val_ap_max=('degraded_val_ap_max', 'mean'),
                 minutes=('minutes', 'sum')).round(4))


def figures(s, pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def series(fn, conds):
        pts, los, his = [], [], []
        for c in conds:
            r = s.over_backbones(lambda b: fn(b, c))
            if r is None:
                return None
            lo, hi = ci(r['boot'])
            pts.append(r['point']), los.append(lo), his.append(hi)
        return np.array(pts), np.array(los), np.array(his)

    def ap_line(model):
        return lambda conds: series(lambda b, c: s.ap(model[0], model[1], b, model[2], 'test', [c]), conds)

    def gain_line(pipe, exp, setting='image+all'):
        return lambda conds: series(lambda b, c: s.gain(pipe, exp, b, 'test', [c], setting), conds)

    def severity_plot(lines, title, ylabel, name, zero=False):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        panels = [(axes[0], ['clean'] + C.BLUR_CONDITIONS, [0] + [float(c.split('_s')[1]) for c in C.BLUR_CONDITIONS], 'Gaussian blur sigma (px at 384 px)'),
                  (axes[1], ['clean'] + C.DOWN_CONDITIONS, [1] + [float(c.split('_x')[1]) for c in C.DOWN_CONDITIONS], 'Downsampling factor')]
        drawn = False
        for ax, conds, xs, xlabel in panels:
            for label, fn, style in lines:
                r = fn(conds)
                if r is None:
                    continue
                drawn = True
                ax.plot(xs, r[0], style, marker='o', label=label)
                ax.fill_between(xs, r[1], r[2], alpha=0.12)
            if zero:
                ax.axhline(0, color='grey', lw=0.8)
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.3)
        if drawn:
            axes[0].set_ylabel(ylabel)
            axes[0].legend(fontsize=8)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(C.FIGURES_DIR / name, dpi=150)
        plt.close(fig)

    for pipe in PIPELINES:
        severity_plot([('clinical only', ap_line(('clinical_only', 'A', 'clinical')), ':'),
                       ('image only', ap_line((pipe, 'B', 'image')), '-'),
                       ('image + clinical', ap_line((pipe, 'B', 'image+all')), '-'),
                       ('image + shuffled clinical', ap_line((pipe, 'B', 'image+shuffled')), '--')],
                      f'Experiment B, {PIPE_LABEL[pipe]} fusion: test AP (mean of backbones, 95% CI)', 'Test AP', f'fig_B_ap_{pipe}.png')
    severity_plot([('two-stage', gain_line('two_stage', 'B'), '-'), ('end-to-end', gain_line('end_to_end', 'B'), '-'),
                   ('two-stage, shuffled clinical', gain_line('two_stage', 'B', 'image+shuffled'), '--'),
                   ('end-to-end, shuffled clinical', gain_line('end_to_end', 'B', 'image+shuffled'), '--')],
                  'Gain from clinical features: AP(image + clinical) - AP(image)', 'AP gain', 'fig_B_gain.png', zero=True)
    severity_plot([(f'{v} image only', ap_line(('end_to_end', v, 'image')), '--' if v == 'B' else '-') for v in ['B', 'C1', 'C2']],
                  'Experiment C: image-only AP under each training regime', 'Test AP', 'fig_C_image.png')
    severity_plot([(f'{PIPE_LABEL[p]} {v}', gain_line(p, v), '-' if p == 'two_stage' else '--') for p in PIPELINES for v in ['B', 'C1', 'C2']],
                  'Experiment C: gain from clinical features by training regime', 'AP gain', 'fig_C_gain.png', zero=True)
    severity_plot([('image only (L)', ap_line(('two_stage', 'C2', 'image')), ':'), ('plain fusion (L, x)', ap_line(('two_stage', 'C2', 'image+all')), '--'),
                   ('quality main effect (L, x, q)', ap_line(('two_stage', 'D', 'image+all+q')), '-'),
                   ('quality x image (L, x, q, qL)', ap_line(('two_stage', 'D', 'image+all+qL')), '-'),
                   ('quality interactions (L, x, q, qL, qx)', ap_line(('two_stage', 'D', 'image+all+qint')), '-'),
                   ('oracle interactions', ap_line(('two_stage', 'D', 'image+all+qoracle')), '-.')],
                  'Experiment D: quality-aware two-stage fusion (C2 networks)', 'Test AP', 'fig_D_quality.png')

    # P2 by condition
    rows = []
    for c in C.TEST_CONDITIONS:
        r = s.compare(('two_stage', 'B', 'image+all'), ('end_to_end', 'B', 'image+all'), 'test', [c])
        if r is not None:
            rows.append((c, r['point'], *ci(r['boot'])))
    if rows:
        fig, ax = plt.subplots(figsize=(6, 6))
        ys = np.arange(len(rows))[::-1]
        ax.errorbar([r[1] for r in rows], ys, xerr=[[max(0.0, r[1] - r[2]) for r in rows], [max(0.0, r[3] - r[1]) for r in rows]], fmt='o', capsize=3)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.axvline(0, color='grey', lw=0.8)
        ax.set_xlabel('AP two-stage - AP end-to-end (image + clinical)')
        ax.set_title('Pipelines by condition (mean of backbones, 95% CI)')
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / 'fig_B_pipelines.png', dpi=150)
        plt.close(fig)

    # degradations outside the blur / downsampling family
    conds = ['clean'] + UNSEEN + CONTRAST
    models = [('image only', ('end_to_end', 'B', 'image')), ('two-stage image + clinical', ('two_stage', 'B', 'image+all')),
              ('end-to-end image + clinical', ('end_to_end', 'B', 'image+all'))]
    bars = {label: series(lambda b, c, m=m: s.ap(m[0], m[1], b, m[2], 'test', [c]), conds) for label, m in models}
    if all(v is not None for v in bars.values()):
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(conds))
        for i, (label, v) in enumerate(bars.items()):
            ax.bar(x + (i - 1) * 0.27, v[0], width=0.27, label=label)
        ax.set_xticks(x, conds, rotation=30)
        ax.set_ylabel('Test AP')
        ax.legend(fontsize=8)
        ax.set_title('Experiment B: JPEG, noise and low contrast (never used in training)')
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / 'fig_B_other_degradations.png', dpi=150)
        plt.close(fig)

    # which clinical feature helps: add-one gain per condition (two-stage, experiment B)
    grid = []
    for f in C.CLINICAL + ['all']:
        setting = 'image+all' if f == 'all' else f'image+{f}'
        r = series(lambda b, c: s.gain('two_stage', 'B', b, 'test', [c], setting), C.TEST_CONDITIONS)
        grid.append(None if r is None else r[0])
    if all(g is not None for g in grid):
        grid = np.array(grid)
        limit = max(0.01, float(np.abs(grid).max()))
        fig, ax = plt.subplots(figsize=(12, 3.8))
        image = ax.imshow(grid, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(len(C.TEST_CONDITIONS)), C.TEST_CONDITIONS, rotation=40, ha='right')
        ax.set_yticks(range(len(grid)), C.CLINICAL + ['all five'])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f'{grid[i, j]:+.3f}', ha='center', va='center', fontsize=7)
        fig.colorbar(image, label='AP gain over image only')
        ax.set_title('Two-stage fusion: AP gain from adding each clinical feature to the image logit (mean of backbones)')
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / 'fig_B_features.png', dpi=150)
        plt.close(fig)

    # permutation importance, both pipelines
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    drawn = False
    for ax, pipe in zip(axes, PIPELINES):
        x = np.arange(len(C.CLINICAL))
        for i, (name, conds) in enumerate([('clean', ['clean']), ('blur / downsampling', SEEN)]):
            vals, errs = [], [[], []]
            for f in C.CLINICAL:
                r = s.over_backbones(lambda b: s.sub(s.ap(pipe, 'B', b, PERM_REFERENCE, 'test', conds), s.ap(pipe, 'B', b, PERM_SETTINGS[f], 'test', conds)))
                if r is None:
                    vals = None
                    break
                lo, hi = ci(r['boot'])
                vals.append(r['point']), errs[0].append(max(0.0, r['point'] - lo)), errs[1].append(max(0.0, hi - r['point']))
            if vals is None:
                continue
            drawn = True
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.38, yerr=errs, capsize=2, label=name)
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xticks(x, C.CLINICAL, rotation=25)
        ax.set_title(PIPE_LABEL[pipe])
        ax.grid(alpha=0.3, axis='y')
    if drawn:
        axes[0].set_ylabel('AP drop when the feature is permuted')
        axes[0].legend(fontsize=8)
        fig.suptitle('Permutation importance of each clinical feature (image + clinical, mean of backbones, 95% CI)')
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / 'fig_B_permutation.png', dpi=150)
    plt.close(fig)

    # reliability of the raw probabilities (appendix)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    drawn = False
    for ax, (title, conds) in zip(axes, [('clean', ['clean']), ('blur / downsampling', SEEN)]):
        for pipe in PIPELINES:
            g = pred[(pred.pipeline == pipe) & (pred.experiment == 'B') & (pred.setting == 'image+all') &
                     (pred.classifier == MAIN_CLF[pipe]) & (pred.eval_set == 'test') & pred.condition.isin(conds)]
            if g.empty:
                continue
            drawn = True
            bins = np.clip((g.prob.to_numpy() * 10).astype(int), 0, 9)
            m = g.groupby(bins).agg(p=('prob', 'mean'), y=('y', 'mean'))
            ax.plot(m.p, m.y, marker='o', label=PIPE_LABEL[pipe])
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
        ax.set_title(title)
        ax.set_xlabel('Predicted probability (raw)')
        ax.grid(alpha=0.3)
    if drawn:
        axes[0].set_ylabel('Observed DR frequency')
        axes[0].legend(fontsize=8)
        fig.suptitle('Reliability, image + clinical (raw probabilities; class-weighted training shifts them upwards)')
        fig.tight_layout()
        fig.savefig(C.FIGURES_DIR / 'fig_calibration.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--val-only', action='store_true', help='Use validation photos only; the test set stays unopened.')
    parser.add_argument('--backbones', nargs='+', choices=list(C.BACKBONES), default=list(C.BACKBONES))
    parser.add_argument('--allow-incomplete', action='store_true', help='Warn instead of stopping when predictions are missing.')
    parser.add_argument('--dose-gate', action='store_true', help='Exit with code 3 if the step-0 dose check fails.')
    args = parser.parse_args()
    C.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    table = C.load_table()
    pred = load_predictions()
    eval_sets = ['val'] if args.val_only else ['test', 'inadequate', 'val']
    print(f'{len(pred)} prediction rows, {pred.groupby(KEY).ngroups} models; evaluation sets: {eval_sets}', flush=True)
    problems = check_complete(pred, table, args.backbones, eval_sets, final=not args.val_only)
    if problems:
        print(f'{len(problems)} completeness problem(s):\n  ' + '\n  '.join(problems[:40])
              + ('\n  ...' if len(problems) > 40 else ''), flush=True)
        if not args.allow_incomplete:
            raise SystemExit('Predictions are incomplete or inconsistent; finish the missing steps (rerun run_all.py) '
                             'or pass --allow-incomplete for a provisional look.')
    else:
        print('completeness check passed', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        metrics, boot, ctx = evaluate(pred, table, eval_sets)
        s = Summary(metrics, boot, args.backbones)
        keys = key_results(s)
        suffix = '_val_only' if args.val_only else ''
        metrics.to_csv(C.SUMMARY_DIR / f'metrics{suffix}.csv', index=False)
        keys.to_csv(C.SUMMARY_DIR / f'key_results{suffix}.csv', index=False)
        if not args.val_only:
            feature_results(s).to_csv(C.SUMMARY_DIR / 'feature_results.csv', index=False)
            for name, frame in condition_tables(s).items():
                frame.to_csv(C.SUMMARY_DIR / f'{name}.csv', index_label=frame.index.name or 'condition')
            severity = severity_table(pred, ctx, args.backbones)
            if severity is not None:
                severity.to_csv(C.SUMMARY_DIR / 'table_severity.csv', index=False)
            try:
                figures(s, pred)
            except ImportError:
                print('matplotlib not installed: figures skipped.')
    training = training_table()
    if training is not None:
        training.to_csv(C.SUMMARY_DIR / 'training.csv')
    counts = table.groupby('split').agg(photos=('image_id', 'size'), patients=('patient_id', 'nunique'), dr_positive_photos=(C.TARGET, 'sum'))
    counts['dr_positive_patients'] = table[table[C.TARGET] == 1].groupby('split').patient_id.nunique()
    test_patients = set(table.loc[table.split == 'test', 'patient_id'])
    counts['patients_also_in_test'] = pd.Series({split: (np.nan if split == 'test' else len(set(p) & test_patients))
                                                 for split, p in table.groupby('split').patient_id})
    for c in C.CLINICAL:
        counts[f'missing_{c}'] = table.groupby('split')[c].apply(lambda v: v.isna().mean()).round(4)
    counts.to_csv(C.SUMMARY_DIR / 'dataset.csv')

    print(f'\nKey results (mean of the backbones; * = pre-specified primary comparison with a {C.PRIMARY_LEVEL:.2f}% '
          'Bonferroni interval, others exploratory with 95% intervals):')
    layer = None
    for r in keys.itertuples(index=False):
        if r.layer != layer:
            layer = r.layer
            print(f'\n  --- {LAYERS[layer]}')
        print(f"  {'*' if r.primary else ' '} {r.id:38s} {r.estimate:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  {r.reading:14s} "
              f"{r.label} | {r.conditions}{' | ' + r.note if r.note else ''}")
    print(f'\nSaved {len(metrics)} model x set x condition rows and {len(keys)} summaries to {C.SUMMARY_DIR}', flush=True)

    # step 0: is the degradation strong enough for the experiment to be informative?
    dose = keys[keys.id == 'VAL_dose_drop'] if len(keys) else keys
    if len(dose):
        drop = float(dose.estimate.iloc[0])
        prevalence = float(table.loc[table.split == 'val', C.TARGET].mean())
        if drop >= C.DOSE_MIN_DROP:
            print(f'\nDose check passed: averaged over the backbones and the two strongest levels, blur / downsampling lower the '
                  f'image-only validation AP by {drop:.3f} (AP of a constant score = {prevalence:.3f}). This is a development check, '
                  'not evidence for the hypotheses; report the rule and its result in the methods or appendix.', flush=True)
        else:
            print(f'\nDOSE CHECK NOT PASSED: averaged over the backbones and the two strongest levels, blur / downsampling lower '
                  f'the image-only validation AP by only {drop:.3f} (rule of thumb: {C.DOSE_MIN_DROP}). 0.05 is a practical threshold, '
                  'not a scientific criterion. Check, in this order: (1) is the clean image model trained properly '
                  '(VAL_image_ap_clean, training.csv)? (2) are the degradations applied correctly (look at a few degraded photos)? '
                  '(3) only then consider stronger levels in src/common.py. Decide on validation results only, never to obtain a '
                  'positive clinical gain, change the levels at most once, and report the check, its result and any change.'
                  + ('' if args.val_only else ' The test results above were computed anyway; report this check with them.'), flush=True)
            if args.dose_gate:
                sys.exit(3)


if __name__ == '__main__':
    main()
