"""Cross-fitted image logits for the two-stage fusion.

Variant Bj scores the end-to-end networks B_clin with an all-zero clinical input: since their classifier is
linear, this returns the image part of their logit (w_image . embedding + bias). The fold-k network never saw
the patients of fold k either, so these logits are out-of-fold too. Refitting the two-stage LR on them gives
the "hybrid" pipeline used to decompose P2.

For one experiment variant (B, C1, C2) the four fold networks of its image-only model set score:
  training photos of fold k  -> the fold-k network only (out-of-fold). For C2 each training photo also gets
                                TRAIN_COPIES degraded copies, degraded like the C2 training data. Weights:
                                clean CLEAN_WEIGHT + copies COPY_WEIGHT each = 1 per photo, so clean : degraded
                                = 1 : 1 (like the end-to-end networks) and every photo counts once in the LR.
  validation photos (clean + the 8 blur / downsampling levels, degraded on the fly, deterministic), test
  photos (every test condition, pre-degraded cache), inadequate photos (clean)
                             -> all four fold networks
Output: features/<variant>/<backbone>.npz with one row per photo x condition:
  image_id, split, condition, patient_id, fold, weight, logits (N, 4; NaN where a fold network did not score
  the row), q_lap (sharpness of the actual input photo), q_blur / q_down (true severity of a blur /
  downsampling degradation, 0 for clean, NaN for other degradations and inadequate photos).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C
import degrade
import nets


def true_severity(kind, value):
    if kind == 'clean':
        return 0.0, 0.0
    if kind == 'blur':
        return degrade.severity(kind, value), 0.0
    if kind == 'down':
        return 0.0, degrade.severity(kind, value)
    return np.nan, np.nan


def build_rows(variant, table, quality):
    family = C.degradation_family(C.MODEL_SETS[C.FEATURE_VARIANTS[variant]]['aug'])
    rows = []
    for r in table.itertuples(index=False):
        base = {'image_id': r.image_id, 'patient_id': r.patient_id, 'split': r.split, 'fold': r.fold}
        if r.split == 'test':
            for cond in C.TEST_CONDITIONS:
                qb, qd = true_severity(*degrade.parse(cond))
                rows.append({**base, 'condition': cond, 'source': 'test', 'kind': cond, 'value': 0.0, 'weight': 1.0,
                             'q_blur': qb, 'q_down': qd})
            continue
        known = 0.0 if r.split != 'inadequate' else np.nan
        copies = r.split == 'train' and family
        rows.append({**base, 'condition': 'clean', 'source': 'base', 'kind': 'clean', 'value': 0.0,
                     'weight': C.CLEAN_WEIGHT if copies else 1.0,
                     'q_blur': known, 'q_down': known})
        if r.split == 'val':   # step-0 rehearsal: the same blur / downsampling levels as the test set
            for cond in C.SEEN_CONDITIONS:
                kind, value = degrade.parse(cond)
                qb, qd = true_severity(kind, value)
                rows.append({**base, 'condition': cond, 'source': 'base', 'kind': kind, 'value': value, 'weight': 1.0,
                             'q_blur': qb, 'q_down': qd})
        if copies:
            rng = np.random.default_rng(C.stable_seed(variant, r.image_id))
            for copy in range(C.TRAIN_COPIES):
                kind, value = degrade.sample_spec(rng, family, force=True)
                qb, qd = true_severity(kind, value)
                rows.append({**base, 'condition': f'aug{copy}_{kind}{value:.2f}', 'source': 'base', 'kind': kind,
                             'value': value, 'weight': C.COPY_WEIGHT, 'q_blur': qb, 'q_down': qd})
    rows = pd.DataFrame(rows).merge(quality.reset_index()[['image_id', 'condition', 'q_lap']],
                                    on=['image_id', 'condition'], how='left')
    # q_lap of degraded training copies and degraded validation photos is computed while scoring
    computed = (rows.condition.str.startswith('aug') | ((rows.split == 'val') & (rows.condition != 'clean'))).to_numpy()
    if rows.loc[~computed, 'q_lap'].isna().any():
        raise ValueError('The quality table is missing rows; rebuild the cache (build_cache.py).')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=list(C.FEATURE_VARIANTS), required=True)
    parser.add_argument('--backbones', nargs='+', choices=list(C.BACKBONES), default=list(C.BACKBONES))
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    device = nets.resolve_device(args.device)
    model_set = C.FEATURE_VARIANTS[args.variant]
    out_dir = C.FEATURES_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    table = C.load_table()
    rows = build_rows(args.variant, table, nets.quality_table())
    labels = table.set_index('image_id')[C.TARGET].reindex(rows.image_id).to_numpy()
    print(f'{args.variant}: {len(rows)} photo x condition rows', flush=True)

    for backbone in args.backbones:
        target = out_dir / f'{backbone}.npz'
        if target.is_file() and not args.force:
            print(f'[skip] features {args.variant} {backbone}: already extracted', flush=True)
            continue
        logits = np.full((len(rows), C.FOLDS), np.nan, dtype=np.float32)
        q_lap = rows.q_lap.to_numpy().copy()
        provenance = {'variant': args.variant, 'model_set': model_set, 'backbone': backbone, 'folds': {}}
        for k in range(C.FOLDS):
            path = C.model_dir(model_set, backbone, k) / 'best.pt'
            model, config = nets.load_model(path, device)
            wanted = np.flatnonzero(((rows.split == 'train') & (rows.fold == k)).to_numpy() | (rows.split != 'train').to_numpy())
            sub = rows.iloc[wanted]
            n_clinical = len(config['clinical_features'] or [])
            dataset = nets.PhotoSet(sub.image_id, labels[wanted], config['normalization']['mean'], config['normalization']['std'],
                                    sources=sub.source, kinds=sub.kind, values=sub.value, quality='lap', q_fixed=sub.q_lap,
                                    clinical=np.zeros((len(sub), n_clinical), dtype=np.float32) if n_clinical else None)
            started = time.perf_counter()
            lg, qs = nets.score(model, nets.make_loader(dataset, C.BATCH_SIZE, args.workers, False, device=device), device)
            logits[wanted, k] = lg
            q_lap[wanted] = np.where(np.isnan(q_lap[wanted]), qs, q_lap[wanted])
            provenance['folds'][k] = {'checkpoint': str(path), 'rows_scored': int(len(wanted))}
            print(f'{args.variant} {backbone} fold {k}: scored {len(wanted)} rows in {time.perf_counter() - started:.0f}s', flush=True)
            del model
        arrays = {c: rows[c].to_numpy(str) for c in ['image_id', 'split', 'condition', 'patient_id']}
        arrays.update(fold=rows.fold.to_numpy(int), weight=rows.weight.to_numpy(float), logits=logits, q_lap=q_lap,
                      q_blur=rows.q_blur.to_numpy(float), q_down=rows.q_down.to_numpy(float))
        tmp = out_dir / f'.{backbone}.tmp.npz'
        np.savez_compressed(tmp, **arrays)
        tmp.replace(target)
        (out_dir / f'{backbone}.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print(f'{args.variant} {backbone}: saved {target.name}', flush=True)


if __name__ == '__main__':
    main()
