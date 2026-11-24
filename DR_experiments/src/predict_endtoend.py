"""End-to-end predictions: the four fold networks of a model set score validation (clean + the 8 blur /
downsampling levels), test (every condition) and inadequate photos; the predicted probabilities are averaged over the folds.

Image-only model sets reuse the logits already computed by extract_features.py (same networks, same photos).
For B_clin, the test photos under PERM_CONDITIONS are also scored with one clinical feature permuted between
patients (PERM_REPEATS permutations per feature): permutation importance for the end-to-end pipeline. The
classifier is linear in [embedding, clinical], so each photo goes through the network once (image part of
the logit, all-zero clinical input) and every permutation only recomputes the clinical term v . x. The
unpermuted reference 'perm:none' is computed the same way, so reference and permutations share the same
numerics.
Output: predictions/end_to_end/<model_set>/<backbone>.csv.gz
"""
import argparse

import numpy as np
import pandas as pd

import common as C
import degrade
import nets


def eval_rows(table):
    rows = []
    for r in table.itertuples(index=False):
        if r.split == 'train':
            continue
        for cond in C.eval_conditions(r.split):
            if r.split == 'test':     # pre-degraded test cache, addressed by the condition name
                source, kind, value = 'test', cond, 0.0
            else:                     # base cache, degraded on the fly (deterministic for blur / downsampling)
                source, (kind, value) = 'base', degrade.parse(cond)
            rows.append({'image_id': r.image_id, 'patient_id': r.patient_id, 'eval_set': r.split, 'condition': cond,
                         'source': source, 'kind': kind, 'value': value, 'y': getattr(r, C.TARGET)})
    return pd.DataFrame(rows)


def finish(rows, fold_probs, spec, backbone):
    out = rows[['image_id', 'patient_id', 'eval_set', 'condition', 'y']].copy()
    out.insert(0, 'classifier', 'joint')
    out.insert(0, 'setting', spec['setting'])
    out.insert(0, 'backbone', backbone)
    out.insert(0, 'experiment', spec['experiment'])
    out.insert(0, 'pipeline', 'end_to_end')
    out['prob'] = fold_probs.mean(axis=1)
    for k in range(C.FOLDS):
        out[f'prob_f{k}'] = fold_probs[:, k]
    return out


def score_rows(model_set, backbone, rows, clinical_frame, device, args):
    fold_probs = np.full((len(rows), C.FOLDS), np.nan)
    for k in range(C.FOLDS):
        model, config = nets.load_model(C.model_dir(model_set, backbone, k) / 'best.pt', device)
        encoder = C.ClinicalEncoder(config['clinical_preprocessing'])
        dataset = nets.PhotoSet(rows.image_id, rows.y, config['normalization']['mean'], config['normalization']['std'],
                                sources=rows.source, kinds=rows.kind, values=rows.value, clinical=encoder.transform(clinical_frame))
        logits, _ = nets.score(model, nets.make_loader(dataset, C.BATCH_SIZE, args.workers, False, device=device), device)
        fold_probs[:, k] = C.sigmoid(logits)
        del model
    return fold_probs


def permutation_predictions(model_set, backbone, table, rows, spec, device, args):
    """perm:none and perm:<feature>:r<k> for the given rows, from one network pass per fold."""
    variants = [('perm:none', table)] + [(f'perm:{f}:r{r}', C.permuted_feature(table, f, r))
                                         for f in C.CLINICAL for r in range(C.PERM_REPEATS)]
    frames = {name: t.set_index('image_id').loc[rows.image_id].reset_index() for name, t in variants}
    probs = {name: np.full((len(rows), C.FOLDS), np.nan) for name, _ in variants}
    for k in range(C.FOLDS):
        model, config = nets.load_model(C.model_dir(model_set, backbone, k) / 'best.pt', device)
        encoder = C.ClinicalEncoder(config['clinical_preprocessing'])
        n_clinical = len(encoder.names)
        dataset = nets.PhotoSet(rows.image_id, rows.y, config['normalization']['mean'], config['normalization']['std'],
                                sources=rows.source, kinds=rows.kind, values=rows.value,
                                clinical=np.zeros((len(rows), n_clinical), dtype=np.float32))
        image_part, _ = nets.score(model, nets.make_loader(dataset, C.BATCH_SIZE, args.workers, False, device=device), device)
        v = model.classifier.weight.detach().float().cpu().numpy()[0, C.EMBEDDING_SIZE:]
        for name, frame in frames.items():
            probs[name][:, k] = C.sigmoid(image_part + encoder.transform(frame) @ v)
        del model
    return [finish(rows, p, {**spec, 'setting': name}, backbone) for name, p in probs.items()]


def from_features(model_set, spec, backbone):
    variant = next(v for v, m in C.FEATURE_VARIANTS.items() if m == model_set)
    f = np.load(C.FEATURES_DIR / variant / f'{backbone}.npz')
    frame = pd.DataFrame({k: f[k] for k in ['image_id', 'split', 'condition', 'patient_id']})
    keep = (frame.split != 'train').to_numpy()
    rows = frame[keep].rename(columns={'split': 'eval_set'}).reset_index(drop=True)
    table = C.load_table().set_index('image_id')
    rows['y'] = table[C.TARGET].reindex(rows.image_id).to_numpy()
    return finish(rows, C.sigmoid(f['logits'][keep]), spec, backbone)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-set', choices=list(C.MODEL_SETS), required=True)
    parser.add_argument('--backbones', nargs='+', choices=list(C.BACKBONES), default=list(C.BACKBONES))
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    spec = C.MODEL_SETS[args.model_set]
    out_dir = C.PREDICTIONS_DIR / 'end_to_end' / args.model_set
    out_dir.mkdir(parents=True, exist_ok=True)
    device = nets.resolve_device(args.device)

    for backbone in args.backbones:
        target = out_dir / f'{backbone}.csv.gz'
        if target.is_file() and not args.force:
            print(f'[skip] end-to-end {args.model_set} {backbone}: predictions exist', flush=True)
            continue
        if spec['clinical'] == 'none':
            out = from_features(args.model_set, spec, backbone)
        else:
            table = C.load_table()
            if spec['clinical'] == 'shuffled':
                table = C.shuffled_clinical(table)
            rows = eval_rows(table)
            clinical_frame = table.set_index('image_id').loc[rows.image_id].reset_index()
            out = finish(rows, score_rows(args.model_set, backbone, rows, clinical_frame, device, args), spec, backbone)
            print(f'end-to-end {args.model_set} {backbone}: scored {len(rows)} photos x {C.FOLDS} folds', flush=True)
            if args.model_set == 'B_clin':
                perm_rows = rows[(rows.eval_set == 'test') & rows.condition.isin(C.PERM_CONDITIONS)].reset_index(drop=True)
                out = pd.concat([out] + permutation_predictions(args.model_set, backbone, table, perm_rows, spec, device, args),
                                ignore_index=True)
                print(f'end-to-end {args.model_set} {backbone}: permutation importance scored', flush=True)
        tmp = out_dir / f'.{backbone}.tmp.csv.gz'
        out.to_csv(tmp, index=False, compression='gzip')
        tmp.replace(target)
        print(f'end-to-end {args.model_set} {backbone}: {len(out)} predictions saved', flush=True)


if __name__ == '__main__':
    main()
