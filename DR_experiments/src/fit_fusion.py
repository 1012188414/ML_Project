"""Two-stage fusion and the non-network baselines. Nothing here is tuned: every model has a fixed
configuration, so no selection takes place on the training or validation data.

  --stage A           clinical-only LR, random forest, XGBoost (fixed small configurations)
  --stage B | C1 | C2 two-stage LR on that experiment's image networks:
                        image | image+all | image + one feature (x5) | image + all but one feature (x5)
                      B only: image + missing indicators only (does record completeness, rather than the
                      clinical values, carry the gain?), image + shuffled clinical (SHUFFLE_REPEATS
                      permutations; negative control), the late average (mean of the clinical-LR and image
                      probabilities), a sensitivity analysis of the fixed regularisation (C = 0.1 and 10), and
                      permutation importance of each clinical feature for image+all (test, PERM_CONDITIONS)
  --stage Bj          "hybrid" pipeline for decomposing P2: the same two-stage LR (image, image+all) fitted on the
                      image part of the end-to-end networks' logit (see extract_features.py)
  --stage D           quality-aware LR on the C2 logits (L = image logit, x = clinical, q = quality signal):
                        image+q             L, q, qL            (quality without clinical features)
                        image+all+q         L, x, q             (quality main effect only)
                        image+all+qL        L, x, q, qL         (quality modulates the image weight only)
                        image+all+qint      L, x, q, qL, qx     (main model: also modulates the clinical weights)
                        image+all+qoracle   same with q = known synthetic blur / downsampling severity (appendix; a reference,
                                            not guaranteed to be an upper bound)
                      compared with the plain C2 fusion image+all (L, x).

Fusion model: logistic regression, L2, C = 1, balanced class weights. L is one fold network's logit,
standardised on the (weighted) training rows; x the encoded clinical features (common.ClinicalEncoder, fitted
on all training photos); q_lap standardised on the (weighted) training rows. Training photos use their out-of-fold logit; for
validation / test / inadequate photos the fitted model is applied with each fold network's logit and the four
probabilities are averaged. Every fitted fusion model is saved in full next to the predictions
(*_models.joblib: LR, standardisation constants, clinical preprocessing state, feature order), and its
coefficients in readable form (*_models.json).
Output: predictions/<pipeline>/<stage>/<backbone>.csv.gz
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common as C


def fusion_lr(c=C.LR_C):
    return LogisticRegression(C=c, class_weight='balanced', solver='liblinear', max_iter=5000)


def output_frame(rows, fold_probs, pipeline, experiment, backbone, setting, classifier):
    out = pd.DataFrame({'pipeline': pipeline, 'experiment': experiment, 'backbone': backbone, 'setting': setting,
                        'classifier': classifier, 'image_id': rows.image_id.to_numpy(),
                        'patient_id': rows.patient_id.to_numpy(), 'eval_set': rows.eval_set.to_numpy(),
                        'condition': rows.condition.to_numpy(), 'y': rows.y.to_numpy()})
    fold_probs = np.asarray(fold_probs, dtype=float)
    if fold_probs.ndim == 1:
        out['prob'] = fold_probs
        for k in range(C.FOLDS):
            out[f'prob_f{k}'] = np.nan
    else:
        out['prob'] = fold_probs.mean(axis=1)
        for k in range(C.FOLDS):
            out[f'prob_f{k}'] = fold_probs[:, k]
    return out


def save(frames, path, info=None, models=None):
    """Model files first, predictions last: the predictions file marks the step as finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if models is not None:
        joblib.dump(models, path.with_name(path.name.replace('.csv.gz', '_models.joblib')))
    if info is not None:
        path.with_name(path.name.replace('.csv.gz', '_models.json')).write_text(json.dumps(info, indent=2, default=str) + '\n')
    tmp = path.with_name('.' + path.name + '.tmp')
    pd.concat(frames, ignore_index=True).to_csv(tmp, index=False, compression='gzip')
    tmp.replace(path)


# ---------------------------------------------------------------- A: clinical only
def clinical_models():
    from sklearn.ensemble import RandomForestClassifier
    models = {'lr': fusion_lr(),
              'rf': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20, class_weight='balanced',
                                           random_state=C.SEED, n_jobs=-1)}
    try:
        from xgboost import XGBClassifier  # macOS: brew install libomp
        models['xgb'] = XGBClassifier(objective='binary:logistic', tree_method='hist', n_estimators=100, learning_rate=0.1,
                                      max_depth=2, min_child_weight=10, subsample=0.8, random_state=C.SEED, n_jobs=1)
    except ImportError as error:
        print(f'clinical xgb: skipped ({error})', flush=True)
    return models


def stage_a(args):
    path = C.PREDICTIONS_DIR / 'clinical_only' / 'A.csv.gz'
    if path.is_file() and not args.force:
        print('[skip] clinical-only predictions exist', flush=True)
        return
    table = C.load_table()
    tr = (table.split == 'train').to_numpy()
    encoder = C.ClinicalEncoder().fit(table[tr])
    X = encoder.transform(table)
    y = table[C.TARGET].to_numpy()
    rows = []
    for r in table[~tr].itertuples(index=False):
        for cond in C.eval_conditions(r.split):
            rows.append({'image_id': r.image_id, 'patient_id': r.patient_id, 'eval_set': r.split, 'condition': cond})
    rows = pd.DataFrame(rows)
    idx = pd.Series(np.arange(len(table)), index=table.image_id).reindex(rows.image_id).to_numpy()
    rows['y'] = y[idx]
    frames, info, fitted = [], {}, {'clinical_preprocessing': encoder.state, 'features': encoder.names}
    for name, model in clinical_models().items():
        if name == 'xgb':
            model.set_params(scale_pos_weight=float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())))
        model.fit(X[tr], y[tr])
        frames.append(output_frame(rows, model.predict_proba(X[idx])[:, 1], 'clinical_only', 'A', 'no_image', 'clinical', name))
        info[name] = {'features': encoder.names}
        fitted[name] = model
        if name == 'lr':
            info[name]['coefficients'] = dict(zip(encoder.names, model.coef_[0].round(4).tolist()))
        print(f'clinical {name}: fitted', flush=True)
    save(frames, path, info, fitted)


# ---------------------------------------------------------------- two-stage fusion
def load_features(variant, backbone, table):
    f = np.load(C.FEATURES_DIR / variant / f'{backbone}.npz')
    frame = pd.DataFrame({k: f[k] for k in ['image_id', 'split', 'condition', 'patient_id', 'fold', 'weight',
                                            'q_lap', 'q_blur', 'q_down']})
    frame['y'] = table.set_index('image_id')[C.TARGET].reindex(frame.image_id).to_numpy()
    return frame, f['logits'].astype(float)


def design(L, X, x_names, Q, q_names, interact):
    """interact: 'none' (L, x, q) | 'image' (+ qL) | 'all' (+ qL, qx)."""
    parts, names = [L[:, None]], ['image_logit']
    if X is not None:
        parts.append(X)
        names += x_names
    if Q is not None:
        parts.append(Q)
        names += q_names
        for j, q in enumerate(q_names):
            if interact in ('image', 'all'):
                parts.append((L * Q[:, j])[:, None])
                names.append(f'image_logit*{q}')
            if interact == 'all' and X is not None:
                parts.append(X * Q[:, [j]])
                names += [f'{x}*{q}' for x in x_names]
    return np.hstack(parts), names


def weighted_mean_sd(values, weights):
    mean = np.average(values, weights=weights)
    return float(mean), float(np.sqrt(np.average((values - mean) ** 2, weights=weights)) or 1.0)


def two_stage(frame, logits, clinical_table, features, quality=None, interact='none', c=C.LR_C, permuted=None):
    """Fit on the out-of-fold training rows; predict the evaluation rows with each fold network's logit.
    features: list of clinical features, or ['missing'] for the missing indicators only.
    quality: None | 'lap' | 'oracle'. permuted: [(setting, clinical table with one feature permuted), ...] ->
    extra predictions of the same fitted model for the test photos under PERM_CONDITIONS."""
    tr = (frame.split == 'train').to_numpy()
    weights = frame.weight.to_numpy()[tr]
    Q, q_names, q_stats = None, [], None
    if quality == 'lap':
        raw = frame.q_lap.to_numpy(dtype=float)
        q_stats = weighted_mean_sd(raw[tr], weights)
        Q, q_names = ((raw - q_stats[0]) / q_stats[1])[:, None], ['q_lap']
    elif quality == 'oracle':
        Q, q_names = frame[['q_blur', 'q_down']].to_numpy(dtype=float), ['q_blur', 'q_down']
    ev = ~tr if Q is None else ~tr & np.isfinite(Q).all(axis=1)
    train_idx = np.flatnonzero(tr)
    L_train = logits[train_idx, frame.fold.to_numpy()[train_idx]]
    mu, sd = weighted_mean_sd(L_train, weights)
    X, x_names, encoder, cols = None, [], None, []
    if features:
        clin_rows = clinical_table.set_index('image_id').loc[frame.image_id].reset_index()
        encoder = C.ClinicalEncoder().fit(clin_rows[tr & (frame.condition == 'clean').to_numpy()])
        if features == ['missing']:
            cols = [i for i, n in enumerate(encoder.names) if n.endswith('_missing')]
        else:
            cols = encoder.columns_for(features)
        X, x_names = encoder.transform(clin_rows)[:, cols], [encoder.names[i] for i in cols]

    def rows_of(mask, L, Xs=None):
        Xs = X if Xs is None else Xs
        return design((L - mu) / sd, None if Xs is None else Xs[mask], x_names, None if Q is None else Q[mask], q_names, interact)

    X_train, names = rows_of(tr, L_train)
    model = fusion_lr(c).fit(X_train, frame.y.to_numpy()[tr], sample_weight=weights)
    probs = np.zeros((ev.sum(), C.FOLDS))
    for k in range(C.FOLDS):
        probs[:, k] = model.predict_proba(rows_of(ev, logits[ev, k])[0])[:, 1]
    info = {'C': c, 'coefficients': dict(zip(names, model.coef_[0].round(4).tolist())),
            'intercept': round(float(model.intercept_[0]), 4)}
    full = {'model': model, 'feature_names': names, 'logit_mean': mu, 'logit_sd': sd, 'quality': quality,
            'q_mean_sd': q_stats, 'interactions': interact, 'clinical_preprocessing': encoder.state if encoder else None,
            'clinical_columns': [encoder.names[i] for i in cols] if encoder else []}
    extra = []
    if permuted:
        mask = ev & (frame.split == 'test').to_numpy() & frame.condition.isin(C.PERM_CONDITIONS).to_numpy()
        for setting, table_p in permuted:
            rows_p = table_p.set_index('image_id').loc[frame.image_id].reset_index()
            X_p = encoder.transform(rows_p)[:, cols]
            probs_p = np.stack([model.predict_proba(rows_of(mask, logits[mask, k], X_p)[0])[:, 1] for k in range(C.FOLDS)], axis=1)
            extra.append((setting, frame[mask].rename(columns={'split': 'eval_set'}), probs_p))
    return frame[ev].rename(columns={'split': 'eval_set'}), probs, info, full, extra


def fusion_jobs(stage, table):
    """(setting, clinical features, clinical table, quality, interactions, C, classifier name)"""
    if stage == 'Bj':
        return [('image', [], table, None, 'none', C.LR_C, 'lr'), ('image+all', C.CLINICAL, table, None, 'none', C.LR_C, 'lr')]
    if stage == 'D':
        return [('image+q', [], table, 'lap', 'image', C.LR_C, 'lr'),
                ('image+all+q', C.CLINICAL, table, 'lap', 'none', C.LR_C, 'lr'),
                ('image+all+qL', C.CLINICAL, table, 'lap', 'image', C.LR_C, 'lr'),
                ('image+all+qint', C.CLINICAL, table, 'lap', 'all', C.LR_C, 'lr'),
                ('image+all+qoracle', C.CLINICAL, table, 'oracle', 'all', C.LR_C, 'lr')]
    jobs = [('image', [], table, None, 'none', C.LR_C, 'lr'), ('image+all', C.CLINICAL, table, None, 'none', C.LR_C, 'lr')]
    jobs += [(f'image+{c}', [c], table, None, 'none', C.LR_C, 'lr') for c in C.CLINICAL]
    jobs += [(f'image+all-{c}', [f for f in C.CLINICAL if f != c], table, None, 'none', C.LR_C, 'lr') for c in C.CLINICAL]
    if stage == 'B':
        jobs.append(('image+missing', ['missing'], table, None, 'none', C.LR_C, 'lr'))
        for r in range(C.SHUFFLE_REPEATS):   # repeat 0 is the permutation the end-to-end control uses
            jobs.append(('image+shuffled' if r == 0 else f'image+shuffled_r{r}', C.CLINICAL,
                         C.shuffled_clinical(table, repeat=r), None, 'none', C.LR_C, 'lr'))
        for c in C.LR_C_SENSITIVITY:
            jobs += [('image', [], table, None, 'none', c, f'lr_c{c:g}'), ('image+all', C.CLINICAL, table, None, 'none', c, f'lr_c{c:g}')]
    return jobs


def stage_two(stage, args):
    table = C.load_table()
    variant = 'C2' if stage == 'D' else stage
    pipeline, experiment = ('hybrid', 'B') if stage == 'Bj' else ('two_stage', stage)
    for backbone in args.backbones:
        path = C.PREDICTIONS_DIR / pipeline / experiment / f'{backbone}.csv.gz'
        late_path = C.PREDICTIONS_DIR / 'late_average' / 'B' / f'{backbone}.csv.gz'
        need_fusion = args.force or not path.is_file()
        need_late = stage == 'B' and (args.force or not late_path.is_file())   # checked on its own, so an
        if not need_fusion and not need_late:                                   # interrupted run resumes it
            print(f'[skip] {pipeline} {stage} {backbone}: predictions exist', flush=True)
            continue
        frame, logits = load_features(variant, backbone, table)
        if need_fusion:
            jobs = fusion_jobs(stage, table)
            frames, info, models = [], {}, {}
            for setting, features, clin_table, quality, interact, c, clf in jobs:
                permuted = None
                if stage == 'B' and setting == 'image+all' and clf == 'lr':
                    permuted = [('perm:none', table)] + [(f'perm:{f}:r{r}', C.permuted_feature(table, f, r))
                                                         for f in C.CLINICAL for r in range(C.PERM_REPEATS)]
                rows, probs, fit_info, full, extra = two_stage(frame, logits, clin_table, features, quality, interact, c, permuted)
                frames.append(output_frame(rows, probs, pipeline, experiment, backbone, setting, clf))
                for perm_setting, perm_rows, perm_probs in extra:
                    frames.append(output_frame(perm_rows, perm_probs, pipeline, experiment, backbone, perm_setting, clf))
                info[f'{clf}:{setting}'] = fit_info
                models[f'{clf}:{setting}'] = full
            save(frames, path, info, models)
            print(f'{pipeline} {stage} {backbone}: {len(jobs)} fusion models fitted', flush=True)
        if need_late:
            late_average(frame, logits, backbone, late_path)


def late_average(frame, logits, backbone, path):
    """Baseline without learning: mean of the clinical-LR probability and the image probability."""
    clinical = pd.read_csv(C.PREDICTIONS_DIR / 'clinical_only' / 'A.csv.gz', dtype={'image_id': str, 'patient_id': str})
    clinical = clinical[(clinical.classifier == 'lr') & (clinical.condition == 'clean')].set_index('image_id').prob
    ev = (frame.split != 'train').to_numpy()
    rows = frame[ev].rename(columns={'split': 'eval_set'})
    p_clin = clinical.reindex(rows.image_id).to_numpy()[:, None]
    save([output_frame(rows, (C.sigmoid(logits[ev]) + p_clin) / 2, 'late_average', 'B', backbone, 'late_average', 'mean')], path)
    print(f'late average B {backbone}: saved', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['A', 'B', 'Bj', 'C1', 'C2', 'D'], required=True)
    parser.add_argument('--backbones', nargs='+', choices=list(C.BACKBONES), default=list(C.BACKBONES))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.stage == 'A':
        stage_a(args)
    else:
        stage_two(args.stage, args)


if __name__ == '__main__':
    main()
