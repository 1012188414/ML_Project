"""Run every experiment (140 networks: 7 model sets x 5 backbones x 4 cross-fitting folds).

    python run_all.py                       # everything, ending with the final evaluation on test + inadequate
    python run_all.py --smoke               # tiny end-to-end test (minutes); writes to results/main_smoke/

Folders (see README.md and src/dr_paths.py): code in src/, data in data/, pretrained and trained weights in
weights/, everything a run produces in results/<run>/. The default run is 'main' with five backbones: the four of
the original main run and PVT-v2. PVT-v2 was trained later with the same protocol, as its own run
(python run_all.py --backbones pvt_v2_b1 --results-name pvt), and then merged into results/main and
weights/trained/main (see results/main/SIGNATURE_NOTE.md). Finished networks are never retrained.

The experiments form a ladder (see README.md):
  step 0   checks on validation photos: is the degradation strong enough? rehearsal of the analysis
  layer 1  each modality alone                      (A, B)
  layer 2  does clinical information compensate?    (B: gain, P1, negative control)
  layer 3  how to fuse, which features              (B: P2, feature analyses)
  layer 4  after robust training                    (C1, C2)
  layer 5  quality-aware fusion                     (D)

Step by step (optional):
    python run_all.py --stages check cache B    # trains B; evaluates on VALIDATION photos only (step 0)
    python run_all.py --stages C D              # likewise
    python run_all.py --stages evaluate         # final evaluation on test + inadequate

Optional post-processing (trains nothing):
    python run_all.py --stages extra            # supplementary analyses -> results/extra_analysis/

Dose gate: in a full run, right after stage B, the validation-only evaluation checks that the strongest blur
/ downsampling levels lower the image-only validation AP by at least DOSE_MIN_DROP. If not, the run stops
before C, D and before any test performance is computed (--skip-dose-check overrides).

Test predictions are generated along the way, but test performance is computed only by a full run (no
--stages) or when `evaluate` is named explicitly, and is never used for model selection; any other run ends
with a validation-only evaluation. The final evaluation stops if the predictions are incomplete.

The run is resumable: finished networks, features and predictions are skipped, so after an interruption
start the same command again. Results are tied to the code that produced them (results/<run>/signature.json):
  * settings, degradation, network, training or cache code, or the dataset CSVs changed -> the run stops;
    --force marks every network as stale and retrains it (stage by stage, also across separate runs);
  * feature-extraction, fusion or end-to-end scoring code changed -> those steps are recomputed for every
    stage automatically, the trained networks are kept.
The console output is also saved to results/<run>/logs/run_all.log.

Stages
  check     data check (files, splits, folds, patient separation, contradictory records)
  cache     resize every photo once; pre-degrade the test photos (16 conditions); quality signal
  B         A + B: image-only, end-to-end + clinical, end-to-end + shuffled clinical networks; cross-fitted
            logits; clinical-only models, two-stage fusion, late average; end-to-end predictions;
            permutation importance; hybrid pipeline (two-stage LR on the end-to-end image logit)
  C         C1 (basic augmentation) and C2 (basic + blur / downsampling): image-only and end-to-end networks,
            two-stage fusion on their logits
  D         quality-aware two-stage fusion on the C2 logits (no new networks)
  evaluate  completeness check, metrics with a shared patient bootstrap, summaries by layer, tables, figures
  extra     (optional) src/analysis_extra.py: mechanism and complementarity analyses -> results/extra_analysis/
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
STAGES = ['check', 'cache', 'B', 'C', 'D', 'evaluate']
POST_STAGES = ['extra']   # optional post-processing of finished runs; never part of the default run
EXTRA_IN_MAIN = ['pvt_v2_b1']   # trained after the four main backbones, same protocol, merged into run 'main'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', nargs='+', choices=STAGES + POST_STAGES, default=None,
                        help='Default: all experiment stages (not the optional extra).')
    parser.add_argument('--backbones', nargs='+', default=None,
                        help='Default: the five backbones of run main (four main backbones + PVT-v2).')
    parser.add_argument('--results-name', default=None,
                        help='Run name: results go to results/<name>/, trained networks to weights/trained/<name>/ '
                             '(default: main). Required when --backbones names a backbone that is not part of run main, '
                             'so that the finished main results are left untouched.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--workers', type=int, default=4, help='DataLoader / cache worker processes.')
    parser.add_argument('--val-only', action='store_true',
                        help='Never open the test set in this run, even for a full run.')
    parser.add_argument('--skip-dose-check', action='store_true',
                        help='Full run: continue to C, D and the test evaluation even if the step-0 dose check fails.')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Final evaluation: warn instead of stopping when predictions are missing.')
    parser.add_argument('--smoke', action='store_true', help='Tiny end-to-end test on a few photos.')
    parser.add_argument('--no-pretrained', action='store_true', help='Debug: random weights (no download).')
    parser.add_argument('--force', action='store_true', help='Redo finished steps instead of skipping them.')
    args = parser.parse_args()

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env.setdefault('PYTHONWARNINGS', 'ignore::FutureWarning')   # keeps scikit-learn deprecation notices out of the log
    if args.smoke:
        env['DR_SMOKE'] = '1'
    if args.no_pretrained:
        env['DR_NO_PRETRAINED'] = '1'
    if args.results_name:
        env['DR_RESULTS'] = args.results_name
    post = [s for s in POST_STAGES if s in (args.stages or [])]
    if args.stages and not set(args.stages) - set(POST_STAGES):   # only post-processing requested
        run_post(post, env, args.force)
        return
    os.environ.update({k: env[k] for k in ('DR_SMOKE', 'DR_NO_PRETRAINED', 'DR_RESULTS') if k in env})
    sys.path.insert(0, str(SRC))
    import common as C  # noqa: E402  (reads the environment set above)

    run_backbones = list(C.DEFAULT_BACKBONES) + EXTRA_IN_MAIN
    backbones = args.backbones or run_backbones
    unknown = set(backbones) - set(C.BACKBONES)
    if unknown:
        parser.error(f'Unknown backbones: {sorted(unknown)}')
    extra = sorted(set(backbones) - set(run_backbones))
    if extra and C.RESULTS_NAME == C.DEFAULT_RUN:
        parser.error(f'{extra} are not part of run main: give their run its own name, e.g. --results-name {extra[0]}')
    requested = [s for s in (args.stages or STAGES) if s in STAGES]
    stages = [s for s in STAGES if s in requested]
    open_test = (args.stages is None or 'evaluate' in args.stages) and not args.val_only
    if 'evaluate' not in stages:
        stages.append('evaluate')
    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(C.LOG_DIR / 'run_all.log', 'a', encoding='utf-8')
    started = time.perf_counter()

    # results belong to the code that produced them: track which stages still need recomputing
    sig_path = C.RESULTS / 'signature.json'
    train_sig, down_sig = C.training_signature(), C.downstream_signature()
    state = json.loads(sig_path.read_text()) if sig_path.is_file() else {}
    train_stale, down_stale = set(state.get('train_stale', [])), set(state.get('downstream_stale', []))
    if state and state.get('training') != train_sig:
        if not args.force:
            parser.error(f'{C.RESULTS} was produced with different settings, data, network or training code. '
                         'Rerun with --force to retrain everything, or move the old results folder away.')
        train_stale, down_stale = {'cache', 'B', 'C', 'D'}, {'B', 'C', 'D'}
    elif args.force:
        train_stale, down_stale = {'cache', 'B', 'C', 'D'}, {'B', 'C', 'D'}
    if state and state.get('downstream') != down_sig:
        down_stale |= {'B', 'C', 'D'}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    import importlib.metadata as md
    versions = {}
    for package in ['torch', 'torchvision', 'timm', 'scikit-learn', 'numpy', 'pandas', 'xgboost']:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None

    def save_state():
        sig_path.write_text(json.dumps({'training': train_sig, 'downstream': down_sig,
                                        'train_stale': sorted(train_stale), 'downstream_stale': sorted(down_stale),
                                        'python': sys.version.split()[0], 'packages': versions,
                                        'updated': time.strftime('%Y-%m-%d %H:%M')}, indent=2) + '\n')
    save_state()   # written before any work, so an interrupted run keeps the stale marks

    def say(text):
        print(text, flush=True)
        log.write(text + '\n')
        log.flush()

    def run(script, *extra, device=False, workers=False, force=False, models=True):
        cmd = [sys.executable, str(SRC / script), *map(str, extra)]
        if models:
            cmd += ['--backbones', *backbones]
        if device:
            cmd += ['--device', args.device]
        if workers:
            cmd += ['--workers', str(args.workers)]
        if force:
            cmd.append('--force')
        say(f'\n$ {" ".join(cmd[1:])}')
        t0 = time.perf_counter()
        process = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        log.flush()
        if process.wait() == 3 and '--dose-gate' in cmd:
            say('STOPPED by the dose check (see the message above). Nothing after stage B was run and no test performance '
                'was computed. Adjust the degradation levels and rerun with --force, or pass --skip-dose-check.')
            sys.exit(3)
        if process.returncode != 0:
            say(f'FAILED ({script}, exit code {process.returncode}). Fix the problem and rerun the same command; '
                'finished steps will be skipped.')
            sys.exit(process.returncode)
        say(f'--- done in {time.perf_counter() - t0:.0f}s')

    def train(model_sets, redo):
        for model_set in model_sets:
            run('train.py', '--model-set', model_set, device=True, workers=True, force=redo)

    say(f'=== run_all {time.strftime("%Y-%m-%d %H:%M")} | stages {stages} | backbones {backbones} | '
        f'smoke {args.smoke} | test performance {"computed at the end" if open_test else "not computed"} | '
        f'code {train_sig}/{down_sig} | stale: train {sorted(train_stale) or "-"}, downstream {sorted(down_stale) or "-"} | results {C.RESULTS}')
    if state and state.get('downstream') != down_sig:
        say('feature / fusion / scoring code changed: those steps will be recomputed for every stage; networks are kept')
    for stage in stages:
        say(f'\n=================== stage {stage} ===================')
        redo_train, redo_down = stage in train_stale, stage in down_stale
        if stage == 'check':
            run('check_data.py', models=False)
        elif stage == 'cache':
            run('build_cache.py', workers=True, models=False, force=redo_train)
        elif stage == 'B':
            train(C.STAGE_MODEL_SETS['B'], redo_train)
            run('extract_features.py', '--variant', 'B', device=True, workers=True, force=redo_down)
            run('fit_fusion.py', '--stage', 'A', force=redo_down)
            run('fit_fusion.py', '--stage', 'B', force=redo_down)
            for model_set in C.STAGE_MODEL_SETS['B']:
                run('predict_endtoend.py', '--model-set', model_set, device=True, workers=True, force=redo_down)
            run('extract_features.py', '--variant', 'Bj', device=True, workers=True, force=redo_down)   # P2 decomposition
            run('fit_fusion.py', '--stage', 'Bj', force=redo_down)
            if args.stages is None and not args.smoke and not args.skip_dose_check:
                say('\nstep 0: validation-only evaluation with the dose gate (no test performance is computed)')
                run('evaluate_all.py', '--val-only', '--dose-gate')
        elif stage == 'C':
            train(C.STAGE_MODEL_SETS['C'], redo_train)
            for variant in ['C1', 'C2']:
                run('extract_features.py', '--variant', variant, device=True, workers=True, force=redo_down)
                run('fit_fusion.py', '--stage', variant, force=redo_down)
            for model_set in C.STAGE_MODEL_SETS['C']:
                run('predict_endtoend.py', '--model-set', model_set, device=True, workers=True, force=redo_down)
        elif stage == 'D':
            run('fit_fusion.py', '--stage', 'D', force=redo_down)
        elif stage == 'evaluate':
            extra = [] if open_test else ['--val-only']
            if args.allow_incomplete:
                extra.append('--allow-incomplete')
            say('final evaluation: test + inadequate performance' if open_test else
                'validation-only evaluation (test performance not computed; run `--stages evaluate` at the end)')
            run('evaluate_all.py', *extra)
        train_stale.discard(stage)
        down_stale.discard(stage)
        save_state()
    say(f'\nAll requested stages finished in {(time.perf_counter() - started) / 60:.1f} min.')
    say(f'Results: {C.SUMMARY_DIR} (key_results.csv, metrics.csv, figures/); log: {C.LOG_DIR / "run_all.log"}')
    log.close()
    if post:
        run_post(post, env, args.force)


def run_post(post, env, force):
    """Optional post-processing of finished runs (no training): extra analyses."""
    scripts = {'extra': ROOT / 'src' / 'analysis_extra.py'}
    for stage in post:
        cmd = [sys.executable, str(scripts[stage])]
        print(f'\n=================== {stage} ===================\n$ {" ".join(cmd[1:])}', flush=True)
        code = subprocess.call(cmd, cwd=ROOT, env=env)
        if code != 0:
            print(f'FAILED ({scripts[stage].name}, exit code {code}).', flush=True)
            sys.exit(code)


if __name__ == '__main__':
    main()
