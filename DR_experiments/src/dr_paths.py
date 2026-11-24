"""Every folder location of the project, in one place.

common.py imports these names, so the other scripts keep using C.RESULTS, C.CACHE, C.MODELS_DIR, ...
This file is deliberately NOT part of the training signature (common.TRAINING_FILES): moving folders is not an
experimental change and must not force retraining. Settings that change an experiment stay in common.py.

Layout (project root = the folder that contains run_all.py):
    data/                     brset_diabetic*.csv, dr_grade.csv, images/
    data/cache/               resized and pre-degraded photos, quality signal (rebuilt by the `cache` stage)
    weights/pretrained/       ImageNet weights of the backbones (Hugging Face cache, offline use)
    weights/trained/<run>/    trained networks: <model_set>/<backbone>/fold<k>/best.pt + metrics.json
    results/<run>/            features, predictions, summary, logs, signature.json of one run
A run is selected with run_all.py --results-name (environment variable DR_RESULTS); the default run is 'main'
(five backbones: the four of the original main run and PVT-v2, which was trained later as run 'pvt' and then
merged into 'main').
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
IMAGES = DATA / 'images'                       # flat folder: data/images/<image_id>.jpg
DIABETIC_CSV = DATA / 'brset_diabetic.csv'      # train / val / test (diabetic patients, adequate quality)
INADEQUATE_CSV = DATA / 'brset_diabetic_inadequate.csv'   # real low-quality photos, unseen patients
GRADE_CSV = DATA / 'dr_grade.csv'               # ICDR grade 0-4 per photo (analysis only, never a model input)
WEIGHTS = ROOT / 'weights'
HF_CACHE = WEIGHTS / 'pretrained'               # pretrained weights for offline use
if 'HF_HOME' not in os.environ and HF_CACHE.is_dir():
    os.environ['HF_HOME'] = str(HF_CACHE)

SMOKE = os.environ.get('DR_SMOKE') == '1'               # tiny end-to-end check (run_all.py --smoke)
NO_PRETRAINED = os.environ.get('DR_NO_PRETRAINED') == '1'   # debugging only
# Run name: 'main' unless run_all.py --results-name sets DR_RESULTS. An extra backbone is run under its own name,
# so the finished four-backbone results (and their signature.json) are never touched.
DEFAULT_RUN = 'main'
RESULTS_NAME = os.environ.get('DR_RESULTS') or DEFAULT_RUN
_RUN = RESULTS_NAME + '_smoke' if SMOKE else RESULTS_NAME
RESULTS = ROOT / 'results' / _RUN
CACHE = DATA / ('cache_smoke' if SMOKE else 'cache')
MODELS_DIR = WEIGHTS / 'trained' / _RUN
FEATURES_DIR = RESULTS / 'features'
PREDICTIONS_DIR = RESULTS / 'predictions'
SUMMARY_DIR = RESULTS / 'summary'
FIGURES_DIR = SUMMARY_DIR / 'figures'
LOG_DIR = RESULTS / 'logs'
