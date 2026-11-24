# signature.json of this run

On 2026-11-23 the project folders were reorganised (see README.md in the project root). The folder paths moved
from src/common.py to src/dr_paths.py; nothing else in the training code changed. Because run_all.py hashes
src/common.py, the stored training signature was updated from `ed997d540a2fdd6a` to `600220d1fbbfaa83`.
The downstream signature (`310b4fcdb9031579`) did not change. The previous signature.json is kept in
archive/pre_reorg/signatures/main.json.
The stored hash did NOT match the code before the reorganisation either: this run was made before src/common.py and src/nets.py were extended with the extra backbones (PVT-v2 and others; see results/combined_5bb/README_5bb.md). Its four networks were not retrained.

Software recorded in signature.json before the reorganisation (the versions this run's results were produced
with; run_all.py overwrites them with the current environment's versions at every run):
python 3.10.11; torch 2.10.0+cu128, torchvision 0.25.0+cu128, timm 1.0.31, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, xgboost 3.2.0

## PVT-v2 merged into this run (2026-11-23)

PVT-v2 was trained after the four main backbones, with the same protocol and code, as its own run
(`run_all.py --backbones pvt_v2_b1 --results-name pvt`). Its networks, features and predictions were then moved here
(run 'main'); its five-backbone evaluation (formerly results/combined_5bb/) is now results/main/summary/.
Both runs had the same code fingerprints (training `600220d1fbbfaa83`, downstream `310b4fcdb9031579`).
The clinical-only predictions (experiment A) of the two runs are the same up to floating-point rounding; the main
run's copy is used, PVT's is in archive/merged_pvt_run/.

The PVT run's own note (archive/merged_pvt_run/SIGNATURE_NOTE_pvt.md) records its software:
python 3.10.11; torch 2.14.1+cu132, torchvision 0.29.1+cu132, timm 1.0.31, scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, xgboost 3.2.0
