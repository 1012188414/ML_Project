# extra_analysis: extra analyses (exploratory, experiments A-C)

Made by `src/analysis_extra.py` from the saved predictions in results/main; no network is trained and
nothing in that folder is changed. Test photos; 95% intervals from a patient-level bootstrap; no
multiplicity correction. All summaries: per backbone first, then the mean over the backbones analysed (default: all 5, as in
results/main/summary).

## (2) Mechanism of compensation (two-stage fusion)
The fused logit splits exactly into an image term (fusion coefficient x standardised image logit) and a
clinical term (clinical features + intercept). The clinical term of a photo is the same in every condition
(`checks.csv`, `clinical_term_max_range_across_conditions`, should be about 1e-4 or less).
* `mechanism_by_backbone.csv`, `mechanism_summary.csv` (backbone mean), per experiment (B, C1, C2) and condition:
  - `image_term_separation_d`: how well the image term still separates DR from no DR (mean difference / pooled SD)
  - `clinical_share_of_variance`: Var(clinical term) / (Var(clinical term) + Var(image term))
  - `clinical_gain_AP`: AP(fusion) - AP(image); should equal `table_C_gain.csv` of results/main/summary (`checks.csv`)
* `fig_mechanism.png`: the three quantities against degradation level, B and C2.

## (3) Complementarity of robust training (C2) and clinical information
2 x 2: B / C2 training x image only / + clinical (two-stage), split by ICDR grade.
* `complementarity_ap_by_grade.csv`: AP of each grade's DR photos against all no-DR photos; gains from clinical,
  from robust training, from both, and the interaction (both - clinical - robust; about 0 = additive).
  Clean and the mean of the 8 blur / downsampling conditions. `..._per_backbone.csv`: point values.
* `complementarity_rescue_summary.csv` (mean of the 8 degraded conditions and the backbones) and
  `..._by_condition.csv`: at each model's fixed threshold (Youden's J, clean validation photos), the share of
  DR photos missed by the B image model that each remedy flags, their overlap, and overlap / (product of the two
  shares): below 1 means the remedies tend to rescue different photos. No-DR row: false-positive rates.
* `fig_complementarity.png`: AP gains by grade with 95% intervals.

Small groups (mild NPDR: 18 test photos, 13 patients): descriptive only.
