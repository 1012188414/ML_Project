"""Extra analyses for the DR report (experiments A-C only). Reads saved predictions; trains nothing.

  (2) Mechanism: how the two-stage fusion compensates. The fusion logit splits exactly into
      image term = coef_image * z  and  clinical term = everything else (constant for a photo, because the
      clinical features do not change with degradation). Per condition it reports how well the image term
      still separates DR from no DR, and how much of the fused logit's spread comes from the clinical term.
  (3) Complementarity: why robust training (C2) and clinical information add up. 2 x 2 design
      (B / C2 training x image only / + clinical), split by ICDR grade:
        a) AP of each grade's DR photos against all no-DR photos (threshold-free), with patient bootstrap
        b) which missed DR photos each remedy rescues at the fixed (Youden, clean validation) threshold

Usage (from anywhere; also `python run_all.py --stages extra`):
    python src/analysis_extra.py                       # all 5 backbones of run main
    python src/analysis_extra.py --backbones convnextv2_nano densenet169 efficientnet_b3 mobilenetv3_large_150d --out results/extra_analysis_4bb
Each backbone's predictions are taken from the first folder in --results-dirs that has them (default: results/main,
which holds all five backbones).
Output: results/extra_analysis/ (or --out). Everything here is exploratory: test photos, 95% intervals, no correction.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

BACKBONES = ["convnextv2_nano", "densenet169", "efficientnet_b3", "mobilenetv3_large_150d", "pvt_v2_b1"]
LEVELS = {"blur": ["blur_s1", "blur_s2", "blur_s3", "blur_s5"], "down": ["down_x2", "down_x3", "down_x4", "down_x6"]}
MAIN8 = LEVELS["blur"] + LEVELS["down"]
SEVERITY = {"clean": 0, "blur_s1": 1, "blur_s2": 2, "blur_s3": 3, "blur_s5": 4,
            "down_x2": 1, "down_x3": 2, "down_x4": 3, "down_x6": 4}
GRADES = {"mild NPDR (ICDR 1)": [1], "moderate NPDR (ICDR 2)": [2], "severe NPDR / PDR (ICDR 3-4)": [3, 4],
          "all DR (ICDR 1-4)": [1, 2, 3, 4]}
COLS = ["setting", "classifier", "image_id", "patient_id", "eval_set", "condition", "y", "prob",
        "prob_f0", "prob_f1", "prob_f2", "prob_f3"]
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def average_precision(y, s):
    """Same definition as sklearn.metrics.average_precision_score (step-wise, ties grouped)."""
    y = np.asarray(y)
    n_pos = y.sum()
    if n_pos == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = distinct + 1 - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def load(results, experiment, backbone):
    """Two-stage image-only and image+all predictions (LR, C = 1) of one experiment and backbone."""
    folder = results / "predictions" / "two_stage" / experiment
    coefs = json.loads((folder / f"{backbone}_models.json").read_text(encoding="utf-8"))
    for key in ("lr:image", "lr:image+all"):
        if key not in coefs:
            raise KeyError(f"{key} missing in {folder / (backbone + '_models.json')}")
    frames = []
    for chunk in pd.read_csv(folder / f"{backbone}.csv.gz", usecols=COLS, chunksize=400_000):
        frames.append(chunk[(chunk.classifier == "lr") & chunk.setting.isin(["image", "image+all"])
                            & chunk.eval_set.isin(["val", "test"])])
    data = pd.concat(frames, ignore_index=True)
    key = ["eval_set", "condition", "image_id"]
    if data.duplicated(key + ["setting"]).any():
        raise ValueError(f"Duplicate predictions in {experiment}/{backbone}; filter on another column")
    img = data[data.setting == "image"].set_index(key).sort_index()
    fus = data[data.setting == "image+all"].set_index(key).sort_index()
    if not img.index.equals(fus.index):
        raise ValueError(f"Image-only and fusion rows do not match in {experiment}/{backbone}")

    c_img = coefs["lr:image"]["coefficients"]["image_logit"]
    b_img = coefs["lr:image"]["intercept"]
    c_fus = coefs["lr:image+all"]["coefficients"]["image_logit"]
    # per fold: logit(p_img) = c_img * z + b_img ; logit(p_fus) = c_fus * z + clinical term (incl. intercept)
    z = np.stack([(logit(img[f"prob_f{f}"]) - b_img) / c_img for f in range(4)], axis=1)
    clin = np.stack([logit(fus[f"prob_f{f}"]) for f in range(4)], axis=1) - c_fus * z
    out = img[["patient_id", "y"]].copy()
    out["p_image"], out["p_fusion"] = img["prob"].to_numpy(), fus["prob"].to_numpy()
    out["image_term"] = c_fus * z.mean(axis=1)
    out["clinical_term"] = clin.mean(axis=1)
    out = out.reset_index()
    # the clinical term must not depend on the photo's condition
    spread = out[out.eval_set == "test"].groupby("image_id")["clinical_term"].agg(lambda v: v.max() - v.min())
    out.attrs["clinical_term_max_range"] = float(spread.max())
    return out


# ----------------------------------------------------------------------------------------------- (2)
def separation(y, s):
    pos, neg = s[y == 1], s[y == 0]
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2)
    return float((pos.mean() - neg.mean()) / pooled)


def mechanism(preds, backbones, experiments):
    rows = []
    for exp in experiments:
        for bb in backbones:
            test = preds[exp, bb].query("eval_set == 'test'")
            for cond, g in test.groupby("condition"):
                y = g.y.to_numpy()
                ti, tc = g.image_term.to_numpy(), g.clinical_term.to_numpy()
                rows.append({
                    "experiment": exp, "backbone": bb, "condition": cond,
                    "family": next((f for f, lv in LEVELS.items() if cond in lv), "clean" if cond == "clean" else "other"),
                    "severity_level": SEVERITY.get(cond, np.nan),
                    "AP_image": average_precision(y, g.p_image.to_numpy()),
                    "AP_fusion": average_precision(y, g.p_fusion.to_numpy()),
                    "image_term_separation_d": separation(y, ti),
                    "image_term_sd": ti.std(ddof=1),
                    "image_term_mean_abs": np.abs(ti).mean(),
                    "clinical_term_sd": tc.std(ddof=1),
                    "clinical_share_of_variance": tc.var(ddof=1) / (tc.var(ddof=1) + ti.var(ddof=1)),
                    "corr_image_clinical_term": np.corrcoef(ti, tc)[0, 1],
                })
    table = pd.DataFrame(rows)
    table["clinical_gain_AP"] = table.AP_fusion - table.AP_image
    metrics = [c for c in table.columns if c not in ("experiment", "backbone", "condition", "family", "severity_level")]
    summary = (table.groupby(["experiment", "condition", "family", "severity_level"], dropna=False)[metrics]
               .mean().reset_index().sort_values(["experiment", "family", "severity_level"]))
    return table, summary


# ----------------------------------------------------------------------------------------------- (3)
def youden_threshold(frame):
    clean_val = frame.query("eval_set == 'val' and condition == 'clean'")
    out = {}
    for col in ("p_image", "p_fusion"):
        fpr, tpr, thr = roc_curve(clean_val.y, clean_val[col])
        out[col] = float(thr[np.argmax(tpr - fpr)])
    return out


MODELS = {"B image": ("B", "p_image"), "B + clinical": ("B", "p_fusion"),
          "C2 image": ("C2", "p_image"), "C2 + clinical": ("C2", "p_fusion")}


def complementarity(preds, backbones, grades, n_boot, seed):
    """a) AP per grade (grade's DR photos vs all no-DR photos) with a shared patient bootstrap;
       b) rescue overlap at fixed thresholds."""
    rng = np.random.default_rng(seed)
    first = preds["B", backbones[0]].query("eval_set == 'test' and condition == 'clean'")
    patients = np.sort(first.patient_id.unique())
    draws = rng.integers(0, len(patients), size=(n_boot, len(patients)))   # shared by all models
    conds = ["clean"] + MAIN8

    ap_rows, rescue_rows = [], []
    point = {}                     # (backbone, grade, cond, model) -> AP
    boot = {}                      # same key -> array over draws
    for bb in backbones:
        tests = {name: preds[exp, bb].query("eval_set == 'test'") for name, (exp, _) in MODELS.items()}
        thresholds = {name: youden_threshold(preds[exp, bb])[col] for name, (exp, col) in MODELS.items()}
        for cond in conds:
            frames = {}
            for name in MODELS:      # plain mask: query("... @cond") fails inside comprehensions on Python < 3.12
                t = tests[name]
                frames[name] = t[t.condition == cond].set_index("image_id").sort_index()
            base = frames["B image"]
            y = base.y.to_numpy()
            grade = base.dr_icdr.to_numpy()
            pid = base.patient_id.to_numpy()
            scores = {name: frames[name][MODELS[name][1]].to_numpy() for name in MODELS}
            # patient -> row indices, for the bootstrap
            pos_of = {p: np.flatnonzero(pid == p) for p in patients}
            boot_idx = [np.concatenate([pos_of[patients[k]] for k in d]) for d in draws]
            for gname, gvals in grades.items():
                keep = (y == 0) | np.isin(grade, gvals)
                for name in MODELS:
                    s = scores[name]
                    point[bb, gname, cond, name] = average_precision(y[keep], s[keep])
                    vals = np.empty(n_boot)
                    for b, idx in enumerate(boot_idx):
                        idx = idx[keep[idx]]
                        vals[b] = average_precision(y[idx], s[idx])
                    boot[bb, gname, cond, name] = vals

            # b) rescue analysis at fixed thresholds, only for DR photos missed by the B image model
            flagged = {name: scores[name] >= thresholds[name] for name in MODELS}
            for gname, gvals in list(grades.items()) + [("no DR (false positives)", [0])]:
                in_group = np.isin(grade, gvals) if gvals != [0] else (y == 0)
                if gvals == [0]:
                    rescue_rows.append({"backbone": bb, "condition": cond, "group": gname, "n_photos": int(in_group.sum()),
                                        **{f"flagged: {m}": flagged[m][in_group].mean() for m in MODELS}})
                    continue
                missed = in_group & ~flagged["B image"]
                n_missed = int(missed.sum())
                clin = flagged["B + clinical"][missed]
                rob = flagged["C2 image"][missed]
                both_models = flagged["C2 + clinical"][missed]
                rescue_rows.append({
                    "backbone": bb, "condition": cond, "group": gname, "n_photos": int(in_group.sum()),
                    **{f"flagged: {m}": flagged[m][in_group].mean() for m in MODELS},
                    "missed by B image": n_missed,
                    "rescued by clinical (B + clinical flags it)": clin.mean() if n_missed else np.nan,
                    "rescued by robust training (C2 image flags it)": rob.mean() if n_missed else np.nan,
                    "rescued by either remedy": (clin | rob).mean() if n_missed else np.nan,
                    "rescued by both remedies (overlap)": (clin & rob).mean() if n_missed else np.nan,
                    "both / expected if independent": ((clin & rob).mean() / (clin.mean() * rob.mean()))
                    if n_missed and clin.mean() * rob.mean() > 0 else np.nan,
                    "rescued by the combined model (C2 + clinical)": both_models.mean() if n_missed else np.nan,
                })

    # summaries: average over backbones (and over the 8 degraded conditions), then contrasts
    def contrast(get, gname, cset):
        """returns point and bootstrap array of the backbone-mean of the condition-mean of get(...)"""
        pt = np.mean([np.mean([get(point, bb, gname, c) for c in cset]) for bb in backbones])
        bt = np.mean([np.mean([get(boot, bb, gname, c) for c in cset], axis=0) for bb in backbones], axis=0)
        return pt, bt

    effects = {
        "AP, B image": lambda d, bb, g, c: d[bb, g, c, "B image"],
        "AP, B + clinical": lambda d, bb, g, c: d[bb, g, c, "B + clinical"],
        "AP, C2 image": lambda d, bb, g, c: d[bb, g, c, "C2 image"],
        "AP, C2 + clinical": lambda d, bb, g, c: d[bb, g, c, "C2 + clinical"],
        "gain from clinical (B + clinical - B image)": lambda d, bb, g, c: d[bb, g, c, "B + clinical"] - d[bb, g, c, "B image"],
        "gain from robust training (C2 image - B image)": lambda d, bb, g, c: d[bb, g, c, "C2 image"] - d[bb, g, c, "B image"],
        "gain from both (C2 + clinical - B image)": lambda d, bb, g, c: d[bb, g, c, "C2 + clinical"] - d[bb, g, c, "B image"],
        "interaction (both - clinical - robust)": lambda d, bb, g, c: (d[bb, g, c, "C2 + clinical"] - d[bb, g, c, "C2 image"]
                                                                      - d[bb, g, c, "B + clinical"] + d[bb, g, c, "B image"]),
    }
    for cname, cset in (("clean", ["clean"]), ("mean of 8 blur / downsampling", MAIN8)):
        for gname in grades:
            for ename, get in effects.items():
                pt, bt = contrast(get, gname, cset)
                lo, hi = np.nanpercentile(bt, [2.5, 97.5])
                ap_rows.append({"condition": cname, "group": gname, "quantity": ename,
                                "estimate": pt, "ci95_low": lo, "ci95_high": hi})
    ap_table = pd.DataFrame(ap_rows)
    per_bb = pd.DataFrame([{"backbone": bb, "group": g, "condition": c, "model": m, "AP": v}
                           for (bb, g, c, m), v in point.items()])
    rescue = pd.DataFrame(rescue_rows)
    rescue_summary = rescue.assign(degraded=rescue.condition.isin(MAIN8)).query("degraded").drop(
        columns=["degraded", "backbone", "condition"]).groupby("group", sort=False).mean(numeric_only=True).reset_index()
    return ap_table, per_bb, rescue, rescue_summary


README = """# extra_analysis: extra analyses (exploratory, experiments A-C)

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
"""


# ----------------------------------------------------------------------------------------------- figures
def figures(mech_summary, ap_table, out, n_backbones):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for exp, style in (("B", "-"), ("C2", "--")):
        for fam, color in (("blur", "tab:blue"), ("down", "tab:orange")):
            s = mech_summary[(mech_summary.experiment == exp) & mech_summary.condition.isin(["clean"] + LEVELS[fam])]
            s = s.sort_values("severity_level")
            label = f"{exp}, {fam}"
            axes[0].plot(s.severity_level, s.image_term_separation_d, style, marker="o", color=color, label=label)
            axes[1].plot(s.severity_level, s.clinical_share_of_variance, style, marker="o", color=color, label=label)
            axes[2].plot(s.severity_level, s.clinical_gain_AP, style, marker="o", color=color, label=label)
    for ax, title in zip(axes, ("Image evidence: DR vs no-DR separation (d)",
                                "Share of fused-logit variance from clinical term",
                                "Clinical gain in AP (fusion - image)")):
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["clean", "1", "2", "3", "4"])
        ax.set_xlabel("degradation level (blur s1/2/3/5, down x2/3/4/6)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Two-stage fusion: mean of {n_backbones} backbone(s), test photos (exploratory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_mechanism.png", dpi=200)
    plt.close(fig)

    sub = ap_table[(ap_table.condition == "mean of 8 blur / downsampling")
                   & ap_table.quantity.isin(["gain from clinical (B + clinical - B image)",
                                             "gain from robust training (C2 image - B image)",
                                             "gain from both (C2 + clinical - B image)"])]
    groups = list(GRADES)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    for i, (q, color) in enumerate(zip(sub.quantity.unique(), ("tab:green", "tab:purple", "tab:gray"))):
        s = sub[sub.quantity == q].set_index("group").loc[groups]
        x = np.arange(len(groups)) + (i - 1) * width
        ax.bar(x, s.estimate, width, color=color, label=q.split(" (")[0])
        ax.errorbar(x, s.estimate, yerr=[s.estimate - s.ci95_low, s.ci95_high - s.estimate], fmt="none",
                    ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=8)
    ax.set_ylabel("AP gain over B image (degraded, mean of 8)")
    ax.set_title("Where each remedy helps: AP of each grade's DR photos vs all no-DR photos (95% CI)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_complementarity.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parent.parent),
                        help="project root (contains data/ and results/); default: the folder above src/")
    parser.add_argument("--results-dirs", nargs="+", default=["results/main"],
                        help="folders searched in order for each backbone's predictions")
    parser.add_argument("--backbones", nargs="+", default=BACKBONES)
    parser.add_argument("--out", default="results/extra_analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.project)
    out = root / args.out
    dirs = [root / d for d in args.results_dirs]
    if any(out.resolve() == d.resolve() for d in dirs) or out.name == "results":
        raise SystemExit("--out must be a new folder; this script never writes into a results folder")
    source = {}
    for bb in args.backbones:
        found = [d for d in dirs if (d / "predictions" / "two_stage" / "B" / f"{bb}.csv.gz").exists()]
        if not found:
            raise SystemExit(f"No predictions for {bb} in {args.results_dirs}")
        source[bb] = found[0]
        print(f"{bb}: predictions from {found[0].name}/", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    grade = pd.read_csv(root / "data" / "dr_grade.csv").set_index("image_id")["dr_icdr"]

    preds, checks = {}, []
    for exp in ("B", "C1", "C2"):
        for bb in args.backbones:
            frame = load(source[bb], exp, bb)
            frame["dr_icdr"] = frame.image_id.map(grade)
            if frame.dr_icdr.isna().any():
                raise ValueError("ICDR grade missing for some photos")
            if not ((frame.dr_icdr > 0).astype(int) == frame.y).all():
                raise ValueError(f"ICDR grade and DR label disagree in {exp}/{bb}")
            preds[exp, bb] = frame
            checks.append({"experiment": exp, "backbone": bb,
                           "clinical_term_max_range_across_conditions": frame.attrs["clinical_term_max_range"]})
            print(f"loaded {exp}/{bb}: {len(frame):,} rows", flush=True)

    mech, mech_summary = mechanism(preds, args.backbones, ("B", "C1", "C2"))
    mech.to_csv(out / "mechanism_by_backbone.csv", index=False)
    mech_summary.to_csv(out / "mechanism_summary.csv", index=False)
    print("mechanism done", flush=True)

    ap_table, per_bb, rescue, rescue_summary = complementarity(preds, args.backbones, GRADES, args.n_boot, args.seed)
    ap_table.to_csv(out / "complementarity_ap_by_grade.csv", index=False)
    per_bb.to_csv(out / "complementarity_ap_by_grade_per_backbone.csv", index=False)
    rescue.to_csv(out / "complementarity_rescue_by_condition.csv", index=False)
    rescue_summary.to_csv(out / "complementarity_rescue_summary.csv", index=False)
    print("complementarity done", flush=True)

    # consistency with the main pipeline: these AP gains should equal table_C_gain.csv (two-stage)
    main4 = BACKBONES[:4]
    if sorted(args.backbones) == sorted(BACKBONES):
        ref = root / "results" / "main" / "summary" / "table_C_gain.csv"
    elif sorted(args.backbones) == sorted(main4):
        ref = root / "archive" / "merged_pvt_run" / "summary_main_4bb" / "table_C_gain.csv"
    else:
        ref = None
    if ref is not None and ref.exists():
        mine = mech_summary.pivot(index="condition", columns="experiment", values="clinical_gain_AP")
        theirs = pd.read_csv(ref).set_index("condition")
        diff = {exp: float((mine[exp] - theirs[f"two-stage {exp} gain"]).reindex(mine.index).abs().max())
                for exp in ("B", "C1", "C2")}
        checks.append({"experiment": "all", "backbone": f"{len(args.backbones)}-backbone mean vs {ref.relative_to(root).as_posix()}",
                       "max_abs_diff_vs_table_C_gain": max(diff.values())})
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)

    figures(mech_summary, ap_table, out, len(args.backbones))
    (out / "README_extra.md").write_text(README, encoding="utf-8")
    print(f"done in {time.time() - started:.0f} s; results in {out}", flush=True)


if __name__ == "__main__":
    main()
