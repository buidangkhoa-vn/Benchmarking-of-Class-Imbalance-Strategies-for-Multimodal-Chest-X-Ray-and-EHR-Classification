"""Tables, tests and figures of the paper from the experiment results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import KEY, best_rows, corrected_t, paired, select, strategy_effects, summary_rows

NAMES = {"ehr": "EHR only", "img_scores": "Image: 11 scores", "fusion_scores": "Fusion: 11 scores", "img_densenet": "Image: DenseNet",
         "fusion_densenet": "Fusion: DenseNet", "img_raddino": "Image: RAD-DINO", "fusion_raddino": "Fusion: RAD-DINO",
         "history_a": "History", "history_c": "History", "fusion_history_a": "Fusion: RAD-DINO + history",
         "fusion_history_c": "Fusion: RAD-DINO + history"}
GROUP = {"ehr": "EHR only", "img_scores": "Image only", "img_densenet": "Image only", "img_raddino": "Image only",
         "fusion_scores": "Early fusion", "fusion_densenet": "Early fusion", "fusion_raddino": "Early fusion"}


def load(run_dir: Path, name: str):
    f = run_dir / name / "results.csv"
    if not f.exists():
        return None, None
    return pd.read_csv(f), pd.read_csv(run_dir / name / "baselines.csv")


def baseline_summary(B: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["trivial_f2", "lookup_f2", "lookup_hit_rate", "prior_rule_f2"] if c in B]
    return B.groupby(["protocol", "seed", "fold"])[cols].mean().groupby("protocol").mean().reset_index()


def results_table(R: pd.DataFrame, n_tests: int) -> pd.DataFrame:
    """Table IV / V rows: configuration selected on inner validation, macro over targets, corrected tests over folds."""
    T = summary_rows(select(R), n_tests)
    T.insert(2, "representation", T.feature_set.map(NAMES).fillna(T.feature_set))
    return T


def strategy_by_representation(R: pd.DataFrame) -> pd.DataFrame:
    """Fig. 2: change in test F2 against no augmentation, fold-level means over targets, classifiers and the
    representations of each group (EHR only, image only, early fusion)."""
    R = R[(R.strategy != "native") & R.feature_set.isin(GROUP)]
    idx = ["protocol", "feature_set", "target", "classifier", "fold"]
    base = R[R.strategy == "none"].set_index(idx).test_f2
    rows = []
    for s in ["class_weight", "ros", "smote", "smote_nc"]:
        d = R[R.strategy == s].set_index(idx).test_f2; i = d.index.intersection(base.index)
        if not len(i):
            continue
        diff = (d.loc[i] - base.loc[i]).reset_index(); diff["group"] = diff.feature_set.map(GROUP)
        for (p, g), dp in diff.groupby(["protocol", "group"]):
            m, lo, hi, pv = corrected_t(dp.groupby("fold").test_f2.mean().values)
            rows.append({"protocol": p, "group": g, "strategy": s, "mean": m, "lo": lo, "hi": hi, "p": pv})
    return pd.DataFrame(rows)


def strategy_pooled(R: pd.DataFrame) -> pd.DataFrame:
    """Each strategy against no augmentation pooled over all representations of a protocol (fold-level means over
    representations, targets and classifiers), as quoted in the text for P1 and P1-A."""
    R = R[R.strategy != "native"]
    idx = ["protocol", "feature_set", "target", "classifier", "seed", "fold"]
    base = R[R.strategy == "none"].set_index(idx).test_f2
    rows = []
    for (p, s), d in R[R.strategy != "none"].groupby(["protocol", "strategy"]):
        d = d.set_index(idx).test_f2; i = d.index.intersection(base.index)
        if not len(i):
            continue
        m, lo, hi, pv = corrected_t((d.loc[i] - base.loc[i]).groupby(level=["seed", "fold"]).mean().values)
        rows.append({"protocol": p, "strategy": s, "dF2": m, "lo": lo, "hi": hi, "p": pv})
    return pd.DataFrame(rows)


def comparisons(S: pd.DataFrame, pairs) -> pd.DataFrame:
    rows = []
    for (pa, fa), (pb, fb) in pairs:
        a = S[(S.protocol == pa) & (S.feature_set == fa)]; b = S[(S.protocol == pb) & (S.feature_set == fb)]
        if len(a) and len(b):
            for metric in ["test_f2", "test_roc_auc"]:
                rows.append({"a": f"{pa}:{fa}", "b": f"{pb}:{fb}", **paired(a, b, metric)})
    return pd.DataFrame(rows)


def seen_unseen(S: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["test_f2", "seen_f2", "unseen_f2", "unseenclean_f2", "unseen_trivial_f2", "seen_n", "unseen_n"] if c in S]
    fm = S[S.split == "record"].groupby(["protocol", "feature_set", "fold"])[cols].mean()
    return fm.groupby(["protocol", "feature_set"]).agg(["mean", "std"]).round(4).reset_index()


def per_target(S: pd.DataFrame) -> pd.DataFrame:
    return S.groupby(KEY)[["test_f2", "test_trivial_f2", "test_roc_auc", "test_pr_auc"]].mean().reset_index()


def tuning(R: pd.DataFrame) -> dict:
    """Default vs tuned configurations, chosen on inner validation within a family or across families."""
    by = ("classifier",)
    sel = {"LR_default": R[R.classifier == "LR"], "LR_tuned": select(R[R.family == "LR"], by)}
    for f in ["RF", "ET", "LGBM"]:
        sel[f"{f}_default"] = R[R.classifier == f]; sel[f"{f}_tuned"] = select(R[R.family == f], by)
    sel["trees_default"] = select(R[R.classifier.isin(["RF", "ET", "LGBM"])], by)
    sel["trees_tuned"] = select(R[R.family.isin(["RF", "ET", "LGBM"])], by)
    sel["overall_tuned"] = select(R, by)
    summary = pd.concat([summary_rows(S).assign(selection=k) for k, S in sel.items()], ignore_index=True)
    tests = []
    for a, b in [("trees_tuned", "LR_tuned"), ("trees_tuned", "trees_default"), ("LR_tuned", "LR_default")]:
        for (p, fs), Sa in sel[a].groupby(["protocol", "feature_set"]):
            Sb = sel[b][(sel[b].protocol == p) & (sel[b].feature_set == fs)]
            for metric in ["test_f2", "test_roc_auc"]:
                tests.append({"protocol": p, "feature_set": fs, "comparison": f"{a} - {b}", **paired(Sa, Sb, metric)})
    picked = select(R, by).drop_duplicates(KEY)
    chosen = picked.groupby(["protocol", "family"]).size().unstack(fill_value=0).reset_index()
    chosen_cfg = picked.groupby(["protocol", "classifier"]).size().unstack(fill_value=0).reset_index()
    rec = R[R.split == "record"].groupby(["feature_set", "classifier"])[["test_f2", "seen_f2", "unseen_f2", "unseen_trivial_f2"]].mean()
    al = sel["overall_tuned"]
    al = al[al.split == "patient"].groupby(["protocol", "feature_set"])[["aligned_f2", "aligned_trivial_f2", "other_f2", "other_trivial_f2"]].mean()
    al["aligned_margin"] = al.aligned_f2 - al.aligned_trivial_f2; al["other_margin"] = al.other_f2 - al.other_trivial_f2
    S = sel["overall_tuned"]; sub = []
    for p in sorted(S[S.split == "patient_aligned"].protocol.unique()):
        for a, b in [("img_raddino", "ehr"), ("fusion_raddino", "ehr")]:
            sub.append(comparisons(S, [((p, a), (p, b))]))
    return {"tuning_summary": summary, "tuning_tests": pd.DataFrame(tests), "tuning_family_chosen": chosen, "tuning_classifier_chosen": chosen_cfg,
            "tuning_P1_per_config": rec.reset_index(), "tuning_aligned_vs_other": al.reset_index(),
            "tuning_aligned_subset_tests": pd.concat(sub, ignore_index=True) if sub else pd.DataFrame()}


def trees(R: pd.DataFrame) -> dict:
    """Number of trees / boosting rounds (100, 300, 1000) for each tree family, no augmentation: macro results and paired
    changes against the paper's 300 (same target and fold, corrected t-test over folds)."""
    m = R.classifier.str.extract(r"^(?P<base>.+?)(?:_n(?P<n>\d+))?$")
    R = R.assign(base=m.base, n_trees=np.where(R.family == "LR", 0, m.n.fillna(300).astype(float)).astype(int))
    cols = ["test_f2", "test_trivial_f2", "test_roc_auc", "test_pr_auc", "test_brier", "test_brier_prev"]
    fm = R.groupby(["protocol", "feature_set", "base", "n_trees", "seed", "fold"])[cols].mean().reset_index()
    summary = fm.groupby(["protocol", "feature_set", "base", "n_trees"])[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["delta"] = summary.test_f2_mean - summary.test_trivial_f2_mean
    rows = []
    for (p, fs, b), g in fm[fm.n_trees > 0].groupby(["protocol", "feature_set", "base"]):
        piv = g.set_index(["seed", "fold", "n_trees"])[["test_f2", "test_roc_auc", "test_pr_auc", "test_brier"]].unstack("n_trees")
        for n in sorted(g.n_trees.unique()):
            if n == 300 or 300 not in set(g.n_trees):
                continue
            for met in ["test_f2", "test_roc_auc", "test_pr_auc", "test_brier"]:
                mm, lo, hi, pv = corrected_t((piv[(met, n)] - piv[(met, 300)]).dropna().values)
                rows.append({"protocol": p, "feature_set": fs, "classifier": b, "comparison": f"n{n} - n300", "metric": met,
                             "diff": mm, "lo": lo, "hi": hi, "p": pv})
    return {"trees_summary": summary, "trees_tests": pd.DataFrame(rows)}


def sensitivity(R: pd.DataFrame) -> dict:
    """PCA size, age vs prior-diagnosis flags, a second fold assignment (repeated CV) and Brier scores (logistic
    regression)."""
    N = R[R.strategy == "none"]
    fm = N.groupby(["protocol", "feature_set", "seed", "fold"])[["test_f2", "test_trivial_f2", "test_roc_auc", "test_brier",
                                                                 "test_brier_prev", "pca_explained_var"]].mean().reset_index()
    macro = fm.groupby(["protocol", "feature_set", "seed"]).mean(numeric_only=True).drop(columns="fold").reset_index()
    def cmp(p, a, b, seeds, metric):
        x = N[(N.protocol == p) & (N.feature_set == a) & N.seed.isin(seeds)]
        y = N[(N.protocol == p) & (N.feature_set == b) & N.seed.isin(seeds)]
        return {"protocol": p, "a": a, "b": b, "seeds": "+".join(map(str, seeds)), **paired(x, y, metric)}
    rows = []
    for p in ["P2", "P4", "P4-C"]:
        for g in ["img_raddino", "fusion_raddino"]:
            for v in ["_pc32", "_pc128", "_full"]:
                for met in ["test_f2", "test_roc_auc"]:
                    rows.append(cmp(p, g + v, g, [42], met))
    for p, h in [("P4", "a"), ("P4-C", "c")]:
        for a, b in [("age_own", "age_released"), (f"prior_dx_{h}", "ehr"), (f"history_{h}", f"prior_dx_{h}"), (f"history_{h}", "ehr")]:
            for met in ["test_f2", "test_roc_auc"]:
                rows.append(cmp(p, a, b, [42], met))
        for a in ["img_raddino", "fusion_raddino", f"history_{h}"]:
            for met in ["test_f2", "test_roc_auc"]:
                rows.append(cmp(p, a, "ehr", [42, 43], met))
    for a in ["img_raddino", "fusion_raddino"]:
        for met in ["test_f2", "test_roc_auc"]:
            rows.append(cmp("P2", a, "ehr", [42, 43], met))
    P2 = R[(R.protocol == "P2") & R.feature_set.isin(["ehr", "fusion_raddino"])]
    for fs in ["ehr", "fusion_raddino"]:
        for st in ["class_weight", "ros", "smote_nc"]:
            x = P2[(P2.feature_set == fs) & (P2.strategy == st)]; y = P2[(P2.feature_set == fs) & (P2.strategy == "none")]
            for met in ["test_f2", "test_brier"]:
                rows.append({"protocol": "P2", "a": f"{fs}:{st}", "b": f"{fs}:none", "seeds": "42+43", **paired(x, y, met)})
    seeds = macro.pivot_table(index=["protocol", "feature_set"], columns="seed", values="test_f2").dropna()
    if seeds.shape[1] == 2:
        seeds["abs_diff"] = (seeds.iloc[:, 0] - seeds.iloc[:, 1]).abs()
    return {"sensitivity_macro": macro, "sensitivity_tests": pd.DataFrame(rows), "sensitivity_seed_variation": seeds.reset_index()}


RARE_TARGETS = ["scoliosis", "tuberculosis", "lung_infection"]      # fewest positive patients (45, 315, 336)


def without_rare_targets(R: pd.DataFrame, rare=RARE_TARGETS) -> dict:
    """Patient-separated cross-validation without the rarest targets: macro F2, margin over all-positive and
    strategy effects."""
    R = R[~R.target.isin(rare) & (R.split == "patient")]
    return {"rare_excluded_table": summary_rows(select(R)), "rare_excluded_strategies": strategy_effects(R)}


def per_fold_selection(R: pd.DataFrame) -> pd.DataFrame:
    """Selection within each outer fold (that fold's inner validation set only) against selection over all folds."""
    by = ["strategy", "classifier"]
    best = best_rows(R, KEY + ["seed", "fold"], by)
    a = best.groupby(["protocol", "feature_set", "fold"]).test_f2.mean().groupby(["protocol", "feature_set"]).mean()
    b = select(R, by).groupby(["protocol", "feature_set", "fold"]).test_f2.mean().groupby(["protocol", "feature_set"]).mean()
    return pd.DataFrame({"per_fold_selection": a, "selection_over_folds": b, "difference": a - b}).reset_index()


# ---------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e"})
    return plt


def figure_targets(pt_main: pd.DataFrame, pt_adm: pd.DataFrame | None, path):
    """Fig. 1: per-target margin of the selected EHR-only model over all-positive under P1, P1-A and P3."""
    plt = _style()
    order = ["pneumonia", "lung_cancer", "fracture", "bronchitis", "arthritis", "lung_infection", "tuberculosis", "scoliosis"]
    def m(T, p, t):
        r = T[(T.protocol == p) & (T.feature_set == "ehr") & (T.target == t)] if T is not None else T
        return float((r.test_f2 - r.test_trivial_f2).iloc[0]) if r is not None and len(r) else np.nan

    order = [t for t in order if t in set(pt_main.target)]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    for i, t in enumerate(order[::-1]):
        a1, a3 = m(pt_main, "P1", t), m(pt_main, "P3", t)
        aa = m(pt_adm, "P1-A", t) if pt_adm is not None else np.nan
        ax.plot([np.nanmin([a1, a3, aa]), np.nanmax([a1, a3, aa])], [i, i], "-", color="#c3c2b7", lw=1.2, zorder=1)
        ax.plot(a1, i, "o", ms=4.5, color="#eb6834", mec="white", mew=0.6, zorder=3, label="P1, released labels" if i == 0 else None)
        if pt_adm is not None:
            ax.plot(aa, i, "D", ms=4.0, color="#1baf7a", mec="#0b0b0b", mew=0.5, zorder=4, label="P1-A, admission-aligned" if i == 0 else None)
        ax.plot(a3, i, "o", ms=4.5, color="#2a78d6", mec="white", mew=0.6, zorder=3, label="P3 patient-level" if i == 0 else None)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([t.replace("_", " ").capitalize() + ("$^*$" if t == "scoliosis" else "") for t in order[::-1]])
    ax.set_xlabel("F2 margin over all-positive (EHR-only)"); ax.xaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.40, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


def figure_strategies(FS: pd.DataFrame, path):
    """Fig. 2: strategy effects with corrected 95% CIs, one panel per representation group."""
    plt = _style()
    names = {"class_weight": "Class weighting", "ros": "Random oversampling", "smote": "SMOTE (naive)", "smote_nc": "SMOTE-NC (type-aware)"}
    style = {"P1": ("s", "#eb6834", 0.24, "P1 record split"), "P2": ("^", "#1baf7a", 0.0, "P2 patient-grouped"),
             "P3": ("o", "#2a78d6", -0.24, "P3 patient-level")}
    strats = list(names)
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 2.15), sharey=True)
    lo, hi = min(-0.03, FS.lo.min() * 1.08), max(0.03, FS.hi.max() * 1.1)
    for ax, g in zip(axes, ["EHR only", "Image only", "Early fusion"]):
        for p, (mk, col, off, label) in style.items():
            d = FS[(FS.protocol == p) & (FS.group == g)].set_index("strategy")
            ss = [s for s in strats if s in d.index]
            if not ss:
                continue
            ys = [len(strats) - 1 - strats.index(s) + off for s in ss]
            ax.errorbar(d.loc[ss, "mean"], ys, xerr=[d.loc[ss, "mean"] - d.loc[ss, "lo"], d.loc[ss, "hi"] - d.loc[ss, "mean"]],
                        fmt=mk, ms=3.2, color=col, elinewidth=0.8, capsize=1.2, mec="white", mew=0.4, label=label)
        ax.axvline(0, color="#0b0b0b", lw=0.6); ax.set_xlim(lo, hi); ax.set_title(g, fontsize=7, pad=2)
        ax.xaxis.grid(True, color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
        ax.set_xticks([-0.08, -0.04, 0]); ax.set_xticks([0.03], minor=True); ax.set_xticklabels(["\u2212.08", "\u2212.04", "0"])
    axes[0].set_yticks(range(len(strats))); axes[0].set_yticklabels([names[s] for s in strats[::-1]])
    axes[1].set_xlabel(r"$\Delta$ test F2 vs. no augmentation")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=6, handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.6, 1.01))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93), w_pad=0.4); fig.savefig(path); plt.close(fig)


MAIN_PAIRS = [(("P3", "img_raddino"), ("P3", "img_scores")), (("P3", "img_raddino"), ("P3", "ehr")), (("P3", "fusion_raddino"), ("P3", "ehr")),
              (("P2", "img_raddino"), ("P2", "ehr")), (("P2", "fusion_raddino"), ("P2", "ehr"))]
ADM_PAIRS = [(("P4", "img_raddino"), ("P4", "ehr")), (("P4", "fusion_raddino"), ("P4", "ehr")),
             (("P4-C", "img_raddino"), ("P4-C", "ehr")), (("P4-C", "fusion_raddino"), ("P4-C", "img_raddino"))]
CXH_PAIRS = [(("P2-CX", "img_raddino"), ("P2-CX", "ehr")), (("P2-CX", "img_raddino"), ("P2-CX", "img_scores")),
             (("P2-CX", "fusion_raddino"), ("P2-CX", "img_raddino")), (("P4-H", "history_a"), ("P4", "ehr")),
             (("P4-C-H", "history_c"), ("P4-C", "ehr")), (("P4-H", "fusion_history_a"), ("P4-H", "history_a")),
             (("P4-C-H", "fusion_history_c"), ("P4-C-H", "history_c"))]


def prior_rule_comparison(S_hist: pd.DataFrame, B_adm: pd.DataFrame) -> pd.DataFrame:
    """History models against the prior-diagnosis rule (same folds and targets)."""
    rows = []
    for p_rule, p_model, fs in [("P4", "P4-H", "history_a"), ("P4-C", "P4-C-H", "history_c")]:
        h = B_adm[B_adm.protocol == p_rule].groupby("fold").prior_rule_f2.mean()
        m = S_hist[(S_hist.protocol == p_model) & (S_hist.feature_set == fs)].groupby("fold").test_f2.mean()
        i = h.index.intersection(m.index)
        mm, lo, hi, p = corrected_t((m.loc[i] - h.loc[i]).values)
        rows.append({"a": f"{p_model}:{fs}", "b": f"{p_rule}:prior-diagnosis rule", "diff": mm, "lo": lo, "hi": hi, "p": p})
    return pd.DataFrame(rows)
