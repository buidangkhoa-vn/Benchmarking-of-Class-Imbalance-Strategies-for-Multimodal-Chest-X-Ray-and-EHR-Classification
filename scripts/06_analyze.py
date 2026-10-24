import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance import cohort as C  # noqa: E402
from cxr_ehr_imbalance import report as rp  # noqa: E402
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.io import atomic_csv  # noqa: E402
from cxr_ehr_imbalance.stats import select, strategy_effects, temporal_bootstrap  # noqa: E402

args, cfg = cli("Tables, statistical tests and figures from the experiment results.")
runs, out = cfg["dirs"]["runs"], cfg["dirs"]["analysis"]


def save(df: pd.DataFrame, name: str) -> None:
    """Writes outputs/analysis/<name>.csv and prints the table."""
    atomic_csv(df.round(5), out / f"{name}.csv")
    print(f"\n== {name}\n{df.round(4).to_string(index=False)}")


R1, B1 = rp.load(runs, "main")
R2, B2 = rp.load(runs, "admission")
R3, B3 = rp.load(runs, "chexpert_history")
R4, _ = rp.load(runs, "tuning")
R5, _ = rp.load(runs, "sensitivity")
R6, _ = rp.load(runs, "trees")
R7, B7 = rp.load(runs, "frontal")
R8, B8 = rp.load(runs, "densenet_admission")
R9, _ = rp.load(runs, "joint_tuning")
R10, _ = rp.load(runs, "seed_variation")
R11, _ = rp.load(runs, "reference_rerun")
R12, _ = rp.load(runs, "densenet_history")
R13, _ = rp.load(runs, "densenet_rerun")

# Table III, Fig. 1, per-target margins, Sections V-A to V-C.
if R1 is not None:
    S1 = select(R1)
    save(rp.results_table(R1, n_tests=17), "table4_protocols")
    save(rp.baseline_summary(B1), "table4_baselines")
    save(rp.seen_unseen(S1), "P1_seen_unseen")
    save(rp.comparisons(S1, rp.MAIN_PAIRS), "comparisons_main")
    save(strategy_effects(R1), "strategy_effects_main")
    save(rp.strategy_pooled(R1), "strategy_effects_pooled_main")
    save(rp.best_per_strategy(R1), "best_classifier_per_strategy_main")
    save(rp.matched_split_comparison(R1), "matched_P1_P2")
    FS = rp.strategy_by_representation(R1)
    save(FS, "fig2_strategy_effects")
    rp.figure_strategies(FS, out / "fig2_strategy_effects.pdf")
    sampler = (
        R1[R1.strategy.isin(["smote", "smote_nc"])]
        .groupby(["protocol", "feature_set", "strategy"])[["syn_bad_row_frac"]]
        .mean()
    )
    save(sampler.reset_index(), "sampler_audit")
    selected = S1.drop_duplicates(rp.KEY + ["seed", "fold"]).groupby(["protocol", "classifier"]).size()
    save(selected.unstack(fill_value=0).reset_index(), "selected_classifiers_main")

# Table IV (admission-aligned and rule-refined labels, calendar-time split), Section V-D.
if R2 is not None:
    R2_cv = R2[~R2.protocol.str.startswith("P5")]
    S2 = select(R2)
    save(rp.results_table(R2_cv, n_tests=14), "table5_aligned_clean")
    save(rp.baseline_summary(B2), "table5_baselines")
    save(rp.seen_unseen(S2), "P1A_seen_unseen")
    save(rp.comparisons(S2, rp.ADM_PAIRS), "comparisons_admission")
    save(strategy_effects(R2_cv), "strategy_effects_admission")
    save(rp.strategy_pooled(R2_cv), "strategy_effects_pooled_admission")
    boots, rng = [], np.random.default_rng(cfg["random_state"])
    for p, cols in [("P5-R", C.TARGETS), ("P5-A", [f"adm_{t}" for t in C.TARGETS])]:
        split_file = runs / "admission" / "splits" / f"{p}__s0__f0.npz"
        if not (R2.protocol == p).any() or not split_file.exists():
            continue
        df = C.load_cohort(cfg["paths"]["cohort_csv"])
        src = df if p == "P5-R" else pd.read_csv(cfg["dirs"]["labels"] / "admission_labels.csv.gz")
        labels = {"subject_id": df.subject_id.values}
        labels.update({t: src[c].values for t, c in zip(C.TARGETS, cols, strict=True)})
        boots.append(temporal_bootstrap(R2[R2.protocol == p], runs / "admission" / "groups", split_file, labels, rng))
    if boots:
        save(pd.concat(boots, ignore_index=True), "table5_P5_bootstrap")
    if R1 is not None:
        rp.figure_targets(rp.per_target(S1), rp.per_target(S2), out / "fig1_per_target_margins.pdf")
elif R1 is not None:
    rp.figure_targets(rp.per_target(S1), None, out / "fig1_per_target_margins.pdf")

# Table IV (time-aligned history), CheXpert labels, Sections V-D and V-E.
if R3 is not None:
    S3 = select(R3)
    save(rp.results_table(R3[R3.protocol.str.endswith("-H")], n_tests=14), "table5_history")
    save(rp.results_table(R3[R3.protocol.str.endswith("CX")], n_tests=4), "table6_chexpert")
    save(rp.baseline_summary(B3), "table6_baselines")
    save(rp.seen_unseen(S3), "P1CX_seen_unseen")
    save(strategy_effects(R3), "strategy_effects_chexpert_history")
    if R2 is not None:
        S23 = pd.concat([S2, S3], ignore_index=True)
        save(rp.comparisons(S23, rp.CXH_PAIRS), "comparisons_chexpert_history")
        save(rp.prior_rule_comparison(S3, B2), "history_vs_prior_rule")

# Robustness checks (Section V-F, Table VI).
if R4 is not None:
    for k, v in rp.tuning(R4).items():
        save(v, k)

if R6 is not None:
    for k, v in rp.trees(R6).items():
        save(v, k)

if R5 is not None:
    for k, v in rp.sensitivity(R5).items():
        save(v, k)

if R7 is not None:
    R_all = pd.concat([R for R in [R1, R2, R7.iloc[:0]] if R is not None], ignore_index=True)
    for k, v in rp.frontal_only(R7, R_all).items():
        save(v, k)
    save(rp.baseline_summary(B7), "frontal_baselines")

if R8 is not None and R2 is not None:
    for k, v in rp.encoder_comparison(R8, R2).items():
        save(v, k)

if R13 is not None and R8 is not None:
    save(rp.rerun_difference(R13, R8), "densenet_rerun_difference")
if R12 is not None and R8 is not None and R2 is not None and R3 is not None:
    for k, v in rp.encoder_control(pd.concat([R8, R12], ignore_index=True), R2, R3, B2).items():
        save(v, k)

if R9 is not None:
    for k, v in rp.joint_tuning(R9).items():
        save(v, k)

if R10 is not None and R2 is not None and R3 is not None:
    R2_ref = R2 if R11 is None else pd.concat([R2[~R2.protocol.isin(R11.protocol.unique())], R11], ignore_index=True)
    ref = pd.concat([R2_ref, R3], ignore_index=True)
    for k, v in rp.seed_variation(R10, ref, B2).items():
        save(v, k)

if R11 is not None and R2 is not None:
    save(rp.rerun_difference(R11, R2[R2.protocol.isin(R11.protocol.unique())]), "reference_rerun_difference")

CV = [R for R in [R1, R2[~R2.protocol.str.startswith("P5")] if R2 is not None else None, R3] if R is not None]
if CV:
    PF = pd.concat([rp.per_fold_selection(R) for R in CV], ignore_index=True)
    save(PF, "selection_over_folds_vs_within_fold")
    save(pd.concat([rp.threshold_diagnostics(select(R)) for R in CV], ignore_index=True), "threshold_diagnostics")
    print(
        f"selection over folds minus within-fold selection: {PF.difference.min():.4f} to {PF.difference.max():.4f}, "
        f"mean {PF.difference.mean():.4f}"
    )
    for k, v in rp.without_rare_targets(pd.concat(CV, ignore_index=True)).items():
        save(v, k)
