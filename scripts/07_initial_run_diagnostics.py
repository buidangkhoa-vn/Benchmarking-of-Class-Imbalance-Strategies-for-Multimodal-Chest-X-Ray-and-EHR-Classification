"""Diagnostics of the initial record-split benchmark from its saved predictions, without retraining (Sec. III-D, Table II):
patient overlap, exact EHR matches, the no-learning lookup table, seen vs. unseen patients and late fusion.
Needs paths.initial_run_predictions (validation and test predictions of the initial run)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance import cohort as C  # noqa: E402
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.io import atomic_csv, atomic_json  # noqa: E402
from cxr_ehr_imbalance.modeling import choose_threshold, f2_score  # noqa: E402

args, cfg = cli(__doc__)
out = cfg["dirs"]["analysis"]
df = C.load_cohort(cfg["paths"]["cohort_csv"])
P = pd.read_csv(cfg["paths"]["initial_run_predictions"],
                usecols=["dataset_position", "split", "seed", "target", "feature_set", "selection_role", "threshold", "y_true", "y_probability"])
te = np.sort(P[P.split == "test"].dataset_position.unique())
va = np.sort(P[P.split == "validation"].dataset_position.unique())
tr = np.setdiff1d(np.arange(len(df)), np.union1d(te, va))
dev = np.union1d(tr, va)
sid = df.subject_id.values
key = pd.util.hash_pandas_object(df[C.EHR].fillna(-999), index=False).values
lookup = []
for t in C.TARGETS:
    src = pd.Series(df[t].values[dev], index=key[dev]); src = src[~src.index.duplicated()]
    lookup.append(f2_score(df[t].values[te], pd.Series(key[te]).map(src).fillna(0).astype(int).values))
summary = {"train_val_test": [len(tr), len(va), len(te)],
           "test_patient_seen_in_development": float(np.isin(sid[te], list(set(sid[dev]))).mean()),
           "test_exact_ehr_match_in_development": float(np.isin(key[te], key[dev]).mean()),
           "lookup_macro_f2": float(np.mean(lookup))}

unseen_va = ~np.isin(sid[va], list(set(sid[tr])))
unseen_te = ~np.isin(sid[te], list(set(sid[dev])))
summary.update({"unseen_validation_radiographs": int(unseen_va.sum()), "unseen_test_radiographs": int(unseen_te.sum())})
rows, probs = [], {}
for (fs, t, seed), g in P[P.selection_role.str.contains("feature_winner")].groupby(["feature_set", "target", "seed"]):
    gv = g[g.split == "validation"].set_index("dataset_position").reindex(va)
    gt = g[g.split == "test"].set_index("dataset_position").reindex(te)
    thr = float(gt.threshold.iloc[0]); yt, pt = gt.y_true.values, gt.y_probability.values
    yv, pv = gv.y_true.values, gv.y_probability.values
    probs[(fs, t, seed)] = (pv, pt)
    u = unseen_te
    thr_u = choose_threshold(yv[unseen_va], pv[unseen_va])       # threshold re-selected on unseen validation patients
    rows.append({"feature_set": fs, "target": t, "seed": seed, "f2_all": f2_score(yt, pt >= thr), "f2_seen": f2_score(yt[~u], pt[~u] >= thr),
                 "f2_unseen": f2_score(yt[u], pt[u] >= thr), "f2_unseen_clean_threshold": f2_score(yt[u], pt[u] >= thr_u),
                 "pr_auc_unseen": average_precision_score(yt[u], pt[u]), "auroc_unseen": roc_auc_score(yt[u], pt[u]),
                 "all_positive_unseen": 5 * yt[u].mean() / (4 * yt[u].mean() + 1)})
W = pd.DataFrame(rows)
fusion = []
for (fs, t, seed), (pv_i, pt_i) in probs.items():
    if fs != "image_only" or ("ehr_only", t, seed) not in probs:
        continue
    pv_e, pt_e = probs[("ehr_only", t, seed)]
    yv, yt = df[t].values[va][unseen_va], df[t].values[te][unseen_te]
    def score(w):
        p = w * pv_i[unseen_va] + (1 - w) * pv_e[unseen_va]
        return f2_score(yv, p >= choose_threshold(yv, p))
    w = max(np.round(np.arange(0, 1.01, 0.1), 1), key=score)
    pv = w * pv_i[unseen_va] + (1 - w) * pv_e[unseen_va]; pt = w * pt_i[unseen_te] + (1 - w) * pt_e[unseen_te]
    fusion.append({"target": t, "seed": seed, "weight_image": w, "f2_unseen_clean_threshold": f2_score(yt, pt >= choose_threshold(yv, pv)),
                   "pr_auc_unseen": average_precision_score(yt, pt), "auroc_unseen": roc_auc_score(yt, pt)})
table2 = W.groupby(["feature_set", "target"]).mean(numeric_only=True).groupby("feature_set").mean().drop(columns="seed")
if fusion:
    table2.loc["late_fusion"] = pd.DataFrame(fusion).groupby("target").mean(numeric_only=True).mean()
atomic_csv(table2.reset_index().round(4), out / "table2_initial_run.csv")
atomic_json(summary, out / "initial_run_summary.json")
print(summary); print(table2.round(3).to_string())
