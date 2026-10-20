"""Diagnostics of the ORIGINAL (record-split) benchmark from its saved predictions — no retraining.
Reproduces Section III-D / Table 2 of the paper: patient overlap, EHR exact-match rate, the no-learning
EHR lookup table, and seen- vs unseen-patient performance of the feature-set winners (incl. late fusion).

Inputs (not distributed; MIMIC-derived, PhysioNet credentialed access required):
  --csv   x_ray_and_features_data.csv            (29,083 x 48, hash da85519958e4bccb)
  --preds selected_predictions.csv.gz            (validation + test predictions of the original benchmark)
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

T = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
EHR = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc", "diabete", "hiv", "smoke", "cough"]
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

def f2(y, yh):
    y, yh = np.asarray(y).astype(bool), np.asarray(yh).astype(bool)
    tp = (y & yh).sum(); fp = (~y & yh).sum(); fn = (y & ~yh).sum(); d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0

def best_thr(y, p):
    """F2-optimal threshold on a grid (ties: higher F2, recall, precision, then closest to 0.5)."""
    yb = np.asarray(y).astype(bool); P = p[None, :] >= GRID[:, None]
    tp = (P & yb).sum(1); fp = (P & ~yb).sum(1); fn = (~P & yb).sum(1)
    d = 5 * tp + 4 * fn + fp; f = np.where(d > 0, 5 * tp / np.maximum(d, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0); pre = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return float(GRID[np.lexsort((np.abs(GRID - 0.5), -pre, -rec, -f))[0]])

def main(a):
    df = pd.read_csv(a.csv, low_memory=False)
    P = pd.read_csv(a.preds, usecols=["dataset_position", "subject_id", "split", "seed", "target", "feature_set", "sampler",
                                      "classifier", "selection_role", "threshold", "y_true", "y_probability"])
    te_pos = np.sort(P[P.split == "test"].dataset_position.unique()); va_pos = np.sort(P[P.split == "validation"].dataset_position.unique())
    tr_pos = np.setdiff1d(np.arange(len(df)), np.union1d(te_pos, va_pos))
    sid = df.subject_id.values
    s_tr, s_trva = set(sid[tr_pos]), set(sid[np.union1d(tr_pos, va_pos)])
    print(f"split sizes train/val/test: {len(tr_pos)}/{len(va_pos)}/{len(te_pos)}")
    print(f"test radiographs whose patient is in train or val: {np.isin(sid[te_pos], list(s_trva)).mean():.3f}")
    key = pd.util.hash_pandas_object(df[EHR].fillna(-999), index=False).values
    dev = np.union1d(tr_pos, va_pos)
    print(f"test radiographs with an exact EHR match in train+val: {np.isin(key[te_pos], key[dev]).mean():.3f}")
    # ---- lookup table: copy the label of the exact EHR match (first match), predict negative otherwise ----
    lk = []
    for t in T:
        src = pd.Series(df[t].values[dev], index=key[dev]); src = src[~src.index.duplicated()]
        yhat = pd.Series(key[te_pos]).map(src).fillna(0).astype(int).values
        lk.append(f2(df[t].values[te_pos], yhat))
    print(f"EHR lookup table macro test F2: {np.mean(lk):.3f}")
    # ---- seen vs unseen patients for the feature-set winners ----
    W = P[P.selection_role.str.contains("feature_winner")]
    unseen_va = ~np.isin(sid[va_pos], list(s_tr)); unseen_te = ~np.isin(sid[te_pos], list(s_trva))
    print(f"unseen validation radiographs: {unseen_va.sum()} | unseen test radiographs: {unseen_te.sum()}")
    rows, probs = [], {}
    for (fs, t, seed), g in W.groupby(["feature_set", "target", "seed"]):
        gv = g[g.split == "validation"].set_index("dataset_position").reindex(va_pos)
        gt = g[g.split == "test"].set_index("dataset_position").reindex(te_pos)
        thr = float(gt.threshold.iloc[0]); yt, pt = gt.y_true.values, gt.y_probability.values
        yv, pv = gv.y_true.values, gv.y_probability.values
        thr_clean = best_thr(yv[unseen_va], pv[unseen_va])
        probs[(fs, t, seed)] = (pv, pt)
        u = unseen_te
        rows.append(dict(feature_set=fs, target=t, seed=seed, f2_all=f2(yt, pt >= thr), f2_seen=f2(yt[~u], pt[~u] >= thr),
                         f2_unseen=f2(yt[u], pt[u] >= thr), f2_unseen_clean=f2(yt[u], pt[u] >= thr_clean),
                         prauc_unseen=average_precision_score(yt[u], pt[u]), auroc_unseen=roc_auc_score(yt[u], pt[u]),
                         trivial_unseen=5 * yt[u].mean() / (4 * yt[u].mean() + 1)))
    R = pd.DataFrame(rows)
    # ---- late fusion of image and EHR winners: weight and threshold chosen on unseen validation patients ----
    lf = []
    for (fs, t, seed), (pv_i, pt_i) in probs.items():
        if fs != "image_only" or ("ehr_only", t, seed) not in probs:
            continue
        pv_e, pt_e = probs[("ehr_only", t, seed)]
        yv = df[t].values[va_pos][unseen_va]; yt = df[t].values[te_pos][unseen_te]
        def score(w):
            p = w * pv_i[unseen_va] + (1 - w) * pv_e[unseen_va]; return f2(yv, p >= best_thr(yv, p))
        w = max(np.round(np.arange(0, 1.01, 0.1), 1), key=score)
        pv = w * pv_i[unseen_va] + (1 - w) * pv_e[unseen_va]; pt = w * pt_i[unseen_te] + (1 - w) * pt_e[unseen_te]
        lf.append(dict(target=t, seed=seed, f2_unseen_clean=f2(yt, pt >= best_thr(yv, pv)),
                       prauc_unseen=average_precision_score(yt, pt), auroc_unseen=roc_auc_score(yt, pt)))
    M = R.groupby(["feature_set", "target"]).mean(numeric_only=True).groupby("feature_set").mean()
    print("\nfeature-set winners, macro over 8 targets (mean of seeds):")
    print(M[["f2_all", "f2_seen", "f2_unseen", "f2_unseen_clean", "trivial_unseen", "prauc_unseen", "auroc_unseen"]].round(3).to_string())
    if lf:
        L = pd.DataFrame(lf).groupby("target").mean(numeric_only=True).mean()
        print("\nlate fusion (unseen, clean weight+threshold):", L[["f2_unseen_clean", "prauc_unseen", "auroc_unseen"]].round(3).to_dict())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--preds", required=True)
    main(ap.parse_args())
