from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# n_test / n_train of the corrected t-test: 20% test and 64% inner training data per outer fold.
N_TEST_OVER_N_TRAIN = 0.2 / 0.64
KEY = ["protocol", "feature_set", "target"]
# Grid order of the strategies, used to break ties in validation F2.
STRATEGY_ORDER = ["none", "class_weight", "ros", "smote", "smote_nc", "native"]


def corrected_t(x, ratio=N_TEST_OVER_N_TRAIN) -> tuple[float, float, float, float]:
    """Mean, 95% CI and two-sided p-value of fold-level values x (n folds, or n = k x r for r repetitions)."""
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se = np.sqrt((1 / n + ratio) * x.var(ddof=1)) if n > 1 else np.nan
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 1.0
    return m, m - h, m + h, p


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def _grid_rank(R: pd.DataFrame, by) -> pd.DataFrame:
    """Grid position of each strategy (STRATEGY_ORDER) and classifier (order of first appearance, i.e. config order)."""
    out = pd.DataFrame(index=R.index)
    for c in by:
        order = STRATEGY_ORDER if c == "strategy" else []
        seen = list(dict.fromkeys(R[c]))
        pos = {v: i for i, v in enumerate([v for v in order if v in seen] + [v for v in seen if v not in order])}
        out[f"_{c}_rank"] = R[c].map(pos)
    return out


def best_rows(R: pd.DataFrame, group, by) -> pd.DataFrame:
    """Row with the highest validation F2 per group; ties go to the configuration listed first in the grid."""
    rk = _grid_rank(R, by)
    R = pd.concat([R, rk], axis=1)
    R = R.sort_values(["val_f2"] + list(rk), ascending=[False] + [True] * rk.shape[1], kind="mergesort")
    return R.groupby(group).head(1).drop(columns=list(rk))


def select(R: pd.DataFrame, by=("strategy", "classifier"), within_fold=True) -> pd.DataFrame:
    """Configuration selected for each protocol, feature set and target, using inner-validation F2 only.

    within_fold=True (main analysis): each outer fold uses the configuration that is best on its own inner validation
    set, so no information from the fold's test patients enters the choice.
    within_fold=False (robustness check): one configuration per target, best on inner-validation F2 averaged over the
    folds. In both cases ties go to the configuration listed first in the grid.
    """
    by = list(by)
    if within_fold:
        return best_rows(R, KEY + ["seed", "fold"], by)
    g = R.groupby(KEY + by, sort=False).val_f2.mean().reset_index()
    best = best_rows(g, KEY, by)[KEY + by]
    return R.merge(best, on=KEY + by)


FOLD_METRICS = ("test_f2", "test_trivial_f2", "test_pr_auc", "test_roc_auc", "test_brier", "test_brier_prev")


def fold_macro(S: pd.DataFrame, cols=FOLD_METRICS) -> pd.DataFrame:
    """Macro average over targets for every protocol, feature set, seed and fold."""
    cols = [c for c in cols if c in S.columns]
    return S.groupby(["protocol", "feature_set", "seed", "fold"])[cols].mean().reset_index()


def summary_rows(S: pd.DataFrame, n_tests: int | None = None) -> pd.DataFrame:
    """Macro F2 (mean, SD, corrected CI), margin over all-positive (corrected CI and p-value) and threshold-free
    metrics; `significant` applies a Bonferroni correction over `n_tests` rows."""
    fm = fold_macro(S)
    rows = []
    for (p, fs), g in fm.groupby(["protocol", "feature_set"]):
        m, lo, hi, _ = corrected_t(g.test_f2)
        d, dlo, dhi, dp = corrected_t(g.test_f2 - g.test_trivial_f2)
        r = {
            "protocol": p,
            "feature_set": fs,
            "n_folds": len(g),
            "f2": m,
            "f2_sd": g.test_f2.std(ddof=1),
            "f2_lo": lo,
            "f2_hi": hi,
            "delta": d,
            "delta_lo": dlo,
            "delta_hi": dhi,
            "delta_p": dp,
        }
        for c, k in [
            ("test_pr_auc", "pr_auc"),
            ("test_roc_auc", "auroc"),
            ("test_brier", "brier"),
            ("test_brier_prev", "brier_prev"),
        ]:
            if c in g:
                r[k] = g[c].mean()
                r[k + "_sd"] = g[c].std(ddof=1)
        rows.append(r)
    T = pd.DataFrame(rows)
    if n_tests:
        T["significant"] = T.delta_p < 0.05 / n_tests
    return T


def paired(Sa: pd.DataFrame, Sb: pd.DataFrame, metric="test_f2") -> dict:
    """Paired fold-level comparison of two selected configurations (same folds and seeds)."""
    x = Sa.groupby(["seed", "fold"])[metric].mean()
    y = Sb.groupby(["seed", "fold"])[metric].mean()
    i = x.index.intersection(y.index)
    if not len(i):
        return {"metric": metric, "diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": 0}
    m, lo, hi, p = corrected_t((x.loc[i] - y.loc[i]).values)
    return {"metric": metric, "diff": m, "lo": lo, "hi": hi, "p": p, "n": len(i)}


def strategy_effects(R: pd.DataFrame) -> pd.DataFrame:
    """Each strategy against no augmentation with the same classifier, target and fold.

    The paired differences are averaged over targets and classifiers within each fold and tested with the corrected
    t-test; `p_holm` applies the Holm correction over all rows of the table.
    """
    R = R[R.strategy != "native"]
    idx = ["protocol", "feature_set", "target", "classifier", "seed", "fold"]
    base = R[R.strategy == "none"].set_index(idx)
    rows = []
    for (p, fs, st), d in R[R.strategy != "none"].groupby(["protocol", "feature_set", "strategy"]):
        d = d.set_index(idx)
        i = d.index.intersection(base.index)
        if not len(i):
            continue
        diff = (d.loc[i, "test_f2"] - base.loc[i, "test_f2"]).groupby(level=["seed", "fold"]).mean()
        dpr = (d.loc[i, "test_pr_auc"] - base.loc[i, "test_pr_auc"]).groupby(level=["seed", "fold"]).mean()
        dbr = (d.loc[i, "test_brier"] - base.loc[i, "test_brier"]).groupby(level=["seed", "fold"]).mean()
        m, lo, hi, pv = corrected_t(diff.values)
        rows.append(
            {
                "protocol": p,
                "feature_set": fs,
                "strategy": st,
                "dF2": m,
                "lo": lo,
                "hi": hi,
                "p": pv,
                "dPR_AUC": dpr.mean(),
                "dBrier": dbr.mean(),
                "n": len(diff),
            }
        )
    T = pd.DataFrame(rows)
    if len(T):
        T["p_holm"] = holm(T.p.values)
    return T


def temporal_bootstrap(
    R: pd.DataFrame, preds_dir, split_file, labels: dict, rng: np.random.Generator, B=1000
) -> pd.DataFrame:
    """Calendar-time split (P5, a single split): 95% CIs of macro F2 and of its margin over all-positive from B
    patient-level bootstrap resamples of the test set.

    The published analysis draws the P5-R resamples first and the P5-A resamples next from one generator (seed 42).
    """
    from .modeling import choose_threshold, f2_score

    sp = np.load(split_file)
    te, va = sp["test"], sp["val"]
    sid = labels["subject_id"][te]
    us, inv = np.unique(sid, return_inverse=True)
    members = [np.where(inv == j)[0] for j in range(len(us))]
    draws = [np.concatenate([members[j] for j in rng.integers(0, len(us), len(us))]) for _ in range(B)]
    rows = []
    for fs, bb in select(R).groupby("feature_set"):
        bb = bb.drop_duplicates("target")
        per_t = []
        for r in bb.itertuples():
            z = np.load(f"{preds_dir}/{r.protocol}__s0__f0__{fs}.npz")
            key = f"{r.target}__{r.strategy}__{r.classifier}"
            y = labels[r.target]
            thr = choose_threshold(y[va], z[f"{key}__p_va"])
            per_t.append((y[te].astype(bool), z[f"{key}__p_te"] >= thr))
        bs = np.array(
            [
                (
                    np.mean([f2_score(yt[i], yh[i]) for yt, yh in per_t]),
                    np.mean([f2_score(yt[i], yh[i]) - 5 * yt[i].mean() / (4 * yt[i].mean() + 1) for yt, yh in per_t]),
                )
                for i in draws
            ]
        )
        point = np.mean([f2_score(yt, yh) for yt, yh in per_t])
        triv = np.mean([5 * yt.mean() / (4 * yt.mean() + 1) for yt, _ in per_t])
        rows.append(
            {
                "protocol": bb.protocol.iloc[0],
                "feature_set": fs,
                "f2": point,
                "f2_lo": np.percentile(bs[:, 0], 2.5),
                "f2_hi": np.percentile(bs[:, 0], 97.5),
                "delta": point - triv,
                "delta_lo": np.percentile(bs[:, 1], 2.5),
                "delta_hi": np.percentile(bs[:, 1], 97.5),
            }
        )
    return pd.DataFrame(rows)
