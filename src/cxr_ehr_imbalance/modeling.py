"""Preprocessing, imbalance strategies, classifiers, threshold selection and metrics."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


# ---------------------------------------------------------------- preprocessing
def build_preprocessor(fs: dict, pca_default=64, knn_neighbors=5) -> ColumnTransformer:
    """Continuous: standardise, then distance-weighted kNN imputation with missingness indicators.
    Embeddings: standardise, then PCA. Binary: unchanged. All steps are fitted on the inner training set."""
    trs = []
    if fs["cont"]:
        trs.append(("cont", Pipeline([("sc", StandardScaler()),
                                      ("imp", KNNImputer(n_neighbors=knn_neighbors, weights="distance", add_indicator=True))]),
                    fs["cont"]))
    if fs["emb"]:
        n_pc = pca_default if fs.get("pca", "default") == "default" else fs["pca"]
        steps = [("sc", StandardScaler())]
        if n_pc:
            steps.append(("pca", PCA(n_components=min(n_pc, len(fs["emb"])), random_state=0)))
        trs.append(("emb", Pipeline(steps), fs["emb"]))
    if fs["bin"]:
        trs.append(("bin", "passthrough", fs["bin"]))
    return ColumnTransformer(trs, remainder="drop", sparse_threshold=0)


def discrete_columns(ct: ColumnTransformer, indicators_first=False) -> tuple[np.ndarray, int, float]:
    """Positions of binary and missingness-indicator columns in the transformed matrix (the categorical columns of
    SMOTE-NC), the number of columns, and the variance retained by PCA (nan without embeddings, 1.0 without PCA)."""
    pos, cont_ind, binary, evr = 0, [], [], np.nan
    for name, trans, cols in ct.transformers_:
        if name == "cont":
            imp = trans.named_steps["imp"]
            n_ind = len(imp.indicator_.features_) if getattr(imp, "indicator_", None) is not None else 0
            pos += len(cols); cont_ind += list(range(pos, pos + n_ind)); pos += n_ind
        elif name == "emb":
            if "pca" in trans.named_steps:
                evr = float(trans.named_steps["pca"].explained_variance_ratio_.sum()); pos += trans.named_steps["pca"].n_components_
            else:
                evr = 1.0; pos += len(cols)
        elif name == "bin":
            binary += list(range(pos, pos + len(cols))); pos += len(cols)
    return np.array(cont_ind + binary if indicators_first else binary + cont_ind, dtype=int), pos, evr


# ---------------------------------------------------------------- imbalance strategies
def resample(strategy: str, X, y, disc, seed: int):
    """none / class_weight / native: data unchanged. ros: random oversampling. smote: SMOTE on all columns.
    smote_nc: SMOTE-NC with binary and indicator columns declared categorical. Samplers balance the classes."""
    audit = {"syn_rows": 0, "syn_bad_row_frac": np.nan, "syn_bad_cell_frac": np.nan}
    if strategy in ("none", "class_weight", "native"):
        return X, y, audit
    from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
    n_min = int(min(y.sum(), len(y) - y.sum()))
    k = max(1, min(5, n_min - 1))
    if strategy == "ros":
        sampler = RandomOverSampler(random_state=seed)
    elif strategy == "smote":
        sampler = SMOTE(k_neighbors=k, random_state=seed)
    elif strategy == "smote_nc":
        sampler = SMOTENC(categorical_features=list(disc), k_neighbors=k, random_state=seed)
    else:
        raise ValueError(strategy)
    Xr, yr = sampler.fit_resample(X, y)
    syn = Xr[len(X):]
    audit["syn_rows"] = len(syn)
    if len(disc) and len(syn):
        bad = ~np.isin(syn[:, disc], [0.0, 1.0])
        audit["syn_bad_row_frac"] = float(bad.any(1).mean()); audit["syn_bad_cell_frac"] = float(bad.mean())
    return Xr, yr, audit


# ---------------------------------------------------------------- classifiers
def make_classifier(spec: dict, seed: int, class_weight: bool, pos_weight: float, lr_max_iter=2000):
    """spec: {"family": LR|RF|ET|LGBM|XGB|BRF|EasyEnsemble, **hyperparameters}; defaults are the paper's settings."""
    hp = {k: v for k, v in spec.items() if k != "family"}
    cw = "balanced" if class_weight else None
    fam = spec["family"]
    if fam == "LR":
        return LogisticRegression(C=hp.get("C", 1.0), max_iter=lr_max_iter, class_weight=cw)
    if fam == "RF":
        return RandomForestClassifier(**{"n_estimators": 300, **hp}, n_jobs=1, random_state=seed, class_weight=cw)
    if fam == "ET":
        return ExtraTreesClassifier(**{"n_estimators": 300, **hp}, n_jobs=1, random_state=seed, class_weight=cw)
    if fam == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{"n_estimators": 300, "learning_rate": 0.05, **hp}, n_jobs=1, random_state=seed, verbose=-1,
                              class_weight=cw)
    if fam == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(**{"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, **hp}, tree_method="hist", n_jobs=1,
                             random_state=seed, eval_metric="logloss", scale_pos_weight=(pos_weight if class_weight else 1.0))
    if fam == "BRF":
        from imblearn.ensemble import BalancedRandomForestClassifier
        return BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all", replacement=True, bootstrap=False,
                                              n_jobs=1, random_state=seed)
    if fam == "EasyEnsemble":
        from imblearn.ensemble import EasyEnsembleClassifier
        return EasyEnsembleClassifier(n_estimators=10, n_jobs=1, random_state=seed)
    raise ValueError(fam)


# ---------------------------------------------------------------- thresholds and metrics
def choose_threshold(y, p) -> float:
    """Threshold on the grid 0.01..0.99 that maximises F2 (ties: higher recall, higher precision, closer to 0.5)."""
    yb = y.astype(bool); P = p[None, :] >= THRESHOLDS[:, None]
    tp = (P & yb).sum(1); fp = (P & ~yb).sum(1); fn = (~P & yb).sum(1)
    d = 5 * tp + 4 * fn + fp
    f2 = np.where(d > 0, 5 * tp / np.maximum(d, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    pre = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return float(THRESHOLDS[np.lexsort((np.abs(THRESHOLDS - 0.5), -pre, -rec, -f2))[0]])


def f2_score(y, yhat) -> float:
    y = np.asarray(y).astype(bool); yhat = np.asarray(yhat).astype(bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum(); d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0


def metrics(y, p, thr, prefix, train_prevalence=np.nan) -> dict:
    y = np.asarray(y).astype(bool); pred = p >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = len(y) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    d2 = 5 * tp + 4 * fn + fp
    prev = float(y.mean()) if len(y) else np.nan
    out = {"n": len(y), "pos": int(y.sum()), "prev": prev, "f2": 5 * tp / d2 if d2 else 0.0,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "recall": rec, "precision": prec,
           "bal_acc": (rec + spec) / 2, "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
           "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
           "brier_prev": float(np.mean((train_prevalence - y) ** 2)) if len(y) else np.nan,
           "trivial_f2": 5 * prev / (4 * prev + 1) if len(y) else np.nan, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {f"{prefix}_{k}": v for k, v in out.items()}
