# %% [markdown]
# # STEP 8 (v8) — PCA sensitivity, age/history baselines, second CV seed, Brier scores
#
# **Purpose** (answers the merged review, B2/B4/B6/B7). Logistic regression only, so the run is short.
# 1. **PCA sensitivity (B4).** RAD-DINO image and fusion inputs with 32, 64 and 128 principal components and without PCA
#    (768 standardised dimensions), under P2 (released labels), P4 (admission-aligned) and P4-C (clean).
# 2. **What the history gain comes from (B6).** Under P4 / P4-C: released age only, age at the radiograph's own admission only,
#    prior-diagnosis flags only, and age-at-own-admission + flags (the paper's history vector). Released vs own-admission age
#    is the same variable before and after time alignment.
# 3. **Second cross-validation seed (B2).** Patient-grouped folds rebuilt with seed 43 (same algorithm as seed 42), for EHR,
#    RAD-DINO image/fusion (64 PCs) and the history vector; under P2 also class weighting, ROS and SMOTE/SMOTE-NC. The same
#    configurations are run on the seed-42 folds in this notebook, so the two repetitions can be compared with the corrected
#    repeated k-fold t-test.
# 4. **Brier score (B7)** for every fit, next to the Brier score of predicting the training prevalence.
#
# **How to run** — secondary account (Colab Pro), Runtime **CPU, High-RAM** → **Run all**. No BigQuery, nothing to edit.
# - Needs on Drive (all from earlier runs): `data_aug/x_ray_and_features_data.csv`, `data_aug/embeddings/raddino`,
#   v2 fold manifests, STEP5 labels (`MMMDA_runs/v4_2026-10-19_admlabels/v4_labels`), v3 BigQuery tables (`MMMDA_runs/v3_2026-10-19_P4P5/v3_bq`).
# - Expected time: ~20–30 min with 8+ cores. If the session drops: Run all again (finished tasks are skipped).
# - Done when `FLUSH DONE` appears. Download **one file** `v8_for_claude.zip` to `D:\project_TTNT_Data_aug\analysis_v8\`.
#
# Every folder/file this notebook creates carries the version prefix `v8`.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE      = "full"                # "full" | "preflight" (1 fold, 2 targets, separate folder)
VERSION       = "v8_2026-10-22"

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
EXPECTED_HASH = "da85519958e4bccb"
RADDINO_DIR   = f"{DRIVE_ROOT}/embeddings/raddino"
V2_FOLDS      = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_subject_s42.csv"   # seed-42 patient folds (read only)
V4_ADMLABELS  = f"{DRIVE_ROOT}/MMMDA_runs/v4_2026-10-19_admlabels/v4_labels/v4_admission_labels.csv.gz"
V3_BQ_DIR     = f"{DRIVE_ROOT}/MMMDA_runs/v3_2026-10-19_P4P5/v3_bq"
OUT_ROOT      = f"{DRIVE_ROOT}/MMMDA_runs"                               # -> MMMDA_runs/v8_2026-10-22_sensitivity/
LOCAL_CACHE   = "/content/mmmda_v8_cache"
N_PAR         = None                  # parallel processes; None = all cores

# %%
# ============================== CELL 2 — INSTALL + MOUNT ==============================
import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("V8_LOCAL_TEST") == "1"           # offline test only
if LOCAL_TEST:
    CSV_PATH = os.environ["V8_CSV"]; OUT_ROOT = os.environ["V8_OUT"]; LOCAL_CACHE = os.environ["V8_CACHE"]
    RADDINO_DIR = os.environ["V8_EMB"]; V2_FOLDS = os.environ.get("V8_V2FOLDS", ""); V4_ADMLABELS = os.environ["V8_ADMLABELS"]
    V3_BQ_DIR = os.environ["V8_BQDIR"]; RUN_MODE = os.environ.get("V8_MODE", RUN_MODE); N_PAR = int(os.environ.get("V8_NPAR", "2"))
if IN_COLAB:
    for mod, pkg in [("iterstrat", "iterative-stratification"), ("imblearn", "imbalanced-learn")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
print("cores:", os.cpu_count(), "| RUN_MODE:", RUN_MODE, "| VERSION:", VERSION)

# %%
# ============================== CELL 3 — IMPORTS + HELPERS ==============================
import json, time, hashlib, traceback, warnings, platform, glob, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
warnings.filterwarnings("ignore")

def atomic_to_csv(df, path):
    path = str(path); tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip" if path.endswith(".gz") else None); os.replace(tmp, path)

def atomic_savez(path, **arrays):
    path = str(path); tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def atomic_json(obj, path):
    path = str(path); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

V = "v8"
OUT = Path(OUT_ROOT) / (f"{VERSION}_sensitivity" + ("_preflight" if RUN_MODE == "preflight" else ""))
CACHE = Path(LOCAL_CACHE) / OUT.name
DIRS = {k: OUT / f"{V}_{k}" for k in ["manifests", "results", "errors", "analysis"]}
for d in list(DIRS.values()) + [CACHE / f"{V}_X", CACHE / f"{V}_splits"]:
    d.mkdir(parents=True, exist_ok=True)
print("OUT  :", OUT)

# %%
# ============================== CELL 4 — LOAD + LOCK DATA, LABELS, TIME-ALIGNED AGE ==============================
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV differs from the benchmarked file — STOP"
TARGETS  = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
EHR_CONT = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc"]
EHR_BIN  = ["diabete", "hiv", "smoke", "cough"]
EHR      = EHR_CONT + EHR_BIN
df["dataset_position"] = np.arange(len(df))
S = df.groupby("subject_id")[TARGETS].first()
AL = pd.read_csv(V4_ADMLABELS)
assert (AL.dataset_position.values == np.arange(len(df))).all() and (AL.dicom.values == df.dicom.values).all()
adm_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_admissions.csv.gz", parse_dates=["admittime"]).drop_duplicates("hadm_id").set_index("hadm_id")
pts_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_patients.csv.gz").drop_duplicates("subject_id").set_index("subject_id")
ay = adm_bq.admittime.reindex(df.hadm_id).dt.year.values.astype(float)
ay = np.where(np.isnan(ay), pd.to_datetime(df.admittime).dt.year.values, ay)
age_own = (pts_bq.anchor_age.reindex(df.subject_id).values + (ay - pts_bq.anchor_year.reindex(df.subject_id).values)).astype(float)
print("age at own admission differs from released age:", round(float((age_own != df.age.values).mean()), 4))
lab = {}
for t in TARGETS:
    lab[f"rel__{t}"] = df[t].values.astype(np.int8)
    lab[f"adm__{t}"] = AL[f"adm_{t}"].values.astype(np.int8)
    lab[f"cln__{t}"] = AL[f"cln_{t}"].values.astype(np.int8)
np.savez(CACHE / f"{V}_labels.npz", **lab)

# %%
# ============================== CELL 5 — TABLE, FEATURE SETS, SPLITS (seed 42 and seed 43), TASKS ==============================
E = np.load(f"{RADDINO_DIR}/emb.npy").astype(np.float32)
ids = pd.read_csv(f"{RADDINO_DIR}/ids.csv")
posm = df.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
assert posm.notna().all(), f"{int(posm.isna().sum())} radiographs without a RAD-DINO embedding"
EMB = [f"raddino_{j}" for j in range(E.shape[1])]
HIST_A = [f"hist_{t}" for t in TARGETS]; HIST_C = [f"histcln_{t}" for t in TARGETS]
rec = pd.concat([df[["dataset_position", "subject_id"] + EHR].reset_index(drop=True),
                 pd.DataFrame({"age_own": age_own}),
                 AL[HIST_A + HIST_C].reset_index(drop=True).astype(np.int8),
                 pd.DataFrame(E[posm.astype(int).values], columns=EMB)], axis=1)
rec.to_pickle(CACHE / f"{V}_rec.pkl")
def fs(cont=(), emb=False, bin_=(), pca=64):
    return dict(cont=list(cont), emb=EMB if emb else [], bin=list(bin_), pca=pca)
FEATSETS = {"ehr": fs(EHR_CONT, bin_=EHR_BIN),
            "img_pc32": fs(emb=True, pca=32), "img_pc64": fs(emb=True, pca=64), "img_pc128": fs(emb=True, pca=128), "img_full": fs(emb=True, pca=None),
            "fus_pc32": fs(EHR_CONT, True, EHR_BIN, 32), "fus_pc64": fs(EHR_CONT, True, EHR_BIN, 64),
            "fus_pc128": fs(EHR_CONT, True, EHR_BIN, 128), "fus_full": fs(EHR_CONT, True, EHR_BIN, None),
            "age_rel": fs(["age"]), "age_own": fs(["age_own"]),
            "hist_a": fs(bin_=HIST_A), "hist_c": fs(bin_=HIST_C),
            "agehist_a": fs(["age_own"], bin_=HIST_A), "agehist_c": fs(["age_own"], bin_=HIST_C)}

def build_folds(seed):
    Ys = S[TARGETS].values; parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(Ys)), Ys)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed * 100 + k).split(np.zeros(len(tr)), Ys[tr]))
        role = np.empty(len(Ys), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"subject_id": S.index.values, "fold": k, "role": role}))
    return pd.concat(parts, ignore_index=True)
if V2_FOLDS and os.path.exists(V2_FOLDS):
    m42 = pd.read_csv(V2_FOLDS); print("seed-42 patient folds read from", V2_FOLDS)
else:
    print("seed-42 patient folds: v2 file not found -> rebuilt (same seed, same algorithm)"); m42 = build_folds(42)
m43 = build_folds(43)
atomic_to_csv(m43.groupby(["fold", "role"]).size().rename("patients").reset_index(), DIRS["manifests"] / f"{V}_seed43_fold_sizes.csv")
same = (m42.sort_values(["fold", "subject_id"]).role.values == m43.sort_values(["fold", "subject_id"]).role.values).mean()
print(f"seed 43 vs seed 42: identical role assignments {same:.1%} (should be far from 100%)")
for sname, man in [("P2", m42), ("P2s43", m43)]:
    for k in range(5):
        role = rec.subject_id.map(man[man.fold == k].set_index("subject_id").role).values
        assert pd.notna(role).all()
        idx = {r: np.where(role == r)[0] for r in ("train", "val", "test")}
        s_tr, s_va, s_te = [set(rec.subject_id.values[idx[r]]) for r in ("train", "val", "test")]
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        np.savez(CACHE / f"{V}_splits" / f"{sname}__f{k}.npz", idx_tr=idx["train"], idx_va=idx["val"], idx_te=idx["test"])

STRAT4 = ["none", "class_weight", "ROS", "SMOTE*"]          # SMOTE* = SMOTE-NC (type-aware) if the input has discrete columns, else SMOTE
PCA_FEATS = ["img_pc32", "img_pc64", "img_pc128", "img_full", "fus_pc32", "fus_pc64", "fus_pc128", "fus_full"]
PROTO = {}
for seed_tag, split in [("", "P2"), ("_s43", "P2s43")]:
    PROTO[f"P2_rel{seed_tag}"] = dict(split=split, label="rel", feats={"ehr": STRAT4, "fus_pc64": STRAT4, "img_pc64": ["none"]})
    PROTO[f"P4_adm{seed_tag}"] = dict(split=split, label="adm", feats={"ehr": ["none"], "img_pc64": ["none"], "fus_pc64": ["none"], "agehist_a": ["none"]})
    PROTO[f"P4C_cln{seed_tag}"] = dict(split=split, label="cln", feats={"ehr": ["none"], "img_pc64": ["none"], "fus_pc64": ["none"], "agehist_c": ["none"]})
for p in ["P2_rel", "P4_adm", "P4C_cln"]:                    # PCA sensitivity (seed 42)
    for f in PCA_FEATS:
        PROTO[p]["feats"].setdefault(f, ["none"])
for p, h in [("P4_adm", "a"), ("P4C_cln", "c")]:             # age / history decomposition (seed 42)
    for f in ["age_rel", "age_own", f"hist_{h}"]:
        PROTO[p]["feats"].setdefault(f, ["none"])
FOLDS = list(range(5))
TGTS = TARGETS
if RUN_MODE == "preflight" or (LOCAL_TEST and os.environ.get("V8_TINY") == "1"):
    FOLDS = [0]; TGTS = ["pneumonia", "fracture"]
TASKS, skipped = [], []
for p, cfg in PROTO.items():
    for t in TGTS:
        y = lab[f"{cfg['label']}__{t}"]
        for k in FOLDS:
            sp = np.load(CACHE / f"{V}_splits" / f"{cfg['split']}__f{k}.npz")
            ptr, pva, pte = int(y[sp["idx_tr"]].sum()), int(y[sp["idx_va"]].sum()), int(y[sp["idx_te"]].sum())
            if ptr < 10 or pva < 3 or pte < 3:
                skipped.append({"protocol": p, "fold": k, "target": t, "pos_tr": ptr, "pos_va": pva, "pos_te": pte}); continue
            for ft in cfg["feats"]:
                TASKS.append((p, k, ft, t))
atomic_to_csv(pd.DataFrame(skipped, columns=["protocol", "fold", "target", "pos_tr", "pos_va", "pos_te"]), DIRS["manifests"] / f"{V}_skipped.csv")
n_fits = sum(len(PROTO[p]["feats"][ft]) for p, _, ft, _ in TASKS)
print(f"tasks {len(TASKS)} | fits {n_fits} | skipped (too few positives) {len(skipped)}")

# %%
# ============================== CELL 6 — MODEL / SAMPLER / METRIC FUNCTIONS (same as the paper) ==============================
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

def build_preprocessor(fset):
    trs = []
    if fset["cont"]:
        trs.append(("cont", Pipeline([("sc", StandardScaler()),
                                      ("imp", KNNImputer(n_neighbors=5, weights="distance", add_indicator=True))]), fset["cont"]))
    if fset["emb"]:
        steps = [("sc", StandardScaler())]
        if fset["pca"]:
            steps.append(("pca", PCA(n_components=fset["pca"], random_state=0)))
        trs.append(("emb", Pipeline(steps), fset["emb"]))
    if fset["bin"]:
        trs.append(("bin", "passthrough", fset["bin"]))
    return ColumnTransformer(trs, remainder="drop", sparse_threshold=0)

def discrete_columns(ct):
    pos, disc, evr = 0, [], np.nan
    for name, trans, cols in ct.transformers_:
        if name == "cont":
            imp = trans.named_steps["imp"]
            n_ind = len(imp.indicator_.features_) if getattr(imp, "indicator_", None) is not None else 0
            pos += len(cols); disc += list(range(pos, pos + n_ind)); pos += n_ind
        elif name == "emb":
            if "pca" in trans.named_steps:
                evr = float(trans.named_steps["pca"].explained_variance_ratio_.sum()); pos += trans.named_steps["pca"].n_components_
            else:
                evr = 1.0; pos += len(cols)
        elif name == "bin":
            disc += list(range(pos, pos + len(cols))); pos += len(cols)
    return np.array(disc, dtype=int), pos, evr

def resample(strategy, X, y, disc, seed):
    if strategy in ("none", "class_weight"):
        return X, y
    n_min = int(min(y.sum(), len(y) - y.sum())); k = max(1, min(5, n_min - 1))
    if strategy == "ROS":
        s = RandomOverSampler(random_state=seed)
    elif strategy == "SMOTE*":
        s = SMOTENC(categorical_features=list(disc), k_neighbors=k, random_state=seed) if len(disc) else SMOTE(k_neighbors=k, random_state=seed)
    else:
        raise ValueError(strategy)
    return s.fit_resample(X, y)

def choose_threshold(y, p):
    yb = y.astype(bool); P = p[None, :] >= GRID[:, None]
    tp = (P & yb).sum(1); fp = (P & ~yb).sum(1); fn = (~P & yb).sum(1)
    d = 5 * tp + 4 * fn + fp
    f2 = np.where(d > 0, 5 * tp / np.maximum(d, 1), 0.0)
    rec_ = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    pre_ = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return float(GRID[np.lexsort((np.abs(GRID - 0.5), -pre_, -rec_, -f2))[0]])

def metric_block(y, p, thr, prefix, p_train):
    y = np.asarray(y).astype(bool); pred = p >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    d2 = 5 * tp + 4 * fn + fp; prev = float(y.mean())
    out = {"n": len(y), "pos": int(y.sum()), "prev": prev, "f2": 5 * tp / d2 if d2 else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0, "precision": tp / (tp + fp) if tp + fp else 0.0,
           "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
           "brier": float(np.mean((p - y) ** 2)), "brier_prev": float(np.mean((p_train - y) ** 2)),
           "trivial_f2": 5 * prev / (4 * prev + 1)}
    return {f"{prefix}_{k}": v for k, v in out.items()}

# %%
# ============================== CELL 7 — STAGE A: PREPROCESSING CACHE ==============================
def prep_task(key, cache, featsets):
    sname, fold, feat = key
    out = Path(cache) / f"{V}_X" / f"{sname}__f{fold}__{feat}.npz"
    if out.exists():
        return key, "cached"
    tab = pd.read_pickle(Path(cache) / f"{V}_rec.pkl")
    sp = np.load(Path(cache) / f"{V}_splits" / f"{sname}__f{fold}.npz")
    ct = build_preprocessor(featsets[feat])
    Xtr = ct.fit_transform(tab.iloc[sp["idx_tr"]]).astype(np.float32)
    Xva = ct.transform(tab.iloc[sp["idx_va"]]).astype(np.float32)
    Xte = ct.transform(tab.iloc[sp["idx_te"]]).astype(np.float32)
    disc, ncol, evr = discrete_columns(ct); assert Xtr.shape[1] == ncol
    if len(disc):
        assert np.isin(Xtr[:, disc], [0.0, 1.0]).all()
    atomic_savez(out, Xtr=Xtr, Xva=Xva, Xte=Xte, disc=disc, evr=np.array([evr]))
    return key, "built"

NP = N_PAR or os.cpu_count()
def task_done(task):
    p, k, ft, t = task
    return (DIRS["results"] / f"{V}_{p}__f{k}__{ft}__{t}.csv").exists()
TODO = [t for t in TASKS if not task_done(t)]
print(f"tasks finished in earlier sessions: {len(TASKS) - len(TODO)} | remaining: {len(TODO)}")
prep_keys = sorted({(PROTO[p]["split"], k, ft) for p, k, ft, t in TODO})
t0 = time.time()
Parallel(n_jobs=min(NP, max(1, len(prep_keys))))(delayed(prep_task)(k, str(CACHE), FEATSETS) for k in tqdm(prep_keys, desc="preprocess"))
print(f"stage A: {len(prep_keys)} caches in {(time.time() - t0) / 60:.1f} min")

# %%
# ============================== CELL 8 — STAGE B: LOGISTIC REGRESSION FITS (resume-safe: 1 file per task) ==============================
def model_task(task, cfg):
    protocol, fold, feat, target = task
    tag = f"{protocol}__f{fold}__{feat}__{target}"
    res_path = Path(cfg["results"]) / f"{V}_{tag}.csv"
    if res_path.exists():
        return task, 0, 0
    pc = cfg["proto"][protocol]
    Xd = np.load(Path(cfg["cache"]) / f"{V}_X" / f"{pc['split']}__f{fold}__{feat}.npz")
    sp = np.load(Path(cfg["cache"]) / f"{V}_splits" / f"{pc['split']}__f{fold}.npz")
    y = np.load(Path(cfg["cache"]) / f"{V}_labels.npz")[f"{pc['label']}__{target}"].astype(int)
    ytr, yva, yte = y[sp["idx_tr"]], y[sp["idx_va"]], y[sp["idx_te"]]
    Xtr, Xva, Xte, disc = Xd["Xtr"], Xd["Xva"], Xd["Xte"], Xd["disc"]
    rows, errors = [], []
    for strategy in pc["feats"][feat]:
        job_key = f"{tag}__{strategy}__LR"; t0 = time.time()
        try:
            Xs, ys = resample(strategy, Xtr, ytr, disc, 42)
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced" if strategy == "class_weight" else None)
            clf.fit(Xs, ys)
            pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
            thr = choose_threshold(yva, pva); ptrain = float(ytr.mean())
            row = {"version": V, "data_hash": cfg["data_hash"], "job_key": job_key, "protocol": protocol, "split": pc["split"],
                   "seed": 43 if pc["split"].endswith("s43") else 42, "label_type": pc["label"], "fold": fold, "feature_set": feat,
                   "target": target, "strategy": strategy, "classifier": "LR", "threshold": thr, "n_features": Xtr.shape[1],
                   "pca_explained_var": float(Xd["evr"][0]), "n_train": len(ytr), "pos_train": int(ytr.sum())}
            row.update(metric_block(yva, pva, thr, "val", ptrain)); row.update(metric_block(yte, pte, thr, "test", ptrain))
            row["fit_seconds"] = time.time() - t0
            rows.append(row)
        except Exception as e:
            errors.append({"job_key": job_key, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
    if rows:
        atomic_to_csv(pd.DataFrame(rows), res_path)
    if errors:
        atomic_to_csv(pd.DataFrame(errors), Path(cfg["errors"]) / f"{V}_{tag}.csv")
    return task, len(rows), len(errors)

CFG = {"cache": str(CACHE), "results": str(DIRS["results"]), "errors": str(DIRS["errors"]), "proto": PROTO, "data_hash": data_hash}
t0 = time.time(); tot_new = tot_err = 0
try:
    gen = Parallel(n_jobs=NP, return_as="generator_unordered")(delayed(model_task)(t, CFG) for t in TODO)
except (TypeError, ValueError):
    gen = Parallel(n_jobs=NP, return_as="generator")(delayed(model_task)(t, CFG) for t in TODO)
pbar = tqdm(gen, total=len(TODO), desc="model tasks")
for task, n_new, n_err in pbar:
    tot_new += n_new; tot_err += n_err; pbar.set_postfix(new_fits=tot_new, errors=tot_err)
print(f"stage B: {(time.time() - t0) / 60:.1f} min | new fits {tot_new} | errors {tot_err}")

# %%
# ============================== CELL 9 — MERGE + QUICK SUMMARY + ZIP ==============================
res_files = sorted(glob.glob(str(DIRS["results"] / f"{V}_*.csv")))
R = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True) if res_files else pd.DataFrame()
err_files = sorted(glob.glob(str(DIRS["errors"] / f"{V}_*.csv")))
ERR = pd.concat([pd.read_csv(f) for f in err_files], ignore_index=True) if err_files else pd.DataFrame()
print(f"fits done {len(R)} / expected {n_fits} | errors {len(ERR)}")
if len(ERR):
    print(ERR[["job_key", "error"]].head(10).to_string())
atomic_to_csv(R, DIRS["analysis"] / f"{V}_results_all.csv.gz")
R0 = R[R.strategy == "none"]
fm = R0.groupby(["protocol", "feature_set", "fold"]).agg(f2=("test_f2", "mean"), tv=("test_trivial_f2", "mean"), au=("test_roc_auc", "mean"),
                                                         br=("test_brier", "mean"), brp=("test_brier_prev", "mean"), ev=("pca_explained_var", "mean")).reset_index()
SUM = fm.groupby(["protocol", "feature_set"]).agg(f2=("f2", "mean"), f2_sd=("f2", "std"), triv=("tv", "mean"), auroc=("au", "mean"),
                                                  brier=("br", "mean"), brier_prev=("brp", "mean"), pca_ev=("ev", "mean")).reset_index()
SUM["delta"] = SUM.f2 - SUM.triv
atomic_to_csv(SUM.round(4), DIRS["analysis"] / f"{V}_summary_none.csv")
print("\nMACRO (logistic regression, no augmentation)\n", SUM.round(3).to_string(index=False))
ST = R[R.protocol.str.startswith("P2_rel")].groupby(["protocol", "feature_set", "strategy"]).agg(f2=("test_f2", "mean"), brier=("test_brier", "mean")).reset_index()
atomic_to_csv(ST.round(4), DIRS["analysis"] / f"{V}_strategies_P2.csv")
print("\nP2 strategies (both seeds)\n", ST.pivot_table(index=["protocol", "feature_set"], columns="strategy", values="f2").round(4).to_string())
numbers = {"version": VERSION, "data_hash": data_hash, "n_fits": len(R), "n_expected": n_fits, "n_errors": len(ERR), "skipped": skipped,
           "seed43_identical_roles": float(same),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
atomic_json(numbers, DIRS["analysis"] / f"{V}_paper_numbers.json")
ZIP = OUT / f"{V}_for_claude.zip"
tmpz = str(ZIP) + ".tmp"
with zipfile.ZipFile(tmpz, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f"{OUT.name}/{DIRS[d].name}/{f.name}")
os.replace(tmpz, ZIP)
print("\nFILE TO DOWNLOAD:", ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")

# %%
# ============================== CELL 10 — FLUSH TO DRIVE (required before closing the tab) ==============================
if IN_COLAB:
    print("Flushing all files to Google Drive... do NOT close the tab until 'FLUSH DONE' appears.")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("FLUSH DONE — download", ZIP.name, "from", OUT)
