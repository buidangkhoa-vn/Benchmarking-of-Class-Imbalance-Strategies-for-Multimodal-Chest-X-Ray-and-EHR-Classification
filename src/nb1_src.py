# %% [markdown]
# # MMMDA NB1 — Leakage-aware class-imbalance benchmark (CXR features + EHR)
#
# **What this notebook does**
# 1. Locks the input CSV (shape, hash, subject-level constancy of labels/EHR).
# 2. Builds three evaluation protocols on the *same* data:
#    - `P1_record`  – record-level 5-fold CV that ignores `subject_id` (reproduces the leak of the old benchmark)
#    - `P2_grouped` – record-level rows, folds grouped by subject
#    - `P3_patient` – one row per subject (labels/EHR are subject-level), same subject folds as P2
# 3. Runs imbalance strategies × classifiers × feature sets × targets with an inner validation split
#    for threshold selection (test folds are never used for any choice).
# 4. Computes baselines (predict-all-positive, EHR lookup table), merges results, and produces the
#    paper tables / figures / `paper_numbers.json`.
#
# **How to run (Colab Pro)**
# - Runtime: CPU **High-RAM** (or any GPU runtime – only the CPU cores are used here). More cores = faster.
# - Run cells top to bottom. First run with `RUN_MODE = "preflight"` (~5–15 min), check the printed
#   preview/ETA, then set `RUN_MODE = "full"` and run again.
# - Several Colab sessions can share the work: open this notebook in 2–3 sessions, set the same `RUN_TAG`,
#   `N_WORKERS = k` and a different `WORKER_INDEX` (0..k-1) in each. Everything is resume-safe: if a
#   session dies, just re-run all cells; finished jobs are skipped.
# - When all workers are done, run the **MERGE** and **ANALYSIS** cells once (any session, `WORKER_INDEX`
#   does not matter there).
# - **Stronger image embeddings (NB2):** `EMBEDDINGS` points to NB2's output folders and is used only when
#   `emb.npy` exists. Start NB1 now; when NB2 (full) has finished, simply re-run NB1 with the same `RUN_TAG` in
#   every worker session — finished jobs are skipped and only the new embedding jobs (P2/P3) run.
# - P3 uses frontal (AP/PA) radiographs for patient-level image features when NB2's `dicom_meta.csv` exists.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE      = "preflight"          # "preflight" (tiny grid, separate folder) | "full"
RUN_TAG       = "leakage_aware_v2"   # new tag => new output folder; never reuse a tag for a changed protocol
WORKER_INDEX  = 0                    # 0..N_WORKERS-1 (one value per Colab session)
N_WORKERS     = 1                    # number of Colab sessions sharing the full run
ROLE          = "single"             # "worker": train only | "final": merge + analysis only | "single": everything
REQUIRE_EMBEDDINGS = False           # True: stop if the NB2 embeddings are missing

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
OUT_ROOT      = f"{DRIVE_ROOT}/leakage_aware_v2"
LOCAL_CACHE   = "/content/mmmda_cache"          # fast local disk; rebuilt automatically after a restart
LOCAL_OUT     = "/content/mmmda_out"            # per-job working files (local); each finished task is packed to Drive
EXPECTED_HASH = "da85519958e4bccb"              # hash of the CSV used by the published benchmark

PROTOCOLS     = ["P3_patient", "P2_grouped", "P1_record"]
CV_SEEDS      = [42]                 # add 43 later for repeated CV if time allows
N_FOLDS       = 5
INNER_VAL_FRAC = 0.20

# Stronger image embeddings from NB2 (full run). Folders without emb.npy are ignored, so this run can start
# before NB2 finishes: re-run NB1 later with the same RUN_TAG and only the new embedding jobs are executed.
EMBEDDINGS    = {"raddino": f"{DRIVE_ROOT}/embeddings/raddino", "xrv1024": f"{DRIVE_ROOT}/embeddings/xrv_chex"}
EMB_PROTOCOLS = ["P3_patient", "P2_grouped"]    # embedding feature sets are not run under P1 (leak already shown)
EMB_PCA_DIM   = 64                   # PCA (fit on inner-train only) applied to embeddings
# View metadata written by NB2 (cell 3). Used for P3: patient-level image features = mean over FRONTAL (AP/PA)
# radiographs when the patient has one, otherwise over all radiographs. None/missing file -> plain mean.
VIEW_META     = f"{DRIVE_ROOT}/embeddings/dicom_meta.csv"

N_PAR         = None                 # parallel processes; None -> all CPU cores

# %%
# ============================== CELL 2 — INSTALL + MOUNT ==============================
import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("MMMDA_LOCAL_TEST") == "1"     # only used for offline smoke tests
if LOCAL_TEST:
    CSV_PATH = os.environ["MMMDA_CSV"]; OUT_ROOT = os.environ["MMMDA_OUT"]; LOCAL_CACHE = os.environ["MMMDA_CACHE"]
    LOCAL_OUT = os.environ.get("MMMDA_LOCAL_OUT", LOCAL_CACHE + "_out")
    RUN_MODE = os.environ.get("MMMDA_MODE", RUN_MODE); N_PAR = int(os.environ.get("MMMDA_NPAR", "2"))
    EMBEDDINGS = __import__("json").loads(os.environ.get("MMMDA_EMB", "{}"))
    VIEW_META = os.environ.get("MMMDA_VIEW", "")
if IN_COLAB:
    for mod, pkg in [("iterstrat", "iterative-stratification"), ("imblearn", "imbalanced-learn"),
                     ("lightgbm", "lightgbm"), ("xgboost", "xgboost")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
print("cores:", os.cpu_count(), "| RUN_MODE:", RUN_MODE, "| worker", WORKER_INDEX, "/", N_WORKERS)

# %%
# ============================== CELL 3 — IMPORTS + HELPERS ==============================
import json, time, hashlib, traceback, warnings, platform, glob
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

def atomic_to_csv(df, path):
    path = str(path); tmp = path + ".tmp"
    df.to_csv(tmp, index=False); os.replace(tmp, path)

def atomic_savez(path, **arrays):
    path = str(path); tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def atomic_json(obj, path):
    path = str(path); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)

OUT = Path(OUT_ROOT) / (RUN_TAG + ("_preflight" if RUN_MODE == "preflight" else ""))
CACHE = Path(LOCAL_CACHE) / OUT.name
LOUT = Path(LOCAL_OUT) / OUT.name
for d in [OUT / "manifests", OUT / "results", OUT / "preds_packed", OUT / "errors", OUT / "analysis", CACHE / "X", CACHE / "splits",
          LOUT / "results", LOUT / "preds"]:
    d.mkdir(parents=True, exist_ok=True)
print("OUT  :", OUT); print("CACHE:", CACHE)

# %%
# ============================== CELL 4 — LOAD + LOCK DATA ==============================
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV differs from the benchmarked file — STOP"
assert df.subject_id.nunique() == 10479

TARGETS  = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
IMG_DEAD = ["2", "5", "6", "9", "11", "12", "13"]   # TorchXRayVision heads untrained by CheXpert weights (always ~0)
IMG      = [str(i) for i in range(18) if str(i) not in IMG_DEAD]
EHR_CONT = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc"]
EHR_BIN  = ["diabete", "hiv", "smoke", "cough"]
EHR      = EHR_CONT + EHR_BIN

for c in IMG_DEAD:
    assert df[c].abs().max() < 1e-30, c
for c in TARGETS:
    assert df[c].isna().sum() == 0 and set(df[c].unique()) <= {0, 1}, c
assert df[EHR_BIN].isna().sum().sum() == 0
const = df.groupby("subject_id")[TARGETS + EHR].nunique(dropna=False).max()
assert (const <= 1).all(), "labels/EHR are NOT constant within subject — protocol assumptions break, STOP"

df["dataset_position"] = np.arange(len(df))
subj_lab = df.groupby("subject_id")[TARGETS].first()
data_audit = pd.DataFrame({"pos_records": df[TARGETS].sum(), "pos_subjects": subj_lab.sum(),
                           "prev_records_%": (100 * df[TARGETS].mean()).round(2)})
print(data_audit)
ehr_key = pd.util.hash_pandas_object(df[EHR].fillna(-999), index=False)
print("records/subject: mean %.2f, max %d | unique EHR vectors %d"
      % (df.groupby("subject_id").size().mean(), df.groupby("subject_id").size().max(), ehr_key.nunique()))

# %%
# ============================== CELL 5 — TABLES + FEATURE SETS ==============================
rec = df[["dataset_position", "subject_id", "dicom", "StudyDate"] + TARGETS + EHR + IMG].copy()
EMB_COLS = {}
for name, folder in EMBEDDINGS.items():
    if not os.path.exists(f"{folder}/emb.npy"):
        print(f"embedding {name}: not found at {folder} (skipped — run NB2 full, then re-run NB1)")
        continue
    E = np.load(f"{folder}/emb.npy").astype(np.float32)
    ids = pd.read_csv(f"{folder}/ids.csv")
    pos = rec.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
    assert pos.notna().all(), f"{name}: {int(pos.isna().sum())} dicoms have no embedding"
    cols = [f"{name}_{j}" for j in range(E.shape[1])]
    rec = pd.concat([rec, pd.DataFrame(E[pos.astype(int).values], columns=cols, index=rec.index)], axis=1)
    EMB_COLS[name] = cols
    print(f"embedding {name}: dim {E.shape[1]}")

# view metadata (from NB2): frontal = AP or PA
view_stats = {}
if VIEW_META and os.path.exists(VIEW_META):
    vm = pd.read_csv(VIEW_META, usecols=["dicom", "ViewPosition"])
    assert rec.dicom.isin(vm.dicom).all() and vm.dicom.is_unique, "dicom_meta.csv incomplete — re-run NB2 cell 3"
    rec = rec.merge(vm, on="dicom", how="left")
    assert len(rec) == len(df) and (rec.dataset_position.values == np.arange(len(df))).all()
    rec["frontal"] = rec.ViewPosition.isin(["AP", "PA"])
    view_stats = {"view_counts": rec.ViewPosition.fillna("NA").value_counts().to_dict(),
                  "frontal_records": int(rec.frontal.sum()),
                  "subjects_with_frontal": int(rec.groupby("subject_id").frontal.any().sum())}
    print("views:", view_stats)
else:
    rec["frontal"] = True
    print("no view metadata — patient-level image features use all radiographs")

if REQUIRE_EMBEDDINGS and len(EMB_COLS) < len(EMBEDDINGS):
    raise RuntimeError("Chưa có embedding của NB2 (emb.npy). Hãy chạy NB2 với RUN_MODE='full' cho xong rồi Run all lại notebook này.")
IMG_ALL_COLS = IMG + sum(EMB_COLS.values(), [])
agg = {c: "first" for c in TARGETS + EHR}
pat = rec.groupby("subject_id").agg(agg)
img_all = rec.groupby("subject_id")[IMG_ALL_COLS].mean()
img_front = rec[rec.frontal].groupby("subject_id")[IMG_ALL_COLS].mean()
img_pat = img_all.copy()
img_pat.loc[img_front.index] = img_front
pat = pat.join(img_pat).reset_index()
pat["n_records"] = rec.groupby("subject_id").size().reindex(pat.subject_id).values
pat["has_frontal"] = rec.groupby("subject_id").frontal.any().reindex(pat.subject_id).values
assert len(pat) == 10479

FEATSETS = {"img_xrv": dict(cont=IMG, emb=[], bin=[]),
            "ehr": dict(cont=EHR_CONT, emb=[], bin=EHR_BIN),
            "fusion_xrv": dict(cont=IMG + EHR_CONT, emb=[], bin=EHR_BIN)}
for name, cols in EMB_COLS.items():
    FEATSETS[f"img_{name}"] = dict(cont=[], emb=cols, bin=[])
    FEATSETS[f"fusion_{name}"] = dict(cont=EHR_CONT, emb=cols, bin=EHR_BIN)
TABLE_OF = {"P1_record": "rec", "P2_grouped": "rec", "P3_patient": "pat"}

rec.to_pickle(CACHE / "rec.pkl"); pat.to_pickle(CACHE / "pat.pkl")
np.savez(CACHE / "labels_rec.npz", **{t: rec[t].values.astype(np.int8) for t in TARGETS}, subject_id=rec.subject_id.values)
np.savez(CACHE / "labels_pat.npz", **{t: pat[t].values.astype(np.int8) for t in TARGETS}, subject_id=pat.subject_id.values)
print("feature sets:", list(FEATSETS))

# %%
# ============================== CELL 6 — GRID (preflight vs full) ==============================
CLFS_ALL   = ["LR", "RF", "ET", "LGBM", "XGB"]
NATIVE_ALL = ["BRF", "EasyEns"]

if RUN_MODE == "preflight":
    G_PROTOCOLS = [p for p in ["P3_patient", "P2_grouped"] if p in PROTOCOLS]
    G_SEEDS, G_FOLDS = CV_SEEDS[:1], [0]
    G_FEATS = ["ehr", "fusion_xrv"] + [f"img_{n}" for n in list(EMB_COLS)[:1]]
    G_TARGETS = ["pneumonia", "scoliosis"]
    G_CLFS, G_NATIVE = ["LR", "RF", "LGBM", "XGB"], ["BRF"]
    G_STRATS = {"none", "class_weight", "ROS", "SMOTE", "SMOTE_naive", "SMOTENC_typeaware"}
else:
    G_PROTOCOLS, G_SEEDS, G_FOLDS = PROTOCOLS, CV_SEEDS, list(range(N_FOLDS))
    G_FEATS, G_TARGETS = list(FEATSETS), TARGETS
    G_CLFS, G_NATIVE = CLFS_ALL, NATIVE_ALL
    G_STRATS = {"none", "class_weight", "ROS", "SMOTE", "SMOTE_naive", "SMOTENC_typeaware"}
# P1 only needs to reproduce the leak -> reduced grid
P1_CLFS, P1_STRATS, P1_NATIVE = ["LR", "RF", "ET", "LGBM"], {"none", "ROS", "SMOTE", "SMOTE_naive", "SMOTENC_typeaware"}, []
if LOCAL_TEST and os.environ.get("MMMDA_TINY") == "1":   # offline smoke test only — ignored on Colab
    G_FOLDS = [0, 1]; G_TARGETS = ["pneumonia", "scoliosis"]; G_CLFS = ["LR", "LGBM"]; G_NATIVE = ["BRF"]; P1_CLFS = ["LR", "LGBM"]

def job_grid(protocol, has_discrete):
    clfs = P1_CLFS if protocol == "P1_record" else G_CLFS
    strats = P1_STRATS if protocol == "P1_record" else G_STRATS
    native = P1_NATIVE if protocol == "P1_record" else G_NATIVE
    base = ["none", "class_weight", "ROS", "SMOTE_naive", "SMOTENC_typeaware"] if has_discrete else ["none", "class_weight", "ROS", "SMOTE"]
    jobs = [(s, c) for s in base if s in strats for c in clfs]
    jobs += [("native", c) for c in native]
    return jobs

# %%
# ============================== CELL 7 — FOLD MANIFESTS + SPLITS ==============================
def subject_manifest(seed):
    path = OUT / "manifests" / f"folds_subject_s{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    Ys = pat[TARGETS].values
    parts = []
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(mskf.split(np.zeros(len(Ys)), Ys)):
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed * 100 + k)
        itr, iva = next(msss.split(np.zeros(len(tr)), Ys[tr]))
        role = np.empty(len(Ys), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"subject_id": pat.subject_id.values, "fold": k, "role": role}))
    m = pd.concat(parts, ignore_index=True); atomic_to_csv(m, path); return m

def record_manifest(seed):
    path = OUT / "manifests" / f"folds_record_s{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    Yr = rec[TARGETS].values
    parts = []
    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(mskf.split(np.zeros(len(Yr)), Yr)):
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=seed * 100 + k)
        itr, iva = next(msss.split(np.zeros(len(tr)), Yr[tr]))
        role = np.empty(len(Yr), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"dataset_position": rec.dataset_position.values, "fold": k, "role": role}))
    m = pd.concat(parts, ignore_index=True); atomic_to_csv(m, path); return m

split_report = []
for seed in CV_SEEDS:
    ms = subject_manifest(seed)
    mr = record_manifest(seed) if "P1_record" in PROTOCOLS else None
    for k in range(N_FOLDS):
        rmap = ms[ms.fold == k].set_index("subject_id").role
        roles = {"P3_patient": pat.subject_id.map(rmap).values, "P2_grouped": rec.subject_id.map(rmap).values}
        if mr is not None:
            roles["P1_record"] = mr[mr.fold == k].set_index("dataset_position").role.reindex(rec.dataset_position).values
        for proto, role in roles.items():
            tab = pat if proto == "P3_patient" else rec
            assert pd.notna(role).all()
            i_tr, i_va, i_te = [np.where(role == r)[0] for r in ("train", "val", "test")]
            sid = tab.subject_id.values
            s_tr, s_va, s_te = set(sid[i_tr]), set(sid[i_va]), set(sid[i_te])
            seen_te = np.isin(sid[i_te], list(s_tr | s_va))
            unseen_va = ~np.isin(sid[i_va], list(s_tr))
            if proto != "P1_record":
                assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), f"subject overlap in {proto}"
                assert not seen_te.any()
            np.savez(CACHE / "splits" / f"{proto}__s{seed}__f{k}.npz", idx_tr=i_tr, idx_va=i_va, idx_te=i_te,
                     seen_te=seen_te, unseen_va=unseen_va)
            np.savez(OUT / "manifests" / f"split_{proto}__s{seed}__f{k}.npz", idx_tr=i_tr, idx_va=i_va, idx_te=i_te,
                     seen_te=seen_te, unseen_va=unseen_va)
            ytab = tab[TARGETS].values
            split_report.append({"protocol": proto, "seed": seed, "fold": k, "n_tr": len(i_tr), "n_va": len(i_va),
                                 "n_te": len(i_te), "test_seen_%": round(100 * seen_te.mean(), 1),
                                 "min_test_pos": int(ytab[i_te].sum(0).min()),
                                 "min_val_pos": int(ytab[i_va].sum(0).min()), "min_train_pos": int(ytab[i_tr].sum(0).min())})
split_report = pd.DataFrame(split_report)
atomic_to_csv(split_report, OUT / "manifests" / "split_report.csv")
print(split_report.groupby("protocol")[["n_tr", "n_va", "n_te", "test_seen_%", "min_test_pos", "min_val_pos", "min_train_pos"]].mean().round(1))

# %%
# ============================== CELL 8 — MODEL / SAMPLER / METRIC FUNCTIONS ==============================
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)

def build_preprocessor(fs, pca_dim):
    trs = []
    if fs["cont"]:
        trs.append(("cont", Pipeline([("sc", StandardScaler()),
                                      ("imp", KNNImputer(n_neighbors=5, weights="distance", add_indicator=True))]), fs["cont"]))
    if fs["emb"]:
        trs.append(("emb", Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(pca_dim, len(fs["emb"])), random_state=0))]), fs["emb"]))
    if fs["bin"]:
        trs.append(("bin", "passthrough", fs["bin"]))
    return ColumnTransformer(trs, remainder="drop", sparse_threshold=0)

def column_types(ct):
    pos, out = 0, {"cont": [], "ind": [], "emb": [], "bin": []}
    for name, trans, cols in ct.transformers_:
        if name == "cont":
            imp = trans.named_steps["imp"]
            n_ind = len(imp.indicator_.features_) if getattr(imp, "indicator_", None) is not None else 0
            out["cont"] += list(range(pos, pos + len(cols))); pos += len(cols)
            out["ind"] += list(range(pos, pos + n_ind)); pos += n_ind
        elif name == "emb":
            k = trans.named_steps["pca"].n_components_
            out["emb"] += list(range(pos, pos + k)); pos += k
        elif name == "bin":
            out["bin"] += list(range(pos, pos + len(cols))); pos += len(cols)
    return out, pos

def make_clf(name, seed, cw, pos_weight):
    cwv = "balanced" if cw else None
    if name == "LR":   return LogisticRegression(C=1.0, max_iter=2000, class_weight=cwv)
    if name == "RF":   return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed, class_weight=cwv)
    if name == "ET":   return ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=seed, class_weight=cwv)
    if name == "LGBM": return LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, random_state=seed, verbose=-1, class_weight=cwv)
    if name == "XGB":  return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist", n_jobs=1,
                                            random_state=seed, eval_metric="logloss", scale_pos_weight=(pos_weight if cw else 1.0))
    if name == "BRF":  return BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all", replacement=True,
                                                             bootstrap=False, n_jobs=1, random_state=seed)
    if name == "EasyEns": return EasyEnsembleClassifier(n_estimators=10, n_jobs=1, random_state=seed)
    raise ValueError(name)

def resample(strategy, X, y, disc, seed):
    audit = {"syn_rows": 0, "syn_bad_row_frac": np.nan, "syn_bad_cell_frac": np.nan}
    if strategy in ("none", "class_weight", "native"):
        return X, y, audit
    n_min = int(min(y.sum(), len(y) - y.sum()))
    k = max(1, min(5, n_min - 1))
    if strategy == "ROS":
        s = RandomOverSampler(random_state=seed)
    elif strategy in ("SMOTE", "SMOTE_naive"):
        s = SMOTE(k_neighbors=k, random_state=seed)
    elif strategy == "SMOTENC_typeaware":
        s = SMOTENC(categorical_features=list(disc), k_neighbors=k, random_state=seed)
    else:
        raise ValueError(strategy)
    Xr, yr = s.fit_resample(X, y)
    syn = Xr[len(X):]
    audit["syn_rows"] = len(syn)
    if len(disc) and len(syn):
        bad = ~np.isin(syn[:, disc], [0.0, 1.0])
        audit["syn_bad_row_frac"] = float(bad.any(1).mean()); audit["syn_bad_cell_frac"] = float(bad.mean())
        if strategy == "SMOTENC_typeaware":
            assert not bad.any(), "type-aware sampler produced non-binary values"
    return Xr, yr, audit

def choose_threshold(y, p):
    yb = y.astype(bool); P = p[None, :] >= GRID[:, None]
    tp = (P & yb).sum(1); fp = (P & ~yb).sum(1); fn = (~P & yb).sum(1)
    d = 5 * tp + 4 * fn + fp
    f2 = np.where(d > 0, 5 * tp / np.maximum(d, 1), 0.0)
    rec_ = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    pre_ = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    best = np.lexsort((np.abs(GRID - 0.5), -pre_, -rec_, -f2))[0]
    return float(GRID[best])

def metric_block(y, p, thr, prefix):
    y = np.asarray(y).astype(bool); pred = p >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = len(y) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0; rec_ = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    d2 = 5 * tp + 4 * fn + fp
    prev = float(y.mean()) if len(y) else np.nan
    out = {"n": len(y), "pos": int(y.sum()), "prev": prev,
           "f2": 5 * tp / d2 if d2 else 0.0, "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
           "recall": rec_, "precision": prec, "bal_acc": (rec_ + spec) / 2,
           "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
           "brier": float(np.mean((p - y) ** 2)) if len(y) else np.nan,
           "trivial_f2": 5 * prev / (4 * prev + 1) if len(y) else np.nan,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {f"{prefix}_{k}": v for k, v in out.items()}

# %%
# ============================== CELL 9 — STAGE A: PREPROCESSING CACHE ==============================
def prep_task(key, cfg):
    protocol, seed, fold, feat = key
    out = Path(cfg["cache"]) / "X" / f"{protocol}__s{seed}__f{fold}__{feat}.npz"
    if out.exists():
        return key, "cached", 0.0
    t0 = time.time()
    tab = pd.read_pickle(Path(cfg["cache"]) / f"{cfg['table_of'][protocol]}.pkl")
    sp = np.load(Path(cfg["cache"]) / "splits" / f"{protocol}__s{seed}__f{fold}.npz")
    fs = cfg["featsets"][feat]
    ct = build_preprocessor(fs, cfg["emb_pca_dim"])
    Xtr = ct.fit_transform(tab.iloc[sp["idx_tr"]]).astype(np.float32)
    Xva = ct.transform(tab.iloc[sp["idx_va"]]).astype(np.float32)
    Xte = ct.transform(tab.iloc[sp["idx_te"]]).astype(np.float32)
    types, ncol = column_types(ct)
    assert Xtr.shape[1] == ncol, (Xtr.shape, ncol)
    disc = np.array(types["bin"] + types["ind"], dtype=int)
    if len(disc):
        assert np.isin(Xtr[:, disc], [0.0, 1.0]).all()
    atomic_savez(out, Xtr=Xtr, Xva=Xva, Xte=Xte, disc=disc,
                 ncont=len(types["cont"]), nind=len(types["ind"]), nemb=len(types["emb"]), nbin=len(types["bin"]))
    return key, "built", time.time() - t0

CFG = {"cache": str(CACHE), "out": str(OUT), "lout": str(LOUT), "table_of": TABLE_OF, "featsets": FEATSETS, "emb_pca_dim": EMB_PCA_DIM,
       "targets": TARGETS, "data_hash": data_hash, "run_tag": RUN_TAG}

# Stage-B tasks = (protocol, seed, fold, feat, target); sharded across workers by a stable hash
def feat_allowed(protocol, feat):
    is_emb = any(feat.endswith(f"_{n}") for n in EMB_COLS)
    return (not is_emb) or (protocol in EMB_PROTOCOLS)
ALL_TASKS = [(p, s, f, ft, t) for p in G_PROTOCOLS for s in G_SEEDS for f in G_FOLDS for ft in G_FEATS for t in G_TARGETS
             if feat_allowed(p, ft)]
MY_TASKS = [t for t in ALL_TASKS if stable_hash("|".join(map(str, t))) % N_WORKERS == WORKER_INDEX]
prep_keys = sorted({t[:4] for t in MY_TASKS})
print(f"tasks total {len(ALL_TASKS)} | this worker {len(MY_TASKS)} | preprocessing caches {len(prep_keys)}")

NP = N_PAR or os.cpu_count()
if ROLE != "final":
    t0 = time.time()
    prep_out = Parallel(n_jobs=min(NP, max(1, len(prep_keys))))(delayed(prep_task)(k, CFG) for k in tqdm(prep_keys, desc="preprocess"))
    print("stage A done in %.1f min" % ((time.time() - t0) / 60))
else:
    print("ROLE = final: no training in this notebook")

# %%
# ============================== CELL 10 — STAGE B: MODELS (resume-safe, parallel) ==============================
def pred_keys_on_drive(out, tag):
    """(strategy, classifier) pairs whose predictions are already safely on Drive (packed or legacy layout)."""
    keys = set()
    pp = Path(out) / "preds_packed" / f"{tag}.npz"
    if pp.exists():
        try:
            keys |= {tuple(k.split("__")[:2]) for k in np.load(pp).files if k.endswith("__p_te")}
        except Exception:
            pass                                   # unreadable / partially synced file -> recompute
    old = Path(out) / "preds" / tag                # legacy layout (one npz per job, workers of the first run)
    if old.exists():
        keys |= {tuple(f.stem.split("__")[:2]) for f in old.glob("*.npz") if not f.name.endswith(".tmp.npz")}
    return keys

def model_task(task, cfg, grid_fn_args):
    protocol, seed, fold, feat, target = task
    tag = f"{protocol}__s{seed}__f{fold}__{feat}__{target}"
    res_path = Path(cfg["out"]) / "results" / f"{tag}.csv"          # Drive: final per-task results
    pack_path = Path(cfg["out"]) / "preds_packed" / f"{tag}.npz"    # Drive: all predictions of the task in ONE file
    err_path = Path(cfg["out"]) / "errors" / f"{tag}.csv"
    lres_path = Path(cfg["lout"]) / "results" / f"{tag}.csv"        # local working copy (fast, many writes)
    lpdir = Path(cfg["lout"]) / "preds" / tag; lpdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in (lres_path, res_path):
        if p.exists():
            rows = pd.read_csv(p).to_dict("records"); break
    on_drive = pred_keys_on_drive(cfg["out"], tag)
    local_pred = {tuple(f.stem.split("__")[:2]) for f in lpdir.glob("*.npz") if not f.name.endswith(".tmp.npz")}
    done = {r["job_key"] for r in rows if (r["strategy"], r["classifier"]) in (on_drive | local_pred)}
    Xd = np.load(Path(cfg["cache"]) / "X" / f"{protocol}__s{seed}__f{fold}__{feat}.npz")
    sp = np.load(Path(cfg["cache"]) / "splits" / f"{protocol}__s{seed}__f{fold}.npz")
    lab = np.load(Path(cfg["cache"]) / f"labels_{cfg['table_of'][protocol]}.npz")
    y = lab[target].astype(int)
    ytr, yva, yte = y[sp["idx_tr"]], y[sp["idx_va"]], y[sp["idx_te"]]
    Xtr, Xva, Xte, disc = Xd["Xtr"], Xd["Xva"], Xd["Xte"], Xd["disc"]
    jobs = grid_fn_args[protocol][bool(len(disc))]
    errors, n_new = [], 0
    for strategy, clf_name in jobs:
        job_key = f"{tag}__{strategy}__{clf_name}"
        if job_key in done:
            continue
        t0 = time.time()
        try:
            Xs, ys, aud = resample(strategy, Xtr, ytr, disc, seed)
            pw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
            clf = make_clf(clf_name, seed, strategy == "class_weight", pw)
            clf.fit(Xs, ys)
            pva = clf.predict_proba(Xva)[:, 1].astype(np.float32)
            pte = clf.predict_proba(Xte)[:, 1].astype(np.float32)
            thr = choose_threshold(yva, pva)
            row = {"run_tag": cfg["run_tag"], "data_hash": cfg["data_hash"], "job_key": job_key, "protocol": protocol,
                   "seed": seed, "fold": fold, "feature_set": feat, "target": target, "strategy": strategy,
                   "classifier": clf_name, "threshold": thr, "n_features": Xtr.shape[1], "n_discrete": len(disc),
                   "n_train_before": len(ytr), "pos_train_before": int(ytr.sum()),
                   "n_train_after": len(ys), "pos_train_after": int(np.sum(ys)), **aud}
            row.update(metric_block(yva, pva, thr, "val"))
            row.update(metric_block(yte, pte, thr, "test"))
            if protocol == "P1_record":
                seen, uva = sp["seen_te"], sp["unseen_va"]
                row.update(metric_block(yte[seen], pte[seen], thr, "seen"))
                row.update(metric_block(yte[~seen], pte[~seen], thr, "unseen"))
                thr_clean = choose_threshold(yva[uva], pva[uva]) if yva[uva].sum() > 0 else thr
                row["unseen_cleanthr"] = thr_clean
                row.update(metric_block(yte[~seen], pte[~seen], thr_clean, "unseenclean"))
            row["fit_seconds"] = time.time() - t0
            row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            atomic_savez(lpdir / f"{strategy}__{clf_name}.npz", p_va=pva, p_te=pte, thr=np.float32(thr))
            rows = [r for r in rows if r["job_key"] != job_key] + [row]
            atomic_to_csv(pd.DataFrame(rows), lres_path)
            n_new += 1
        except Exception as e:
            errors.append({"job_key": job_key, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:],
                           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    # ---- publish the task to Drive as 2 files (results CSV + one packed npz) ----
    local_files = sorted(f for f in lpdir.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    if local_files or (rows and not res_path.exists()):
        arrays = {}
        if pack_path.exists():
            try:
                z = np.load(pack_path); arrays = {k: z[k] for k in z.files}
            except Exception:
                arrays = {}
        for f in local_files:
            z = np.load(f)
            for k in ("p_va", "p_te", "thr"):
                arrays[f"{f.stem}__{k}"] = z[k]
        if arrays:
            atomic_savez(pack_path, **arrays)
        if rows:
            atomic_to_csv(pd.DataFrame(rows), res_path)
    if errors:
        old = pd.read_csv(err_path).to_dict("records") if err_path.exists() else []
        atomic_to_csv(pd.DataFrame(old + errors), err_path)
    return task, n_new, len(errors)

GRID_BY_PROTO = {p: {True: job_grid(p, True), False: job_grid(p, False)} for p in PROTOCOLS}
n_jobs_expected = sum(len(GRID_BY_PROTO[t[0]][bool(FEATSETS[t[3]]["bin"])]) for t in MY_TASKS)
print("model jobs for this worker (approx):", n_jobs_expected)

if ROLE != "final":
    t0 = time.time(); tot_new = tot_err = 0
    try:
        gen = Parallel(n_jobs=NP, return_as="generator_unordered")(delayed(model_task)(t, CFG, GRID_BY_PROTO) for t in MY_TASKS)
    except (TypeError, ValueError):
        gen = Parallel(n_jobs=NP, return_as="generator")(delayed(model_task)(t, CFG, GRID_BY_PROTO) for t in MY_TASKS)
    pbar = tqdm(gen, total=len(MY_TASKS), desc="model tasks")
    for task, n_new, n_err in pbar:
        tot_new += n_new; tot_err += n_err
        pbar.set_postfix(new_jobs=tot_new, errors=tot_err)
    print(f"stage B done in {(time.time() - t0) / 60:.1f} min | new jobs {tot_new} | errors {tot_err}")
    if ROLE == "worker":
        print(f"\n>>> WORKER {WORKER_INDEX} XONG. Khi cả {N_WORKERS} worker đều xong, mở notebook FINAL và Run all.")

# %%
# ============================== CELL 11 — BASELINES (trivial + EHR lookup table) ==============================
def f2_at(y, yhat):
    y = y.astype(bool); yhat = yhat.astype(bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum(); d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0

base_rows = []
keys_by_table = {"rec": pd.util.hash_pandas_object(rec[EHR].fillna(-999), index=False).values,
                 "pat": pd.util.hash_pandas_object(pat[EHR].fillna(-999), index=False).values}
for proto in G_PROTOCOLS:
    tname = TABLE_OF[proto]; tab = rec if tname == "rec" else pat; keys = keys_by_table[tname]
    for seed in G_SEEDS:
        for k in G_FOLDS:
            sp = np.load(CACHE / "splits" / f"{proto}__s{seed}__f{k}.npz")
            for t in G_TARGETS:
                y = tab[t].values
                src = pd.Series(y[sp["idx_tr"]], index=keys[sp["idx_tr"]]); src = src[~src.index.duplicated()]
                def lk(idx, fb):
                    m = pd.Series(keys[idx]).map(src)
                    return np.where(m.notna(), m.fillna(fb), fb).astype(int), float(m.notna().mean())
                fb = max([0, 1], key=lambda b: f2_at(y[sp["idx_va"]], lk(sp["idx_va"], b)[0]))
                yhat, hit = lk(sp["idx_te"], fb)
                yte = y[sp["idx_te"]]; prev = yte.mean()
                base_rows.append({"protocol": proto, "seed": seed, "fold": k, "target": t,
                                  "trivial_f2": 5 * prev / (4 * prev + 1), "lookup_f2": f2_at(yte, yhat),
                                  "lookup_hit_rate": hit, "lookup_fallback": fb, "test_prev": prev})
baselines = pd.DataFrame(base_rows)
atomic_to_csv(baselines, OUT / "analysis" / "baselines.csv")
print(baselines.groupby("protocol")[["trivial_f2", "lookup_f2", "lookup_hit_rate"]].mean().round(3))

# %%
# ============================== CELL 12 — MERGE + COMPLETENESS ==============================
res_files = sorted(glob.glob(str(OUT / "results" / "*.csv")))
R = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True) if res_files else pd.DataFrame()
err_files = sorted(glob.glob(str(OUT / "errors" / "*.csv")))
ERR = pd.concat([pd.read_csv(f) for f in err_files], ignore_index=True) if err_files else pd.DataFrame()
if len(R):
    R = R.drop_duplicates("job_key", keep="last")
    if ROLE != "worker":
        atomic_to_csv(R, OUT / "results_all.csv")
expected = []
for t in ALL_TASKS:
    has_disc = bool(FEATSETS[t[3]]["bin"])
    for s, c in GRID_BY_PROTO[t[0]][has_disc]:
        expected.append(f"{t[0]}__s{t[1]}__f{t[2]}__{t[3]}__{t[4]}__{s}__{c}")
expected = pd.Series(expected)
done_keys = set(R.job_key) if len(R) else set()
print(f"jobs done {len(done_keys)} / expected {len(expected)} | errors logged {len(ERR)}")
missing = expected[~expected.isin(done_keys)]
if len(missing):
    def _worker_of(k):
        p, s_, f_, ft, t = k.split("__")[:5]
        return stable_hash("|".join(map(str, (p, int(s_[1:]), int(f_[1:]), ft, t)))) % N_WORKERS
    by_w = missing.map(_worker_of).value_counts().sort_index()
    print("missing examples:", missing.head(5).tolist())
    for w, n in by_w.items():
        print(f"  worker {w}: còn thiếu {n} job -> mở lại notebook worker {w} và Run all (tự chạy tiếp phần còn thiếu)")
    if ROLE == "worker":
        print("  (nếu các worker khác vẫn đang chạy thì thiếu job của họ là bình thường)")
if len(ERR):
    print(ERR[["job_key", "error"]].head(20).to_string())
if len(R):
    print("\nmean fit seconds by classifier:\n", R.groupby("classifier").fit_seconds.mean().round(2))
    if RUN_MODE == "preflight":
        cols = ["protocol", "feature_set", "target", "strategy", "classifier", "val_f2", "test_f2", "test_trivial_f2",
                "test_pr_auc", "syn_bad_row_frac", "fit_seconds"]
        print(R[cols].sort_values(["protocol", "target", "val_f2"], ascending=[True, True, False]).head(40).round(3).to_string())
        # ETA for the full grid on this machine
        def full_job_count(proto):
            clfs = P1_CLFS if proto == "P1_record" else CLFS_ALL
            strats = P1_STRATS if proto == "P1_record" else {"none", "class_weight", "ROS", "SMOTE", "SMOTE_naive", "SMOTENC_typeaware"}
            nat = 0 if proto == "P1_record" else len(NATIVE_ALL)
            n = 0
            for f, fs in FEATSETS.items():
                if not feat_allowed(proto, f):
                    continue
                base = ["none", "class_weight", "ROS", "SMOTE_naive", "SMOTENC_typeaware"] if fs["bin"] else ["none", "class_weight", "ROS", "SMOTE"]
                n += len([x for x in base if x in strats]) * len(clfs) + nat
            return n * len(TARGETS) * N_FOLDS * len(CV_SEEDS)
        sec = R.groupby("protocol").fit_seconds.mean()
        s_rec = sec.get("P2_grouped", sec.mean()); s_pat = sec.get("P3_patient", sec.mean())
        counts = {p: full_job_count(p) for p in PROTOCOLS}
        eta = sum(counts[p] * (s_pat if p == "P3_patient" else s_rec) for p in counts) / NP / 3600
        print(f"\nFULL grid ≈ {counts} jobs | ETA with {NP} cores ≈ {eta:.1f} h for ONE worker "
              f"(divide by the number of parallel Colab sessions)")

# %%
# ============================== CELL 13 — ANALYSIS (tables, figures, paper_numbers.json) ==============================
def run_analysis():
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    AN = OUT / "analysis"
    R = pd.read_csv(OUT / "results_all.csv")
    B = pd.read_csv(AN / "baselines.csv")
    numbers = {"data_hash": data_hash, "n_records": len(rec), "n_subjects": len(pat), "views": view_stats,
               "pos_subjects": data_audit["pos_subjects"].to_dict(), "pos_records": data_audit["pos_records"].to_dict()}

    # --- config selection by inner-validation F2 averaged over folds/seeds (never by test) ---
    KEY = ["protocol", "feature_set", "target"]
    cfg_agg = (R.groupby(KEY + ["strategy", "classifier"])
                 .agg(val_f2=("val_f2", "mean"), test_f2=("test_f2", "mean"), test_f2_sd=("test_f2", "std"),
                      test_pr_auc=("test_pr_auc", "mean"), test_roc_auc=("test_roc_auc", "mean"),
                      test_trivial=("test_trivial_f2", "mean"), test_prev=("test_prev", "mean"), n=("fold", "size"))
                 .reset_index())
    best = cfg_agg.sort_values("val_f2", ascending=False).groupby(KEY).head(1).sort_values(KEY)
    atomic_to_csv(best, AN / "selected_configs.csv")

    # --- Table A: protocol effect ---
    TA = best.groupby(["protocol", "feature_set"])[["test_f2", "test_pr_auc", "test_roc_auc", "test_trivial"]].mean()
    TA = TA.join(B.groupby("protocol")[["lookup_f2", "lookup_hit_rate"]].mean(), on="protocol")
    TA["prauc_lift"] = best.assign(l=best.test_pr_auc / best.test_prev).groupby(["protocol", "feature_set"]).l.mean()
    atomic_to_csv(TA.reset_index().round(4), AN / "tableA_protocol_effect.csv")
    print("TABLE A — macro over targets (selected by inner-val F2)\n", TA.round(3).to_string())
    numbers["tableA"] = TA.round(4).reset_index().to_dict("records")

    # --- Table B: imbalance strategies (P3 and P2), paired against 'none' with the same classifier ---
    TB_rows = []
    for proto in [p for p in ["P3_patient", "P2_grouped"] if p in R.protocol.unique()]:
        Rp = R[(R.protocol == proto) & (R.strategy != "native")]
        base = Rp[Rp.strategy == "none"].set_index(["feature_set", "target", "classifier", "seed", "fold"])
        for (feat, strat, clf), g in Rp[Rp.strategy != "none"].groupby(["feature_set", "strategy", "classifier"]):
            gg = g.set_index(["feature_set", "target", "classifier", "seed", "fold"])
            common = gg.index.intersection(base.index)
            if not len(common):
                continue
            d_f2 = gg.loc[common, "test_f2"] - base.loc[common, "test_f2"]
            d_pr = gg.loc[common, "test_pr_auc"] - base.loc[common, "test_pr_auc"]
            try:
                pval = wilcoxon(d_f2.values).pvalue if (d_f2 != 0).any() else 1.0
            except ValueError:
                pval = np.nan
            TB_rows.append({"protocol": proto, "feature_set": feat, "strategy": strat, "classifier": clf,
                            "dF2_mean": d_f2.mean(), "dF2_wins": int((d_f2 > 0).sum()), "dF2_losses": int((d_f2 < 0).sum()),
                            "dPRAUC_mean": d_pr.mean(), "wilcoxon_p": pval, "n_pairs": len(common)})
        nat = R[(R.protocol == proto) & (R.strategy == "native")]
        for (feat, clf), g in nat.groupby(["feature_set", "classifier"]):
            TB_rows.append({"protocol": proto, "feature_set": feat, "strategy": "native", "classifier": clf,
                            "macro_test_f2": g.groupby("target").test_f2.mean().mean()})
    TB = pd.DataFrame(TB_rows)
    def holm(p):
        p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
        for i, idx in enumerate(order):
            run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
        return adj
    if len(TB) and "wilcoxon_p" in TB and TB.wilcoxon_p.notna().any():
        m = TB.wilcoxon_p.notna()
        TB.loc[m, "holm_p"] = holm(TB.loc[m, "wilcoxon_p"].values)
    atomic_to_csv(TB.round(5), AN / "tableB_strategies_vs_none.csv")
    strat_macro = (R[R.protocol.isin(["P3_patient", "P2_grouped"])]
                   .groupby(["protocol", "feature_set", "strategy", "classifier", "target"]).test_f2.mean()
                   .groupby(["protocol", "feature_set", "strategy", "classifier"]).mean().unstack("classifier"))
    atomic_to_csv(strat_macro.reset_index().round(4), AN / "tableB2_macro_f2_strategy_by_classifier.csv")
    print("\nTABLE B — Δ test F2 vs 'none' (same classifier), paired over target×fold\n",
          TB[TB.strategy != "native"].groupby(["protocol", "feature_set", "strategy"]).dF2_mean.mean().unstack().round(4))

    # --- Sampler audit: invalid synthetic binary values ---
    aud = R[R.strategy.isin(["SMOTE_naive", "SMOTENC_typeaware"])].groupby(["protocol", "feature_set", "strategy"])[
        ["syn_bad_row_frac", "syn_bad_cell_frac"]].mean()
    atomic_to_csv(aud.reset_index().round(4), AN / "sampler_audit.csv")
    print("\nSampler audit (fraction of synthetic rows with a non-binary value in a binary/indicator column)\n", aud.round(3))
    numbers["sampler_audit"] = aud.round(4).reset_index().to_dict("records")

    # --- Table C: P1 seen vs unseen subjects ---
    if "P1_record" in R.protocol.unique():
        sel = best[best.protocol == "P1_record"][KEY + ["strategy", "classifier"]]
        P1 = R.merge(sel, on=KEY + ["strategy", "classifier"])
        TC = P1.groupby(["feature_set", "target"])[["test_f2", "seen_f2", "unseen_f2", "unseenclean_f2", "unseen_trivial_f2",
                                                   "unseen_pr_auc", "unseen_prev", "seen_n", "unseen_n"]].mean()
        TCm = TC.groupby("feature_set").mean()
        atomic_to_csv(TC.reset_index().round(4), AN / "tableC_P1_seen_unseen.csv")
        print("\nTABLE C — P1 record split: seen vs unseen test subjects (macro)\n", TCm.round(3).to_string())
        numbers["tableC_macro"] = TCm.round(4).reset_index().to_dict("records")

    # --- Late fusion (image + EHR), weight and threshold chosen on inner validation ---
    def load_pred(tag, strategy, clf):
        pp = OUT / "preds_packed" / f"{tag}.npz"
        if pp.exists():
            z = np.load(pp); key = f"{strategy}__{clf}"
            if f"{key}__p_te" in z.files:
                return {"p_va": z[f"{key}__p_va"], "p_te": z[f"{key}__p_te"]}
        f = OUT / "preds" / tag / f"{strategy}__{clf}.npz"          # legacy layout
        if f.exists():
            z = np.load(f); return {"p_va": z["p_va"], "p_te": z["p_te"]}
        return None
    def late_fusion(proto, img_feat, ehr_feat="ehr"):
        out = []
        lab = np.load(CACHE / f"labels_{TABLE_OF[proto]}.npz")
        for t in G_TARGETS:
            bi = best[(best.protocol == proto) & (best.feature_set == img_feat) & (best.target == t)]
            be = best[(best.protocol == proto) & (best.feature_set == ehr_feat) & (best.target == t)]
            if not len(bi) or not len(be):
                continue
            bi, be = bi.iloc[0], be.iloc[0]
            for seed in G_SEEDS:
                for k in G_FOLDS:
                    sp = np.load(CACHE / "splits" / f"{proto}__s{seed}__f{k}.npz")
                    a = load_pred(f"{proto}__s{seed}__f{k}__{img_feat}__{t}", bi.strategy, bi.classifier)
                    b = load_pred(f"{proto}__s{seed}__f{k}__{ehr_feat}__{t}", be.strategy, be.classifier)
                    if a is None or b is None:
                        continue
                    y = lab[t]; yva, yte = y[sp["idx_va"]], y[sp["idx_te"]]
                    bestw = max(np.round(np.arange(0, 1.01, 0.1), 1),
                                key=lambda w: metric_block(yva, w * a["p_va"] + (1 - w) * b["p_va"],
                                                           choose_threshold(yva, w * a["p_va"] + (1 - w) * b["p_va"]), "v")["v_f2"])
                    pva = bestw * a["p_va"] + (1 - bestw) * b["p_va"]; pte = bestw * a["p_te"] + (1 - bestw) * b["p_te"]
                    m = metric_block(yte, pte, choose_threshold(yva, pva), "test")
                    out.append({"protocol": proto, "image": img_feat, "target": t, "seed": seed, "fold": k, "w_img": bestw,
                                "test_f2": m["test_f2"], "test_pr_auc": m["test_pr_auc"], "test_roc_auc": m["test_roc_auc"]})
        return pd.DataFrame(out)
    lf_parts = [late_fusion(p, f) for p in R.protocol.unique() for f in FEATSETS
                if f.startswith("img_") and f in R.feature_set.unique()]
    LF = pd.concat(lf_parts, ignore_index=True) if lf_parts else pd.DataFrame()
    if len(LF):
        atomic_to_csv(LF.round(4), AN / "late_fusion.csv")
        LFm = LF.groupby(["protocol", "image", "target"])[["test_f2", "test_pr_auc", "test_roc_auc", "w_img"]].mean().groupby(["protocol", "image"]).mean()
        print("\nLATE FUSION (macro)\n", LFm.round(3))
        numbers["late_fusion"] = LFm.round(4).reset_index().to_dict("records")

    # --- Figures ---
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    order_p = [p for p in ["P1_record", "P2_grouped", "P3_patient"] if p in TA.index.get_level_values(0)]
    feats = [f for f in FEATSETS if f in TA.index.get_level_values(1)]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    w = 0.8 / max(1, len(feats))
    for i, f in enumerate(feats):
        vals = [TA.loc[(p, f), "test_f2"] if (p, f) in TA.index else np.nan for p in order_p]
        ax.bar(np.arange(len(order_p)) + i * w - 0.4 + w / 2, vals, w, label=f)
    for j, p in enumerate(order_p):
        ax.hlines(TA.loc[p].test_trivial.mean(), j - 0.45, j + 0.45, colors="k", linestyles="--", lw=0.8)
        ax.hlines(TA.loc[p].lookup_f2.mean(), j - 0.45, j + 0.45, colors="r", linestyles=":", lw=0.8)
    ax.set_xticks(range(len(order_p))); ax.set_xticklabels([p.split("_")[0] for p in order_p])
    ax.set_ylabel("Macro test F2"); ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title("Dashed: predict-all-positive · dotted: EHR lookup", fontsize=6)
    fig.tight_layout(); fig.savefig(AN / "fig1_protocol_effect.pdf"); fig.savefig(AN / "fig1_protocol_effect.png", dpi=200)

    if {"P1_record", "P3_patient"} <= set(best.protocol):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for i, t in enumerate(TARGETS):
            a1 = best[(best.protocol == "P1_record") & (best.feature_set == "ehr") & (best.target == t)].test_f2
            a3 = best[(best.protocol == "P3_patient") & (best.feature_set == "ehr") & (best.target == t)].test_f2
            tv = best[(best.protocol == "P3_patient") & (best.feature_set == "ehr") & (best.target == t)].test_trivial
            if len(a1) and len(a3):
                ax.plot([a3.iloc[0], a1.iloc[0]], [i, i], "-", color="0.6", lw=1)
                ax.plot(a1.iloc[0], i, "o", color="C3", ms=4, label="P1 record split" if i == 0 else None)
                ax.plot(a3.iloc[0], i, "o", color="C0", ms=4, label="P3 patient-level" if i == 0 else None)
                ax.plot(tv.iloc[0], i, "|", color="k", ms=6, label="all-positive" if i == 0 else None)
        ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([t.replace("_", " ") + (" *" if t == "scoliosis" else "") for t in TARGETS])
        ax.set_xlabel("Test F2 (EHR-only, selected config)"); ax.legend(fontsize=6, frameon=False, loc="center right")
        fig.tight_layout(); fig.savefig(AN / "fig2_targets_P1_vs_P3.pdf"); fig.savefig(AN / "fig2_targets_P1_vs_P3.png", dpi=200)

    if len(TB) and "P3_patient" in TB.protocol.values:
        H = TB[(TB.protocol == "P3_patient") & (TB.strategy != "native")]
        fsets = [f for f in ["ehr", "fusion_xrv"] + [f"fusion_{n}" for n in EMB_COLS] if f in H.feature_set.values]
        fig, axes = plt.subplots(1, len(fsets), figsize=(3.3 * len(fsets), 2.4), squeeze=False)
        fig.subplots_adjust(wspace=1.0)
        for ax, f in zip(axes[0], fsets):
            M = H[H.feature_set == f].pivot(index="strategy", columns="classifier", values="dF2_mean")
            im = ax.imshow(M.values, cmap="RdBu", vmin=-0.05, vmax=0.05)
            ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=45, fontsize=6)
            ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=6); ax.set_title(f, fontsize=7)
            for (i, j), v in np.ndenumerate(M.values):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=5)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="ΔF2 vs none")
        fig.savefig(AN / "fig3_strategy_delta_P3.pdf", bbox_inches="tight"); fig.savefig(AN / "fig3_strategy_delta_P3.png", dpi=200, bbox_inches="tight")

    numbers["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                           "sklearn": sklearn.__version__}
    for mod in ["imblearn", "lightgbm", "xgboost", "iterstrat"]:
        try:
            numbers["versions"][mod] = importlib.import_module(mod).__version__
        except Exception:
            numbers["versions"][mod] = "?"
    numbers["grid"] = {"protocols": G_PROTOCOLS, "seeds": G_SEEDS, "folds": G_FOLDS, "featsets": G_FEATS,
                       "classifiers": G_CLFS, "native": G_NATIVE, "emb_pca_dim": EMB_PCA_DIM}
    atomic_json(numbers, AN / "paper_numbers.json")
    print("\nSaved analysis to", AN)

if ROLE in ("final", "single"):
    run_analysis()
else:
    print("ROLE = worker: phần phân tích chạy trong notebook FINAL")

# %%
# ============================== CELL 14 — FLUSH TO DRIVE (bắt buộc trước khi đóng tab) ==============================
if IN_COLAB and ROLE in ("worker", "final"):
    print("Đang đẩy toàn bộ file lên Google Drive... ĐỪNG đóng tab cho tới khi thấy 'FLUSH XONG'.")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("FLUSH XONG — đã có thể đóng tab.")
