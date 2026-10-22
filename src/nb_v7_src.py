# %% [markdown]
# # STEP 7 (v7) — Ablation siêu tham số + nhóm ảnh "khớp lần nhập viện"
#
# **Mục đích** (trả lời 2 ý review):
# 1. **Siêu tham số.** Các mô hình cây trong bài dùng cấu hình mặc định (RF/ET `max_depth=None`, `min_samples_leaf=1`;
#    LightGBM `max_depth=-1`, `min_child_samples=20`). Notebook này chạy thêm các cấu hình có điều chuẩn
#    (`max_depth` ∈ {3, 5}, `min_samples_leaf` / `min_child_samples` ∈ {20, 50}) và LR với `C` ∈ {1, 0.1, 0.01},
#    chọn cấu hình trên inner-validation như bài. Câu hỏi: cây được điều chuẩn có đuổi kịp LR khi tách bệnh nhân không,
#    và lợi thế của cây dưới chia theo ảnh (P1) có co lại khi cây không còn học thuộc được bệnh nhân không?
# 2. **EHR lệch nhãn.** Với nhãn theo lần nhập viện (A, C), 11 biến EHR vẫn lấy từ một lần nhập viện "nguồn" của bệnh nhân.
#    Chỉ 36% ảnh (một ảnh/bệnh nhân) thuộc đúng lần nhập viện nguồn — ở nhóm này ảnh, EHR và nhãn khớp hoàn toàn.
#    Notebook (a) báo kết quả riêng cho nhóm khớp / không khớp trong tập test của P2/P4/P4-C và (b) chạy lại toàn bộ
#    chỉ trên nhóm khớp (`P2al_adm`, `P2al_cln`, cùng fold bệnh nhân P2).
#
# **Cách chạy** — tài khoản phụ (Colab Pro), Runtime **CPU, High-RAM** (càng nhiều core càng nhanh) → **Run all**.
# Không cần BigQuery, không cần sửa gì.
# - Cần trên Drive (đều đã có từ các lần trước): `data_aug/x_ray_and_features_data.csv`, `data_aug/embeddings/raddino`,
#   fold v2 (`leakage_aware_v2/.../manifests`), nhãn STEP5 (`MMMDA_runs/v4_2026-10-19_admlabels/v4_labels`).
# - Ước tính: ~25–40 phút với 48 core, ~2 giờ với 8 core. Rớt kết nối: Run all lại (tự bỏ qua phần đã xong).
# - Xong khi thấy `FLUSH XONG`. Tải **một file** `v7_for_claude.zip` về `D:\project_TTNT_Data_aug\analysis_v7\`.
#
# Mọi thư mục/file notebook tạo ra đều có tiền tố phiên bản `v7`.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE      = "full"                # "full" | "preflight" (1 fold, 2 nhãn, thư mục riêng)
VERSION       = "v7_2026-10-22"       # tag phiên bản: có trong tên mọi thư mục/file tạo ra

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
EXPECTED_HASH = "da85519958e4bccb"
RADDINO_DIR   = f"{DRIVE_ROOT}/embeddings/raddino"
V2_FOLDS      = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_subject_s42.csv"   # fold P2 (chỉ đọc)
V2_RECFOLDS   = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_record_s42.csv"    # fold P1 (chỉ đọc)
V4_ADMLABELS  = f"{DRIVE_ROOT}/MMMDA_runs/v4_2026-10-19_admlabels/v4_labels/v4_admission_labels.csv.gz"   # nhãn A/C (STEP5)
OUT_ROOT      = f"{DRIVE_ROOT}/MMMDA_runs"                               # -> MMMDA_runs/v7_2026-10-22_ablation/
LOCAL_CACHE   = "/content/mmmda_v7_cache"
N_PAR         = None                  # số process song song; None = tất cả core

# %%
# ============================== CELL 2 — INSTALL + MOUNT ==============================
import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("V7_LOCAL_TEST") == "1"           # chỉ dùng cho test offline
if LOCAL_TEST:
    CSV_PATH = os.environ["V7_CSV"]; OUT_ROOT = os.environ["V7_OUT"]; LOCAL_CACHE = os.environ["V7_CACHE"]
    RADDINO_DIR = os.environ["V7_EMB"]; V2_FOLDS = os.environ.get("V7_V2FOLDS", ""); V2_RECFOLDS = os.environ.get("V7_V2RECFOLDS", "")
    V4_ADMLABELS = os.environ["V7_ADMLABELS"]; RUN_MODE = os.environ.get("V7_MODE", RUN_MODE); N_PAR = int(os.environ.get("V7_NPAR", "2"))
if IN_COLAB:
    for mod, pkg in [("iterstrat", "iterative-stratification"), ("lightgbm", "lightgbm")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
print("cores:", os.cpu_count(), "| RUN_MODE:", RUN_MODE, "| VERSION:", VERSION)

# %%
# ============================== CELL 3 — IMPORTS + HELPERS ==============================
import json, time, hashlib, traceback, warnings, platform, glob, shutil, zipfile
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from lightgbm import LGBMClassifier
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

V = "v7"                                             # file/folder prefix
OUT = Path(OUT_ROOT) / (f"{VERSION}_ablation" + ("_preflight" if RUN_MODE == "preflight" else ""))
CACHE = Path(LOCAL_CACHE) / OUT.name
DIRS = {k: OUT / f"{V}_{k}" for k in ["manifests", "results", "errors", "analysis"]}
for d in list(DIRS.values()) + [CACHE / f"{V}_X", CACHE / f"{V}_splits"]:
    d.mkdir(parents=True, exist_ok=True)
print("OUT  :", OUT)

# %%
# ============================== CELL 4 — LOAD + LOCK DATA, LABELS, ALIGNED SUBSET ==============================
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV khác file đã benchmark — DỪNG"
TARGETS  = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
EHR_CONT = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc"]
EHR_BIN  = ["diabete", "hiv", "smoke", "cough"]
EHR      = EHR_CONT + EHR_BIN
df["dataset_position"] = np.arange(len(df))
S = df.groupby("subject_id")[TARGETS].first()
AL = pd.read_csv(V4_ADMLABELS)
assert (AL.dataset_position.values == np.arange(len(df))).all() and (AL.dicom.values == df.dicom.values).all()
# aligned = the radiograph belongs to the admission from which the released labels AND EHR values were copied
ALIGNED = (AL.label_source.values == "own")
print("aligned radiographs:", int(ALIGNED.sum()), f"({ALIGNED.mean():.1%}) | patients:", df.subject_id[ALIGNED].nunique())
assert df.subject_id[ALIGNED].is_unique, "mỗi bệnh nhân phải có đúng 1 ảnh khớp"
lab = {}
for t in TARGETS:
    lab[f"rel__{t}"] = df[t].values.astype(np.int8)
    lab[f"adm__{t}"] = AL[f"adm_{t}"].values.astype(np.int8)
    lab[f"cln__{t}"] = AL[f"cln_{t}"].values.astype(np.int8)
np.savez(CACHE / f"{V}_labels.npz", **lab)
agree = np.mean([(lab[f"adm__{t}"][ALIGNED] == lab[f"rel__{t}"][ALIGNED]).mean() for t in TARGETS])
print(f"on aligned radiographs, admission-aligned label == released label: {agree:.4%}")

# %%
# ============================== CELL 5 — TABLE, FEATURE SETS, SPLITS, TASKS ==============================
E = np.load(f"{RADDINO_DIR}/emb.npy").astype(np.float32)
ids = pd.read_csv(f"{RADDINO_DIR}/ids.csv")
posm = df.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
assert posm.notna().all(), f"{int(posm.isna().sum())} ảnh không có embedding RAD-DINO"
EMB = [f"raddino_{j}" for j in range(E.shape[1])]
rec = pd.concat([df[["dataset_position", "subject_id"] + EHR].reset_index(drop=True),
                 pd.DataFrame(E[posm.astype(int).values], columns=EMB)], axis=1)
FEATSETS = {"ehr": dict(cont=EHR_CONT, emb=[], bin=EHR_BIN),
            "img_raddino": dict(cont=[], emb=EMB, bin=[]),
            "fusion_raddino": dict(cont=EHR_CONT, emb=EMB, bin=EHR_BIN)}
EMB_PCA_DIM = 64
rec.to_pickle(CACHE / f"{V}_rec.pkl")

def rebuild_folds(Y, ids_, idcol):
    parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Y)), Y)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 * 100 + k).split(np.zeros(len(tr)), Y[tr]))
        role = np.empty(len(Y), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({idcol: ids_, "fold": k, "role": role}))
    return pd.concat(parts, ignore_index=True)
if V2_FOLDS and os.path.exists(V2_FOLDS):
    m2 = pd.read_csv(V2_FOLDS); print("P2 folds: đọc từ", V2_FOLDS)
else:
    print("P2 folds: không thấy file v2 -> dựng lại (cùng seed 42, cùng thuật toán)")
    m2 = rebuild_folds(S[TARGETS].values, S.index.values, "subject_id")
if V2_RECFOLDS and os.path.exists(V2_RECFOLDS):
    m1 = pd.read_csv(V2_RECFOLDS); print("P1 folds: đọc từ", V2_RECFOLDS)
else:
    print("P1 folds: không thấy file v2 -> dựng lại (cùng seed 42, cùng thuật toán)")
    m1 = rebuild_folds(df[TARGETS].values, df.dataset_position.values, "dataset_position")
for k in range(5):
    r1 = m1[m1.fold == k].set_index("dataset_position").role.reindex(rec.dataset_position).values
    r2 = rec.subject_id.map(m2[m2.fold == k].set_index("subject_id").role).values
    for sname, role in [("P1", r1), ("P2", r2), ("P2al", np.where(ALIGNED, r2, None))]:
        idx = {r: np.where(role == r)[0] for r in ("train", "val", "test")}
        if sname != "P2al":
            assert pd.notna(role).all()
        s_tr, s_va, s_te = [set(rec.subject_id.values[idx[r]]) for r in ("train", "val", "test")]
        seen_te = np.isin(rec.subject_id.values[idx["test"]], list(s_tr | s_va))
        if sname != "P1":
            assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te) and not seen_te.any()
        np.savez(CACHE / f"{V}_splits" / f"{sname}__f{k}.npz", idx_tr=idx["train"], idx_va=idx["val"], idx_te=idx["test"],
                 seen_te=seen_te, al_te=ALIGNED[idx["test"]])

PROTO = {"P1_rel":   dict(split="P1",   label="rel", feats=["ehr", "fusion_raddino"]),
         "P2_rel":   dict(split="P2",   label="rel", feats=["ehr", "fusion_raddino"]),
         "P4_adm":   dict(split="P2",   label="adm", feats=["ehr", "fusion_raddino"]),
         "P4C_cln":  dict(split="P2",   label="cln", feats=["ehr", "fusion_raddino"]),
         "P2al_adm": dict(split="P2al", label="adm", feats=["ehr", "img_raddino", "fusion_raddino"]),
         "P2al_cln": dict(split="P2al", label="cln", feats=["ehr", "img_raddino", "fusion_raddino"])}
FOLDS = list(range(5))
# classifier configurations (strategy = none); *_default = the configuration used in the paper
CONFIGS = {"LR_C1_default": ("LR", dict(C=1.0)), "LR_C0.1": ("LR", dict(C=0.1)), "LR_C0.01": ("LR", dict(C=0.01)),
           "RF_default": ("RF", dict(max_depth=None, min_samples_leaf=1)), "RF_depth5": ("RF", dict(max_depth=5, min_samples_leaf=1)),
           "RF_depth3": ("RF", dict(max_depth=3, min_samples_leaf=1)), "RF_leaf20": ("RF", dict(max_depth=None, min_samples_leaf=20)),
           "RF_leaf50": ("RF", dict(max_depth=None, min_samples_leaf=50)),
           "ET_default": ("ET", dict(max_depth=None, min_samples_leaf=1)), "ET_depth5": ("ET", dict(max_depth=5, min_samples_leaf=1)),
           "ET_depth3": ("ET", dict(max_depth=3, min_samples_leaf=1)), "ET_leaf20": ("ET", dict(max_depth=None, min_samples_leaf=20)),
           "ET_leaf50": ("ET", dict(max_depth=None, min_samples_leaf=50)),
           "LGBM_default": ("LGBM", dict(max_depth=-1, num_leaves=31, min_child_samples=20)),
           "LGBM_depth5": ("LGBM", dict(max_depth=5, num_leaves=31, min_child_samples=20)),
           "LGBM_depth3": ("LGBM", dict(max_depth=3, num_leaves=8, min_child_samples=20)),
           "LGBM_child50": ("LGBM", dict(max_depth=-1, num_leaves=31, min_child_samples=50))}
CFG_NAMES = list(CONFIGS)
if RUN_MODE == "preflight":
    FOLDS = [0]; CFG_NAMES = ["LR_C1_default", "RF_depth5", "ET_leaf20", "LGBM_depth3"]
if LOCAL_TEST and os.environ.get("V7_TINY") == "1":       # offline smoke test only — ignored on Colab
    FOLDS = [0]; CFG_NAMES = ["LR_C1_default", "LR_C0.1", "RF_depth3", "ET_leaf50", "LGBM_depth3", "LGBM_default"]

TASKS, skipped = [], []
for p, cfg in PROTO.items():
    tgts = TARGETS
    if RUN_MODE == "preflight" or (LOCAL_TEST and os.environ.get("V7_TINY") == "1"):
        tgts = ["pneumonia", "fracture"]
    for t in tgts:
        y = lab[f"{cfg['label']}__{t}"]
        for k in FOLDS:
            sp = np.load(CACHE / f"{V}_splits" / f"{cfg['split']}__f{k}.npz")
            ptr, pva, pte = int(y[sp["idx_tr"]].sum()), int(y[sp["idx_va"]].sum()), int(y[sp["idx_te"]].sum())
            if ptr < 10 or pva < 3 or pte < 3:
                skipped.append({"protocol": p, "fold": k, "target": t, "pos_tr": ptr, "pos_va": pva, "pos_te": pte}); continue
            for ft in cfg["feats"]:
                TASKS.append((p, k, ft, t))
atomic_to_csv(pd.DataFrame(skipped, columns=["protocol", "fold", "target", "pos_tr", "pos_va", "pos_te"]), DIRS["manifests"] / f"{V}_skipped.csv")
print(f"tasks {len(TASKS)} | configs per task {len(CFG_NAMES)} | fits {len(TASKS) * len(CFG_NAMES)} | skipped (quá ít ca dương) {len(skipped)}")
if skipped:
    print(pd.DataFrame(skipped).to_string(index=False))

# %%
# ============================== CELL 6 — MODEL / METRIC FUNCTIONS (same preprocessing and threshold rule as the paper) ==============================
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

def make_clf(cfg_name, seed):
    fam, hp = CONFIGS[cfg_name]
    if fam == "LR":   return LogisticRegression(C=hp["C"], max_iter=2000)
    if fam == "RF":   return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed, **hp)
    if fam == "ET":   return ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=seed, **hp)
    if fam == "LGBM": return LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, random_state=seed, verbose=-1, **hp)
    raise ValueError(cfg_name)

def choose_threshold(y, p):
    yb = y.astype(bool); P = p[None, :] >= GRID[:, None]
    tp = (P & yb).sum(1); fp = (P & ~yb).sum(1); fn = (~P & yb).sum(1)
    d = 5 * tp + 4 * fn + fp
    f2 = np.where(d > 0, 5 * tp / np.maximum(d, 1), 0.0)
    rec_ = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    pre_ = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return float(GRID[np.lexsort((np.abs(GRID - 0.5), -pre_, -rec_, -f2))[0]])

def metric_block(y, p, thr, prefix):
    y = np.asarray(y).astype(bool); pred = p >= thr
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = len(y) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0; rec_ = tp / (tp + fn) if tp + fn else 0.0
    d2 = 5 * tp + 4 * fn + fp; prev = float(y.mean()) if len(y) else np.nan
    out = {"n": len(y), "pos": int(y.sum()), "prev": prev, "f2": 5 * tp / d2 if d2 else 0.0, "recall": rec_, "precision": prec,
           "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
           "trivial_f2": 5 * prev / (4 * prev + 1) if len(y) else np.nan}
    return {f"{prefix}_{k}": v for k, v in out.items()}

# %%
# ============================== CELL 7 — STAGE A: PREPROCESSING CACHE ==============================
def prep_task(key, cache, featsets, pca_dim):
    sname, fold, feat = key
    out = Path(cache) / f"{V}_X" / f"{sname}__f{fold}__{feat}.npz"
    if out.exists():
        return key, "cached"
    tab = pd.read_pickle(Path(cache) / f"{V}_rec.pkl")
    sp = np.load(Path(cache) / f"{V}_splits" / f"{sname}__f{fold}.npz")
    ct = build_preprocessor(featsets[feat], pca_dim)
    Xtr = ct.fit_transform(tab.iloc[sp["idx_tr"]]).astype(np.float32)
    Xva = ct.transform(tab.iloc[sp["idx_va"]]).astype(np.float32)
    Xte = ct.transform(tab.iloc[sp["idx_te"]]).astype(np.float32)
    atomic_savez(out, Xtr=Xtr, Xva=Xva, Xte=Xte)
    return key, "built"

NP = N_PAR or os.cpu_count()
def task_done(task):
    p, k, ft, t = task
    return (DIRS["results"] / f"{V}_{p}__f{k}__{ft}__{t}.csv").exists()
TODO = [t for t in TASKS if not task_done(t)]
print(f"tasks đã xong từ lần chạy trước: {len(TASKS) - len(TODO)} | còn lại: {len(TODO)}")
prep_keys = sorted({(PROTO[p]["split"], k, ft) for p, k, ft, t in TODO})
t0 = time.time()
Parallel(n_jobs=min(NP, max(1, len(prep_keys))))(delayed(prep_task)(k, str(CACHE), FEATSETS, EMB_PCA_DIM) for k in tqdm(prep_keys, desc="preprocess"))
print(f"stage A: {len(prep_keys)} caches in {(time.time() - t0) / 60:.1f} min")
sec = {"ehr": 2.5, "img_raddino": 8, "fusion_raddino": 9}
scale = {"P1": 1.0, "P2": 1.0, "P2al": 0.4}
eta = sum(sec[ft] * scale[PROTO[p]["split"]] * len(CFG_NAMES) for p, _, ft, _ in TODO) / NP / 60
print(f"ETA ≈ {eta:.0f} phút với {NP} core")

# %%
# ============================== CELL 8 — STAGE B: MODELS (resume-safe: 1 file per task on Drive) ==============================
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
    Xtr, Xva, Xte = Xd["Xtr"], Xd["Xva"], Xd["Xte"]
    rows, errors = [], []
    for cname in cfg["configs"]:
        job_key = f"{tag}__{cname}"; t0 = time.time()
        try:
            clf = make_clf(cname, 42); clf.fit(Xtr, ytr)
            pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
            thr = choose_threshold(yva, pva)
            row = {"version": V, "data_hash": cfg["data_hash"], "job_key": job_key, "protocol": protocol, "split": pc["split"],
                   "label_type": pc["label"], "fold": fold, "feature_set": feat, "target": target, "config": cname,
                   "family": CONFIGS[cname][0], "is_default": cname.endswith("default"), "threshold": thr,
                   "n_train": len(ytr), "pos_train": int(ytr.sum())}
            row.update(metric_block(yva, pva, thr, "val")); row.update(metric_block(yte, pte, thr, "test"))
            if pc["split"] == "P1":
                s = sp["seen_te"]; row.update(metric_block(yte[s], pte[s], thr, "seen")); row.update(metric_block(yte[~s], pte[~s], thr, "unseen"))
            if pc["split"] == "P2":
                a = sp["al_te"]; row.update(metric_block(yte[a], pte[a], thr, "al")); row.update(metric_block(yte[~a], pte[~a], thr, "mis"))
            row["fit_seconds"] = time.time() - t0
            rows.append(row)
        except Exception as e:
            errors.append({"job_key": job_key, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
    if rows:
        atomic_to_csv(pd.DataFrame(rows), res_path)
    if errors:
        atomic_to_csv(pd.DataFrame(errors), Path(cfg["errors"]) / f"{V}_{tag}.csv")
    return task, len(rows), len(errors)

CFG = {"cache": str(CACHE), "results": str(DIRS["results"]), "errors": str(DIRS["errors"]), "proto": PROTO,
       "configs": CFG_NAMES, "data_hash": data_hash}
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
# ============================== CELL 9 — MERGE + QUICK SUMMARY ==============================
res_files = sorted(glob.glob(str(DIRS["results"] / f"{V}_*.csv")))
R = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True) if res_files else pd.DataFrame()
err_files = sorted(glob.glob(str(DIRS["errors"] / f"{V}_*.csv")))
ERR = pd.concat([pd.read_csv(f) for f in err_files], ignore_index=True) if err_files else pd.DataFrame()
n_expected = len(TASKS) * len(CFG_NAMES)
print(f"fits done {len(R)} / expected {n_expected} | errors {len(ERR)}")
if len(ERR):
    print(ERR[["job_key", "error"]].head(10).to_string())
atomic_to_csv(R, DIRS["analysis"] / f"{V}_results_all.csv.gz")
KEY = ["protocol", "feature_set", "target"]
agg = R.groupby(KEY + ["family", "config"]).agg(val_f2=("val_f2", "mean"), n_folds=("fold", "size")).reset_index()
def pick(sub):                     # configuration with the best mean inner-validation F2 over folds
    return sub.sort_values("val_f2", ascending=False).groupby(KEY).head(1)[KEY + ["config"]]
SEL = {"LR_default": agg[agg.config == "LR_C1_default"][KEY + ["config"]],
       "LR_tuned": pick(agg[agg.family == "LR"]),
       "trees_default": pick(agg[agg.config.isin(["RF_default", "ET_default", "LGBM_default"])]),
       "trees_tuned": pick(agg[agg.family.isin(["RF", "ET", "LGBM"])]),
       "overall_tuned": pick(agg)}
for fam in ["RF", "ET", "LGBM"]:
    SEL[f"{fam}_default"] = agg[agg.config == f"{fam}_default"][KEY + ["config"]]
    SEL[f"{fam}_tuned"] = pick(agg[agg.family == fam])
rows = []
for sname, sel in SEL.items():
    m = R.merge(sel, on=KEY + ["config"])
    fm = m.groupby(["protocol", "feature_set", "fold"]).agg(f2=("test_f2", "mean"), triv=("test_trivial_f2", "mean"),
                                                            au=("test_roc_auc", "mean"), pr=("test_pr_auc", "mean")).reset_index()
    tab = fm.groupby(["protocol", "feature_set"]).agg(f2=("f2", "mean"), f2_sd=("f2", "std"), triv=("triv", "mean"),
                                                      au=("au", "mean"), pr=("pr", "mean")).reset_index()
    tab["selection"] = sname; rows.append(tab)
    atomic_to_csv(sel, DIRS["analysis"] / f"{V}_selected__{sname}.csv")
SUM = pd.concat(rows, ignore_index=True)
SUM["delta"] = SUM.f2 - SUM.triv
atomic_to_csv(SUM.round(4), DIRS["analysis"] / f"{V}_summary_by_selection.csv")
print("\nMACRO test F2 (rows) by selection rule (columns)\n",
      SUM.pivot_table(index=["protocol", "feature_set"], columns="selection", values="f2").round(3).to_string())
cnt = SEL["overall_tuned"].merge(agg[KEY + ["config", "family"]], on=KEY + ["config"]).groupby(["protocol", "family"]).size().unstack(fill_value=0)
print("\nfamily chosen when every configuration competes\n", cnt.to_string())
P1sel = R[R.protocol == "P1_rel"]
if len(P1sel):
    su = []
    for sname in ["LR_default", "trees_default", "trees_tuned", "overall_tuned"]:
        m = P1sel.merge(SEL[sname], on=KEY + ["config"])
        su.append(m.groupby("feature_set")[["test_f2", "seen_f2", "unseen_f2", "unseen_trivial_f2"]].mean().assign(selection=sname))
    SU = pd.concat(su).reset_index(); atomic_to_csv(SU.round(4), DIRS["analysis"] / f"{V}_P1_seen_unseen.csv")
    print("\nP1 seen vs unseen\n", SU.round(3).to_string(index=False))
Ral = R[R.split == "P2"]
if len(Ral):
    m = Ral.merge(SEL["overall_tuned"], on=KEY + ["config"])
    AM = m.groupby(["protocol", "feature_set"])[["test_f2", "al_f2", "al_trivial_f2", "mis_f2", "mis_trivial_f2", "al_roc_auc", "mis_roc_auc"]].mean()
    atomic_to_csv(AM.reset_index().round(4), DIRS["analysis"] / f"{V}_aligned_vs_misaligned.csv")
    print("\naligned vs misaligned test radiographs (selected configs)\n", AM.round(3).to_string())
numbers = {"version": VERSION, "data_hash": data_hash, "n_fits": len(R), "n_expected": n_expected, "n_errors": len(ERR),
           "aligned_radiographs": int(ALIGNED.sum()), "aligned_label_agreement": float(agree), "skipped": skipped,
           "configs": {k: [v[0], v[1]] for k, v in CONFIGS.items()},
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
atomic_json(numbers, DIRS["analysis"] / f"{V}_paper_numbers.json")

# --- one small zip to send back (results + tables; no data, no predictions) ---
ZIP = OUT / f"{V}_for_claude.zip"
tmpz = str(ZIP) + ".tmp"
with zipfile.ZipFile(tmpz, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f"{OUT.name}/{DIRS[d].name}/{f.name}")
os.replace(tmpz, ZIP)
print("\nFILE CẦN TẢI:", ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")

# %%
# ============================== CELL 10 — FLUSH TO DRIVE (bắt buộc trước khi đóng tab) ==============================
if IN_COLAB:
    print("Đang đẩy toàn bộ file lên Google Drive... ĐỪNG đóng tab cho tới khi thấy 'FLUSH XONG'.")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("FLUSH XONG — tải file", ZIP.name, "trong thư mục", OUT)
