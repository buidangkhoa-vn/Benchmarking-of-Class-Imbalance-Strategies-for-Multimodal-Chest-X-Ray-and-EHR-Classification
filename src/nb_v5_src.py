# %% [markdown]
# # STEP 6 (v5): CheXpert report labels and time-aligned EHR history
#
# CheXpert labels (U-zeros) of the same 29,083 radiographs under record folds (P1) and patient folds (P2): inflation from
# the record split, strategy effects, and image vs EHR with labels defined per radiograph.
# Also P4_hist / P4C_hist: A and C labels with age at the radiograph's own admission and 8 prior-diagnosis flags (earlier
# admissions only), alone and fused with RAD-DINO, on the P2 folds.
#
# Run all (CPU, high RAM). Reads mimic-cxr-2.0.0-chexpert.csv.gz and mimic-cxr-2.0.0-metadata.csv.gz from the
# MIMIC-CXR-JPG folder on Drive and falls back to BigQuery if the CheXpert file is missing. Needs the STEP5 outputs
# (MMMDA_runs/v4_2026-10-19_admlabels) and the v3 tables (MMMDA_runs/v3_2026-10-19_P4P5/v3_bq).
# About 90-110 min with 8 cores; a rerun skips finished tasks. Output names start with v5.

# %%
# Cell 1: control panel
RUN_MODE      = "full"                # "full" | "preflight" (1 fold, 2 targets, separate folder)
VERSION       = "v5_2026-10-19"       # version tag: part of every folder/file name this notebook creates
MIMIC_JPG_ROOT = None                 # None = search automatically; or e.g. "/content/drive/MyDrive/MIMIC_CXR/mimic-cxr-jpg-2.1.0.physionet.org"
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # only used if the CheXpert file is not on Drive
BQ_AUTH       = "other_account"

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
EXPECTED_HASH = "da85519958e4bccb"
RADDINO_DIR   = f"{DRIVE_ROOT}/embeddings/raddino"
V2_FOLDS      = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_subject_s42.csv"   # P2 folds (read only)
V2_RECFOLDS   = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_record_s42.csv"    # P1 folds (read only)
OUT_ROOT      = f"{DRIVE_ROOT}/MMMDA_runs"                               # -> MMMDA_runs/v5_2026-10-19_chexpert/
V4_ADMLABELS  = f"{DRIVE_ROOT}/MMMDA_runs/v4_2026-10-19_admlabels/v4_labels/v4_admission_labels.csv.gz"   # A/C labels + prior-diagnosis flags (STEP5)
V3_BQ_DIR     = f"{DRIVE_ROOT}/MMMDA_runs/v3_2026-10-19_P4P5/v3_bq"     # admissions/patients (age at admission)
LOCAL_CACHE   = "/content/mmmda_v5_cache"
LOCAL_OUT     = "/content/mmmda_v5_out"

# 8 CheXpert observations (U-zeros: only 1.0 counts as positive; -1/blank -> 0)
CX_TARGETS = {"cx_pneumonia": "Pneumonia", "cx_lung_opacity": "Lung Opacity", "cx_effusion": "Pleural Effusion",
              "cx_cardiomegaly": "Cardiomegaly", "cx_edema": "Edema", "cx_pneumothorax": "Pneumothorax",
              "cx_fracture": "Fracture", "cx_lung_lesion": "Lung Lesion"}
N_PAR         = None                  # number of parallel processes; None = all cores

# %%
# Cell 2: install + mount
import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("V3_LOCAL_TEST") == "1"           # offline test only
if LOCAL_TEST:
    CSV_PATH = os.environ["V3_CSV"]; OUT_ROOT = os.environ["V3_OUT"]; LOCAL_CACHE = os.environ["V3_CACHE"]
    LOCAL_OUT = LOCAL_CACHE + "_out"; RADDINO_DIR = os.environ["V3_EMB"]; V2_FOLDS = os.environ.get("V3_V2FOLDS", "")
    V2_RECFOLDS = os.environ.get("V3_V2RECFOLDS", ""); V4_ADMLABELS = os.environ["V5_ADMLABELS"]; V3_BQ_DIR = os.environ["V5_BQDIR"]
    RUN_MODE = os.environ.get("V3_MODE", RUN_MODE); N_PAR = int(os.environ.get("V3_NPAR", "2"))
if IN_COLAB:
    for mod, pkg in [("iterstrat", "iterative-stratification"), ("imblearn", "imbalanced-learn"),
                     ("lightgbm", "lightgbm"), ("db_dtypes", "db-dtypes")]:
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    from google.colab import drive
    drive.mount("/content/drive")
print("cores:", os.cpu_count(), "| RUN_MODE:", RUN_MODE, "| VERSION:", VERSION)

# %%
# Cell 3: imports + helpers
import json, time, hashlib, traceback, warnings, platform, glob, re, shutil, zipfile
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
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

def stable_hash(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)

V = "v5"                                             # file/folder prefix
OUT = Path(OUT_ROOT) / (f"{VERSION}_chexpert" + ("_preflight" if RUN_MODE == "preflight" else ""))
CACHE = Path(LOCAL_CACHE) / OUT.name
LOUT = Path(LOCAL_OUT) / OUT.name
DIRS = {k: OUT / f"{V}_{k}" for k in ["bq", "labels", "manifests", "results", "preds", "errors", "analysis"]}
for d in list(DIRS.values()) + [CACHE / f"{V}_X", CACHE / f"{V}_splits", LOUT]:
    d.mkdir(parents=True, exist_ok=True)
print("OUT  :", OUT)

# %%
# Cell 4: load + lock data
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV differs from the benchmarked file - STOP"
TARGETS  = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
IMG_DEAD = ["2", "5", "6", "9", "11", "12", "13"]
IMG      = [str(i) for i in range(18) if str(i) not in IMG_DEAD]
EHR_CONT = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc"]
EHR_BIN  = ["diabete", "hiv", "smoke", "cough"]
EHR      = EHR_CONT + EHR_BIN
assert df.hadm_id.is_unique and df.dicom.is_unique
df["dataset_position"] = np.arange(len(df))
df["hadm_id"] = df.hadm_id.astype("int64"); df["subject_id"] = df.subject_id.astype("int64")
df["adm_date"] = pd.to_datetime(df.admittime).dt.normalize()
df["study_year"] = (df.StudyDate // 10000).astype(int)
S = df.groupby("subject_id")[TARGETS].first()          # subject-level labels (constant per subject, checked in v2)
SIDS = [int(s) for s in S.index]; HIDS = [int(h) for h in df.hadm_id]
print("subjects", len(SIDS), "| admissions with an image", len(HIDS))

# %%
# Cell 5: CheXpert labels
LABF = DIRS["labels"] / f"{V}_chexpert_cohort_studies.csv.gz"     # chexpert rows of the cohort's studies
MAPF = DIRS["labels"] / f"{V}_dicom_study_map.csv.gz"             # dicom -> study_id for the cohort's radiographs
CX_SRC = DIRS["labels"] / f"{V}_chexpert_source.json"
def find_mimic_root():
    name = "mimic-cxr-jpg-2.1.0.physionet.org"
    pats = [f"/content/drive/MyDrive/{name}", f"/content/drive/MyDrive/*/{name}", f"/content/drive/MyDrive/*/*/{name}",
            f"/content/drive/.shortcut-targets-by-id/*/{name}", f"/content/drive/.shortcut-targets-by-id/*/*/{name}",
            f"/content/drive/Shareddrives/*/{name}", f"/content/drive/Shareddrives/*/*/{name}"]
    for p in pats:
        for c in sorted(glob.glob(p)):
            if os.path.exists(f"{c}/mimic-cxr-2.0.0-metadata.csv.gz"):
                return c
    return None
CX_COLS = sorted(set(CX_TARGETS.values()))
if not (LABF.exists() and MAPF.exists()):
    if LOCAL_TEST:
        cxs = pd.read_csv(os.environ["V5_CHEXPERT"]); meta = pd.read_csv(os.environ["V5_META"]); src = {"source": "LOCAL_TEST"}
    else:
        root = MIMIC_JPG_ROOT or find_mimic_root()
        cand = sorted(glob.glob(f"{root}/*chexpert*.csv*")) if root else []
        if root and cand:
            print("MIMIC_JPG_ROOT:", root, "| chexpert:", Path(cand[0]).name)
            cxs = pd.read_csv(cand[0])
            meta = pd.read_csv(f"{root}/mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id", "study_id"])
            src = {"source": "drive", "root": root, "file": Path(cand[0]).name}
        else:
            print("CheXpert file not found on Drive -> reading it from BigQuery")
            SIDS_BQ = [int(s) for s in df.subject_id.unique()]
            from google.cloud import bigquery
            project = GCP_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            if not project:
                raise RuntimeError("GCP_PROJECT is empty in CELL 1 (Google Cloud project id of the account with PhysioNet access on BigQuery).")
            if BQ_AUTH == "other_account" and IN_COLAB:
                print("=" * 90)
                print("SIGN IN TO BIGQUERY WITH THE CREDENTIALED ACCOUNT (the account with PhysioNet access):")
                print(" 1) Open the link printed below -> choose the CREDENTIALED ACCOUNT -> Allow")
                print(" 2) Copy the verification code -> paste it into the input box below -> Enter")
                print("=" * 90)
                from IPython import get_ipython
                get_ipython().system("gcloud auth application-default login --no-launch-browser "
                                     "--scopes=https://www.googleapis.com/auth/cloud-platform")
                import google.auth
                adc = Path(os.environ.get("CLOUDSDK_CONFIG", str(Path.home() / ".config" / "gcloud"))) / "application_default_credentials.json"
                if not adc.exists():
                    raise RuntimeError("Sign-in with the credentialed account failed (no credential found). Rerun this cell; "
                                       "if it still fails, use notebook STEP4a with the credentialed account.")
                creds, _ = google.auth.load_credentials_from_file(str(adc), scopes=["https://www.googleapis.com/auth/cloud-platform"])
                client = bigquery.Client(project=project, credentials=creds)
            else:
                if IN_COLAB:
                    from google.colab import auth
                    auth.authenticate_user()
                client = bigquery.Client(project=project)
            def bq(sql, params=()):
                cfg = bigquery.QueryJobConfig(query_parameters=list(params))
                return client.query(sql, job_config=cfg).to_dataframe(create_bqstorage_client=False)   # REST download: works in BigQuery Sandbox
            try:
                print("BigQuery is using account:", bq("SELECT SESSION_USER() AS u").u.iloc[0])
            except Exception as e:
                raise RuntimeError(f"BigQuery query failed with project '{project}': {repr(e)[:300]}\n"
                                   "Check that GCP_PROJECT belongs to the account that just signed in and that the BigQuery API is enabled.")
            P_S = [bigquery.ArrayQueryParameter("s", "INT64", SIDS)]
            P_H = [bigquery.ArrayQueryParameter("h", "INT64", HIDS)]
            P_S = [bigquery.ArrayQueryParameter("s", "INT64", SIDS_BQ)]
            tabs = {}
            for ds in ["physionet-data.mimic_cxr_jpg", "physionet-data.mimic_cxr"]:
                try:
                    for tb in client.list_tables(ds):
                        tabs.setdefault(tb.table_id, f"{ds}.{tb.table_id}")
                except Exception as e:
                    print(ds, "cannot list tables:", repr(e)[:120])
            print("tables found:", sorted(tabs))
            tcx = next((v for k, v in tabs.items() if "chexpert" in k.lower()), None)
            tmeta = next((v for k, v in tabs.items() if "metadata" in k.lower()), None)
            if not tcx or not tmeta:
                raise RuntimeError(f"CheXpert/metadata tables not found on BigQuery: {sorted(tabs)}. Please report this message.")
            cxs = bq(f"SELECT * FROM `{tcx}` WHERE subject_id IN UNNEST(@s)", P_S)
            meta = bq(f"SELECT dicom_id, subject_id, study_id FROM `{tmeta}` WHERE subject_id IN UNNEST(@s)", P_S)
            cxs.columns = [{c.lower().replace(" ", "_"): c for c in CX_COLS}.get(c.lower(), c) for c in cxs.columns]
            src = {"source": "bigquery", "chexpert_table": tcx, "metadata_table": tmeta}
    meta = meta.rename(columns={"dicom_id": "dicom"})
    mp = df[["dicom", "subject_id"]].merge(meta[["dicom", "study_id"]], on="dicom", how="left")
    assert mp.study_id.notna().all(), f"{int(mp.study_id.isna().sum())} radiographs without study_id in the metadata"
    missing_cols = [c for c in CX_COLS if c not in cxs.columns]
    assert not missing_cols, f"CheXpert file is missing columns {missing_cols}"
    cxs = cxs[cxs.study_id.isin(set(mp.study_id))][["subject_id", "study_id"] + CX_COLS]
    atomic_to_csv(mp, MAPF); atomic_to_csv(cxs, LABF); atomic_json(src, CX_SRC)
mp = pd.read_csv(MAPF); cxs = pd.read_csv(LABF)
assert (mp.dicom.values == df.dicom.values).all()
lab_rows = mp[["study_id"]].merge(cxs.drop(columns="subject_id").drop_duplicates("study_id"), on="study_id", how="left")
no_report = int(lab_rows[CX_COLS].isna().all(1).sum())
img = df[["dataset_position", "subject_id", "hadm_id", "dicom", "StudyDate"]].copy()
cx_rows = []
for t, col in CX_TARGETS.items():
    v = lab_rows[col].values
    img[t] = (v == 1.0).astype(np.int8)                         # U-zeros
    cx_rows.append({"target": t, "chexpert": col, "pos_images": int((v == 1.0).sum()), "uncertain_images": int((v == -1.0).sum()),
                    "prev_%": round(100 * float((v == 1.0).mean()), 2)})
TARGETS_CX = list(CX_TARGETS)
# label varies within patient? (released labels never did)
multi = img.groupby("subject_id").size() > 1
var = img[img.subject_id.isin(multi[multi].index)].groupby("subject_id")[TARGETS_CX].nunique().gt(1)
for r in cx_rows:
    r["multi_img_patients_with_varying_label_%"] = round(100 * float(var[r["target"]].mean()), 1)
    r["pos_patients"] = int(img.groupby("subject_id")[r["target"]].max().sum())
# prior-study rule: positive if any EARLIER radiograph of the patient (by StudyDate) was positive
img = img.sort_values(["subject_id", "StudyDate", "dataset_position"])
for t in TARGETS_CX:
    prev_any = img.groupby("subject_id")[t].transform(lambda s: s.cummax().shift(1)).fillna(0)
    img[f"histcx_{t}"] = prev_any.astype(np.int8)
img = img.sort_values("dataset_position").reset_index(drop=True)
CXSUM = pd.DataFrame(cx_rows)
# ---- time-aligned EHR features for the repaired-label protocols (from STEP5 + v3 BigQuery tables) ----
AL = pd.read_csv(V4_ADMLABELS)
assert (AL.dataset_position.values == np.arange(len(df))).all() and (AL.dicom.values == df.dicom.values).all()
for t in TARGETS:
    for pref in ["adm", "cln", "hist", "histcln"]:
        img[f"{pref}_{t}"] = AL[f"{pref}_{t}"].values.astype(np.int8)
adm_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_admissions.csv.gz", parse_dates=["admittime"]).drop_duplicates("hadm_id").set_index("hadm_id")
pts_bq = pd.read_csv(f"{V3_BQ_DIR}/v3_bq_patients.csv.gz").drop_duplicates("subject_id").set_index("subject_id")
ay = adm_bq.admittime.reindex(img.hadm_id).dt.year.values.astype(float)
ay = np.where(np.isnan(ay), pd.to_datetime(df.admittime).dt.year.values, ay)
img["age_own"] = (pts_bq.anchor_age.reindex(img.subject_id).values + (ay - pts_bq.anchor_year.reindex(img.subject_id).values)).astype(float)
print("age at own admission differs from released age:", round(float((img.age_own != df.age).mean()), 4))
atomic_to_csv(CXSUM, DIRS["labels"] / f"{V}_chexpert_label_summary.csv")
print("radiographs without any CheXpert finding row (all labels -> 0):", no_report)
print(CXSUM.to_string(index=False))

# %%
# Cell 6: table, feature sets, splits
E = np.load(f"{RADDINO_DIR}/emb.npy").astype(np.float32)
ids = pd.read_csv(f"{RADDINO_DIR}/ids.csv")
posm = df.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
assert posm.notna().all(), f"{int(posm.isna().sum())} radiographs without a RAD-DINO embedding"
EMB = [f"raddino_{j}" for j in range(E.shape[1])]
HIST_A = [f"hist_{t}" for t in TARGETS]; HIST_C = [f"histcln_{t}" for t in TARGETS]
rec = pd.concat([df[["dataset_position", "subject_id"] + EHR + IMG].reset_index(drop=True),
                 img[["age_own"] + HIST_A + HIST_C].reset_index(drop=True),
                 pd.DataFrame(E[posm.astype(int).values], columns=EMB)], axis=1)
FEATSETS = {"ehr": dict(cont=EHR_CONT, emb=[], bin=EHR_BIN),
            "img_xrv": dict(cont=IMG, emb=[], bin=[]),
            "img_raddino": dict(cont=[], emb=EMB, bin=[]),
            "fusion_raddino": dict(cont=EHR_CONT, emb=EMB, bin=EHR_BIN),
            # time-aligned EHR: age at the radiograph's own admission + prior-diagnosis flags (earlier admissions only)
            "ehrhist_a": dict(cont=["age_own"], emb=[], bin=HIST_A),
            "fusionhist_a": dict(cont=["age_own"], emb=EMB, bin=HIST_A),
            "ehrhist_c": dict(cont=["age_own"], emb=[], bin=HIST_C),
            "fusionhist_c": dict(cont=["age_own"], emb=EMB, bin=HIST_C)}
EMB_PCA_DIM = 64
rec.to_pickle(CACHE / f"{V}_rec.pkl")
np.savez(CACHE / f"{V}_labels.npz", **{f"cx__{t}": img[t].values.astype(np.int8) for t in TARGETS_CX},
         **{f"histcx__{t}": img[f"histcx_{t}"].values.astype(np.int8) for t in TARGETS_CX},
         **{f"adm__{t}": img[f"adm_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"cln__{t}": img[f"cln_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"histadm__{t}": img[f"hist_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"histcln__{t}": img[f"histcln_{t}"].values.astype(np.int8) for t in TARGETS})
atomic_to_csv(img[["dataset_position", "subject_id", "dicom"] + TARGETS_CX], DIRS["labels"] / f"{V}_chexpert_image_labels.csv.gz")

if V2_FOLDS and os.path.exists(V2_FOLDS):
    m2 = pd.read_csv(V2_FOLDS); print("P2 folds: read from", V2_FOLDS)
else:
    print("P2 folds: v2 file not found -> rebuilt (same seed 42, same algorithm)")
    Ys = S[TARGETS].values; parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Ys)), Ys)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 * 100 + k).split(np.zeros(len(tr)), Ys[tr]))
        role = np.empty(len(Ys), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"subject_id": S.index.values, "fold": k, "role": role}))
    m2 = pd.concat(parts, ignore_index=True)
if V2_RECFOLDS and os.path.exists(V2_RECFOLDS):
    m1 = pd.read_csv(V2_RECFOLDS); print("P1 folds: read from", V2_RECFOLDS)
else:
    print("P1 folds: v2 file not found -> rebuilt (same seed 42, same algorithm)")
    Yr = df[TARGETS].values; parts = []
    for k, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(Yr)), Yr)):
        itr, iva = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 * 100 + k).split(np.zeros(len(tr)), Yr[tr]))
        role = np.empty(len(Yr), dtype=object); role[te] = "test"; role[tr[itr]] = "train"; role[tr[iva]] = "val"
        parts.append(pd.DataFrame({"dataset_position": df.dataset_position.values, "fold": k, "role": role}))
    m1 = pd.concat(parts, ignore_index=True)
SPLITS = {}
for k in range(5):
    SPLITS[("P1", k)] = m1[m1.fold == k].set_index("dataset_position").role.reindex(rec.dataset_position).values
    SPLITS[("P2", k)] = rec.subject_id.map(m2[m2.fold == k].set_index("subject_id").role).values
for (sname, k), role in SPLITS.items():
    assert pd.notna(role).all()
    i_tr, i_va, i_te = [np.where(role == r)[0] for r in ("train", "val", "test")]
    s_tr, s_va, s_te = [set(rec.subject_id.values[i]) for i in (i_tr, i_va, i_te)]
    seen_te = np.isin(rec.subject_id.values[i_te], list(s_tr | s_va))
    if sname != "P1":
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te) and not seen_te.any()
    np.savez(CACHE / f"{V}_splits" / f"{sname}__f{k}.npz", idx_tr=i_tr, idx_va=i_va, idx_te=i_te, seen_te=seen_te)

CX_FEATS = ["ehr", "img_xrv", "img_raddino", "fusion_raddino"]
PROTO = {"P2_chexpert": dict(split="P2", folds=list(range(5)), label="cx", feats=CX_FEATS, targets=TARGETS_CX),
         "P1_chexpert": dict(split="P1", folds=list(range(5)), label="cx", feats=CX_FEATS, targets=TARGETS_CX),
         "P4_hist":     dict(split="P2", folds=list(range(5)), label="adm", feats=["ehrhist_a", "fusionhist_a"], targets=TARGETS),
         "P4C_hist":    dict(split="P2", folds=list(range(5)), label="cln", feats=["ehrhist_c", "fusionhist_c"], targets=TARGETS)}
CLFS = ["LR", "LGBM", "ET"]
G_FEATS = list(FEATSETS)
if RUN_MODE == "preflight":
    for _p in PROTO: PROTO[_p]["folds"] = [0]
    G_FEATS = ["ehr", "img_raddino", "ehrhist_a", "ehrhist_c"]; CLFS = ["LR", "LGBM"]
if LOCAL_TEST and os.environ.get("V3_TINY") == "1":       # offline test only
    for _p in PROTO: PROTO[_p]["folds"] = [0]
    CLFS = ["LR", "LGBM"]; G_FEATS = ["ehr", "img_raddino", "ehrhist_a", "fusionhist_c"]

lab = np.load(CACHE / f"{V}_labels.npz")
TASKS, skipped = [], []
for p, cfg in PROTO.items():
    tgts = cfg["targets"]
    if RUN_MODE == "preflight" or (LOCAL_TEST and os.environ.get("V3_TINY") == "1"):
        tgts = [t for t in tgts if t in ("cx_pneumonia", "cx_effusion", "cx_fracture", "pneumonia", "fracture", "arthritis")][: 2 if RUN_MODE == "preflight" else 3]
    for t in tgts:
        y = lab[f"{cfg['label']}__{t}"]
        for k in cfg["folds"]:
            sp = np.load(CACHE / f"{V}_splits" / f"{cfg['split']}__f{k}.npz")
            ptr, pva, pte = int(y[sp["idx_tr"]].sum()), int(y[sp["idx_va"]].sum()), int(y[sp["idx_te"]].sum())
            if ptr < 10 or pva < 3 or pte < 3:
                skipped.append({"protocol": p, "fold": k, "target": t, "pos_tr": ptr, "pos_va": pva, "pos_te": pte}); continue
            for ft in [f for f in G_FEATS if f in cfg.get("feats", G_FEATS)]:
                TASKS.append((p, k, ft, t))
atomic_to_csv(pd.DataFrame(skipped, columns=["protocol", "fold", "target", "pos_tr", "pos_va", "pos_te"]), DIRS["manifests"] / f"{V}_skipped.csv")
print(f"tasks {len(TASKS)} | skipped (too few positives) {len(skipped)}")
if skipped:
    print(pd.DataFrame(skipped).to_string(index=False))

# %%
# Cell 7: model / sampler / metric functions
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

def make_clf(name, seed, cw):
    cwv = "balanced" if cw else None
    if name == "LR":   return LogisticRegression(C=1.0, max_iter=2000, class_weight=cwv)
    if name == "ET":   return ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=seed, class_weight=cwv)
    if name == "LGBM": return LGBMClassifier(n_estimators=300, learning_rate=0.05, n_jobs=1, random_state=seed, verbose=-1, class_weight=cwv)
    raise ValueError(name)

def resample(strategy, X, y, disc, seed):
    audit = {"syn_rows": 0, "syn_bad_row_frac": np.nan}
    if strategy in ("none", "class_weight"):
        return X, y, audit
    n_min = int(min(y.sum(), len(y) - y.sum())); k = max(1, min(5, n_min - 1))
    if strategy == "ROS":
        s = RandomOverSampler(random_state=seed)
    elif strategy == "SMOTE":
        s = SMOTE(k_neighbors=k, random_state=seed)
    elif strategy == "SMOTENC_typeaware":
        s = SMOTENC(categorical_features=list(disc), k_neighbors=k, random_state=seed)
    else:
        raise ValueError(strategy)
    Xr, yr = s.fit_resample(X, y)
    syn = Xr[len(X):]; audit["syn_rows"] = len(syn)
    if len(disc) and len(syn):
        bad = ~np.isin(syn[:, disc], [0.0, 1.0]); audit["syn_bad_row_frac"] = float(bad.any(1).mean())
        assert strategy != "SMOTENC_typeaware" or not bad.any()
    return Xr, yr, audit

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
    spec = tn / (tn + fp) if tn + fp else 0.0; d2 = 5 * tp + 4 * fn + fp
    prev = float(y.mean()) if len(y) else np.nan
    out = {"n": len(y), "pos": int(y.sum()), "prev": prev, "f2": 5 * tp / d2 if d2 else 0.0,
           "recall": rec_, "precision": prec, "bal_acc": (rec_ + spec) / 2,
           "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
           "pr_auc": average_precision_score(y, p) if y.sum() > 0 else np.nan,
           "trivial_f2": 5 * prev / (4 * prev + 1) if len(y) else np.nan, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {f"{prefix}_{k}": v for k, v in out.items()}

def strategies_for(has_disc):
    return ["none", "class_weight", "ROS", "SMOTENC_typeaware" if has_disc else "SMOTE"]

# %%
# Cell 8: stage A: preprocessing cache
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
    types, ncol = column_types(ct); assert Xtr.shape[1] == ncol
    disc = np.array(types["bin"] + types["ind"], dtype=int)
    if len(disc):
        assert np.isin(Xtr[:, disc], [0.0, 1.0]).all()
    atomic_savez(out, Xtr=Xtr, Xva=Xva, Xte=Xte, disc=disc)
    return key, "built"

NP = N_PAR or os.cpu_count()
def task_done(task):
    p, k, ft, t = task; tag = f"{p}__f{k}__{ft}__{t}"
    return (DIRS["results"] / f"{V}_{tag}.csv").exists() and (DIRS["preds"] / f"{V}_{tag}.npz").exists()
TODO = [t for t in TASKS if not task_done(t)]
print(f"tasks finished in earlier sessions: {len(TASKS) - len(TODO)} | remaining: {len(TODO)}")
prep_keys = sorted({(PROTO[p]["split"], k, ft) for p, k, ft, t in TODO})
t0 = time.time()
Parallel(n_jobs=min(NP, max(1, len(prep_keys))))(delayed(prep_task)(k, str(CACHE), FEATSETS, EMB_PCA_DIM) for k in tqdm(prep_keys, desc="preprocess"))
print(f"stage A: {len(prep_keys)} caches in {(time.time() - t0) / 60:.1f} min")
n_fits = sum(len(strategies_for(bool(FEATSETS[ft]["bin"]))) * len(CLFS) for _, _, ft, _ in TASKS)
sec = {"ehr": 4, "img_xrv": 5, "img_raddino": 12, "fusion_raddino": 15, "ehrhist_a": 2, "ehrhist_c": 2, "fusionhist_a": 13, "fusionhist_c": 13}
eta = sum(sec[ft] * len(strategies_for(bool(FEATSETS[ft]["bin"]))) * len(CLFS) for _, _, ft, _ in TODO) / NP / 60
print(f"model fits: {n_fits} | ETA ~ {eta:.0f} min with {NP} cores")

# %%
# Cell 9: stage B: models
def model_task(task, cfg):
    protocol, fold, feat, target = task
    tag = f"{protocol}__f{fold}__{feat}__{target}"
    res_path = Path(cfg["results"]) / f"{V}_{tag}.csv"; pack_path = Path(cfg["preds"]) / f"{V}_{tag}.npz"
    if res_path.exists() and pack_path.exists():
        try:
            np.load(pack_path).files; return task, 0, 0
        except Exception:
            pass
    pc = cfg["proto"][protocol]
    Xd = np.load(Path(cfg["cache"]) / f"{V}_X" / f"{pc['split']}__f{fold}__{feat}.npz")
    sp = np.load(Path(cfg["cache"]) / f"{V}_splits" / f"{pc['split']}__f{fold}.npz")
    y = np.load(Path(cfg["cache"]) / f"{V}_labels.npz")[f"{pc['label']}__{target}"].astype(int)
    ytr, yva, yte = y[sp["idx_tr"]], y[sp["idx_va"]], y[sp["idx_te"]]
    Xtr, Xva, Xte, disc = Xd["Xtr"], Xd["Xva"], Xd["Xte"], Xd["disc"]
    rows, arrays, errors = [], {}, []
    for strategy in strategies_for(bool(len(disc))):
        for clf_name in cfg["clfs"]:
            job_key = f"{tag}__{strategy}__{clf_name}"; t0 = time.time()
            try:
                Xs, ys, aud = resample(strategy, Xtr, ytr, disc, 42)
                clf = make_clf(clf_name, 42, strategy == "class_weight"); clf.fit(Xs, ys)
                pva = clf.predict_proba(Xva)[:, 1].astype(np.float32); pte = clf.predict_proba(Xte)[:, 1].astype(np.float32)
                thr = choose_threshold(yva, pva)
                row = {"version": V, "data_hash": cfg["data_hash"], "job_key": job_key, "protocol": protocol, "split": pc["split"],
                       "label_type": pc["label"], "fold": fold, "feature_set": feat, "target": target, "strategy": strategy,
                       "classifier": clf_name, "threshold": thr, "n_train": len(ytr), "pos_train": int(ytr.sum()),
                       "n_train_after": len(ys), **aud}
                row.update(metric_block(yva, pva, thr, "val")); row.update(metric_block(yte, pte, thr, "test"))
                if pc["split"] == "P1":
                    seen = sp["seen_te"]
                    row.update(metric_block(yte[seen], pte[seen], thr, "seen")); row.update(metric_block(yte[~seen], pte[~seen], thr, "unseen"))
                row["fit_seconds"] = time.time() - t0
                rows.append(row); arrays[f"{strategy}__{clf_name}__p_va"] = pva; arrays[f"{strategy}__{clf_name}__p_te"] = pte
            except Exception as e:
                errors.append({"job_key": job_key, "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
    if arrays:
        atomic_savez(pack_path, **arrays)
    if rows:
        atomic_to_csv(pd.DataFrame(rows), res_path)
    if errors:
        atomic_to_csv(pd.DataFrame(errors), Path(cfg["errors"]) / f"{V}_{tag}.csv")
    return task, len(rows), len(errors)

CFG = {"cache": str(CACHE), "results": str(DIRS["results"]), "preds": str(DIRS["preds"]), "errors": str(DIRS["errors"]),
       "proto": PROTO, "clfs": CLFS, "data_hash": data_hash}
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
# Cell 10: baselines
def f2_at(y, yhat):
    y = y.astype(bool); yhat = yhat.astype(bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum(); d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0
keys = pd.util.hash_pandas_object(rec[EHR].fillna(-999), index=False).values
base_rows = []
PKT = {(p, k, t) for (p, k, ft, t) in TASKS}
for p, cfg in PROTO.items():
    for t in sorted({t for (pp, k, t) in PKT if pp == p}):
        y = lab[f"{cfg['label']}__{t}"]; hist = lab[f"hist{cfg['label']}__{t}"]
        for k in [k for k in cfg["folds"] if (p, k, t) in PKT]:
            sp = np.load(CACHE / f"{V}_splits" / f"{cfg['split']}__f{k}.npz")
            src_all = pd.Series(y[sp["idx_tr"]], index=keys[sp["idx_tr"]])
            src = src_all[~src_all.index.duplicated()]                  # v2 definition: first matching training row
            src_any = src_all.groupby(level=0).max()                     # any matching training row positive
            def lk(idx, fb):
                m = pd.Series(keys[idx]).map(src)
                return np.where(m.notna(), m.fillna(fb), fb).astype(int), float(m.notna().mean())
            fb = max([0, 1], key=lambda b: f2_at(y[sp["idx_va"]], lk(sp["idx_va"], b)[0]))
            yhat, hit = lk(sp["idx_te"], fb); yte = y[sp["idx_te"]]; prev = yte.mean()
            m_any = pd.Series(keys[sp["idx_te"]]).map(src_any)
            yhat_any = np.where(m_any.notna(), m_any.fillna(fb), fb).astype(int)
            base_rows.append({"protocol": p, "fold": k, "target": t, "test_prev": prev, "trivial_f2": 5 * prev / (4 * prev + 1),
                              "lookup_f2": f2_at(yte, yhat), "lookup_any_f2": f2_at(yte, yhat_any), "lookup_hit_rate": hit,
                              "history_f2": f2_at(yte, hist[sp["idx_te"]]) ,
                              "history_pos_rate": float(hist[sp["idx_te"]].mean())})
BASE = pd.DataFrame(base_rows)
atomic_to_csv(BASE, DIRS["analysis"] / f"{V}_baselines.csv")
print(BASE.groupby("protocol")[["trivial_f2", "lookup_f2", "lookup_any_f2", "lookup_hit_rate", "history_f2"]].mean().round(3))

# %%
# Cell 11: merge + analysis
res_files = sorted(glob.glob(str(DIRS["results"] / f"{V}_*.csv")))
R = pd.concat([pd.read_csv(f) for f in res_files], ignore_index=True) if res_files else pd.DataFrame()
err_files = sorted(glob.glob(str(DIRS["errors"] / f"{V}_*.csv")))
ERR = pd.concat([pd.read_csv(f) for f in err_files], ignore_index=True) if err_files else pd.DataFrame()
print(f"fits done {len(R)} / expected {n_fits} | errors {len(ERR)}")
if len(ERR):
    print(ERR[["job_key", "error"]].head(10).to_string())
atomic_to_csv(R, DIRS["analysis"] / f"{V}_results_all.csv")
KEY = ["protocol", "feature_set", "target"]
agg = (R.groupby(KEY + ["strategy", "classifier"]).agg(val_f2=("val_f2", "mean"), test_f2=("test_f2", "mean"),
       test_pr_auc=("test_pr_auc", "mean"), test_roc_auc=("test_roc_auc", "mean"), test_trivial=("test_trivial_f2", "mean"),
       n_folds=("fold", "size")).reset_index())
best = agg.sort_values("val_f2", ascending=False).groupby(KEY).head(1).sort_values(KEY)
atomic_to_csv(best, DIRS["analysis"] / f"{V}_selected_configs.csv")
TA = best.groupby(["protocol", "feature_set"])[["test_f2", "test_pr_auc", "test_roc_auc", "test_trivial"]].mean()
TA = TA.join(BASE.groupby("protocol")[["lookup_f2", "lookup_any_f2", "history_f2"]].mean(), on="protocol")
TA["n_targets"] = best.groupby(["protocol", "feature_set"]).size()
atomic_to_csv(TA.reset_index().round(4), DIRS["analysis"] / f"{V}_tableA.csv")
if "P1_chexpert" in set(R.protocol):
    sel1 = best[best.protocol == "P1_chexpert"][KEY + ["strategy", "classifier"]]
    P1r = R.merge(sel1, on=KEY + ["strategy", "classifier"])
    TC = P1r.groupby(["feature_set", "target"])[["test_f2", "seen_f2", "unseen_f2", "unseen_trivial_f2", "seen_n", "unseen_n"]].mean()
    atomic_to_csv(TC.reset_index().round(4), DIRS["analysis"] / f"{V}_tableC_P1cx_seen_unseen.csv")
    print("\nP1_chexpert - seen vs unseen patients (macro)\n", TC.groupby("feature_set").mean().round(3).to_string())
print("\nMACRO (configuration selected on inner-validation F2)\n", TA.round(3).to_string())

# --- imbalance strategies vs none (paired, same classifier/target/fold) ---
srows = []
for p in sorted(set(R.protocol)):
    Rp = R[R.protocol == p]; b0 = Rp[Rp.strategy == "none"].set_index(["feature_set", "target", "classifier", "fold"])
    for (fs, st), d in Rp[Rp.strategy != "none"].groupby(["feature_set", "strategy"]):
        dd = d.set_index(["feature_set", "target", "classifier", "fold"]); c = dd.index.intersection(b0.index)
        diff = dd.loc[c, "test_f2"] - b0.loc[c, "test_f2"]
        srows.append({"protocol": p, "feature_set": fs, "strategy": st, "dF2_mean": diff.mean(), "n": len(diff)})
STRAT = pd.DataFrame(srows)
atomic_to_csv(STRAT.round(5), DIRS["analysis"] / f"{V}_strategies_vs_none.csv")
print("\nΔF2 vs no augmentation\n", STRAT.pivot_table(index=["protocol", "feature_set"], columns="strategy", values="dF2_mean").round(4).to_string())

numbers = {"version": VERSION, "data_hash": data_hash, "chexpert_source": json.load(open(CX_SRC)),
           "label_summary": CXSUM.to_dict("records"), "radiographs_without_report_labels": no_report,
           "tableA": TA.round(4).reset_index().to_dict("records"), "n_fits": len(R), "n_errors": len(ERR), "skipped": skipped,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
atomic_json(numbers, DIRS["analysis"] / f"{V}_paper_numbers.json")

# summary zip (tables and admission labels, no predictions)
ZIP = OUT / f"{V}_summary.zip"
tmpz = str(ZIP) + ".tmp"
with zipfile.ZipFile(tmpz, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["labels", "manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp") and "cohort_studies" not in f.name and "dicom_study_map" not in f.name:
                z.write(f, f"{OUT.name}/{DIRS[d].name}/{f.name}")
os.replace(tmpz, ZIP)
print("\nFILE TO DOWNLOAD:", ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")

# %%
# Cell 12: flush to Drive
if IN_COLAB:
    print("Flushing all files to Google Drive... do NOT close the session until 'FLUSH DONE' appears.")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("FLUSH DONE - download", ZIP.name, "from", OUT)
