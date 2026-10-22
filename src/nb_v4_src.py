# %% [markdown]
# # STEP 5 (v4) — Admission-aligned labels (P1/P4/P5) — supersedes STEP4 v3
#
# **Why v4.** The v3 results showed that the labels in `x_ray_and_features_data.csv` were created by keyword matching on the ICD titles
# of **one** admission, after which the labels + EHR of **each patient's last row (in file order)** were copied to all radiographs of
# that patient. The reconstructed rule matches **10,477/10,479** patients. Actual keywords: `pneumonia`, `tuberculosis`, `bronchitis`,
# `arthritis`, `fracture`, `malignant neoplasm` (→ "lung_cancer" is in fact *any* cancer), `scoliosis`, and titles ending in
# `infection` (→ "lung_infection").
#
# This notebook (earlier results are left untouched):
# 1. Reuses the four BigQuery tables extracted in v3 (no new sign-in if the v3 folder is still on Drive).
# 2. Re-checks the label rule and measures label misalignment (share of radiographs carrying another admission's label, earlier/later).
# 3. Assigns **admission-aligned labels** = diagnoses of **the radiograph's own admission** for all 8 targets, then runs:
#    - `P1_admission`: record split as in the original design (v2 P1 folds) — how much leakage remains with aligned labels;
#    - `P4_admission`: patient split (v2 P2 folds);
#    - `P5_temporal` / `P5_temporal_adm`: calendar-time split (as in v3), released / admission-aligned labels;
#    - `P4_admission_clean`: as P4 but without non-diagnostic codes (history, screening, vaccination, "without ...")
#      and with correct definitions of "lung cancer" (lung/bronchus malignancy) and "lung infection" (lung/respiratory infection).
#
# **How to run** — Colab (Pro), Runtime **CPU, High-RAM** → **Run all**. Nothing to edit.
# - If the v3 tables are not found, the notebook asks for a BigQuery sign-in as in STEP4 (link → choose <your PhysioNet-credentialed Google account> → paste the code).
# - Expected time ~90–110 min with 8 cores. If the session drops: Run all again (finished tasks are skipped).
# - Done when `FLUSH DONE` appears. The summary archive `v4_for_claude.zip` is written to the output folder (path printed at the end).
#
# Every folder/file this notebook creates carries the version prefix `v4`.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE      = "full"                # "full" | "preflight" (1 fold, 2 targets, separate folder)
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # BigQuery project of the account with PhysioNet access (<your PhysioNet-credentialed Google account>)
BQ_AUTH       = "other_account"       # "other_account": sign in to BigQuery with a different account than the one running Colab
                                      #                  (paste a verification code; choose the account with PhysioNet access)
                                      # "same_account" : the account running Colab is the account with PhysioNet access
VERSION       = "v4_2026-10-19"       # version tag: part of every folder/file name this notebook creates

DRIVE_ROOT    = "/content/drive/MyDrive/data_aug"
CSV_PATH      = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
EXPECTED_HASH = "da85519958e4bccb"
RADDINO_DIR   = f"{DRIVE_ROOT}/embeddings/raddino"                       # output of STEP1 (NB2)
V2_FOLDS      = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_subject_s42.csv"   # P2 folds (read only)
V2_RECFOLDS   = f"{DRIVE_ROOT}/leakage_aware_v2/leakage_aware_v2/manifests/folds_record_s42.csv"    # P1 folds (read only)
PREV_BQ_DIRS  = [f"{DRIVE_ROOT}/MMMDA_runs/v3_2026-10-19_P4P5/v3_bq"]    # BigQuery tables extracted in v3 (reused)
OUT_ROOT      = f"{DRIVE_ROOT}/MMMDA_runs"                               # -> MMMDA_runs/v4_2026-10-19_admlabels/
LOCAL_CACHE   = "/content/mmmda_v4_cache"
LOCAL_OUT     = "/content/mmmda_v4_out"

# BigQuery datasets tried in turn (the one matching the cohort's hadm_id + admission dates is selected automatically)
BQ_CANDIDATES = [("physionet-data.mimiciv_3_1_hosp", "physionet-data.mimiciv_3_1_hosp"),
                 ("physionet-data.mimiciv_2_2_hosp", "physionet-data.mimiciv_2_2_hosp"),
                 ("physionet-data.mimiciv_hosp",     "physionet-data.mimiciv_hosp"),
                 ("physionet-data.mimiciv_3_0_hosp", "physionet-data.mimiciv_3_0_hosp"),
                 ("physionet-data.mimic_hosp",       "physionet-data.mimic_core")]    # (hosp: diagnoses, core: admissions/patients)

N_PAR         = None                  # number of parallel processes; None = all cores

# %%
# ============================== CELL 2 — INSTALL + MOUNT ==============================
import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("V3_LOCAL_TEST") == "1"           # offline test only
if LOCAL_TEST:
    CSV_PATH = os.environ["V3_CSV"]; OUT_ROOT = os.environ["V3_OUT"]; LOCAL_CACHE = os.environ["V3_CACHE"]
    LOCAL_OUT = LOCAL_CACHE + "_out"; RADDINO_DIR = os.environ["V3_EMB"]; V2_FOLDS = os.environ.get("V3_V2FOLDS", "")
    V2_RECFOLDS = os.environ.get("V3_V2RECFOLDS", ""); PREV_BQ_DIRS = [x for x in os.environ.get("V3_PREVBQ", "").split(",") if x]
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
# ============================== CELL 3 — IMPORTS + HELPERS ==============================
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

V = "v4"                                             # file/folder prefix
OUT = Path(OUT_ROOT) / (f"{VERSION}_admlabels" + ("_preflight" if RUN_MODE == "preflight" else ""))
CACHE = Path(LOCAL_CACHE) / OUT.name
LOUT = Path(LOCAL_OUT) / OUT.name
DIRS = {k: OUT / f"{V}_{k}" for k in ["bq", "labels", "manifests", "results", "preds", "errors", "analysis"]}
for d in list(DIRS.values()) + [CACHE / f"{V}_X", CACHE / f"{V}_splits", LOUT]:
    d.mkdir(parents=True, exist_ok=True)
print("OUT  :", OUT)

# %%
# ============================== CELL 4 — LOAD + LOCK DATA ==============================
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV differs from the benchmarked file — STOP"
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
# ============================== CELL 5 — BIGQUERY EXTRACTION (cached on Drive) ==============================
BQF = {k: DIRS["bq"] / f"{V}_bq_{k}.csv.gz" for k in ["admissions", "patients", "diagnoses", "icd_titles"]}
BQ_SRC = DIRS["bq"] / f"{V}_bq_source.json"
if LOCAL_TEST and os.environ.get("V3_BQ_DIR"):
    for k, f in BQF.items():
        if not f.exists():
            shutil.copy(Path(os.environ["V3_BQ_DIR"]) / f.name, f)
    if not BQ_SRC.exists():
        atomic_json({"hosp": "LOCAL_TEST", "core": "LOCAL_TEST"}, BQ_SRC)

# (0) reuse the tables extracted in the previous run (v3) if they are still on Drive
if not all(f.exists() for f in BQF.values()):
    for pdir in PREV_BQ_DIRS:
        pdir = Path(pdir)
        srcs = {k: sorted(pdir.glob(f"*_bq_{k}.csv.gz")) for k in BQF}
        if all(srcs.values()):
            for k, f in BQF.items():
                shutil.copy(srcs[k][0], str(f) + ".tmp"); os.replace(str(f) + ".tmp", f)
            js = sorted(pdir.glob("*_bq_source.json"))
            info = json.load(open(js[0])) if js else {}
            atomic_json({**info, "copied_from": str(pdir)}, BQ_SRC)
            print("BigQuery: reusing the tables extracted in", pdir); break

# (a) tables already extracted with STEP4a (credentialed account) -> only unzip
if not all(f.exists() for f in BQF.values()):
    for zp in [Path(OUT_ROOT) / f"{V}_bq_extract.zip", OUT / f"{V}_bq_extract.zip", Path(OUT_ROOT).parent / f"{V}_bq_extract.zip",
               Path("/content/drive/MyDrive") / f"{V}_bq_extract.zip", Path("/content") / f"{V}_bq_extract.zip"]:
        if zp.exists():
            with zipfile.ZipFile(zp) as z:
                for m in z.namelist():
                    name = Path(m).name
                    if name.startswith(f"{V}_bq_") and not m.endswith("/"):
                        tmp = DIRS["bq"] / (name + ".tmp")
                        with z.open(m) as fi, open(tmp, "wb") as fo: shutil.copyfileobj(fi, fo)
                        os.replace(tmp, DIRS["bq"] / name)
            print("BigQuery: using tables extracted beforehand from", zp); break

# (b) query BigQuery directly
if not all(f.exists() for f in BQF.values()):
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
    # --- discover extra candidate datasets (may be forbidden; that is fine) ---
    cands = list(BQ_CANDIDATES)
    try:
        for ds in client.list_datasets(project="physionet-data"):
            n = ds.dataset_id
            if re.match(r"mimiciv_.*hosp$", n) and ("physionet-data." + n, "physionet-data." + n) not in cands:
                cands.insert(0, ("physionet-data." + n, "physionet-data." + n))
    except Exception as e:
        print("list_datasets not permitted (skipped):", repr(e)[:120])
    # --- pick the dataset whose admissions match our hadm_id + admission date ---
    checks, errs = [], []
    our = df.set_index("hadm_id")
    study = pd.to_datetime(our.StudyDate.astype(str), format="%Y%m%d")
    for hosp, core in cands:
        try:
            a = bq(f"SELECT hadm_id, DATE(admittime) AS adate, DATE(dischtime) AS ddate FROM `{core}.admissions` WHERE hadm_id IN UNNEST(@h)", P_H)
            a = a.drop_duplicates("hadm_id").set_index("hadm_id").reindex(our.index)
            ad, dd = pd.to_datetime(a.adate), pd.to_datetime(a.ddate)
            found = float(ad.notna().mean())
            date_ok = float((ad == our.adm_date).mean())
            in_adm = float(((study >= ad) & (study <= dd)).mean())       # MIMIC-CXR shares the date shift with MIMIC-IV
            checks.append({"hosp": hosp, "core": core, "found": found, "date_match": date_ok, "study_in_admission": in_adm})
            print(f"{core}: found {found:.4f} | admission-date match {date_ok:.4f} | study inside admission {in_adm:.4f}")
            if found > 0.999 and date_ok > 0.99:
                break
        except Exception as e:
            errs.append(f"{core}: {repr(e)[:200]}"); print(f"{core}: not usable ({repr(e)[:150]})")
    ok = [c for c in checks if c["found"] > 0.98 and max(c["date_match"], c["study_in_admission"]) > 0.95]
    if not ok:
        hint = ("\nIf the error is 403/Access Denied: the signed-in Google account has no access to physionet-data on BigQuery "
                "(physionet.org → MIMIC-IV page → 'Request access using Google BigQuery'), or GCP_PROJECT is wrong.")
        raise RuntimeError(f"No BigQuery dataset matches the cohort.\nchecks={checks}\nerrors={errs}{hint}\nPlease report this full message.")
    src = max(ok, key=lambda c: (c["found"], c["date_match"], c["study_in_admission"]))
    hosp, core = src["hosp"], src["core"]
    print("==> using", hosp, "/", core)
    t0 = time.time()
    adm = bq(f"SELECT subject_id, hadm_id, admittime, dischtime, admission_type FROM `{core}.admissions` WHERE subject_id IN UNNEST(@s)", P_S)
    pts = bq(f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group FROM `{core}.patients` WHERE subject_id IN UNNEST(@s)", P_S)
    dxs = bq(f"SELECT subject_id, hadm_id, seq_num, icd_code, icd_version FROM `{hosp}.diagnoses_icd` WHERE subject_id IN UNNEST(@s)", P_S)
    tit = bq(f"SELECT icd_code, icd_version, long_title FROM `{hosp}.d_icd_diagnoses`")
    for k, d in [("admissions", adm), ("patients", pts), ("diagnoses", dxs), ("icd_titles", tit)]:
        atomic_to_csv(d, BQF[k]); print(k, d.shape)
    atomic_json({**src, "checks": checks, "project": project, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "seconds": round(time.time() - t0, 1)}, BQ_SRC)
else:
    print("BigQuery: using the copy saved on Drive", json.load(open(BQ_SRC)).get("hosp"))

adm = pd.read_csv(BQF["admissions"], parse_dates=["admittime", "dischtime"])
pts = pd.read_csv(BQF["patients"])
dxs = pd.read_csv(BQF["diagnoses"], dtype={"icd_code": str})
tit = pd.read_csv(BQF["icd_titles"], dtype={"icd_code": str})
dxs["icd_code"] = dxs.icd_code.str.strip(); tit["icd_code"] = tit.icd_code.str.strip()
assert set(SIDS) <= set(pts.subject_id), "patients missing from the patients table"
miss_h = set(HIDS) - set(adm.hadm_id)
assert len(miss_h) <= 0.002 * len(HIDS), f"{len(miss_h)} cohort hadm_id values not found in admissions"
print(f"admissions {len(adm)} (cohort subjects, all admissions) | diagnoses {len(dxs)} | titles {len(tit)}")

# %%
# ============================== CELL 6 — EXACT LABEL RULE (reconstructed from v3 tables) ==============================
# Released labels = case-insensitive keyword match on ICD long titles of ONE admission; that admission is the subject's LAST
# row in file order, and its labels (and EHR) were copied to every radiograph of the subject.
LABEL_RULE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis",
              "arthritis": r"arthritis", "fracture": r"fracture", "lung_cancer": r"malignant neoplasm",
              "scoliosis": r"scoliosis", "lung_infection": r"infection\s*$"}
dxt = dxs.merge(tit.drop_duplicates(["icd_version", "icd_code"]), on=["icd_code", "icd_version"], how="left")
ttl = dxt.long_title.fillna("").str.strip()
for t, k in LABEL_RULE.items():
    dxt[t] = ttl.str.contains(k, case=False, regex=True)
ADMLAB = dxt.groupby("hadm_id")[TARGETS].any().astype(np.int8)          # label vector of every admission (all admissions)
def adm_vec(hadms):
    return ADMLAB.reindex(list(hadms)).fillna(0).astype(np.int8).values
last_h = df.groupby("subject_id").hadm_id.last()                          # df is in file order
first_h = df.groupby("subject_id").hadm_id.first()
Ssub = S.loc[last_h.index, TARGETS].values
match_last = (adm_vec(last_h.values) == Ssub).all(1)
match_first = (adm_vec(first_h.values) == Ssub).all(1)
coh_any = pd.DataFrame(adm_vec(df.hadm_id), columns=TARGETS).assign(subject_id=df.subject_id.values).groupby("subject_id").max()
match_any = (coh_any.loc[last_h.index].values == Ssub).all(1)
pt = pts.drop_duplicates("subject_id").set_index("subject_id")
admt = adm.drop_duplicates("hadm_id").set_index("hadm_id").admittime
def age_at(sids, hadms):
    return pt.anchor_age.reindex(sids).values + (pd.to_datetime(admt.reindex(hadms).values).year - pt.anchor_year.reindex(sids).values)
age_src_ok = float((age_at(last_h.index, last_h.values) == df.groupby("subject_id").age.first().loc[last_h.index].values).mean())
RECON = pd.DataFrame([{"target": t, "rule": LABEL_RULE[t],
                       "agree_last_row_adm": float((adm_vec(last_h.values)[:, j] == Ssub[:, j]).mean()),
                       "agree_first_row_adm": float((adm_vec(first_h.values)[:, j] == Ssub[:, j]).mean()),
                       "agree_any_cohort_adm": float((coh_any.loc[last_h.index].values[:, j] == Ssub[:, j]).mean())}
                      for j, t in enumerate(TARGETS)])
recon_summary = {"subjects": len(last_h), "exact_all8_last_row": int(match_last.sum()), "exact_all8_first_row": int(match_first.sum()),
                 "exact_all8_any_cohort_adm": int(match_any.sum()), "age_equals_age_at_last_row_admission": age_src_ok}
atomic_to_csv(RECON.round(5), DIRS["labels"] / f"{V}_label_reconstruction.csv")
atomic_json(recon_summary, DIRS["labels"] / f"{V}_label_reconstruction_summary.json")
print(RECON.round(4).to_string(index=False)); print(recon_summary)
# titles matched by each rule (cohort subjects): shows what the label names really mean
trows = []
for t in TARGETS:
    g = dxt[dxt[t]].groupby("long_title").subject_id.nunique().sort_values(ascending=False)
    for title, n in g.items():
        trows.append({"target": t, "long_title": title, "n_subjects": int(n)})
    print(f"[{t}] {len(g)} titles; top: " + "; ".join(f"{x[:38]} ({n})" for x, n in g.head(4).items()))
atomic_to_csv(pd.DataFrame(trows), DIRS["labels"] / f"{V}_label_rule_titles.csv")
if match_last.mean() < 0.99:
    raise RuntimeError(f"The label rule matches only {match_last.mean():.3f} of patients (<0.99) — stopping; please report the output.")
P4_TARGETS = list(TARGETS)

# ---- clinically cleaned label definition (own design; used only for P4_admission_clean) ----
# drop non-diagnostic titles (history, screening, vaccination, tuberculin reaction, ...) and negated titles
# ("... without (mention of) <disease>"); "lung_cancer" = primary lung/bronchus malignancy; "lung_infection" = lung/respiratory infection.
CLEAN_INCLUDE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis",
                 "arthritis": r"arthritis", "fracture": r"fracture", "lung_cancer": r"^malignant neoplasm.*(lung|bronch)",
                 "scoliosis": r"scoliosis",
                 "lung_infection": r"(infection|abscess|empyema).*(lung|pulmon|respirat)|(lung|pulmon).*(infection|abscess)"}
NONDIAG = r"history|screening|vaccinat|inoculat|tuberculin|exposure|contact with|carrier|encounter|observation"
NEGWORD = {"pneumonia": "pneumonia", "tuberculosis": "tuberculosis", "bronchitis": "bronchitis", "arthritis": "arthritis",
           "fracture": "fracture", "lung_cancer": "neoplasm", "scoliosis": "scoliosis", "lung_infection": "infection"}
nondiag = ttl.str.contains(NONDIAG, case=False, regex=True)
for t in TARGETS:
    dxt[f"c_{t}"] = (ttl.str.contains(CLEAN_INCLUDE[t], case=False, regex=True) & ~nondiag
                     & ~ttl.str.contains(r"without[^;]*" + NEGWORD[t], case=False, regex=True))
ADMLAB_CLEAN = dxt.groupby("hadm_id")[[f"c_{t}" for t in TARGETS]].any().astype(np.int8)
ADMLAB_CLEAN.columns = TARGETS
def adm_vec_clean(hadms):
    return ADMLAB_CLEAN.reindex(list(hadms)).fillna(0).astype(np.int8).values
# semantic audit of the RELEASED labels: share of positive subjects whose label-source admission has a clean diagnosis
src_rule, src_clean = adm_vec(last_h.values), adm_vec_clean(last_h.values)
sem_rows = []
for j, t in enumerate(TARGETS):
    pos = Ssub[:, j] == 1
    m = dxt[dxt.hadm_id.isin(set(last_h.values[pos])) & dxt[t]]
    only_nd = (m.assign(nd=nondiag.loc[m.index] | m.long_title.fillna("").str.contains(r"without[^;]*" + NEGWORD[t], case=False, regex=True))
                .groupby("hadm_id").nd.all())
    sem_rows.append({"target": t, "released_pos_subjects": int(pos.sum()),
                     "pos_with_clean_dx_in_source_adm_%": round(100 * src_clean[pos, j].mean(), 1),
                     "pos_only_nondiagnostic_or_negated_titles_%": round(100 * only_nd.mean(), 1),
                     "clean_rule_titles": int(dxt.loc[dxt[f"c_{t}"], "long_title"].nunique())})
SEM = pd.DataFrame(sem_rows)
atomic_to_csv(SEM, DIRS["labels"] / f"{V}_label_semantics.csv")
ctrows = []
for t in TARGETS:
    g = dxt[dxt[f"c_{t}"]].groupby("long_title").subject_id.nunique().sort_values(ascending=False)
    ctrows += [{"target": t, "long_title": x, "n_subjects": int(n)} for x, n in g.items()]
atomic_to_csv(pd.DataFrame(ctrows), DIRS["labels"] / f"{V}_clean_rule_titles.csv")
print("\nSemantic audit of released labels:\n", SEM.to_string(index=False))

# %%
# ============================== CELL 7 — REPAIRED (OWN-ADMISSION) LABELS + CORRUPTION AUDIT ==============================
img = df[["dataset_position", "subject_id", "hadm_id", "dicom", "StudyDate", "study_year", "age"] + TARGETS].copy()
img["admittime"] = img.hadm_id.map(admt)
img.loc[img.admittime.isna(), "admittime"] = df.loc[img.admittime.isna(), "adm_date"]
own = adm_vec(img.hadm_id); own_c = adm_vec_clean(img.hadm_id)
for j, t in enumerate(TARGETS):
    img[f"adm_{t}"] = own[:, j]; img[f"cln_{t}"] = own_c[:, j]
allv = adm[["subject_id", "hadm_id", "admittime"]].drop_duplicates("hadm_id").reset_index(drop=True)
allv = pd.concat([allv, pd.DataFrame(adm_vec(allv.hadm_id), columns=TARGETS),
                  pd.DataFrame(adm_vec_clean(allv.hadm_id), columns=[f"c_{t}" for t in TARGETS])], axis=1)
for t in TARGETS:                                     # diagnosed in an EARLIER admission (known at imaging time)
    fpos = allv[allv[t] == 1].groupby("subject_id").admittime.min()
    img[f"hist_{t}"] = (img.subject_id.map(fpos) < img.admittime).fillna(False).astype(np.int8)
    fpos_c = allv[allv[f"c_{t}"] == 1].groupby("subject_id").admittime.min()
    img[f"histcln_{t}"] = (img.subject_id.map(fpos_c) < img.admittime).fillna(False).astype(np.int8)
src_h = img.subject_id.map(last_h); src_t = src_h.map(admt)
img["label_source"] = np.select([img.hadm_id.values == src_h.values, (src_t < img.admittime).values, (src_t > img.admittime).values],
                                ["own", "earlier", "later"], "same_time")
img["age_own"] = age_at(img.subject_id.values, img.hadm_id.values)
csv = img[TARGETS].values
crows = []
for j, t in enumerate(TARGETS):
    y, o = csv[:, j], own[:, j]
    crows.append({"target": t, "csv_pos_images": int(y.sum()), "own_pos_images": int(o.sum()),
                  "csv_prev_%": round(100 * y.mean(), 2), "own_prev_%": round(100 * o.mean(), 2),
                  "images_disagree_%": round(100 * (y != o).mean(), 2),
                  "csv_pos_but_own_neg_%": round(100 * ((y == 1) & (o == 0)).sum() / max(1, y.sum()), 1),
                  "own_pos_but_csv_neg_%": round(100 * ((o == 1) & (y == 0)).sum() / max(1, o.sum()), 1),
                  "csv_pos_from_later_adm_only_%": round(100 * ((y == 1) & (o == 0) & (img.label_source.values == "later")).sum() / max(1, y.sum()), 1),
                  "clean_pos_images": int(own_c[:, j].sum()), "clean_prev_%": round(100 * own_c[:, j].mean(), 2)})
CORR = pd.DataFrame(crows)
anyd = (csv != own).any(1)
multi = img.subject_id.map(img.groupby("subject_id").size() > 1).values
corr_summary = {"images": len(img), "label_source_share": img.label_source.value_counts(normalize=True).round(4).to_dict(),
                "images_with_any_label_differing_from_own_adm": int(anyd.sum()), "frac_any_differ": round(float(anyd.mean()), 4),
                "frac_any_differ_multi_adm_subjects": round(float(anyd[multi].mean()), 4),
                "frac_age_differs_from_own_adm": round(float((img.age != img.age_own).mean()), 4)}
atomic_to_csv(CORR, DIRS["labels"] / f"{V}_label_corruption.csv")
atomic_json(corr_summary, DIRS["labels"] / f"{V}_label_corruption_summary.json")
atomic_to_csv(img[["dataset_position", "subject_id", "hadm_id", "dicom", "label_source"] + [f"adm_{t}" for t in TARGETS]
                  + [f"cln_{t}" for t in TARGETS] + [f"hist_{t}" for t in TARGETS] + [f"histcln_{t}" for t in TARGETS]],
              DIRS["labels"] / f"{V}_admission_labels.csv.gz")
print(CORR.to_string(index=False)); print(corr_summary)

# %%
# ============================== CELL 8 — P5 TEMPORAL SPLIT (real calendar time) ==============================
pt = pts.drop_duplicates("subject_id").set_index("subject_id")
g0 = pt.anchor_year_group.astype(str).str.extract(r"(\d{4})")[0].astype(float)
img["est_year"] = (img.subject_id.map(g0) + 1 + (img.study_year - img.subject_id.map(pt.anchor_year))).astype(float)
first_year = img.groupby("subject_id").est_year.min()
assert first_year.notna().all(), f"{int(first_year.isna().sum())} patients without anchor_year_group"
order = sorted(S.index, key=lambda s: (first_year[s], stable_hash(f"v3|{s}")))
n = len(order); n_tr, n_va = int(round(0.64 * n)), int(round(0.16 * n))
role5 = pd.Series("test", index=order); role5.iloc[:n_tr] = "train"; role5.iloc[n_tr:n_tr + n_va] = "val"
man5 = pd.DataFrame({"subject_id": order, "est_first_year": first_year.reindex(order).values, "role": role5.values})
atomic_to_csv(man5, DIRS["manifests"] / f"{V}_folds_P5_temporal.csv")
p5_report = man5.groupby("role").est_first_year.describe()[["count", "min", "50%", "max"]]
p5_report = p5_report.join(S.join(man5.set_index("subject_id").role).groupby("role")[TARGETS].mean().mul(100).round(1))
atomic_to_csv(p5_report.reset_index(), DIRS["manifests"] / f"{V}_P5_split_report.csv")
print("estimated real year of each image:", img.est_year.min(), "–", img.est_year.max())
print(p5_report.to_string())

# %%
# ============================== CELL 9 — TABLE, FEATURE SETS, SPLITS ==============================
E = np.load(f"{RADDINO_DIR}/emb.npy").astype(np.float32)
ids = pd.read_csv(f"{RADDINO_DIR}/ids.csv")
posm = df.dicom.map(dict(zip(ids.dicom, range(len(ids)))))
assert posm.notna().all(), f"{int(posm.isna().sum())} radiographs without a RAD-DINO embedding"
EMB = [f"raddino_{j}" for j in range(E.shape[1])]
rec = pd.concat([df[["dataset_position", "subject_id"] + EHR + IMG].reset_index(drop=True),
                 pd.DataFrame(E[posm.astype(int).values], columns=EMB)], axis=1)
FEATSETS = {"ehr": dict(cont=EHR_CONT, emb=[], bin=EHR_BIN),
            "img_xrv": dict(cont=IMG, emb=[], bin=[]),
            "img_raddino": dict(cont=[], emb=EMB, bin=[]),
            "fusion_raddino": dict(cont=EHR_CONT, emb=EMB, bin=EHR_BIN)}
EMB_PCA_DIM = 64
rec.to_pickle(CACHE / f"{V}_rec.pkl")
np.savez(CACHE / f"{V}_labels.npz", **{f"subj__{t}": img[t].values.astype(np.int8) for t in TARGETS},
         **{f"adm__{t}": img[f"adm_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"hist__{t}": img[f"hist_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"cln__{t}": img[f"cln_{t}"].values.astype(np.int8) for t in TARGETS},
         **{f"histcln__{t}": img[f"histcln_{t}"].values.astype(np.int8) for t in TARGETS})

# P2 folds (exactly the v2 manifest; rebuilt deterministically if the file is not reachable)
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
atomic_to_csv(m2, DIRS["manifests"] / f"{V}_folds_P2_subject_s42_copy.csv")
# P1 record folds (exactly the v2 manifest; rebuilt deterministically if missing)
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
atomic_to_csv(m1, DIRS["manifests"] / f"{V}_folds_P1_record_s42_copy.csv")
SPLITS = {}
for k in range(5):
    SPLITS[("P1", k)] = m1[m1.fold == k].set_index("dataset_position").role.reindex(rec.dataset_position).values
for k in range(5):
    SPLITS[("P2", k)] = rec.subject_id.map(m2[m2.fold == k].set_index("subject_id").role).values
SPLITS[("P5", 0)] = rec.subject_id.map(man5.set_index("subject_id").role).values
for (sname, k), role in SPLITS.items():
    assert pd.notna(role).all()
    i_tr, i_va, i_te = [np.where(role == r)[0] for r in ("train", "val", "test")]
    s_tr, s_va, s_te = [set(rec.subject_id.values[i]) for i in (i_tr, i_va, i_te)]
    seen_te = np.isin(rec.subject_id.values[i_te], list(s_tr | s_va))
    if sname != "P1":
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te) and not seen_te.any()
    np.savez(CACHE / f"{V}_splits" / f"{sname}__f{k}.npz", idx_tr=i_tr, idx_va=i_va, idx_te=i_te, seen_te=seen_te)

PROTO = {"P4_admission":    dict(split="P2", folds=list(range(5)), label="adm"),
         "P5_temporal":     dict(split="P5", folds=[0], label="subj"),
         "P5_temporal_adm": dict(split="P5", folds=[0], label="adm"),
         "P1_admission":    dict(split="P1", folds=list(range(5)), label="adm", feats=["ehr", "img_xrv"]),
         "P4_admission_clean": dict(split="P2", folds=list(range(5)), label="cln")}
CLFS = ["LR", "LGBM", "ET"]
G_FEATS = list(FEATSETS)
if RUN_MODE == "preflight":
    for _p in ["P4_admission", "P1_admission", "P4_admission_clean"]: PROTO[_p]["folds"] = [0]
    G_FEATS = ["ehr", "img_raddino"]; CLFS = ["LR", "LGBM"]
if LOCAL_TEST and os.environ.get("V3_TINY") == "1":       # offline smoke test only — ignored on Colab
    for _p in ["P4_admission", "P1_admission", "P4_admission_clean"]: PROTO[_p]["folds"] = [0]
    CLFS = ["LR", "LGBM"]; G_FEATS = ["ehr", "img_raddino"]

# eligibility: enough positives in train / val / test for every (protocol, fold, target)
lab = np.load(CACHE / f"{V}_labels.npz")
TASKS, skipped = [], []
for p, cfg in PROTO.items():
    tgts = TARGETS if cfg["label"] == "subj" else P4_TARGETS
    if RUN_MODE == "preflight" or (LOCAL_TEST and os.environ.get("V3_TINY") == "1"):
        tgts = [t for t in ["pneumonia", "fracture", "scoliosis"] if t in tgts][: 2 if RUN_MODE == "preflight" else 3]
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
# ============================== CELL 10 — MODEL / SAMPLER / METRIC FUNCTIONS (same as v2) ==============================
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
# ============================== CELL 11 — STAGE A: PREPROCESSING CACHE ==============================
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
sec = {"ehr": 4, "img_xrv": 5, "img_raddino": 12, "fusion_raddino": 15}
eta = sum(sec[ft] * len(strategies_for(bool(FEATSETS[ft]["bin"]))) * len(CLFS) for _, _, ft, _ in TODO) / NP / 60
print(f"model fits: {n_fits} | ETA ≈ {eta:.0f} min with {NP} cores")

# %%
# ============================== CELL 12 — STAGE B: MODELS (resume-safe: 2 files per task on Drive) ==============================
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
# ============================== CELL 13 — BASELINES (all-positive, EHR lookup, history) ==============================
def f2_at(y, yhat):
    y = y.astype(bool); yhat = yhat.astype(bool)
    tp = (y & yhat).sum(); fp = (~y & yhat).sum(); fn = (y & ~yhat).sum(); d = 5 * tp + 4 * fn + fp
    return 5 * tp / d if d else 0.0
keys = pd.util.hash_pandas_object(rec[EHR].fillna(-999), index=False).values
base_rows = []
PKT = {(p, k, t) for (p, k, ft, t) in TASKS}
for p, cfg in PROTO.items():
    for t in sorted({t for (pp, k, t) in PKT if pp == p}):
        y = lab[f"{cfg['label']}__{t}"]; hist = lab[f"{'histcln' if cfg['label'] == 'cln' else 'hist'}__{t}"]
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
                              "history_f2": f2_at(yte, hist[sp["idx_te"]]) if cfg["label"] in ("adm", "cln") else np.nan,
                              "history_pos_rate": float(hist[sp["idx_te"]].mean())})
BASE = pd.DataFrame(base_rows)
atomic_to_csv(BASE, DIRS["analysis"] / f"{V}_baselines.csv")
print(BASE.groupby("protocol")[["trivial_f2", "lookup_f2", "lookup_any_f2", "lookup_hit_rate", "history_f2"]].mean().round(3))

# %%
# ============================== CELL 14 — MERGE + ANALYSIS ==============================
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
if "P1_admission" in set(R.protocol):
    sel1 = best[best.protocol == "P1_admission"][KEY + ["strategy", "classifier"]]
    P1r = R.merge(sel1, on=KEY + ["strategy", "classifier"])
    TC = P1r.groupby(["feature_set", "target"])[["test_f2", "seen_f2", "unseen_f2", "unseen_trivial_f2", "seen_n", "unseen_n"]].mean()
    atomic_to_csv(TC.reset_index().round(4), DIRS["analysis"] / f"{V}_tableC_P1adm_seen_unseen.csv")
    print("\nP1_admission — seen vs unseen patients (macro)\n", TC.groupby("feature_set").mean().round(3).to_string())
print("\nMACRO (configuration selected on inner-validation F2)\n", TA.round(3).to_string())

# --- patient-cluster bootstrap for the single temporal split (P5) ---
rng = np.random.default_rng(42); B = 1000; boot_rows = []
for p in [p for p in PROTO if PROTO[p]["split"] == "P5" and p in set(best.protocol)]:
    sp = np.load(CACHE / f"{V}_splits" / "P5__f0.npz"); te = sp["idx_te"]
    sid = rec.subject_id.values[te]; us, inv = np.unique(sid, return_inverse=True)
    members = [np.where(inv == j)[0] for j in range(len(us))]
    draws = [np.concatenate([members[j] for j in rng.integers(0, len(us), len(us))]) for _ in range(B)]
    for ft in sorted(set(best[best.protocol == p].feature_set)):
        bb = best[(best.protocol == p) & (best.feature_set == ft)]
        per_t = {}
        for _, r in bb.iterrows():
            z = np.load(DIRS["preds"] / f"{V}_{p}__f0__{ft}__{r.target}.npz")
            pva, pte = z[f"{r.strategy}__{r.classifier}__p_va"], z[f"{r.strategy}__{r.classifier}__p_te"]
            y = lab[f"{PROTO[p]['label']}__{r.target}"]
            thr = choose_threshold(y[sp["idx_va"]], pva)
            per_t[r.target] = (y[te].astype(bool), pte >= thr)
        stats_b = []
        for idx in draws:
            f = [f2_at(yt[idx], yh[idx]) for yt, yh in per_t.values()]
            tv = [5 * yt[idx].mean() / (4 * yt[idx].mean() + 1) for yt, _ in per_t.values()]
            stats_b.append((np.mean(f), np.mean(f) - np.mean(tv)))
        stats_b = np.array(stats_b)
        point = np.mean([f2_at(yt, yh) for yt, yh in per_t.values()])
        boot_rows.append({"protocol": p, "feature_set": ft, "macro_f2": point,
                          "ci_lo": np.percentile(stats_b[:, 0], 2.5), "ci_hi": np.percentile(stats_b[:, 0], 97.5),
                          "gain_vs_trivial_lo": np.percentile(stats_b[:, 1], 2.5), "gain_vs_trivial_hi": np.percentile(stats_b[:, 1], 97.5),
                          "n_targets": len(per_t)})
BOOT = pd.DataFrame(boot_rows)
if len(BOOT):
    atomic_to_csv(BOOT.round(4), DIRS["analysis"] / f"{V}_P5_bootstrap.csv")
    print("\nP5 (single calendar-time split) — macro F2, 95% CI from patient-level bootstrap\n", BOOT.round(3).to_string(index=False))

numbers = {"version": VERSION, "data_hash": data_hash, "bq_source": json.load(open(BQ_SRC)),
           "label_reconstruction": RECON.round(4).to_dict("records"), "label_reconstruction_summary": recon_summary,
           "label_corruption": CORR.to_dict("records"), "label_corruption_summary": corr_summary,
           "label_semantics": SEM.to_dict("records"),
           "P4_targets": P4_TARGETS, "P5_split": p5_report.reset_index().to_dict("records"),
           "tableA": TA.round(4).reset_index().to_dict("records"), "P5_bootstrap": BOOT.round(4).to_dict("records"),
           "n_fits": len(R), "n_errors": len(ERR), "skipped": skipped,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
atomic_json(numbers, DIRS["analysis"] / f"{V}_paper_numbers.json")

# --- one small zip to send back (tables + admission labels; no predictions) ---
ZIP = OUT / f"{V}_for_claude.zip"
tmpz = str(ZIP) + ".tmp"
with zipfile.ZipFile(tmpz, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["labels", "manifests", "analysis"]:
        for f in sorted(DIRS[d].glob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                z.write(f, f"{OUT.name}/{DIRS[d].name}/{f.name}")
    z.write(BQ_SRC, f"{OUT.name}/{DIRS['bq'].name}/{BQ_SRC.name}")
os.replace(tmpz, ZIP)
print("\nFILE TO DOWNLOAD:", ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")

# %%
# ============================== CELL 15 — FLUSH TO DRIVE (required before closing the session) ==============================
if IN_COLAB:
    print("Flushing all files to Google Drive... do NOT close the session until 'FLUSH DONE' appears.")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("FLUSH DONE — download", ZIP.name, "from", OUT)
