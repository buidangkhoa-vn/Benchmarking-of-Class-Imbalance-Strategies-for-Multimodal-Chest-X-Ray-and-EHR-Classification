# %% [markdown]
# # STEP 4a (v3): BigQuery extraction with the PhysioNet-credentialed account
#
# Only needed if STEP4 cannot sign in to BigQuery. Open this notebook with the credentialed account (free Colab, CPU,
# about 5 min), set GCP_PROJECT, Run all, upload x_ray_and_features_data.csv when asked and allow access in the sign-in
# window. The browser then downloads v3_bq_extract.zip (about 10 MB). Put it in MyDrive/data_aug/MMMDA_runs/ of the
# account that runs STEP4 and rerun STEP4. Nothing is written to the Drive of the credentialed account.

# %%
# Cell 1: control panel
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # BigQuery project with PhysioNet access
VERSION       = "v3_2026-10-19"
BQ_AUTH       = "same_account"        # run under the credentialed account
EXPECTED_HASH = "da85519958e4bccb"
BQ_CANDIDATES = [("physionet-data.mimiciv_3_1_hosp", "physionet-data.mimiciv_3_1_hosp"),
                 ("physionet-data.mimiciv_2_2_hosp", "physionet-data.mimiciv_2_2_hosp"),
                 ("physionet-data.mimiciv_hosp",     "physionet-data.mimiciv_hosp"),
                 ("physionet-data.mimiciv_3_0_hosp", "physionet-data.mimiciv_3_0_hosp"),
                 ("physionet-data.mimic_hosp",       "physionet-data.mimic_core")]    # (hosp: diagnoses, core: admissions/patients)

# %%
# Cell 2: setup + upload CSV
import os, sys, re, json, time, hashlib, shutil, zipfile, subprocess, importlib
from pathlib import Path
import numpy as np, pandas as pd
IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("V3_LOCAL_TEST") == "1"
V = "v3"
CSV_PATH = "/content/x_ray_and_features_data.csv"
OUT_ROOT = "/content/v3_bq_extract_work"
if LOCAL_TEST:
    CSV_PATH = os.environ["V3_CSV"]; OUT_ROOT = os.environ["V3_OUT"]
if IN_COLAB:
    try:
        importlib.import_module("db_dtypes")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "db-dtypes"], check=True)
    if not os.path.exists(CSV_PATH):
        from google.colab import files
        print("Select x_ray_and_features_data.csv to upload ...")
        up = files.upload()
        name = next(iter(up))
        if os.path.abspath(name) != CSV_PATH:
            os.replace(name, CSV_PATH)

def atomic_to_csv(df, path):
    path = str(path); tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip" if path.endswith(".gz") else None); os.replace(tmp, path)

def atomic_json(obj, path):
    path = str(path); tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

OUT = Path(OUT_ROOT) / f"{VERSION}_P4P5"
DIRS = {"bq": OUT / f"{V}_bq"}
DIRS["bq"].mkdir(parents=True, exist_ok=True)

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
# Cell 5: BigQuery extraction
BQF = {k: DIRS["bq"] / f"{V}_bq_{k}.csv.gz" for k in ["admissions", "patients", "diagnoses", "icd_titles"]}
BQ_SRC = DIRS["bq"] / f"{V}_bq_source.json"
if LOCAL_TEST and os.environ.get("V3_BQ_DIR"):
    for k, f in BQF.items():
        if not f.exists():
            shutil.copy(Path(os.environ["V3_BQ_DIR"]) / f.name, f)
    if not BQ_SRC.exists():
        atomic_json({"hosp": "LOCAL_TEST", "core": "LOCAL_TEST"}, BQ_SRC)

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
    # further candidate datasets (listing may be forbidden)
    cands = list(BQ_CANDIDATES)
    try:
        for ds in client.list_datasets(project="physionet-data"):
            n = ds.dataset_id
            if re.match(r"mimiciv_.*hosp$", n) and ("physionet-data." + n, "physionet-data." + n) not in cands:
                cands.insert(0, ("physionet-data." + n, "physionet-data." + n))
    except Exception as e:
        print("list_datasets not permitted (skipped):", repr(e)[:120])
    # pick the dataset whose admissions match hadm_id and admission date
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
                "(physionet.org -> MIMIC-IV page -> 'Request access using Google BigQuery'), or GCP_PROJECT is wrong.")
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
# Cell 5: zip + download
ZIP = Path(OUT_ROOT) / f"{V}_bq_extract.zip"
with zipfile.ZipFile(str(ZIP) + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(BQF.values()) + [BQ_SRC]:
        z.write(f, f.name)
os.replace(str(ZIP) + ".tmp", ZIP)
print(f"DONE: {ZIP.name} ({ZIP.stat().st_size / 1e6:.1f} MB)")
print("-> Upload this file to the Google Drive of the account that runs STEP4, folder MyDrive/data_aug/MMMDA_runs/, then Run all in STEP4.")
if IN_COLAB:
    from google.colab import files
    files.download(str(ZIP))
