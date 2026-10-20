# %% [markdown]
# # STEP 4a (v3) — Trích 4 bảng MIMIC-IV từ BigQuery bằng TÀI KHOẢN CHÍNH
#
# Chỉ dùng khi tài khoản chạy Colab Pro (STEP4b) **không** có quyền PhysioNet trên BigQuery và cách dán mã ở STEP4b bị lỗi.
#
# 1. Mở notebook này trên Colab **bằng tài khoản <your PhysioNet-credentialed Google account>** (tài khoản có quyền PhysioNet). Colab miễn phí là đủ, runtime CPU, ~5 phút.
# 2. `GCP_PROJECT` đã điền sẵn `YOUR_GCP_PROJECT_ID` → **Run all**.
# 3. Khi được hỏi, upload file `x_ray_and_features_data.csv` (trong `D:\project_TTNT_Data_aug\`); cửa sổ đăng nhập Google: chọn tài khoản chính, tick hết quyền.
# 4. Cuối cùng trình duyệt tự tải về `v3_bq_extract.zip` (~10 MB).
# 5. Upload file zip đó vào Google Drive **của tài khoản chạy STEP4b**, thư mục `MyDrive/data_aug/MMMDA_runs/` → mở STEP4b → Run all.
#
# Notebook không lưu gì lên Drive của tài khoản chính.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # project BigQuery của <your PhysioNet-credentialed Google account> (tài khoản có quyền PhysioNet)
VERSION       = "v3_2026-10-19"
BQ_AUTH       = "same_account"        # notebook này chạy bằng chính tài khoản có quyền -> cửa sổ đăng nhập thường
EXPECTED_HASH = "da85519958e4bccb"
BQ_CANDIDATES = [("physionet-data.mimiciv_3_1_hosp", "physionet-data.mimiciv_3_1_hosp"),
                 ("physionet-data.mimiciv_2_2_hosp", "physionet-data.mimiciv_2_2_hosp"),
                 ("physionet-data.mimiciv_hosp",     "physionet-data.mimiciv_hosp"),
                 ("physionet-data.mimiciv_3_0_hosp", "physionet-data.mimiciv_3_0_hosp"),
                 ("physionet-data.mimic_hosp",       "physionet-data.mimic_core")]    # (hosp: diagnoses, core: admissions/patients)

# %%
# ============================== CELL 2 — SETUP + UPLOAD CSV ==============================
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
        print("Chọn file x_ray_and_features_data.csv (trong D:\\project_TTNT_Data_aug\\) để upload ...")
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
# ============================== CELL 4 — LOAD + LOCK DATA ==============================
df = pd.read_csv(CSV_PATH)
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
print("shape", df.shape, "| hash", data_hash)
assert df.shape == (29083, 48), df.shape
assert data_hash == EXPECTED_HASH, "CSV khác file đã benchmark — DỪNG"
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

# (a) bảng đã được trích sẵn bằng STEP4a (tài khoản chính) -> chỉ cần giải nén
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
            print("BigQuery: dùng bảng trích sẵn từ", zp); break

# (b) hỏi BigQuery trực tiếp
if not all(f.exists() for f in BQF.values()):
    from google.cloud import bigquery
    project = GCP_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise RuntimeError("Chưa điền GCP_PROJECT ở CELL 1 (project id Google Cloud của tài khoản có quyền PhysioNet trên BigQuery).")
    if BQ_AUTH == "other_account" and IN_COLAB:
        print("=" * 90)
        print("ĐĂNG NHẬP BIGQUERY BẰNG TÀI KHOẢN CHÍNH (tài khoản có quyền PhysioNet):")
        print(" 1) Bấm link hiện ra bên dưới -> chọn TÀI KHOẢN CHÍNH -> Allow / Cho phép")
        print(" 2) Copy mã xác nhận hiện ra -> dán vào ô nhập bên dưới -> Enter")
        print("=" * 90)
        from IPython import get_ipython
        get_ipython().system("gcloud auth application-default login --no-launch-browser "
                             "--scopes=https://www.googleapis.com/auth/cloud-platform")
        import google.auth
        adc = Path(os.environ.get("CLOUDSDK_CONFIG", str(Path.home() / ".config" / "gcloud"))) / "application_default_credentials.json"
        if not adc.exists():
            raise RuntimeError("Đăng nhập tài khoản chính chưa thành công (không thấy credential). Chạy lại cell này; "
                               "nếu vẫn lỗi, dùng notebook STEP4a bằng tài khoản chính.")
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
        print("BigQuery đang dùng tài khoản:", bq("SELECT SESSION_USER() AS u").u.iloc[0])
    except Exception as e:
        raise RuntimeError(f"Không chạy được truy vấn BigQuery với project '{project}': {repr(e)[:300]}\n"
                           "Kiểm tra: GCP_PROJECT có thuộc tài khoản vừa đăng nhập không, BigQuery API đã bật chưa.")
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
        print("list_datasets không được phép (bỏ qua):", repr(e)[:120])
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
            errs.append(f"{core}: {repr(e)[:200]}"); print(f"{core}: không dùng được ({repr(e)[:150]})")
    ok = [c for c in checks if c["found"] > 0.98 and max(c["date_match"], c["study_in_admission"]) > 0.95]
    if not ok:
        hint = ("\nNếu lỗi là 403/Access Denied: tài khoản Google đang đăng nhập chưa được cấp quyền physionet-data trên BigQuery "
                "(physionet.org → trang MIMIC-IV → 'Request access using Google BigQuery'), hoặc GCP_PROJECT sai.")
        raise RuntimeError(f"Không dataset BigQuery nào khớp cohort.\nchecks={checks}\nerrors={errs}{hint}\nGửi toàn bộ thông báo này cho Claude.")
    src = max(ok, key=lambda c: (c["found"], c["date_match"], c["study_in_admission"]))
    hosp, core = src["hosp"], src["core"]
    print("==> dùng", hosp, "/", core)
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
    print("BigQuery: dùng bản đã lưu trên Drive", json.load(open(BQ_SRC)).get("hosp"))

adm = pd.read_csv(BQF["admissions"], parse_dates=["admittime", "dischtime"])
pts = pd.read_csv(BQF["patients"])
dxs = pd.read_csv(BQF["diagnoses"], dtype={"icd_code": str})
tit = pd.read_csv(BQF["icd_titles"], dtype={"icd_code": str})
dxs["icd_code"] = dxs.icd_code.str.strip(); tit["icd_code"] = tit.icd_code.str.strip()
assert set(SIDS) <= set(pts.subject_id), "thiếu bệnh nhân trong patients"
miss_h = set(HIDS) - set(adm.hadm_id)
assert len(miss_h) <= 0.002 * len(HIDS), f"{len(miss_h)} hadm_id của cohort không có trong admissions"
print(f"admissions {len(adm)} (cohort subjects, all admissions) | diagnoses {len(dxs)} | titles {len(tit)}")

# %%
# ============================== CELL 5 — ZIP + DOWNLOAD ==============================
ZIP = Path(OUT_ROOT) / f"{V}_bq_extract.zip"
with zipfile.ZipFile(str(ZIP) + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
    for f in list(BQF.values()) + [BQ_SRC]:
        z.write(f, f.name)
os.replace(str(ZIP) + ".tmp", ZIP)
print(f"XONG: {ZIP.name} ({ZIP.stat().st_size / 1e6:.1f} MB)")
print("-> Upload file này vào Google Drive của tài khoản chạy STEP4b, thư mục MyDrive/data_aug/MMMDA_runs/ rồi Run all STEP4b.")
if IN_COLAB:
    from google.colab import files
    files.download(str(ZIP))
