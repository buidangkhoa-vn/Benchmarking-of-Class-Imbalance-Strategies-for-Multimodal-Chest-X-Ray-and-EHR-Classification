"""Build the v3 notebooks: STEP4a (BigQuery extract on the PhysioNet-credentialed account) and STEP4b (P4/P5 run)."""
import re, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from to_ipynb import convert

src = open(HERE + "/nb4_src.py").read()
cells = re.split(r'^(?=# %%)', src, flags=re.M)
cells = [c for c in cells if c.strip()]
def cell_with(marker):
    hit = [c for c in cells if marker in c]
    assert len(hit) == 1, marker
    return hit[0]
CONTROL = cell_with("CELL 1 — CONTROL PANEL")
LOAD = cell_with("CELL 4 — LOAD + LOCK DATA")
BQ = cell_with("CELL 5 — BIGQUERY EXTRACTION")
cand = re.search(r"BQ_CANDIDATES = \[.*?\]\s*#[^\n]*\n", CONTROL, re.S).group(0)

intro = '''# %% [markdown]
# # STEP 4a (v3) — Extract four MIMIC-IV tables from BigQuery with the PhysioNet-credentialed account
#
# Use only if the account that runs Colab (STEP4) has **no** PhysioNet access on BigQuery and the code-paste login in STEP4 fails.
#
# 1. Open this notebook in Colab **with <your PhysioNet-credentialed Google account>** (the account with PhysioNet access). Free Colab is enough; CPU runtime, ~5 min.
# 2. Set `GCP_PROJECT` (placeholder `YOUR_GCP_PROJECT_ID`) → **Run all**.
# 3. When prompted, upload `x_ray_and_features_data.csv`; in the Google sign-in window choose the credentialed account and grant all permissions.
# 4. The browser then downloads `v3_bq_extract.zip` (~10 MB).
# 5. Upload that zip to the Google Drive **of the account that runs STEP4**, folder `MyDrive/data_aug/MMMDA_runs/` → open STEP4 → Run all.
#
# This notebook writes nothing to the Drive of the credentialed account.

'''
control = f'''# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # BigQuery project of <your PhysioNet-credentialed Google account>
VERSION       = "v3_2026-10-19"
BQ_AUTH       = "same_account"        # this notebook runs under the credentialed account itself -> standard sign-in window
EXPECTED_HASH = "da85519958e4bccb"
{cand}
'''
setup = '''# %%
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

'''
finish = '''# %%
# ============================== CELL 5 — ZIP + DOWNLOAD ==============================
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
'''
a_src = intro + control + setup + LOAD + BQ + finish
open(HERE + "/nb4a_src.py", "w").write(a_src)

OUTD = HERE + "/../notebooks"
convert(HERE + "/nb4a_src.py", f"{OUTD}/STEP4a_v3_2026-10-19_BQ_extract_MAIN_ACCOUNT_only_if_needed.ipynb")
convert(HERE + "/nb4_src.py", f"{OUTD}/STEP4_v3_2026-10-19_P4P5_admission_temporal.ipynb")
