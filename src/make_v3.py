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
CONTROL = cell_with("Cell 1: control panel")
LOAD = cell_with("Cell 4: load + lock data")
BQ = cell_with("Cell 5: BigQuery extraction")
cand = re.search(r"BQ_CANDIDATES = \[.*?\]\s*#[^\n]*\n", CONTROL, re.S).group(0)

intro = '''# %% [markdown]
# # STEP 4a (v3): BigQuery extraction with the PhysioNet-credentialed account
#
# Only needed if STEP4 cannot sign in to BigQuery. Open this notebook with the credentialed account (free Colab, CPU,
# about 5 min), set GCP_PROJECT, Run all, upload x_ray_and_features_data.csv when asked and allow access in the sign-in
# window. The browser then downloads v3_bq_extract.zip (about 10 MB). Put it in MyDrive/data_aug/MMMDA_runs/ of the
# account that runs STEP4 and rerun STEP4. Nothing is written to the Drive of the credentialed account.

'''
control = f'''# %%
# Cell 1: control panel
GCP_PROJECT   = "YOUR_GCP_PROJECT_ID"   # BigQuery project with PhysioNet access
VERSION       = "v3_2026-10-19"
BQ_AUTH       = "same_account"        # run under the credentialed account
EXPECTED_HASH = "da85519958e4bccb"
{cand}
'''
setup = '''# %%
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

'''
finish = '''# %%
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
'''
a_src = intro + control + setup + LOAD + BQ + finish
open(HERE + "/nb4a_src.py", "w").write(a_src)

OUTD = HERE + "/../notebooks"
convert(HERE + "/nb4a_src.py", f"{OUTD}/STEP4a_v3_2026-10-19_BQ_extract_optional.ipynb")
convert(HERE + "/nb4_src.py", f"{OUTD}/STEP4_v3_2026-10-19_P4P5_admission_temporal.ipynb")
