"""MIMIC-IV v2.2 hosp tables for the cohort's patients: admissions, patients, diagnoses_icd, d_icd_diagnoses.

Reads the PhysioNet CSV files (mimic_iv.source: local) or queries BigQuery (mimic_iv.source: bigquery)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.io import atomic_csv  # noqa: E402

args, cfg = cli(__doc__)
out = cfg["dirs"]["mimic"]
df = pd.read_csv(cfg["paths"]["cohort_csv"], usecols=["subject_id", "hadm_id", "admittime"])
sids = sorted(int(s) for s in df.subject_id.unique())
COLS = {"admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
        "patients": ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
        "diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        "d_icd_diagnoses": ["icd_code", "icd_version", "long_title"]}

if cfg["mimic_iv"]["source"] == "local":
    hosp = Path(cfg["paths"]["mimic_iv_hosp"])
    tables = {}
    for name, cols in COLS.items():
        t = pd.read_csv(hosp / f"{name}.csv.gz", usecols=cols, dtype={"icd_code": str})
        tables[name] = t[t.subject_id.isin(sids)] if "subject_id" in cols else t
else:
    from google.cloud import bigquery
    client = bigquery.Client(project=cfg["mimic_iv"]["gcp_project"])
    ds = cfg["mimic_iv"]["bigquery_dataset"]
    param = [bigquery.ArrayQueryParameter("s", "INT64", sids)]
    def query(sql, params=()):
        job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=list(params)))
        return job.to_dataframe(create_bqstorage_client=False)
    tables = {name: query(f"SELECT {', '.join(cols)} FROM `{ds}.{name}`" +
                          (" WHERE subject_id IN UNNEST(@s)" if "subject_id" in cols else ""), param if "subject_id" in cols else ())
              for name, cols in COLS.items()}

for name, t in tables.items():
    atomic_csv(t, out / f"{name}.csv.gz")
    print(name, t.shape)

adm = tables["admissions"].drop_duplicates("hadm_id").set_index("hadm_id")
a = adm.reindex(df.hadm_id.astype("int64"))
print(f"cohort admissions found: {a.admittime.notna().mean():.4f} | admission date matches the cohort file: "
      f"{(pd.to_datetime(a.admittime).dt.normalize().values == pd.to_datetime(df.admittime).dt.normalize().values).mean():.4f}")
