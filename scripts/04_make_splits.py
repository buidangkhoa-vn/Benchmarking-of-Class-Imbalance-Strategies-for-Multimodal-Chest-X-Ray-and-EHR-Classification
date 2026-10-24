"""Fold assignments (Sec. IV-A): record folds (P1), patient folds for every seed in cv.seeds (P2 and later; P3 uses the
same patient folds with one row per patient), and the calendar-time split (P5)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance import cohort as C  # noqa: E402
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.io import atomic_csv  # noqa: E402
from cxr_ehr_imbalance.splits import patient_folds, record_folds, temporal_split  # noqa: E402

args, cfg = cli(__doc__)
out = cfg["dirs"]["splits"]
cv = cfg["cv"]
df = C.load_cohort(cfg["paths"]["cohort_csv"])

seed0 = cv["seeds"][0]
atomic_csv(record_folds(df, seed0, cv["n_folds"], cv["inner_val_frac"]), out / f"folds_record_s{seed0}.csv")
for seed in cv["seeds"]:
    atomic_csv(patient_folds(df, seed, cv["n_folds"], cv["inner_val_frac"]), out / f"folds_patient_s{seed}.csv")
patients = pd.read_csv(cfg["dirs"]["mimic"] / "patients.csv.gz")
P5 = temporal_split(df, patients, cv["temporal"]["train"], cv["temporal"]["val"])
atomic_csv(P5, out / "folds_temporal.csv")

rows = []
for f in sorted(out.glob("folds_*.csv")):
    m = pd.read_csv(f)
    rows.append(m.groupby(["fold", "role"]).size().rename("n").reset_index().assign(file=f.name))
report = pd.concat(rows)
atomic_csv(report, out / "split_sizes.csv")
print(report.pivot_table(index=["file", "fold"], columns="role", values="n").to_string())
print("P5 first-year range by role:\n", P5.groupby("role").est_first_year.describe()[["min", "50%", "max"]])
