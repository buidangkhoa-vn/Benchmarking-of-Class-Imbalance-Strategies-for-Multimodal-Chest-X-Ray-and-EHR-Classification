import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance import cohort as C  # noqa: E402
from cxr_ehr_imbalance.config import cli  # noqa: E402
from cxr_ehr_imbalance.io import atomic_csv, atomic_json  # noqa: E402
from cxr_ehr_imbalance.labels import build_admission_labels, chexpert_labels, load_mimic_tables  # noqa: E402

args, cfg = cli("Label provenance audit and label sets (released, admission-aligned, clean, CheXpert).")
out = cfg["dirs"]["labels"]
df = C.load_cohort(cfg["paths"]["cohort_csv"])

# Cohort and leakage audit (Sections III-A and III-B).
per_patient = df.groupby("subject_id").size()
ehr_key = pd.util.hash_pandas_object(df[C.EHR].fillna(-999), index=False)
patients_per_vector = pd.Series(df.subject_id.values).groupby(ehr_key.values).nunique()
const = df.groupby("subject_id")[C.TARGETS + C.EHR].nunique(dropna=False).max()
atomic_json(
    {
        "data_hash": C.data_hash(df.drop(columns="dataset_position")),
        "radiographs": len(df),
        "patients": int(df.subject_id.nunique()),
        "radiographs_per_patient_mean": round(float(per_patient.mean()), 2),
        "radiographs_per_patient_max": int(per_patient.max()),
        "patients_with_several_radiographs": int((per_patient > 1).sum()),
        "distinct_ehr_vectors": int(ehr_key.nunique()),
        "ehr_vectors_shared_by_patients": int((patients_per_vector > 1).sum()),
        "labels_and_ehr_constant_within_patient": bool((const <= 1).all()),
        "missing_%": (100 * df[C.EHR_CONT].isna().mean()).round(1).to_dict(),
    },
    out / "cohort_audit.json",
)

tables = load_mimic_tables(cfg["dirs"]["mimic"])
labels, audit = build_admission_labels(df, tables)
atomic_csv(labels, out / "admission_labels.csv.gz")
atomic_csv(audit["reconstruction"], out / "label_reconstruction.csv")
atomic_csv(audit["matched_titles"], out / "matched_titles.csv")
atomic_csv(audit["icd_mapping"], out / "icd_mapping.csv")
atomic_json(audit["summary"], out / "label_audit_summary.json")
S = df.groupby("subject_id")[C.TARGETS].first()
table1 = pd.DataFrame(
    {
        "target": C.TARGETS,
        "released_positive_radiographs": df[C.TARGETS].sum().values,
        "released_positive_patients": S.sum().values,
        "released_prevalence_%": (100 * df[C.TARGETS].mean()).round(2).values,
    }
)
table1 = table1.merge(audit["alignment"].drop(columns="released_positive_images"), on="target").merge(
    audit["semantics"].drop(columns="released_positive_patients"), on="target"
)
atomic_csv(table1, out / "table1_label_audit.csv")
print(audit["summary"])
print(table1.to_string(index=False))

root = Path(cfg["paths"]["mimic_cxr_jpg"])
cx_file = root / "mimic-cxr-2.0.0-chexpert.csv.gz"
if cx_file.exists():
    cx, cx_summary = chexpert_labels(
        df,
        pd.read_csv(cx_file),
        pd.read_csv(root / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "study_id"]),
    )
    atomic_csv(cx, out / "chexpert_labels.csv.gz")
    atomic_csv(cx_summary, out / "chexpert_label_summary.csv")
    print(cx_summary.to_string(index=False))
else:
    print(f"{cx_file} not found; CheXpert labels are not built")
