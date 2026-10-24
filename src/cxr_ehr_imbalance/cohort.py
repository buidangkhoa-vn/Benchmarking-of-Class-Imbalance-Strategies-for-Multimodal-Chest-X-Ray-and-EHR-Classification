from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "pneumonia",
    "tuberculosis",
    "bronchitis",
    "arthritis",
    "fracture",
    "lung_cancer",
    "scoliosis",
    "lung_infection",
]
CHEXPERT_TARGETS = {
    "cx_pneumonia": "Pneumonia",
    "cx_lung_opacity": "Lung Opacity",
    "cx_effusion": "Pleural Effusion",
    "cx_cardiomegaly": "Cardiomegaly",
    "cx_edema": "Edema",
    "cx_pneumothorax": "Pneumothorax",
    "cx_fracture": "Fracture",
    "cx_lung_lesion": "Lung Lesion",
}

# Columns "0".."17" of the cohort file are the 18 TorchXRayVision outputs. The seven heads listed here have no
# CheXpert training label and are constant zero, which leaves 11 pathology scores.
IMAGE_SCORES_UNUSED = ["2", "5", "6", "9", "11", "12", "13"]
IMAGE_SCORES = [str(i) for i in range(18) if str(i) not in IMAGE_SCORES_UNUSED]
EHR_CONT = ["age", "oxygen", "heart_rate", "temperature", "hemoglobin", "rbc", "wbc"]
EHR_BIN = ["diabete", "hiv", "smoke", "cough"]
EHR = EHR_CONT + EHR_BIN
PRIOR_DX_A = [f"hist_{t}" for t in TARGETS]
PRIOR_DX_C = [f"histcln_{t}" for t in TARGETS]

COLUMN_GROUPS = {
    "ehr_cont": EHR_CONT,
    "ehr_bin": EHR_BIN,
    "image_scores": IMAGE_SCORES,
    "prior_dx_a": PRIOR_DX_A,
    "prior_dx_c": PRIOR_DX_C,
}

LABEL_COLUMNS = {
    "released": TARGETS,
    "A": [f"adm_{t}" for t in TARGETS],
    "C": [f"cln_{t}" for t in TARGETS],
    "CX": list(CHEXPERT_TARGETS),
}

# Rule-based baseline of each label set: A and C, the target was coded in an earlier admission;
# CX, an earlier radiograph of the patient was positive.
PRIOR_COLUMNS = {"A": PRIOR_DX_A, "C": PRIOR_DX_C, "CX": [f"histcx_{t}" for t in CHEXPERT_TARGETS]}


def data_hash(df: pd.DataFrame) -> str:
    """Short SHA-256 fingerprint of a table, used to check that every run uses the same cohort file."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def load_cohort(path) -> pd.DataFrame:
    """Reads the cohort file and adds `dataset_position`, the row index used to join all derived tables."""
    df = pd.read_csv(path)
    df["dataset_position"] = np.arange(len(df))
    return df


def load_embedding(folder) -> tuple[np.ndarray, pd.DataFrame]:
    """Embedding matrix (emb.npy) and the matching radiograph identifiers (ids.csv) written by step 01."""
    folder = Path(folder)
    return np.load(folder / "emb.npy").astype(np.float32), pd.read_csv(folder / "ids.csv")


def attach_embedding(rec: pd.DataFrame, name: str, folder) -> list[str]:
    """Adds the columns <name>_0 ... <name>_<d-1> to `rec` in place, matched on dicom, and returns their names."""
    E, ids = load_embedding(folder)
    pos = rec.dicom.map(dict(zip(ids.dicom, range(len(ids)), strict=True)))
    if pos.isna().any():
        raise ValueError(f"{name}: {int(pos.isna().sum())} radiographs have no embedding")
    cols = [f"{name}_{j}" for j in range(E.shape[1])]
    rec[cols] = pd.DataFrame(E[pos.astype(int).values], columns=cols, index=rec.index)
    return cols


def patient_table(rec: pd.DataFrame, image_cols: list[str], label_cols: list[str]) -> pd.DataFrame:
    """One row per patient (protocol P3).

    Labels and EHR values are constant within a patient. Image features are averaged over the patient's frontal
    (AP/PA) radiographs, or over all of the patient's radiographs when none is frontal.
    """
    const = [c for c in label_cols + EHR if c in rec.columns]
    pat = rec.groupby("subject_id")[const].first()
    img_all = rec.groupby("subject_id")[image_cols].mean()
    img_front = rec[rec.frontal].groupby("subject_id")[image_cols].mean()
    img = img_all.copy()
    img.loc[img_front.index] = img_front
    return pat.join(img).reset_index()


def resolve_feature_set(spec: dict, embeddings: dict[str, list[str]]) -> dict:
    """Expands a feature-set entry of the config ({cont, emb, bin, pca}) into column lists."""

    def cols(x):
        out = []
        items = [x] if isinstance(x, str) else (x or [])
        for item in items:
            out += COLUMN_GROUPS.get(item, [item])
        return out

    emb = spec.get("emb")
    return {
        "cont": cols(spec.get("cont")),
        "emb": embeddings[emb] if emb else [],
        "bin": cols(spec.get("bin")),
        "pca": spec.get("pca", "default"),
    }
