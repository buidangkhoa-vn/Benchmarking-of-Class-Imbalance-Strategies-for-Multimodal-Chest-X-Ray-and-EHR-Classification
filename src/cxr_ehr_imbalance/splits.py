from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .cohort import TARGETS

# Salt of the hash that orders patients with the same estimated first year in the calendar-time split. It is part of
# the published split definition; changing it changes the split.
TEMPORAL_TIE_SALT = "v3"


def multilabel_folds(Y: np.ndarray, ids, id_name: str, seed: int, n_folds=5, inner_val_frac=0.2) -> pd.DataFrame:
    """Iterative multilabel stratification: `n_folds` outer folds and, inside each outer training set, an inner
    validation set (seed 100 * seed + fold). Returns one row per unit and fold with role train, val or test."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

    parts = []
    skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(Y)), Y)):
        inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed * 100 + k)
        itr, iva = next(inner.split(np.zeros(len(tr)), Y[tr]))
        role = np.empty(len(Y), dtype=object)
        role[te] = "test"
        role[tr[itr]] = "train"
        role[tr[iva]] = "val"
        parts.append(pd.DataFrame({id_name: ids, "fold": k, "role": role}))
    return pd.concat(parts, ignore_index=True)


def patient_folds(df: pd.DataFrame, seed: int, n_folds=5, inner_val_frac=0.2) -> pd.DataFrame:
    """Folds over patients (P2, P3 and the admission-aligned, clean and CheXpert variants)."""
    S = df.groupby("subject_id")[TARGETS].first()
    return multilabel_folds(S[TARGETS].values, S.index.values, "subject_id", seed, n_folds, inner_val_frac)


def record_folds(df: pd.DataFrame, seed: int, n_folds=5, inner_val_frac=0.2) -> pd.DataFrame:
    """Folds over radiographs regardless of the patient (P1, as in the initial benchmark)."""
    return multilabel_folds(
        df[TARGETS].values, df.dataset_position.values, "dataset_position", seed, n_folds, inner_val_frac
    )


def _tie_key(subject_id) -> int:
    return int(hashlib.md5(f"{TEMPORAL_TIE_SALT}|{subject_id}".encode()).hexdigest(), 16)


def temporal_split(df: pd.DataFrame, patients: pd.DataFrame, train=0.64, val=0.16) -> pd.DataFrame:
    """Calendar-time split over patients (P5).

    MIMIC shifts dates per patient; `anchor_year_group` gives the real three-year window of `anchor_year`. The
    estimated real year of a radiograph is the first year of that window + 1 + (study year - anchor_year). Patients are
    ordered by the estimated year of their first radiograph, ties broken by a fixed hash of the subject id, and the
    first `train` and next `val` fractions form the training and validation sets.
    """
    pt = patients.drop_duplicates("subject_id").set_index("subject_id")
    g0 = pt.anchor_year_group.astype(str).str.extract(r"(\d{4})")[0].astype(float)
    study_year = (df.StudyDate // 10000).astype(int)
    est_year = (df.subject_id.map(g0) + 1 + (study_year - df.subject_id.map(pt.anchor_year))).astype(float)
    first_year = est_year.groupby(df.subject_id).min()
    order = sorted(first_year.index, key=lambda s: (first_year[s], _tie_key(s)))
    n = len(order)
    n_tr, n_va = int(round(train * n)), int(round(val * n))
    role = pd.Series("test", index=order)
    role.iloc[:n_tr] = "train"
    role.iloc[n_tr : n_tr + n_va] = "val"
    return pd.DataFrame(
        {"subject_id": order, "est_first_year": first_year.reindex(order).values, "fold": 0, "role": role.values}
    )


def roles_for(table: pd.DataFrame, manifest: pd.DataFrame, fold: int) -> np.ndarray:
    """Role (train, val, test) of every row of `table` in one fold of a fold manifest."""
    m = manifest[manifest.fold == fold]
    if "dataset_position" in m.columns:
        return m.set_index("dataset_position").role.reindex(table.dataset_position).values
    return table.subject_id.map(m.set_index("subject_id").role).values


def split_indices(table: pd.DataFrame, roles: np.ndarray, grouped: bool) -> dict:
    """Row indices of the train, val and test partitions; rows without a role (outside a subset) are dropped.

    For patient-grouped splits, a patient found in two partitions raises an error. `seen_te` marks test rows of
    patients present in training or validation, `unseen_va` validation rows of patients absent from training.
    """
    idx = {r: np.where(roles == r)[0] for r in ("train", "val", "test")}
    sid = table.subject_id.values
    s_tr, s_va, s_te = (set(sid[idx[r]]) for r in ("train", "val", "test"))
    if grouped and ((s_tr & s_va) or (s_tr & s_te) or (s_va & s_te)):
        raise ValueError("a patient appears in two partitions of a patient-grouped split")
    idx["seen_te"] = np.isin(sid[idx["test"]], list(s_tr | s_va))
    idx["unseen_va"] = ~np.isin(sid[idx["val"]], list(s_tr))
    return idx
