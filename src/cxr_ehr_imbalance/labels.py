"""Label provenance: reconstruction of the released labels from MIMIC-IV codes, admission-aligned (A) and clean (C)
labels, prior-diagnosis flags, age at the radiograph's own admission, and CheXpert report labels."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .cohort import CHEXPERT_TARGETS, TARGETS

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

# Keyword rule that reproduces the released labels: case-insensitive match on the ICD long titles of one admission.
LABEL_RULE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis",
              "arthritis": r"arthritis", "fracture": r"fracture", "lung_cancer": r"malignant neoplasm",
              "scoliosis": r"scoliosis", "lung_infection": r"infection\s*$"}
# Clean definition (C): drop non-diagnostic and negated titles, restrict lung cancer and lung infection.
CLEAN_INCLUDE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis",
                 "arthritis": r"arthritis", "fracture": r"fracture", "lung_cancer": r"^malignant neoplasm.*(lung|bronch)",
                 "scoliosis": r"scoliosis",
                 "lung_infection": r"(infection|abscess|empyema).*(lung|pulmon|respirat)|(lung|pulmon).*(infection|abscess)"}
NONDIAG = r"history|screening|vaccinat|inoculat|tuberculin|exposure|contact with|carrier|encounter|observation"
NEGWORD = {"pneumonia": "pneumonia", "tuberculosis": "tuberculosis", "bronchitis": "bronchitis", "arthritis": "arthritis",
           "fracture": "fracture", "lung_cancer": "neoplasm", "scoliosis": "scoliosis", "lung_infection": "infection"}


def load_mimic_tables(folder) -> dict[str, pd.DataFrame]:
    adm = pd.read_csv(f"{folder}/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    pts = pd.read_csv(f"{folder}/patients.csv.gz")
    dxs = pd.read_csv(f"{folder}/diagnoses_icd.csv.gz", dtype={"icd_code": str})
    tit = pd.read_csv(f"{folder}/d_icd_diagnoses.csv.gz", dtype={"icd_code": str})
    dxs["icd_code"] = dxs.icd_code.str.strip(); tit["icd_code"] = tit.icd_code.str.strip()
    return {"admissions": adm, "patients": pts, "diagnoses": dxs, "titles": tit}


class AdmissionLabels:
    """Label vectors (rule R and clean rule C) for every admission of the cohort's patients."""

    def __init__(self, diagnoses: pd.DataFrame, titles: pd.DataFrame):
        dxt = diagnoses.merge(titles.drop_duplicates(["icd_version", "icd_code"]), on=["icd_code", "icd_version"], how="left")
        ttl = dxt.long_title.fillna("").str.strip()
        nondiag = ttl.str.contains(NONDIAG, case=False, regex=True)
        for t in TARGETS:
            dxt[t] = ttl.str.contains(LABEL_RULE[t], case=False, regex=True)
            dxt[f"c_{t}"] = (ttl.str.contains(CLEAN_INCLUDE[t], case=False, regex=True) & ~nondiag
                             & ~ttl.str.contains(r"without[^;]*" + NEGWORD[t], case=False, regex=True))
        self.dx = dxt
        self.nondiag = nondiag
        self.rule = dxt.groupby("hadm_id")[TARGETS].any().astype(np.int8)
        clean = dxt.groupby("hadm_id")[[f"c_{t}" for t in TARGETS]].any().astype(np.int8)
        clean.columns = TARGETS
        self.clean = clean

    def vec(self, hadms, clean=False) -> np.ndarray:
        tab = self.clean if clean else self.rule
        return tab.reindex(list(hadms)).fillna(0).astype(np.int8).values


def age_at_admission(patients: pd.DataFrame, admissions: pd.DataFrame, subject_ids, hadm_ids, fallback_dates=None):
    """MIMIC-IV age at an admission: anchor_age + (admission year - anchor_year)."""
    pt = patients.drop_duplicates("subject_id").set_index("subject_id")
    admt = admissions.drop_duplicates("hadm_id").set_index("hadm_id").admittime
    year = pd.to_datetime(admt.reindex(hadm_ids).values).year.values.astype(float)
    if fallback_dates is not None:
        year = np.where(np.isnan(year), pd.DatetimeIndex(pd.to_datetime(fallback_dates)).year.values, year)
    return pt.anchor_age.reindex(subject_ids).values + (year - pt.anchor_year.reindex(subject_ids).values)


def build_admission_labels(df: pd.DataFrame, tables: dict) -> tuple[pd.DataFrame, dict]:
    """Per-radiograph labels A and C, prior-diagnosis flags, label source and own-admission age, plus audit tables."""
    adm, pts = tables["admissions"], tables["patients"]
    df = df.assign(hadm_id=df.hadm_id.astype("int64"), subject_id=df.subject_id.astype("int64"))
    AL = AdmissionLabels(tables["diagnoses"], tables["titles"])
    S = df.groupby("subject_id")[TARGETS].first()
    last_h = df.groupby("subject_id").hadm_id.last()                 # the cohort file is not in chronological order
    first_h = df.groupby("subject_id").hadm_id.first()
    released = S.loc[last_h.index, TARGETS].values
    match_last = (AL.vec(last_h.values) == released).all(1)
    match_first = (AL.vec(first_h.values) == released).all(1)
    any_adm = pd.DataFrame(AL.vec(df.hadm_id), columns=TARGETS).assign(subject_id=df.subject_id.values).groupby("subject_id").max()
    match_any = (any_adm.loc[last_h.index].values == released).all(1)
    age_last = age_at_admission(pts, adm, last_h.index, last_h.values)
    audit = {"reconstruction": pd.DataFrame([{
                "target": t, "rule": LABEL_RULE[t],
                "agree_last_row_admission": float((AL.vec(last_h.values)[:, j] == released[:, j]).mean()),
                "agree_first_row_admission": float((AL.vec(first_h.values)[:, j] == released[:, j]).mean()),
                "agree_any_cohort_admission": float((any_adm.loc[last_h.index].values[:, j] == released[:, j]).mean())}
                for j, t in enumerate(TARGETS)]),
             "summary": {"subjects": len(last_h), "exact_all8_last_row": int(match_last.sum()),
                         "exact_all8_first_row": int(match_first.sum()), "exact_all8_any_cohort_admission": int(match_any.sum()),
                         "age_equals_age_at_last_row_admission": float((age_last == df.groupby("subject_id").age.first()
                                                                         .loc[last_h.index].values).mean())}}
    # titles matched by each rule
    rows = []
    for t in TARGETS:
        for flag, name in [(t, "rule"), (f"c_{t}", "clean")]:
            g = AL.dx[AL.dx[flag]].groupby("long_title").subject_id.nunique().sort_values(ascending=False)
            rows += [{"target": t, "definition": name, "long_title": x, "n_subjects": int(n)} for x, n in g.items()]
    audit["matched_titles"] = pd.DataFrame(rows)
    # semantics: positive patients whose source admission has a clean code
    src_clean = AL.vec(last_h.values, clean=True)
    sem = []
    for j, t in enumerate(TARGETS):
        pos = released[:, j] == 1
        m = AL.dx[AL.dx.hadm_id.isin(set(last_h.values[pos])) & AL.dx[t]]
        negated = m.long_title.fillna("").str.contains(r"without[^;]*" + NEGWORD[t], case=False, regex=True)
        only_nd = m.assign(nd=AL.nondiag.loc[m.index] | negated).groupby("hadm_id").nd.all()
        sem.append({"target": t, "released_positive_patients": int(pos.sum()),
                    "positive_with_clean_code_in_source_admission_%": round(100 * src_clean[pos, j].mean(), 1),
                    "positive_only_nondiagnostic_or_negated_%": round(100 * only_nd.mean(), 1)})
    audit["semantics"] = pd.DataFrame(sem)

    img = df[["dataset_position", "subject_id", "hadm_id", "dicom", "age"] + TARGETS].copy()
    admt = adm.drop_duplicates("hadm_id").set_index("hadm_id").admittime
    img["admittime"] = img.hadm_id.map(admt)
    missing = img.admittime.isna()
    img.loc[missing, "admittime"] = pd.to_datetime(df.loc[missing, "admittime"]).dt.normalize()
    own, own_c = AL.vec(img.hadm_id), AL.vec(img.hadm_id, clean=True)
    for j, t in enumerate(TARGETS):
        img[f"adm_{t}"] = own[:, j]; img[f"cln_{t}"] = own_c[:, j]
    allv = adm[["subject_id", "hadm_id", "admittime"]].drop_duplicates("hadm_id").reset_index(drop=True)
    allv = pd.concat([allv, pd.DataFrame(AL.vec(allv.hadm_id), columns=TARGETS),
                      pd.DataFrame(AL.vec(allv.hadm_id, clean=True), columns=[f"c_{t}" for t in TARGETS])], axis=1)
    for t in TARGETS:                       # coded in an admission that started before the radiograph's admission
        first_pos = allv[allv[t] == 1].groupby("subject_id").admittime.min()
        img[f"hist_{t}"] = (img.subject_id.map(first_pos) < img.admittime).fillna(False).astype(np.int8)
        first_pos_c = allv[allv[f"c_{t}"] == 1].groupby("subject_id").admittime.min()
        img[f"histcln_{t}"] = (img.subject_id.map(first_pos_c) < img.admittime).fillna(False).astype(np.int8)
    src_h = img.subject_id.map(last_h); src_t = src_h.map(admt)
    img["label_source"] = np.select([img.hadm_id.values == src_h.values, (src_t < img.admittime).values,
                                     (src_t > img.admittime).values], ["own", "earlier", "later"], "same_time")
    img["aligned"] = (img.label_source == "own").astype(np.int8)
    img["age_own"] = age_at_admission(pts, adm, img.subject_id.values, img.hadm_id.values, fallback_dates=df.admittime).astype(float)

    released_img = img[TARGETS].values
    rows = []
    for j, t in enumerate(TARGETS):
        y, o = released_img[:, j], own[:, j]
        rows.append({"target": t, "released_positive_images": int(y.sum()), "A_positive_images": int(o.sum()),
                     "A_prevalence_%": round(100 * o.mean(), 2), "C_positive_images": int(own_c[:, j].sum()),
                     "C_prevalence_%": round(100 * own_c[:, j].mean(), 2),
                     "A_positive_patients": int(img.groupby("subject_id")[f"adm_{t}"].max().sum()),
                     "C_positive_patients": int(img.groupby("subject_id")[f"cln_{t}"].max().sum()),
                     "released_positive_but_own_negative_%": round(100 * ((y == 1) & (o == 0)).sum() / max(1, y.sum()), 1),
                     "of_which_from_later_admission_%": round(100 * ((y == 1) & (o == 0) & (img.label_source.values == "later")).sum()
                                                             / max(1, y.sum()), 1)})
    audit["alignment"] = pd.DataFrame(rows)
    differ = (released_img != own).any(1)
    audit["summary"].update({"label_source_share": img.label_source.value_counts(normalize=True).round(4).to_dict(),
                             "images_with_any_label_differing_from_own_admission": int(differ.sum()),
                             "fraction_differing": round(float(differ.mean()), 4)})
    keep = (["dataset_position", "subject_id", "hadm_id", "dicom", "label_source", "aligned", "age_own"]
            + [f"{p}_{t}" for p in ("adm", "cln", "hist", "histcln") for t in TARGETS])
    return img[keep], audit


def chexpert_labels(df: pd.DataFrame, chexpert: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CheXpert report labels of each radiograph's study (U-zeros: only 1.0 is positive) and the prior-study rule."""
    meta = metadata.rename(columns={"dicom_id": "dicom"})[["dicom", "study_id"]]
    mp = df[["dataset_position", "subject_id", "dicom", "StudyDate"]].merge(meta, on="dicom", how="left")
    if mp.study_id.isna().any():
        raise ValueError(f"{int(mp.study_id.isna().sum())} radiographs have no study_id in the metadata")
    cols = list(CHEXPERT_TARGETS.values())
    lab = mp[["study_id"]].merge(chexpert[["study_id"] + cols].drop_duplicates("study_id"), on="study_id", how="left")
    out = mp[["dataset_position", "subject_id", "dicom", "StudyDate"]].copy()
    summary = []
    for t, col in CHEXPERT_TARGETS.items():
        v = lab[col].values
        out[t] = (v == 1.0).astype(np.int8)
        summary.append({"target": t, "chexpert": col, "positive_images": int((v == 1.0).sum()),
                        "uncertain_images": int((v == -1.0).sum()), "prevalence_%": round(100 * float((v == 1.0).mean()), 2),
                        "positive_patients": int(out.groupby("subject_id")[t].max().sum())})
    multi = out.groupby("subject_id").size() > 1
    varies = out[out.subject_id.isin(multi[multi].index)].groupby("subject_id")[list(CHEXPERT_TARGETS)].nunique().gt(1)
    for r in summary:
        r["multi_image_patients_with_varying_label_%"] = round(100 * float(varies[r["target"]].mean()), 1)
    out = out.sort_values(["subject_id", "StudyDate", "dataset_position"])
    for t in CHEXPERT_TARGETS:               # an earlier radiograph of the patient was positive
        out[f"histcx_{t}"] = out.groupby("subject_id")[t].transform(lambda s: s.cummax().shift(1)).fillna(0).astype(np.int8)
    out = out.sort_values("dataset_position").reset_index(drop=True).drop(columns="StudyDate")
    return out, pd.DataFrame(summary)
