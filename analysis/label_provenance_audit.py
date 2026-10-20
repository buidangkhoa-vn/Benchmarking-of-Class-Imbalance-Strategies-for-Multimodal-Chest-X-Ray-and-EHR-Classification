"""Label provenance audit (paper Section III-C, Table 1 audit columns).

Reconstructs the released labels of x_ray_and_features_data.csv from MIMIC-IV v2.2 diagnosis codes and quantifies
(i) which admission each label comes from, (ii) how many radiographs carry another admission's labels, and
(iii) what the matched ICD titles mean. Same logic as STEP5 (cells "exact label rule" and "repaired labels").

Inputs (not distributed; PhysioNet credentialed access required):
  --csv     x_ray_and_features_data.csv
  --bq-dir  folder with the four BigQuery exports written by STEP4 (any prefix):
            *_bq_admissions.csv.gz, *_bq_patients.csv.gz, *_bq_diagnoses.csv.gz, *_bq_icd_titles.csv.gz
  --out     output folder for the audit tables (contains MIMIC-derived identifiers: keep private)
"""
import argparse, glob, json, os, warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
import numpy as np, pandas as pd

T = ["pneumonia", "tuberculosis", "bronchitis", "arthritis", "fracture", "lung_cancer", "scoliosis", "lung_infection"]
# reconstructed rule of the released labels (case-insensitive substring on ICD long titles of ONE admission)
LABEL_RULE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis", "arthritis": r"arthritis",
              "fracture": r"fracture", "lung_cancer": r"malignant neoplasm", "scoliosis": r"scoliosis", "lung_infection": r"infection\s*$"}
# clinically cleaned variant used for the C labels
CLEAN_INCLUDE = {"pneumonia": r"pneumonia", "tuberculosis": r"tuberculosis", "bronchitis": r"bronchitis", "arthritis": r"arthritis",
                 "fracture": r"fracture", "lung_cancer": r"^malignant neoplasm.*(lung|bronch)", "scoliosis": r"scoliosis",
                 "lung_infection": r"(infection|abscess|empyema).*(lung|pulmon|respirat)|(lung|pulmon).*(infection|abscess)"}
NONDIAG = r"history|screening|vaccinat|inoculat|tuberculin|exposure|contact with|carrier|encounter|observation"
NEGWORD = {"pneumonia": "pneumonia", "tuberculosis": "tuberculosis", "bronchitis": "bronchitis", "arthritis": "arthritis",
           "fracture": "fracture", "lung_cancer": "neoplasm", "scoliosis": "scoliosis", "lung_infection": "infection"}

def one(bq_dir, kind):
    f = sorted(glob.glob(os.path.join(bq_dir, f"*_bq_{kind}.csv.gz")))
    assert f, f"missing *_bq_{kind}.csv.gz in {bq_dir}"
    return f[0]

def main(a):
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, low_memory=False)
    df["dataset_position"] = np.arange(len(df))
    S = df.groupby("subject_id")[T].first()
    adm = pd.read_csv(one(a.bq_dir, "admissions"), parse_dates=["admittime"])
    pts = pd.read_csv(one(a.bq_dir, "patients")).drop_duplicates("subject_id").set_index("subject_id")
    dx = pd.read_csv(one(a.bq_dir, "diagnoses"), dtype={"icd_code": str})
    tit = pd.read_csv(one(a.bq_dir, "icd_titles"), dtype={"icd_code": str})
    dx["icd_code"] = dx.icd_code.str.strip(); tit["icd_code"] = tit.icd_code.str.strip()
    dx = dx.merge(tit.drop_duplicates(["icd_version", "icd_code"]), on=["icd_code", "icd_version"], how="left")
    ttl = dx.long_title.fillna("").str.strip()
    nondiag = ttl.str.contains(NONDIAG, case=False, regex=True)
    for t in T:
        dx[t] = ttl.str.contains(LABEL_RULE[t], case=False, regex=True)
        dx[f"c_{t}"] = (ttl.str.contains(CLEAN_INCLUDE[t], case=False, regex=True) & ~nondiag
                        & ~ttl.str.contains(r"without[^;]*" + NEGWORD[t], case=False, regex=True))
    A = dx.groupby("hadm_id")[T].any().astype(np.int8)
    C = dx.groupby("hadm_id")[[f"c_{t}" for t in T]].any().astype(np.int8); C.columns = T
    vec = lambda M, h: M.reindex(list(h)).fillna(0).astype(np.int8).values
    # 1) which admission reproduces the released labels?
    last_h = df.groupby("subject_id").hadm_id.last(); first_h = df.groupby("subject_id").hadm_id.first()
    Ssub = S.loc[last_h.index, T].values
    anyc = pd.DataFrame(vec(A, df.hadm_id), columns=T).assign(subject_id=df.subject_id.values).groupby("subject_id").max()
    admt = adm.drop_duplicates("hadm_id").set_index("hadm_id").admittime
    age_last = pts.anchor_age.reindex(last_h.index).values + (pd.to_datetime(admt.reindex(last_h.values).values).year - pts.anchor_year.reindex(last_h.index).values)
    summary = {"subjects": len(last_h),
               "exact_all8_last_row_admission": int((vec(A, last_h.values) == Ssub).all(1).sum()),
               "exact_all8_first_row_admission": int((vec(A, first_h.values) == Ssub).all(1).sum()),
               "exact_all8_any_cohort_admission": int((anyc.loc[last_h.index].values == Ssub).all(1).sum()),
               "age_equals_age_at_last_row_admission": float((age_last == df.groupby("subject_id").age.first().loc[last_h.index].values).mean())}
    # 2) admission misalignment per radiograph
    img = df[["dataset_position", "subject_id", "hadm_id", "age"] + T].copy()
    img["admittime"] = img.hadm_id.map(admt)
    own, own_c = vec(A, img.hadm_id), vec(C, img.hadm_id)
    src_h = img.subject_id.map(last_h); src_t = src_h.map(admt)
    img["label_source"] = np.select([img.hadm_id.values == src_h.values, (src_t < img.admittime).values, (src_t > img.admittime).values],
                                    ["own", "earlier", "later"], "same_time")
    age_own = pts.anchor_age.reindex(img.subject_id).values + (img.admittime.dt.year.values - pts.anchor_year.reindex(img.subject_id).values)
    rel = img[T].values
    summary.update({"label_source_share": img.label_source.value_counts(normalize=True).round(4).to_dict(),
                    "radiographs_with_any_label_differing_from_own_admission": int((rel != own).any(1).sum()),
                    "frac_age_differs_from_own_admission": round(float((img.age.values != age_own).mean()), 4)})
    corr = []
    for j, t in enumerate(T):
        y, o = rel[:, j], own[:, j]
        corr.append({"target": t, "released_pos_radiographs": int(y.sum()), "released_pos_patients": int(S[t].sum()),
                     "released_prev_%": round(100 * y.mean(), 2),
                     "released_pos_own_neg_%": round(100 * ((y == 1) & (o == 0)).sum() / max(1, y.sum()), 1),
                     "released_pos_from_later_adm_%": round(100 * ((y == 1) & (o == 0) & (img.label_source.values == "later")).sum() / max(1, y.sum()), 1),
                     "clean_prev_%": round(100 * own_c[:, j].mean(), 2)})
    # 3) semantics: positive patients whose source admission has a clinically valid (clean) code
    src_c = vec(C, last_h.values)
    for j, r in enumerate(corr):
        pos = Ssub[:, j] == 1
        r["valid_pos_patients_%"] = round(100 * src_c[pos, j].mean(), 1)
    CORR = pd.DataFrame(corr)
    CORR.to_csv(os.path.join(a.out, "label_audit_table1.csv"), index=False)
    json.dump(summary, open(os.path.join(a.out, "label_audit_summary.json"), "w"), indent=2)
    pd.DataFrame({"dataset_position": img.dataset_position, "label_source": img.label_source,
                  **{f"adm_{t}": own[:, j] for j, t in enumerate(T)}, **{f"cln_{t}": own_c[:, j] for j, t in enumerate(T)}}
                 ).to_csv(os.path.join(a.out, "admission_aligned_labels.csv.gz"), index=False)
    print(json.dumps(summary, indent=2)); print(CORR.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--bq-dir", required=True); ap.add_argument("--out", required=True)
    main(ap.parse_args())
