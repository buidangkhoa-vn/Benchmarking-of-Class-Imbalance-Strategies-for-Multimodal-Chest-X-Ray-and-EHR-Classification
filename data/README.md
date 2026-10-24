# Data

Nothing in this folder is distributed. Access to MIMIC-IV and MIMIC-CXR-JPG requires a credentialed PhysioNet account.

Place the cohort file here (or set `paths.cohort_csv` in the config):

* `x_ray_and_features_data.csv`: 29,083 radiographs x 48 columns (`subject_id`, `hadm_id`, `dicom`, `StudyDate`, `admittime`,
  the eight targets, eleven EHR variables, and the 18 TorchXRayVision outputs `0`..`17`).

Other inputs are read from the paths in the config:

* `paths.mimic_cxr_jpg`: MIMIC-CXR-JPG v2.1.0 with `files/`, `mimic-cxr-2.0.0-metadata.csv.gz` and `mimic-cxr-2.0.0-chexpert.csv.gz`.
* `paths.mimic_iv_hosp`: MIMIC-IV v2.2 `hosp` module (`admissions.csv.gz`, `patients.csv.gz`, `diagnoses_icd.csv.gz`,
  `d_icd_diagnoses.csv.gz`), or set `mimic_iv.source: bigquery` to query `physionet-data.mimiciv_2_2_hosp`.
