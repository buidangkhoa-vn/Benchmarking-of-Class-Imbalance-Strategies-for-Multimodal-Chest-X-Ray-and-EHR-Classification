# Leakage- and label-aware benchmarking of class-imbalance strategies for chest X-ray and EHR classification

Code for the paper *Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification* (MMMDA workshop, IEEE BigData 2026).

The pipeline compares imbalance strategies (class weighting, random oversampling, SMOTE, type-aware SMOTE-NC, balanced ensembles) with several classifiers and representations (EHR variables, TorchXRayVision pathology scores, DenseNet and RAD-DINO embeddings, early fusion) under record-level, patient-level and calendar-time splits, with released, admission-aligned, clean and CheXpert labels.

## Data access

The repository contains no data. It uses MIMIC-IV (v2.2) and MIMIC-CXR-JPG (v2.1.0), which require:

* the PhysioNet credentialed health data training (CITI "Data or Specimens Only Research");
* a credentialed PhysioNet account and a signed data use agreement for each database.

Please follow the PhysioNet data use agreements and do not publish MIMIC-derived files (labels, predictions, embeddings, split files with subject identifiers). `.gitignore` excludes `outputs/` and the usual data formats.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions used for the paper
pip install -e .
```

Python 3.10 or later. Step 01 needs PyTorch, TorchXRayVision and Transformers (a GPU is recommended); the other steps run on CPU.

## Inputs

| Input | Used by |
|---|---|
| `x_ray_and_features_data.csv`: the analysed cohort (29,083 radiographs, 10,479 patients) from an earlier transfer-learning study ([cohort code](https://github.com/clemence-mottez/mimic_iv)) | all steps |
| MIMIC-CXR-JPG v2.1.0: `files/`, `mimic-cxr-2.0.0-metadata.csv.gz`, `mimic-cxr-2.0.0-chexpert.csv.gz` | 01, 03 |
| MIMIC-IV v2.2 `hosp`: `admissions`, `patients`, `diagnoses_icd`, `d_icd_diagnoses` (local CSV files or BigQuery) | 02 |
| Saved predictions of the initial record-split benchmark (optional) | 07 |

Copy `configs/default.yaml` to `configs/local.yaml` and set the paths. Every path, seed and setting is in this file.

## Pipeline

Run the scripts in order; each takes `--config configs/local.yaml`. Outputs go to `outputs/` (set in the config).

| Script | Output | Paper |
|---|---|---|
| `scripts/01_extract_embeddings.py` | RAD-DINO and DenseNet embeddings, view positions, provenance check of the 18 image columns | III-A, IV-C |
| `scripts/02_extract_mimic_tables.py` | the four MIMIC-IV tables for the cohort's patients | III-C |
| `scripts/03_build_labels.py` | label reconstruction and audit (Table I), admission-aligned (A) and clean (C) labels, prior-diagnosis flags, own-admission age, CheXpert labels | III-B, III-C |
| `scripts/04_make_splits.py` | record folds, patient folds (seeds 42 and 43), calendar-time split | IV-A |
| `scripts/05_run_experiments.py` | model fits and baselines for the experiments `main`, `admission`, `chexpert_history`, `tuning`, `trees`, `sensitivity` | IV, V |
| `scripts/06_analyze.py` | Tables IV-VI, strategy tests, paired comparisons, robustness checks, Fig. 1-2 | V |
| `scripts/07_initial_run_diagnostics.py` | diagnostics of the initial benchmark from its saved predictions (Table II) | III-D |

`05_run_experiments.py --experiment main --folds 0 --targets pneumonia` runs a small subset as a quick check. Finished parts are not refitted when a script is run again.

## Protocols

| Protocol | Split | Labels | Experiment |
|---|---|---|---|
| P1 | record folds | released | main, tuning, trees |
| P2 | patient folds | released | main, tuning, trees, sensitivity |
| P3 | patient folds, one row per patient | released | main |
| P1-A | record folds | A | admission |
| P4, P4-C | patient folds | A, C | admission, tuning, trees, sensitivity |
| P5-R, P5-A | calendar-time split | released, A | admission |
| P1-CX, P2-CX | record / patient folds | CheXpert | chexpert_history |
| P4-H, P4-C-H | patient folds, time-aligned history features | A, C | chexpert_history |
| P2-al-A, P2-al-C | patient folds restricted to aligned radiographs | A, C | tuning |

Splits (`src/cxr_ehr_imbalance/splits.py`):

* 5-fold iterative multilabel stratification on the eight targets (`MultilabelStratifiedKFold`, seed 42; seed 43 for the second fold assignment). Record folds stratify radiographs; patient folds stratify patients, so a patient's radiographs stay in one partition.
* Inside each outer training set, 20% of the units (radiographs or patients, as in the outer split) form the inner validation set (`MultilabelStratifiedShuffleSplit`, seed 100 x seed + fold). Per fold: 64% inner training, 16% inner validation, 20% test.
* P5 orders patients by the estimated calendar year of their first radiograph and assigns the first 64% to training, the next 16% to validation and the last 20% to testing.

Scaling, imputation and PCA are fitted on the inner training set; thresholds and configurations are chosen on the inner validation set; test folds are used only for evaluation.

## Reproducibility

* Package versions: `requirements.txt`. Seeds: `random_state` and `cv.seeds` in the config.
* The published numbers come from an earlier notebook version of this code (Google Colab, package versions as in `requirements.txt`). The scripts give the same labels, splits and baselines, and `06_analyze.py` applied to the results of those runs reproduces the tables and tests of the paper.
* Model selection breaks ties in inner-validation F2 by grid order (strategies: none, class weighting, ROS, SMOTE, SMOTE-NC; classifiers in config order).
* Trees on PCA-reduced embeddings can vary slightly between runs (up to 0.007 macro F2 for a fixed configuration in our reruns).
* Statistics: one value per outer fold and the corrected resampled t-test (Nadeau & Bengio, 2003), see `src/cxr_ehr_imbalance/stats.py`.

## Repository structure

```
configs/default.yaml          paths, splits, feature sets, classifiers, experiments
data/README.md                expected input files
scripts/01_...07_*.py         pipeline steps
src/cxr_ehr_imbalance/
  cohort.py                   cohort columns, embeddings, patient-level table, feature sets
  labels.py                   label rule, A/C labels, prior diagnoses, CheXpert labels
  splits.py                   record, patient and calendar-time splits
  modeling.py                 preprocessing, samplers, classifiers, thresholds, metrics
  runner.py                   experiment runner and baselines
  stats.py                    selection, corrected t-test, Holm, bootstrap
  report.py                   tables and figures
```

## Citation

```
@inproceedings{mmmda2026imbalance,
  title     = {Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification},
  booktitle = {MMMDA Workshop, IEEE International Conference on Big Data},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`. This covers the code only; MIMIC-IV, MIMIC-CXR-JPG, RAD-DINO and TorchXRayVision remain under their own licenses and data use agreements.
