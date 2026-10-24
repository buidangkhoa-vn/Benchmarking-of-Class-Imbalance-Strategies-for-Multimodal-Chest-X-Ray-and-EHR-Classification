# Leakage- and label-aware benchmarking of class-imbalance strategies for chest X-ray and EHR classification

Code for the paper *Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification*.

The pipeline audits a multimodal MIMIC-CXR / MIMIC-IV benchmark for patient leakage and label provenance, and re-benchmarks class-imbalance strategies (class weighting, random oversampling, SMOTE, type-aware SMOTE-NC, balanced ensembles) with several classifiers and representations (EHR variables, TorchXRayVision pathology scores, DenseNet and RAD-DINO embeddings, early fusion) under record-level, patient-level and calendar-time splits, with released, admission-aligned, clean and CheXpert labels.

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
|---|:---:|
| `x_ray_and_features_data.csv`: the analysed cohort (29,083 radiographs, 10,479 patients) from an earlier transfer-learning study ([cohort code](https://github.com/clemence-mottez/mimic_iv)) | all steps |
| MIMIC-CXR-JPG v2.1.0: `files/`, `mimic-cxr-2.0.0-metadata.csv.gz`, `mimic-cxr-2.0.0-chexpert.csv.gz` | 01, 03 |
| MIMIC-IV v2.2 `hosp`: `admissions`, `patients`, `diagnoses_icd`, `d_icd_diagnoses` (local CSV files or BigQuery) | 02 |
| Saved predictions of the initial record-split benchmark (optional) | 07 |

Copy `configs/default.yaml` to `configs/local.yaml` and set the paths. Every path, seed and setting is in this file.

## Repository structure

```
.
├── configs/
│   └── default.yaml                    # every path, seed, feature set, classifier and experiment
├── data/
│   └── README.md                       # expected input files
├── scripts/
│   ├── 01_extract_embeddings.py        # RAD-DINO and DenseNet embeddings, view positions
│   ├── 02_extract_mimic_tables.py      # MIMIC-IV hosp tables for the cohort's patients
│   ├── 03_build_labels.py              # label audit; A, C and CheXpert labels
│   ├── 04_make_splits.py               # record, patient and calendar-time folds
│   ├── 05_run_experiments.py           # model fits and baselines
│   ├── 06_analyze.py                   # tables, statistical tests and figures
│   └── 07_initial_run_diagnostics.py   # diagnostics of the initial benchmark
├── src/
│   └── cxr_ehr_imbalance/
│       ├── cohort.py                   # cohort columns, embeddings, patient table, feature sets
│       ├── config.py                   # configuration file and command-line interface
│       ├── io.py                       # atomic file writes
│       ├── labels.py                   # label rule, A and C labels, prior diagnoses, CheXpert labels
│       ├── splits.py                   # record, patient and calendar-time splits
│       ├── modeling.py                 # preprocessing, samplers, classifiers, thresholds, metrics
│       ├── runner.py                   # experiment runner and baselines
│       ├── stats.py                    # model selection, corrected t-test, Holm, bootstrap
│       └── report.py                   # tables and figures
├── tests/
│   └── test_core.py                    # unit tests on synthetic inputs (no MIMIC data)
├── pyproject.toml
├── requirements.txt                    # package versions used for the paper
└── LICENSE
```

## Pipeline

Run the scripts in order. Each takes `--config configs/local.yaml` and writes to `outputs/` (set in the config); finished parts are not refitted when a script is run again.

| Step | Script | Outputs in `outputs/` | Paper |
|:---:|---|---|:---:|
| 1 | `01_extract_embeddings.py` | `embeddings/`: RAD-DINO and DenseNet features, view positions, provenance check of the image columns | III-A, IV-C |
| 2 | `02_extract_mimic_tables.py` | `mimic/`: admissions, patients and diagnoses of the cohort's patients | III-C |
| 3 | `03_build_labels.py` | `labels/`: label audit (Table I), A and C labels, prior-diagnosis flags, ICD code mapping, CheXpert labels | III-B, III-C |
| 4 | `04_make_splits.py` | `splits/`: record folds, patient folds (seeds 42 and 43), calendar-time split | IV-A |
| 5 | `05_run_experiments.py` | `runs/<experiment>/`: results, baselines and skipped cells | IV, V |
| 6 | `06_analyze.py` | `analysis/`: Tables IV-VII, strategy tests, paired comparisons, Fig. 1-2 | V |
| 7 | `07_initial_run_diagnostics.py` | `analysis/`: diagnostics of the initial run (Table II) | III-D |

`05_run_experiments.py --experiment main --folds 0 --targets pneumonia` runs a small subset as a quick check.

## Paper tables and result files

Files in `outputs/analysis/`, written by `06_analyze.py` unless another step is given.

| Paper | Files |
|---|---|
| Table I | `outputs/labels/table1_label_audit.csv`, `outputs/labels/label_audit_summary.json`, `outputs/labels/icd_mapping.csv` (step 3) |
| Table II | `table2_initial_run.csv`, `initial_run_summary.json` (step 7) |
| Table IV, Section V-A | `table4_protocols.csv`, `table4_baselines.csv`, `P1_seen_unseen.csv`, `matched_P1_P2.csv` |
| Fig. 1 | `fig1_per_target_margins.pdf` |
| Fig. 2, Section V-B | `fig2_strategy_effects.pdf`, `fig2_strategy_effects.csv`, `strategy_effects_main.csv`, `strategy_effects_pooled_main.csv`, `best_classifier_per_strategy_main.csv`, `sampler_audit.csv` |
| Section V-C | `comparisons_main.csv`, `selected_classifiers_main.csv` |
| Table V, Section V-D | `table5_aligned_clean.csv`, `table5_baselines.csv`, `table5_P5_bootstrap.csv`, `table5_history.csv`, `comparisons_admission.csv`, `history_vs_prior_rule.csv`, `P1A_seen_unseen.csv`, `strategy_effects_admission.csv` |
| Table VI, Section V-E | `table6_chexpert.csv`, `table6_baselines.csv`, `P1CX_seen_unseen.csv`, `comparisons_chexpert_history.csv`, `strategy_effects_chexpert_history.csv` |
| Table VII, Section V-F | `tuning_*.csv`, `trees_*.csv`, `sensitivity_*.csv`, `frontal_*.csv`, `encoder_*.csv`, `joint_tuning_*.csv`, `selection_over_folds_vs_within_fold.csv`, `threshold_diagnostics.csv`, `rare_excluded_*.csv` |

## Protocols

Pat.: patients separated across training, validation and test. Label, EHR: taken from the radiograph's own admission (✓) or from one admission per patient (–).

| Protocol | Folds (rows) | Labels | Pat. | Label | EHR | Experiment |
|---|---|---|:---:|:---:|:---:|---|
| P1 | record | released | – | – | – | main, tuning, trees |
| P2 | patient | released | ✓ | – | – | main, tuning, trees, sensitivity |
| P3 | patient (one row per patient) | released | ✓ | – | – | main |
| P1-A | record | A | – | ✓ | – | admission |
| P4, P4-C | patient | A, C | ✓ | ✓ | – | admission, tuning, trees, sensitivity, densenet_admission |
| P5-R, P5-A | calendar-time split | released, A | ✓ | –, ✓ | – | admission |
| P1-CX, P2-CX | record / patient | CheXpert | –, ✓ | ✓ | – | chexpert_history |
| P4-H, P4-C-H | patient | A, C | ✓ | ✓ | history | chexpert_history |
| P2-al-A, P2-al-C | patient (aligned radiographs only) | A, C | ✓ | ✓ | same admission | tuning |
| P2-F, P4-F, P4-C-F | patient (frontal radiographs only) | released, A, C | ✓ | –, ✓, ✓ | – | frontal |
| P4-C-JT | patient | C | ✓ | ✓ | – | joint_tuning |

P4-H and P4-C-H replace the released EHR values by age and prior diagnoses known at imaging time. Under P2-al, the EHR values come from the radiograph's own admission, but laboratory and vital-sign values are not restricted to measurements taken before the radiograph.

Label sets (`src/cxr_ehr_imbalance/labels.py`):

* Released (R): the labels of the cohort file. A keyword match on the ICD long titles of one admission per patient reproduces them exactly for 10,477 of 10,479 patients.
* Admission-aligned (A): the same keyword rule applied to each radiograph's own admission.
* Clean (C): as A, without non-diagnostic titles (history, screening, exposure, titles starting with "Encounter for") and negated titles ("... without ... fracture"). ICD-10 injury titles such as "..., initial encounter for closed fracture" are kept. Lung cancer is restricted to primary lung or bronchus malignancy and lung infection to lung and respiratory infection titles, including lower respiratory infection.
* CheXpert (CX): report labels of each radiograph's study (U-zeros).

All labels are rule-defined. The rules are the constants `LABEL_RULE`, `CLEAN_INCLUDE`, `NONDIAG` and `NEGWORD` in `labels.py`, applied in this order:

1. R and A: an ICD long title counts for a target if it matches `LABEL_RULE` (case-insensitive).
2. C: a title counts if it matches `CLEAN_INCLUDE`, does not match `NONDIAG` and is not negated ("without" followed by `NEGWORD`). Exclusions take precedence over inclusions.
3. An admission is positive for a target if any of its titles counts.
4. Prior-diagnosis flags use only earlier admissions that ended before the radiograph's admission began.

Step 3 writes the resulting code-level mapping, `icd_mapping.csv` (ICD version, code and title, the released-rule and clean match, and why a released-rule match is not clean), and the titles matched by each definition, `matched_titles.csv`. Both are built from MIMIC-IV and are therefore not part of the repository.

Splits (`src/cxr_ehr_imbalance/splits.py`):

* 5-fold iterative multilabel stratification on the eight targets (`MultilabelStratifiedKFold`, seed 42; seed 43 for the second fold assignment). Record folds stratify radiographs; patient folds stratify patients, so a patient's radiographs stay in one partition.
* Inside each outer training set, 20% of the units (radiographs or patients, as in the outer split) form the inner validation set (`MultilabelStratifiedShuffleSplit`, seed 100 x seed + fold). Per fold: 64% inner training, 16% inner validation, 20% test.
* P5 orders patients by the estimated calendar year of their first radiograph and assigns the first 64% to training, the next 16% to validation and the last 20% to testing.

Scaling, imputation and PCA are fitted on the inner training set; thresholds and configurations are chosen on the inner validation set; test folds are used only for evaluation.

## Reproducibility

* Release `v1.0.0` is the code version that accompanies the paper.
* Package versions: `requirements.txt`. Seeds: `random_state` and `cv.seeds` in the config.
* The clean-label (C) and frontal-only experiments were run with these scripts. The remaining results were produced with an earlier notebook-based version of the same code (Google Colab, package versions as in `requirements.txt`); the scripts reproduce its labels, splits and baselines, and `06_analyze.py` applied to those results reproduces the tables and tests of the paper.
* Model selection (`stats.select`): for each protocol, target, representation and outer fold, the configuration (strategy, classifier) with the highest F2 on that fold's inner validation set is evaluated on that fold's test set. `select(..., within_fold=False)` instead keeps one configuration per target chosen on validation F2 averaged over the folds (robustness check). Ties are broken by grid order (strategies: none, class weighting, ROS, SMOTE, SMOTE-NC; classifiers in config order).
* Thresholds: the value in {0.01, ..., 0.99} that maximises inner-validation F2; ties go to higher recall, then higher precision, then the value closest to 0.5.
* Trees on PCA-reduced embeddings can vary slightly between runs (up to 0.007 macro F2 for a fixed configuration in our reruns).
* Statistics: one value per outer fold and the corrected resampled t-test (Nadeau & Bengio, 2003), see `src/cxr_ehr_imbalance/stats.py`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE`. This covers the code only; MIMIC-IV, MIMIC-CXR-JPG, RAD-DINO and TorchXRayVision remain under their own licenses and data use agreements.
