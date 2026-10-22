# Class-imbalance strategies for chest X-ray + EHR classification under leakage- and label-aware evaluation

Code for *"Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification"* (MMMDA workshop, IEEE BigData 2026).

The notebooks and scripts here produced every number, table and figure in the paper. **The repository contains no data.**

## Data access

The project uses MIMIC-IV (v2.2) and MIMIC-CXR-JPG (v2.1.0), which hold clinical records and chest radiographs of patients treated at the Beth Israel Deaconess Medical Center.

> ⚠️ **Data access**
>
> These datasets cannot be downloaded publicly. To use them you need:
> * the PhysioNet credentialed health data training (CITI "Data or Specimens Only Research");
> * a credentialed PhysioNet account and a signed data use agreement for each database;
> * for the BigQuery steps, a Google account linked to that PhysioNet account.
>
> Follow the PhysioNet data use agreements. Do not upload MIMIC-derived files (labels, predictions, embeddings, split files with subject identifiers) to a public repository; `.gitignore` blocks the usual file types.

## Files

| Step | File | Purpose | Paper |
|---|---|---|---|
| 0 | `analysis/original_benchmark_diagnostics.py` | Checks on the initial record-split run from its saved predictions: patient overlap, exact EHR matches, lookup baseline, seen vs. unseen patients, late fusion | Sec. III-D |
| 1 | `notebooks/STEP1_NB2_embeddings_FULL.ipynb` (GPU) | Re-extracts the 18 TorchXRayVision outputs (provenance check), 1024-d DenseNet features and 768-d RAD-DINO embeddings for the 29,083 radiographs; view metadata | Sec. III-A, IV-C |
| 2 | `notebooks/STEP2_NB1_worker{0,1,2}.ipynb` (3 CPU sessions) | P1 record split, P2 patient-grouped, P3 patient-level: 7 strategies × 5 classifiers × 7 representations × 8 targets | Sec. IV, V-A–C |
| 3 | `notebooks/STEP3_NB1_FINAL_merge_analysis.ipynb` | Merges the worker outputs; baselines, tables, figures | Sec. V |
| 3b | `analysis/analyze_v2_merge.py`, `analysis/figs_v2.py` | Final P1–P3 analysis (15,680 fits), Wilcoxon + Holm tests, sampler audit; `figs_v2.py` draws draft figures that 5c and 9c replace | Sec. V-A–C |
| 4 | `notebooks/STEP4_v3_2026-10-19_P4P5_admission_temporal.ipynb` | Extracts MIMIC-IV `admissions`, `patients`, `diagnoses_icd`, `d_icd_diagnoses` for the cohort from BigQuery (`mimiciv_2_2_hosp`) | Sec. III-C |
| 4a | `notebooks/STEP4a_v3_..._BQ_extract_MAIN_ACCOUNT_only_if_needed.ipynb` | Optional: the same extraction run from another Google account | — |
| 5 | `notebooks/STEP5_v4_2026-10-19_admission_labels.ipynb` | Rebuilds the released labels from ICD codes, builds admission-aligned (A) and clean (C) labels; runs P1-A, P4, P4-C and the calendar-time split P5 | Sec. III-C, IV-E, V-D |
| 5b | `analysis/label_provenance_audit.py` | Stand-alone label audit from the BigQuery exports | Sec. III-C, Table I |
| 5c | `analysis/analyze_v4.py`, `analysis/fig2_v4.py` | A/C-label analysis, modality and strategy tests; per-target margin figure | Sec. V-A, V-D, Fig. 1 |
| 6 | `notebooks/STEP6_v5_2026-10-19_chexpert_labels.ipynb` | CheXpert report labels of the same radiographs (P1 vs. P2); history features (age + prior diagnoses) with A and C labels | Sec. IV-F, V-D, V-E |
| 6b | `analysis/analyze_v5.py` | CheXpert and history analysis | Sec. V-D, V-E |
| 7 | `notebooks/STEP7_v7_2026-10-22_ablation_hparams_aligned.ipynb` | Tuned classifiers (RF / Extra Trees / LightGBM depth and leaf grids, LR with C ∈ {1, 0.1, 0.01}) under P1, P2, P4, P4-C; fully aligned subset | Sec. V-F |
| 7b | `analysis/analyze_v7.py` | Analysis of step 7 | Sec. V-F |
| 8 | `notebooks/STEP8_v8_2026-10-22_pca_age_seed_brier.ipynb` | PCA size (32/64/128/full), age vs. prior-diagnosis features, a second CV seed (43), Brier scores; logistic regression | Sec. V-F |
| 8b | `analysis/analyze_v8.py` | Analysis of step 8, repeated-CV tests | Sec. V-F |
| 9 | `analysis/tables_sd_ci.py` | Mean ± SD, 95% CIs and Brier scores for the result tables (no refitting) | Tables IV–VI |
| 9b | `analysis/corrected_t_tests.py` | Corrected resampled t-tests (Nadeau & Bengio, 2003) for the main paired comparisons | Sec. V |
| 9c | `analysis/fig3_strategy_by_representation.py` | Strategy-effect figure, one panel per representation | Fig. 2 |
| 9d | `analysis/selection_and_timing_checks.py` | Per-fold configuration selection, results without the rarest targets, P5 overlap, timing of prior diagnoses | Sec. IV-E/F, V-F |

`src/` holds the Python source of each notebook and the scripts that build the notebooks from it:

| Builder | Notebooks |
|---|---|
| `python src/make_variants.py` | STEP1–3 |
| `python src/make_v3.py` | STEP4, STEP4a |
| `python src/make_v5.py` | STEP6 |
| `python src/to_ipynb.py src/nb_v4_src.py notebooks/STEP5_...ipynb` | STEP5 (same for `nb_v7_src.py` → STEP7, `nb_v8_src.py` → STEP8) |

## Inputs

* `x_ray_and_features_data.csv`: the cohort (29,083 radiographs × 48 columns, 10,479 patients), taken from an earlier transfer-learning study (cohort code: <https://github.com/clemence-mottez/mimic_iv>). Each notebook checks its hash and stops if the file differs; SHA-256 values are in `splits/MANIFEST_SHA256.txt`.
* MIMIC-CXR-JPG v2.1.0: images, `mimic-cxr-2.0.0-metadata.csv.gz`, `mimic-cxr-2.0.0-chexpert.csv.gz`.
* MIMIC-IV v2.2 `hosp` tables on BigQuery (`physionet-data.mimiciv_2_2_hosp`).
* `selected_predictions.csv.gz`: saved predictions of the initial record-split run (step 0 only).

## Running

We ran the notebooks on Google Colab Pro, with the data on Google Drive under `MyDrive/data_aug/`. The first code cell of each notebook is a control panel with the paths; edit it for another layout, and set `GCP_PROJECT` before STEP4/STEP6 (the authors' project id and account were replaced by placeholders). A notebook that is restarted skips finished tasks. Outputs go to a folder named after the step's version (`MMMDA_runs/v3_...`, `v4_...`, …, `v8_...`).

Order: STEP1 → STEP2 (three sessions in parallel) → STEP3 → STEP4 → STEP5 → STEP6 → STEP7 → STEP8.

The analysis scripts read the notebook outputs. Paths are set with environment variables:

```bash
export MMMDA_DATA_DIR=/path/to/data          # x_ray_and_features_data.csv and analysis/ from STEP3 (incl. manifests/)
export MMMDA_WORKER0_DIR=/path/to/worker0    # results/results_worker0_5125.csv (see Notes)
export MMMDA_OUT=out_v2 MMMDA_FIG_DIR=figures
python analysis/analyze_v2_merge.py && python analysis/figs_v2.py
export MMMDA_V4_ANALYSIS=/path/to/v4_2026-10-19_admlabels/v4_analysis MMMDA_V4_OUT=out_v4
python analysis/analyze_v4.py && python analysis/fig2_v4.py
export MMMDA_V5_ANALYSIS=/path/to/v5_2026-10-19_chexpert/v5_analysis MMMDA_V5_OUT=out_v5
python analysis/analyze_v5.py
export MMMDA_V7_ANALYSIS=/path/to/v7_2026-10-22_ablation/v7_analysis MMMDA_V7_OUT=out_v7
python analysis/analyze_v7.py
export MMMDA_V8_ANALYSIS=/path/to/v8_2026-10-22_sensitivity/v8_analysis MMMDA_V8_OUT=out_v8
python analysis/analyze_v8.py
python analysis/tables_sd_ci.py               # uses MMMDA_OUT, MMMDA_V4_ANALYSIS, MMMDA_V5_ANALYSIS
python analysis/corrected_t_tests.py
python analysis/fig3_strategy_by_representation.py
export MMMDA_V3_BQ=/path/to/v3_bq MMMDA_P5_MANIFEST=/path/to/v3_folds_P5_temporal.csv
python analysis/selection_and_timing_checks.py
python analysis/original_benchmark_diagnostics.py --csv x_ray_and_features_data.csv --preds selected_predictions.csv.gz
python analysis/label_provenance_audit.py --csv x_ray_and_features_data.csv --bq-dir /path/to/v3_bq --out audit_out
```

## Settings

* Folds: 5-fold iterative multilabel stratification, grouped by patient except in P1; seed 42 (STEP8 adds seed 43). Inner split 80/20 of each training fold, grouped the same way.
* Model selection: configuration with the highest mean inner-validation F2 over folds; decision threshold chosen on the inner validation set.
* SMOTE variants: `k_neighbors = min(5, n_minority − 1)`; global seed 42.
* Library versions are in `requirements.txt`; hyperparameters are listed in Sec. IV-D of the paper and in the control panel of each notebook.

## Notes

* **Splits.** Split files are not distributed because they list MIMIC subject identifiers. They are regenerated deterministically; compare them with `splits/MANIFEST_SHA256.txt`.
* **Worker 0 of STEP2.** Its per-task files did not finish syncing to Google Drive before the session closed, so the merge in STEP3 saw 10,555 of 15,680 fits. The other 5,125 rows come from worker 0's own `results_all.csv`, and `analyze_v2_merge.py` combines both. The predictions of those tasks were lost, so validation-weighted late fusion appears only as a limitation.
* **Determinism.** Linear and EHR-only models are deterministic. Tree models on PCA-reduced embeddings can vary between runs: up to 0.007 macro F2 for a fixed configuration and 0.012 when the variation changes which configuration is selected.
* **Labels.** `label_provenance_audit.py` reproduces the released labels for 10,477 of 10,479 patients with the rule described in the paper (keyword match on the ICD long titles of the patient's last-listed admission).
* **Language.** Comments, markdown cells and printed messages were translated from Vietnamese to English after the experiments were run. The code itself was not changed: the Python token stream of every source file and notebook code cell, excluding comments and strings, is identical to the version that was run.

## Citation

```
@inproceedings{mmmda2026imbalance,
  title     = {Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification},
  booktitle = {MMMDA Workshop, IEEE International Conference on Big Data},
  year      = {2026}
}
```

## License

To be chosen by the authors.
