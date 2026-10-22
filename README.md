# Leakage- and label-aware benchmark of class-imbalance strategies for CXR + EHR — code

Code for the paper *"Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification"* (MMMDA workshop, IEEE BigData 2026).

This repository contains the notebooks and scripts that produced every number, table and figure of the paper. **It contains no data.** All inputs are derived from MIMIC-IV and MIMIC-CXR-JPG and require PhysioNet credentialed access; do not commit derived files (predictions, labels, split manifests with subject identifiers) to a public repository.

## Data sources and access notice

This project relies on the MIMIC-IV (v2.2) and MIMIC-CXR-JPG (v2.1.0) databases, which contain clinical records and chest X-ray images of patients admitted to the Beth Israel Deaconess Medical Center.

> ⚠️ **Important notice on data access**
>
> The datasets used in this project are **not publicly downloadable**, and this repository contains **no data**. Access to MIMIC-IV and MIMIC-CXR-JPG requires:
> * completion of the official PhysioNet credentialed health data training (CITI "Data or Specimens Only Research");
> * an approved (credentialed) PhysioNet account and a signed data use agreement for each database;
> * for the BigQuery steps, a Google account linked to that PhysioNet account.
>
> All users must comply with the data use agreements and ethical guidelines defined by PhysioNet. Do not upload MIMIC-derived files (labels, predictions, embeddings, split manifests with subject identifiers) to public repositories; `.gitignore` blocks the common file types.

## What the code does

| Step | File | What it does | Paper |
|---|---|---|---|
| 0 | `analysis/original_benchmark_diagnostics.py` | Diagnostics of the initial record-split run from its saved predictions (patient overlap, EHR exact-match rate, no-learning lookup table, seen vs. unseen patients, late fusion) | Sec. III-D, table "Initial run, seen vs. unseen" |
| 1 | `notebooks/STEP1_NB2_embeddings_FULL.ipynb` (GPU) | Re-extracts the 18 TorchXRayVision (`densenet121-res224-chex`) outputs to verify provenance; extracts 1024-d DenseNet features and 768-d RAD-DINO CLS embeddings for the 29,083 radiographs; writes view metadata | Sec. III-A, IV-C |
| 2 | `notebooks/STEP2_NB1_worker{0,1,2}.ipynb` (3 CPU sessions) | P1 record split, P2 patient-grouped, P3 patient-level; 7 imbalance strategies × 5 classifiers × 7 representations × 8 targets; resume-safe, sharded by a stable hash | Sec. IV, V-A–C |
| 3 | `notebooks/STEP3_NB1_FINAL_merge_analysis.ipynb` | Merges the worker outputs, baselines, tables and figures | Sec. V |
| 3b | `analysis/analyze_v2_merge.py`, `analysis/figs_v2.py` | Final v2 analysis (15,680 fits): protocol table, P1 seen/unseen table, strategy tests (Wilcoxon + Holm), sampler audit, Figs. 1 and 3 | Sec. V-A–C |
| 4 | `notebooks/STEP4_v3_2026-10-19_P4P5_admission_temporal.ipynb` | Extracts MIMIC-IV `admissions`, `patients`, `diagnoses_icd`, `d_icd_diagnoses` for the cohort from BigQuery (auto-selects the dataset version matching the cohort; here `mimiciv_2_2_hosp`) | Sec. III-C |
| 4a | `notebooks/STEP4a_v3_..._BQ_extract_MAIN_ACCOUNT_only_if_needed.ipynb` | Optional: the same BigQuery extraction from a different Google account | — |
| 5 | `notebooks/STEP5_v4_2026-10-19_admission_labels.ipynb` | Reconstructs the released labels from ICD codes, audits them, builds admission-aligned (A) and clean (C) labels, runs P1-A, P4, P4-C and the calendar-time split P5 | Sec. III-C, IV-E, V-D |
| 5b | `analysis/label_provenance_audit.py` | Stand-alone label audit (audit columns of the cohort table) from the BigQuery exports | Sec. III-C, cohort table |
| 5c | `analysis/analyze_v4.py`, `analysis/fig2_v4.py` | Analysis with admission-aligned and clean labels (table "Admission-aligned (A) and clean (C) labels"), modality and strategy tests, Fig. 2 | Sec. V-D |
| 6 | `notebooks/STEP6_v5_2026-10-19_chexpert_labels.ipynb` | Second label source: MIMIC-CXR CheXpert report labels of the same radiographs (P1 vs. P2); time-aligned EHR-history features with admission-aligned and clean labels | Sec. IV-F, V-D, V-E |
| 6b | `analysis/analyze_v5.py` | CheXpert-label and EHR-history analysis (tables, paired tests, strategy effects) | Sec. V-D, V-E |
| 7 | `notebooks/STEP7_v7_2026-10-22_ablation_hparams_aligned.ipynb` | Hyperparameter ablation (regularised Random Forest / Extra Trees / LightGBM, logistic regression with C ∈ {1, 0.1, 0.01}) under P1, P2, P4, P4-C, and the fully aligned subset (radiographs whose own admission is the source of the released labels and EHR values) | Sec. V-F |
| 7b | `analysis/analyze_v7.py` | Ablation and aligned-subset analysis | Sec. V-F |
| 8 | `analysis/tables_sd_ci.py` | Mean ± SD and 95% CIs for the protocol tables (no refits) | Tables IV–VII |
| 8b | `analysis/selection_and_timing_checks.py` | Per-fold configuration selection, results without the rarest targets, P5 temporal overlap, timing of prior-diagnosis flags | Sec. IV-E/F, V-F |

`src/` holds the Python sources of all notebooks and the builders that generate them (`make_variants.py` → STEP1–3, `make_v3.py` → STEP4/4a, `make_v5.py` → STEP6; STEP5 and STEP7 are `to_ipynb.py nb_v4_src.py` and `to_ipynb.py nb_v7_src.py`). Running the builders regenerates the notebooks in `notebooks/` byte-for-byte.

## Inputs

* `x_ray_and_features_data.csv` — the analysed cohort (29,083 radiographs × 48 columns, 10,479 patients), inherited from a preceding transfer-learning study (cohort code: <https://github.com/clemence-mottez/mimic_iv>). Checksums are in `splits/MANIFEST_SHA256.txt`; every notebook stops if the file differs.
* MIMIC-CXR-JPG v2.1.0 (images, `mimic-cxr-2.0.0-metadata.csv.gz`, `mimic-cxr-2.0.0-chexpert.csv.gz`).
* MIMIC-IV v2.2 `hosp` tables on BigQuery (`physionet-data.mimiciv_2_2_hosp`), project of a PhysioNet-credentialed account.
* `selected_predictions.csv.gz` — saved validation/test predictions of the initial record-split run (step 0 only).

## Running

The notebooks were run on Google Colab (Pro) with the data on Google Drive under `MyDrive/data_aug/` and MIMIC-CXR-JPG under `MyDrive/.../mimic-cxr-jpg-2.1.0.physionet.org`. Edit the **control panel** (first code cell) of each notebook for other layouts. All notebooks are resume-safe (finished tasks are skipped after a restart) and write every output into a version-tagged folder (`MMMDA_runs/v3_...`, `v4_...`, `v5_...`). Fill `GCP_PROJECT` before running STEP4/STEP6 (placeholders replace the project id and account used by the authors; nothing else was changed).

Order: STEP1 → STEP2 (three sessions in parallel) → STEP3 → STEP4 → STEP5 → STEP6 → STEP7. The analysis scripts read the notebook outputs; paths are set with environment variables:

```bash
export MMMDA_DATA_DIR=/path/to/data          # x_ray_and_features_data.csv and analysis/ from STEP3 (incl. manifests/)
export MMMDA_WORKER0_DIR=/path/to/worker0    # results/results_worker0_5125.csv (see note below)
export MMMDA_OUT=out_v2 MMMDA_FIG_DIR=figures
python analysis/analyze_v2_merge.py && python analysis/figs_v2.py
export MMMDA_V4_ANALYSIS=/path/to/v4_2026-10-19_admlabels/v4_analysis MMMDA_V4_OUT=out_v4
python analysis/analyze_v4.py && python analysis/fig2_v4.py          # overwrites Fig. 2 with the final version
export MMMDA_V5_ANALYSIS=/path/to/v5_2026-10-19_chexpert/v5_analysis MMMDA_V5_OUT=out_v5
python analysis/analyze_v5.py
export MMMDA_V7_ANALYSIS=/path/to/v7_2026-10-22_ablation/v7_analysis MMMDA_V7_OUT=out_v7
python analysis/analyze_v7.py
python analysis/tables_sd_ci.py                                        # uses MMMDA_OUT, MMMDA_V4_ANALYSIS, MMMDA_V5_ANALYSIS
export MMMDA_V3_BQ=/path/to/v3_bq MMMDA_P5_MANIFEST=/path/to/v3_folds_P5_temporal.csv
python analysis/selection_and_timing_checks.py
python analysis/original_benchmark_diagnostics.py --csv x_ray_and_features_data.csv --preds selected_predictions.csv.gz
python analysis/label_provenance_audit.py --csv x_ray_and_features_data.csv --bq-dir /path/to/v3_bq --out audit_out
```

## Notes

* **Splits.** Fold manifests are generated deterministically (iterative multilabel stratification, seed 42) and are not distributed because they list MIMIC subject identifiers; compare regenerated files with `splits/MANIFEST_SHA256.txt`.
* **Worker 0 of STEP2.** Its per-task files did not finish syncing to Google Drive before the session closed, so the FINAL merge saw 10,555 of 15,680 fits. The remaining 5,125 rows were recovered from worker 0's own `results_all.csv`; `analyze_v2_merge.py` merges both (15,680/15,680). Predictions of those tasks were lost, which is why validation-weighted late fusion is reported only as a limitation.
* **Determinism.** Linear and EHR-only models are deterministic. Tree models on PCA-reduced embeddings can differ slightly between runs (we observed up to 0.012 macro F2 on the single calendar-time split).
* **Labels.** `label_provenance_audit.py` reproduces the released labels for 10,477 of 10,479 patients with the rule documented in the paper (keyword match on the ICD long titles of the patient's last-listed admission).

## Requirements

See `requirements.txt` (versions used on Colab for the main runs).

## Citation

```
@inproceedings{mmmda2026imbalance,
  title     = {Leakage- and Label-Aware Benchmarking of Class-Imbalance Strategies for Multimodal Chest X-Ray and EHR Classification},
  booktitle = {MMMDA Workshop, IEEE International Conference on Big Data},
  year      = {2026}
}
```

## License

To be chosen by the authors before making the repository public.
