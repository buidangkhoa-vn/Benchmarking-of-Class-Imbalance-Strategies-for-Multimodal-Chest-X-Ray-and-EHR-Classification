# Image features

Step 01 (`scripts/01_extract_embeddings.py`) extracts two embeddings of the 29,083 radiographs. The features are derived from MIMIC-CXR and are not part of the repository. This file lists which features each result of the paper uses, how each feature set was extracted, and the checksums for verifying a re-extraction. *Not recorded* marks information that was not saved at extraction time and cannot be recovered.

## Feature sets

| Set | Encoder and output | Extraction | Device | Packages | `emb_sha256` |
|---|---|---|---|---|---|
| DenseNet, original | TorchXRayVision `densenet121-res224-chex`, pooled `features2` output (1024-d) | Colab notebook from which step 01 was derived, October 2026; same weights, preprocessing (`normalize(·, 255)`, `XRayCenterCrop`, `XRayResizer(224)`), fp16 autocast and float16 storage | CUDA GPU, model not recorded | Not recorded. TorchXRayVision was installed from PyPI without a version pin; PyTorch, torchvision, Pillow and NumPy were those preinstalled on Colab | Not recorded; the files are no longer available |
| DenseNet, re-extracted | as above | Step 01, 2026-11-04 | NVIDIA RTX PRO 6000 Blackwell Server Edition, fp16 autocast | Python 3.13.16, PyTorch 2.11.0+cu130, TorchXRayVision 1.5.7, NumPy 2.4.6 (recorded by the fits in the same notebook session); torchvision and Pillow not recorded | `108b4a30bfce1b4f718ffe331edec23eef2a5c8947eef46ecf3cbb530fccd322` |
| RAD-DINO, original | `microsoft/rad-dino`, `pooler_output` (768-d) | Same notebook, October 2026; checkpoint revision not pinned | CUDA GPU, model not recorded | Not recorded | Not recorded; the files are no longer available |
| RAD-DINO, re-extracted | `microsoft/rad-dino` at revision `110cbc18d5133582e320b43d53bf5c44e410c936` (`BitImageProcessor`), `pooler_output` (768-d) | Step 01, 2026-11-01 | NVIDIA RTX PRO 6000 Blackwell Server Edition, fp16 autocast | PyTorch 2.11.0+cu130, Transformers 5.21.0; Python, torchvision, Pillow and NumPy not recorded | `ceadc17c836401ba655ad7c2c424a148b35843bec1640bd459d63371203bd0fd` |

`emb_sha256` is the SHA-256 of the float16 matrix as stored in `emb.npy` (C order, rows in the order of the cohort file and of `ids.csv`):

```python
import hashlib, numpy as np
hashlib.sha256(np.load("emb.npy").tobytes()).hexdigest()
```

Both re-extractions used the cohort file with hash `da85519958e4bccb` (`cohort.data_hash`). For the re-extracted DenseNet, the 18 pathology outputs reproduce the cohort file's image columns (Pearson r ≥ 0.999999 for every non-constant column, `provenance_check.json`).

## Features behind each result

| Result | Experiment (`runs/`) | DenseNet | RAD-DINO |
|---|---|---|---|
| Table III (P2, P3), Fig. 1 | `main` | original | original |
| Table IV, P4 and P4-C: Image: DenseNet, Fusion: DenseNet | `densenet_admission` | original | – |
| Table IV, P4-H and P4-C-H: Fusion: DenseNet + history | `densenet_history` | re-extracted | – |
| Table IV, all RAD-DINO rows (P4, P4-C, P4-H, P4-C-H, P5) | `admission`, `chexpert_history` | – | original |
| Table V, first three contrasts | `densenet_admission`, `admission` | original | original |
| Table V, DenseNet + history − history | `densenet_history`, `chexpert_history` | re-extracted | – |
| Table VI, all rows except model seeds and repeated runs; Section V-E | `tuning`, `trees`, `sensitivity`, `frontal`, `joint_tuning`, `admission`, `chexpert_history` | – | original |
| Table VI, model seeds | `seed_variation`, `reference_rerun` | – | re-extracted |
| Table VI, repeated runs | `reference_rerun`, `densenet_rerun`, compared with `admission` and `densenet_admission` | re-extracted vs. original | re-extracted vs. original |

## Agreement between the feature sets

* RAD-DINO: the 1,728 default-seed fits with the re-extracted features (`reference_rerun`) are identical to the published fits.
* DenseNet: the default-seed image fits with the re-extracted features (`densenet_rerun`, 480 fits each under P4 and P4-C) reproduce the published macro F2 within 0.001 and macro AUROC within 0.004. Single fits are not identical: 36% (P4) and 46% (P4-C) of the per-target F2 values agree exactly, and the largest single-fit differences are 0.057 F2 and 0.225 AUROC (`analysis/densenet_rerun_difference.csv`). The fit environment also differed (NumPy 2.4.6 instead of 2.1.3), so the two causes cannot be separated.
* The model fits of every other experiment used the package versions in `requirements.txt`.

## Extracting again

Step 01 writes `extraction_meta.json` next to each `emb.npy`: checkpoint and revision, matrix shape, `emb_sha256`, a checksum of `ids.csv`, the cohort hash, the device, whether fp16 autocast was used, and the versions of Python, PyTorch, torchvision, TorchXRayVision or Transformers, NumPy, Pillow and CUDA. fp16 autocast is applied only on CUDA; on the CPU the forward pass runs in float32 and the features are not identical, so step 01 stops on the CPU unless `embeddings.allow_cpu: true`. Identical checksums are not guaranteed across GPU models or library versions either; compare a new extraction with the checksums above before reusing published results.
