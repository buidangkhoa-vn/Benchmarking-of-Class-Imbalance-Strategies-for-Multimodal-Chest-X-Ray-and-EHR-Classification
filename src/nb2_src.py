# %% [markdown]
# # MMMDA NB2 — Strong image embeddings for the 29,083 radiographs (GPU, optional)
#
# **Output** (used by NB1 through `EMBEDDINGS`):
# `{EMB_ROOT}/raddino/emb.npy + ids.csv`, `{EMB_ROOT}/xrv_chex/emb.npy + ids.csv` (1024-d DenseNet features),
# plus `xrv_chex/logits18.npy` to verify which TorchXRayVision weights produced the CSV's 18 image features.
#
# **Needs**: MIMIC-CXR-JPG access (PhysioNet credentialed) and `mimic-cxr-2.0.0-metadata.csv.gz`
# (maps dicom_id → study_id, ViewPosition). Image source, choose one in the control panel:
# - `"drive"`: a folder on Drive with the MIMIC-CXR-JPG layout `files/pXX/pXXXXXXXX/sXXXXXXXX/<dicom>.jpg`
#   (a resized copy is fine — it is the fastest option);
# - `"gcs"`: the PhysioNet Google Cloud bucket (your Google account must be linked to PhysioNet and granted
#   GCS access; requester-pays → needs a GCP project id). Fast (~10–20 min for 29k images);
# - `"physionet"`: direct HTTPS download with your PhysioNet username/password (slow; multithreaded).
#
# Run `RUN_MODE="preflight"` first (256 images: demo images, timing, provenance check), then `"full"`.
# Everything is resume-safe (per-shard files); re-run after a disconnect.

# %%
# ============================== CELL 1 — CONTROL PANEL ==============================
RUN_MODE         = "preflight"          # "preflight" (256 images) | "full"
DRIVE_ROOT       = "/content/drive/MyDrive/data_aug"
CSV_PATH         = f"{DRIVE_ROOT}/x_ray_and_features_data.csv"
# Folder "mimic-cxr-jpg-2.1.0.physionet.org" (shared with you on Drive). None = auto-detect after mounting.
# If auto-detect fails: drive.google.com → "Shared with me" → right-click the folder → Organize → Add shortcut → My Drive.
MIMIC_JPG_ROOT   = None                  # or e.g. "/content/drive/MyDrive/mimic-cxr-jpg-2.1.0.physionet.org"
IMAGE_SOURCE     = "drive"               # "drive" (the shared folder above) | "gcs" | "physionet"
GCS_BUCKET       = "gs://mimic-cxr-jpg-2.1.0.physionet.org"  # only for "gcs"
GCP_PROJECT      = ""                    # only for "gcs" (requester-pays billing project)
PHYSIONET_URL    = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"   # only for "physionet"
LOCAL_IMG        = "/content/cxr_jpg"    # flat local copy: <dicom>.jpg
EMB_ROOT         = f"{DRIVE_ROOT}/embeddings"
MODELS           = ["xrv_chex", "raddino"]   # add "xrv_all" to test the other TorchXRayVision weights
BATCH            = 64
NUM_WORKERS      = 8
SHARD            = 2048
DL_THREADS       = 32                    # download threads (I/O only)

# %%
# ============================== CELL 2 — INSTALL + MOUNT + GPU ==============================
import os, sys, subprocess, importlib
for mod, pkg in [("torchxrayvision", "torchxrayvision"), ("transformers", "transformers")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
from google.colab import drive
drive.mount("/content/drive")
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", "| cores:", os.cpu_count())
if DEV != "cuda":
    print("WARNING: no GPU — switch runtime to a GPU (L4/A100) for the full run")

import glob
def find_mimic_root():
    name = "mimic-cxr-jpg-2.1.0.physionet.org"
    pats = [f"/content/drive/MyDrive/{name}", f"/content/drive/MyDrive/*/{name}", f"/content/drive/MyDrive/*/*/{name}",
            f"/content/drive/.shortcut-targets-by-id/*/{name}", f"/content/drive/.shortcut-targets-by-id/*/*/{name}",
            f"/content/drive/Shareddrives/*/{name}", f"/content/drive/Shareddrives/*/*/{name}"]
    for p in pats:
        for c in sorted(glob.glob(p)):
            if os.path.exists(f"{c}/mimic-cxr-2.0.0-metadata.csv.gz"):
                return c
    return None

if MIMIC_JPG_ROOT is None:
    MIMIC_JPG_ROOT = find_mimic_root()
if MIMIC_JPG_ROOT is None or not os.path.exists(f"{MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"):
    raise FileNotFoundError(
        "Không thấy thư mục mimic-cxr-jpg-2.1.0.physionet.org trong Drive đã mount.\n"
        "Mở drive.google.com → 'Shared with me' → chuột phải thư mục mimic-cxr-jpg-2.1.0.physionet.org → "
        "Organize → Add shortcut → My Drive, rồi chạy lại cell này (hoặc gán MIMIC_JPG_ROOT bằng tay).")
META_PATH = f"{MIMIC_JPG_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
IMAGE_ROOT_DRIVE = f"{MIMIC_JPG_ROOT}/files"
print("MIMIC_JPG_ROOT  :", MIMIC_JPG_ROOT)
print("metadata exists :", os.path.exists(META_PATH), "| files/ exists:", os.path.isdir(IMAGE_ROOT_DRIVE))

# %%
# ============================== CELL 3 — DICOM LIST + METADATA ==============================
import json, time, glob, shutil
from pathlib import Path
import numpy as np, pandas as pd
from tqdm.auto import tqdm

df = pd.read_csv(CSV_PATH, usecols=["dicom", "subject_id", "StudyDate", "pneumonia", "lung_cancer", "fracture"] +
                 [str(i) for i in range(18)])
assert len(df) == 29083 and df.dicom.is_unique
meta = pd.read_csv(META_PATH, usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
meta = meta.rename(columns={"dicom_id": "dicom", "subject_id": "subject_id_meta"})
m = df.merge(meta, on="dicom", how="left")
missing = m.study_id.isna().sum()
print("dicoms without metadata:", missing)
assert missing == 0, "metadata does not cover every dicom — check META_PATH"
assert (m.subject_id == m.subject_id_meta).all()
m["relpath"] = ("p" + m.subject_id.astype(str).str[:2] + "/p" + m.subject_id.astype(str) + "/s" +
                m.study_id.astype(int).astype(str) + "/" + m.dicom + ".jpg")
view = m.ViewPosition.fillna("NA").value_counts()
print("ViewPosition of the 29,083 benchmark radiographs:\n", view)
Path(EMB_ROOT).mkdir(parents=True, exist_ok=True)
_meta_path = f"{EMB_ROOT}/dicom_meta.csv"
if not (os.path.exists(_meta_path) and len(pd.read_csv(_meta_path, usecols=["dicom"])) == len(m)):
    m[["dicom", "subject_id", "study_id", "ViewPosition", "relpath"]].to_csv(_meta_path + ".tmp", index=False)
    os.replace(_meta_path + ".tmp", _meta_path)   # atomic: NB1 may read this file concurrently
view.rename("n").to_frame().to_csv(f"{EMB_ROOT}/view_position_counts.csv")

work = m if RUN_MODE == "full" else m.sample(256, random_state=0)
print("images to process:", len(work))

# %%
# ============================== CELL 4 — FETCH IMAGES TO LOCAL DISK (I/O threads) ==============================
from concurrent.futures import ThreadPoolExecutor, as_completed
Path(LOCAL_IMG).mkdir(parents=True, exist_ok=True)
todo = [r for r in work[["dicom", "relpath"]].itertuples(index=False) if not os.path.exists(f"{LOCAL_IMG}/{r.dicom}.jpg")]
print("already local:", len(work) - len(todo), "| to fetch:", len(todo))

def _done(dicom):
    return os.path.exists(f"{LOCAL_IMG}/{dicom}.jpg")

if todo and IMAGE_SOURCE == "drive":
    # probe: first images must exist on Drive; measure speed for an ETA
    t_probe = time.time(); sizes = []
    for r in todo[:20]:
        p = f"{IMAGE_ROOT_DRIVE}/{r.relpath}"
        assert os.path.exists(p), f"not found on Drive: {p}"
        sizes.append(os.path.getsize(p))
    print(f"probe OK: 20 files, mean {np.mean(sizes) / 1e6:.2f} MB, {time.time() - t_probe:.1f}s for stat()")
    def cp(r):
        src = f"{IMAGE_ROOT_DRIVE}/{r.relpath}"
        tmp = f"{LOCAL_IMG}/{r.dicom}.jpg.tmp"
        shutil.copyfile(src, tmp); os.replace(tmp, f"{LOCAL_IMG}/{r.dicom}.jpg")
        return r.dicom
    fails = []
    with ThreadPoolExecutor(DL_THREADS) as ex:
        futs = {ex.submit(cp, r): r for r in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="copy from Drive"):
            if f.exception() is not None:
                fails.append(futs[f].relpath)
    print("failed copies:", len(fails), fails[:5])

elif todo and IMAGE_SOURCE == "gcs":
    from google.colab import auth
    auth.authenticate_user()
    assert GCP_PROJECT, "set GCP_PROJECT (requester-pays)"
    test = subprocess.run(["gsutil", "-u", GCP_PROJECT, "ls", f"{GCS_BUCKET}/files/{todo[0].relpath}"], capture_output=True, text=True)
    print(test.stdout or test.stderr)
    lst = "/content/gcs_list.txt"
    with open(lst, "w") as f:
        for r in todo:
            f.write(f"{GCS_BUCKET}/files/{r.relpath}\n")
    t0 = time.time()
    subprocess.run(f"cat {lst} | gsutil -u {GCP_PROJECT} -m -q cp -I {LOCAL_IMG}/", shell=True, check=False)
    print("gsutil done in %.1f min" % ((time.time() - t0) / 60))

elif todo and IMAGE_SOURCE == "physionet":
    import requests, getpass
    user = input("PhysioNet username: "); pw = getpass.getpass("PhysioNet password: ")
    sess = requests.Session(); sess.auth = (user, pw)
    def dl(r):
        url = f"{PHYSIONET_URL}/files/{r.relpath}"
        tmp = f"{LOCAL_IMG}/{r.dicom}.jpg.tmp"
        with sess.get(url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        os.replace(tmp, f"{LOCAL_IMG}/{r.dicom}.jpg")
    fails = []
    with ThreadPoolExecutor(DL_THREADS) as ex:
        futs = {ex.submit(dl, r): r for r in todo}
        for f in tqdm(as_completed(futs), total=len(futs), desc="download PhysioNet"):
            if f.exception() is not None:
                fails.append(futs[f].relpath)
    print("failed downloads:", len(fails), fails[:5])

still = [d for d in work.dicom if not _done(d)]
print("missing local images:", len(still))
assert not still, "some images are missing — re-run this cell (resume-safe) or check access"

# %%
# ============================== CELL 5 — DEMO IMAGES (check before the long run) ==============================
import matplotlib.pyplot as plt
from PIL import Image
demo = work.head(6)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, r in zip(axes, demo.itertuples()):
    im = Image.open(f"{LOCAL_IMG}/{r.dicom}.jpg")
    ax.imshow(im, cmap="gray"); ax.axis("off")
    ax.set_title(f"{r.ViewPosition} | pna={r.pneumonia} lc={r.lung_cancer}\n{im.size[0]}x{im.size[1]}", fontsize=8)
plt.show()

# %%
# ============================== CELL 6 — EXTRACT EMBEDDINGS (resume-safe shards) ==============================
import torchxrayvision as xrv
import torchvision
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

class XRVSet(Dataset):
    def __init__(self, dicoms):
        self.d = list(dicoms)
        self.tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        a = np.asarray(Image.open(f"{LOCAL_IMG}/{self.d[i]}.jpg").convert("L"), dtype=np.float32)
        a = xrv.datasets.normalize(a, 255)[None, ...]
        return torch.from_numpy(self.tf(a).astype(np.float32))

class RadDinoSet(Dataset):
    def __init__(self, dicoms, processor):
        self.d = list(dicoms); self.p = processor
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        im = Image.open(f"{LOCAL_IMG}/{self.d[i]}.jpg").convert("RGB")
        return self.p(images=im, return_tensors="pt")["pixel_values"][0]

def run_model(name, dicoms):
    out = Path(EMB_ROOT) / (name if RUN_MODE == "full" else f"{name}_preflight"); (out / "shards").mkdir(parents=True, exist_ok=True)
    if name.startswith("xrv"):
        weights = {"xrv_chex": "densenet121-res224-chex", "xrv_all": "densenet121-res224-all"}[name]
        model = xrv.models.DenseNet(weights=weights).to(DEV).eval()
        make_ds = lambda ds: XRVSet(ds)
    else:
        proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        model = AutoModel.from_pretrained("microsoft/rad-dino").to(DEV).eval()
        make_ds = lambda ds: RadDinoSet(ds, proc)
    dicoms = list(dicoms); n_sh = (len(dicoms) + SHARD - 1) // SHARD
    t0 = time.time()
    for s in tqdm(range(n_sh), desc=name):
        fe = out / "shards" / f"{s:04d}_emb.npy"
        if fe.exists():
            continue
        ds = dicoms[s * SHARD:(s + 1) * SHARD]
        dl = DataLoader(make_ds(ds), batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=True)
        embs, logits = [], []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            for x in dl:
                x = x.to(DEV, non_blocking=True)
                if name.startswith("xrv"):
                    f = model.features2(x)
                    embs.append(f.float().cpu().numpy()); logits.append(model.classifier(f).float().cpu().numpy())
                else:
                    embs.append(model(pixel_values=x).pooler_output.float().cpu().numpy())
        E = np.concatenate(embs).astype(np.float16)
        np.save(out / "shards" / f"{s:04d}_emb.tmp.npy", E); os.replace(out / "shards" / f"{s:04d}_emb.tmp.npy", fe)
        if logits:
            L = np.concatenate(logits).astype(np.float32)
            np.save(out / "shards" / f"{s:04d}_logits.tmp.npy", L); os.replace(out / "shards" / f"{s:04d}_logits.tmp.npy", out / "shards" / f"{s:04d}_logits.npy")
        if s == 0:
            el = time.time() - t0
            print(f"{name}: first shard {el:.0f}s → ETA ≈ {el * n_sh / 60:.0f} min")
    E = np.concatenate([np.load(out / "shards" / f"{s:04d}_emb.npy") for s in range(n_sh)])
    assert len(E) == len(dicoms) and np.isfinite(E.astype(np.float32)).all()
    np.save(out / "emb.npy", E)
    pd.DataFrame({"dicom": dicoms}).to_csv(out / "ids.csv", index=False)
    if name.startswith("xrv"):
        L = np.concatenate([np.load(out / "shards" / f"{s:04d}_logits.npy") for s in range(n_sh)])
        np.save(out / "logits18.npy", L)
    json.dump({"model": name, "n": len(dicoms), "dim": int(E.shape[1]), "run_mode": RUN_MODE,
               "minutes": (time.time() - t0) / 60}, open(out / "meta.json", "w"), indent=2)
    print(f"{name}: saved {E.shape} to {out}")
    del model; torch.cuda.empty_cache()
    return out

outs = {name: run_model(name, work.dicom.values) for name in MODELS}

# %%
# ============================== CELL 7 — PROVENANCE CHECK (which weights made the CSV's 18 features?) ==============================
for name, out in outs.items():
    if not name.startswith("xrv"):
        continue
    L = np.load(out / "logits18.npy"); ids = pd.read_csv(out / "ids.csv")
    ref = work.set_index("dicom").loc[ids.dicom, [str(i) for i in range(18)]].values
    r = [np.corrcoef(L[:, j], ref[:, j])[0, 1] if ref[:, j].std() > 1e-12 and L[:, j].std() > 1e-12 else np.nan for j in range(18)]
    print(f"{name}: Pearson r per column (CSV vs recomputed logits):", np.round(r, 3))
    print(f"{name}: median r over non-constant columns = {np.nanmedian(r):.3f}; "
          f"columns constant in recomputed logits: {[j for j in range(18) if L[:, j].std() < 1e-12]}")
print("\nNext: in NB1 set EMBEDDINGS = {name: folder} for the folders printed above (full run only).")
