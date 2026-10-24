import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxr_ehr_imbalance.config import cli  # noqa: E402

args, cfg = cli("Image embeddings and view positions of the cohort's radiographs.")
root = Path(cfg["paths"]["mimic_cxr_jpg"])
out = cfg["dirs"]["embeddings"]
E = cfg["embeddings"]
device = E["device"] if torch.cuda.is_available() else "cpu"

df = pd.read_csv(cfg["paths"]["cohort_csv"], usecols=["dicom", "subject_id"] + [str(i) for i in range(18)])
meta = pd.read_csv(
    root / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"]
)
m = df.merge(meta.rename(columns={"dicom_id": "dicom", "subject_id": "subject_id_meta"}), on="dicom", how="left")
m["path"] = [
    str(root / "files" / f"p{str(s)[:2]}" / f"p{s}" / f"s{int(st)}" / f"{d}.jpg")
    for s, st, d in zip(m.subject_id, m.study_id, m.dicom, strict=True)
]
m[["dicom", "subject_id", "study_id", "ViewPosition"]].to_csv(out / "dicom_meta.csv", index=False)
print("view positions:", m.ViewPosition.fillna("NA").value_counts().to_dict())


class Images(Dataset):
    """Radiograph files, loaded and transformed by the encoder's own preprocessing."""

    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(self.paths[i])


def densenet():
    """TorchXRayVision DenseNet-121 (CheXpert weights): pooled features and the 18 pathology outputs."""
    import torchvision
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device).eval()
    tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    def load(p):
        a = xrv.datasets.normalize(np.asarray(Image.open(p).convert("L"), dtype=np.float32), 255)[None, ...]
        return torch.from_numpy(tf(a).astype(np.float32))

    def forward(x):
        f = model.features2(x)
        return f, model.classifier(f)

    return load, forward


def raddino():
    """RAD-DINO (ViT-B/14): CLS embedding."""
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino").to(device).eval()

    def load(p):
        return proc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"][0]

    def forward(x):
        return model(pixel_values=x).pooler_output, None

    return load, forward


for name in E["models"]:
    load, forward = {"densenet": densenet, "raddino": raddino}[name]()
    loader = DataLoader(Images(m.path, load), batch_size=E["batch_size"], num_workers=E["num_workers"], pin_memory=True)
    embs, logits = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
        for x in tqdm(loader, desc=name):
            f, lg = forward(x.to(device, non_blocking=True))
            embs.append(f.float().cpu().numpy())
            if lg is not None:
                logits.append(lg.float().cpu().numpy())
    (out / name).mkdir(exist_ok=True)
    np.save(out / name / "emb.npy", np.concatenate(embs).astype(np.float16))
    m[["dicom"]].to_csv(out / name / "ids.csv", index=False)
    if logits:
        L = np.concatenate(logits)
        np.save(out / name / "logits18.npy", L)
        ref = m[[str(i) for i in range(18)]].values
        r = [
            float(np.corrcoef(L[:, j], ref[:, j])[0, 1]) if ref[:, j].std() > 1e-12 and L[:, j].std() > 1e-12 else None
            for j in range(18)
        ]
        with open(out / name / "provenance_check.json", "w") as f:
            json.dump({"pearson_r_per_column": r}, f, indent=2)
        print(f"{name}: Pearson r with the released image columns:", np.round([x or np.nan for x in r], 3))
    print(f"{name}: saved {len(m)} embeddings")
