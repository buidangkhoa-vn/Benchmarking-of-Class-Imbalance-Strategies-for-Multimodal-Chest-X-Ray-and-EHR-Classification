"""Small file helpers: every write goes to a temporary file first, so an interrupted run never leaves half a file."""
import json
import os

import numpy as np


def atomic_csv(df, path, **kw):
    path = str(path); tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip" if path.endswith(".gz") else None, **kw)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = str(path); tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = str(path); tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
