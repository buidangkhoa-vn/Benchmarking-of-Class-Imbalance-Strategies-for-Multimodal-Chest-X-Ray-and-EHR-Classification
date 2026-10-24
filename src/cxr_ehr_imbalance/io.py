import json
import os

import numpy as np


def atomic_csv(df, path, **kw):
    """Writes a DataFrame as CSV (gzip-compressed when the path ends in .gz)."""
    path = str(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip" if path.endswith(".gz") else None, **kw)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    """Writes arrays to a compressed .npz file."""
    path = str(path)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(obj, path):
    """Writes an object as indented JSON."""
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
