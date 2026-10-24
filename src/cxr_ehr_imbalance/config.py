from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path) -> dict:
    """Reads the YAML config, resolves relative paths against the repository root and creates the output folders.

    `cfg["dirs"]` maps each pipeline stage (embeddings, mimic, labels, splits, runs, analysis) to its output folder.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent.parent  # configs/<file>.yaml -> repository root
    for k, v in list(cfg.get("paths", {}).items()):
        if v is not None and not Path(v).is_absolute():
            cfg["paths"][k] = str((base / v).resolve())
    out = Path(cfg["paths"]["output_dir"])
    cfg["dirs"] = {name: out / name for name in ["embeddings", "mimic", "labels", "splits", "runs", "analysis"]}
    for d in cfg["dirs"].values():
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def cli(description: str, extra=None):
    """Command-line interface shared by the scripts: --config plus optional script-specific arguments."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, load_config(args.config)
