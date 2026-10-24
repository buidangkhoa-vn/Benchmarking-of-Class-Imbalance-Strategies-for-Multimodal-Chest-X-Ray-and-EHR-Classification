"""Runs one experiment from the config: every (protocol, seed, fold, feature set) is preprocessed once, then all
targets, strategies and classifiers are fitted on it. Finished groups are skipped when the script is run again."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from . import cohort as C
from .io import atomic_csv, atomic_npz
from .modeling import (build_preprocessor, choose_threshold, discrete_columns, f2_score, make_classifier, metrics,
                       resample)
from .splits import roles_for, split_indices


# ---------------------------------------------------------------- inputs
def load_tables(cfg: dict, need_patient_table=False) -> dict:
    """Radiograph table with EHR, image scores, embeddings, time-aligned features and all label sets."""
    d = cfg["dirs"]
    df = C.load_cohort(cfg["paths"]["cohort_csv"])
    rec = df[["dataset_position", "subject_id", "dicom"] + C.TARGETS + C.EHR + C.IMAGE_SCORES].copy()
    embeddings = {}
    for name in cfg["embeddings"]["models"]:
        if (d["embeddings"] / name / "emb.npy").exists():
            embeddings[name] = C.attach_embedding(rec, name, d["embeddings"] / name)
    lab_file = d["labels"] / "admission_labels.csv.gz"
    if lab_file.exists():
        AL = pd.read_csv(lab_file)
        assert (AL.dataset_position.values == rec.dataset_position.values).all()
        extra = [c for c in AL.columns if c.startswith(("adm_", "cln_", "hist_", "histcln_"))] + ["age_own", "aligned"]
        rec[extra] = AL[extra].values
    cx_file = d["labels"] / "chexpert_labels.csv.gz"
    if cx_file.exists():
        CX = pd.read_csv(cx_file)
        assert (CX.dataset_position.values == rec.dataset_position.values).all()
        cx_cols = [c for c in CX.columns if c.startswith(("cx_", "histcx_"))]
        rec[cx_cols] = CX[cx_cols].values
    out = {"rec": rec, "embeddings": embeddings, "data_hash": C.data_hash(df.drop(columns="dataset_position"))}
    if need_patient_table:
        meta = d["embeddings"] / "dicom_meta.csv"
        if meta.exists():
            vm = pd.read_csv(meta, usecols=["dicom", "ViewPosition"])
            rec["frontal"] = rec.dicom.map(vm.set_index("dicom").ViewPosition).isin(["AP", "PA"]).values
        else:
            rec["frontal"] = True
        out["pat"] = C.patient_table(rec, C.IMAGE_SCORES + sum(embeddings.values(), []), C.TARGETS)
    return out


def manifest(cfg: dict, split: str, seed: int) -> pd.DataFrame:
    d = cfg["dirs"]["splits"]
    f = {"record": d / f"folds_record_s{seed}.csv", "patient": d / f"folds_patient_s{seed}.csv",
         "patient_aligned": d / f"folds_patient_s{seed}.csv", "temporal": d / "folds_temporal.csv"}[split]
    return pd.read_csv(f)


# ---------------------------------------------------------------- one group = (protocol, seed, fold, feature set)
def run_group(group: dict, opts: dict) -> tuple[int, int]:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):           # one BLAS thread per process: results do not depend on the number of workers
        return _run_group(group, opts)


def _run_group(group: dict, opts: dict) -> tuple[int, int]:
    out = Path(group["out"])
    if out.exists():
        return 0, 0
    tab = pd.read_pickle(group["table_file"])
    idx = np.load(group["split_file"])
    fs = group["feature_set"]
    ct = build_preprocessor(fs, opts["pca_components"], opts["knn_neighbors"])
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xtr = ct.fit_transform(tab.iloc[tr]).astype(np.float32)
    Xva = ct.transform(tab.iloc[va]).astype(np.float32)
    Xte = ct.transform(tab.iloc[te]).astype(np.float32)
    disc, ncol, evr = discrete_columns(ct, indicators_first=opts.get("smote_nc_indicators_first", False))
    assert Xtr.shape[1] == ncol
    rows, preds, errors = [], {}, []
    for target in group["targets"]:
        y = tab[group["label_cols"][target]].values.astype(int)
        ytr, yva, yte = y[tr], y[va], y[te]
        for strategy, clf_name, spec in group["jobs"]:
            if strategy == "smote_nc" and not len(disc):
                if opts.get("smote_nc_without_discrete", "skip") == "skip":
                    continue
                strategy = "smote"
            t0 = time.time()
            try:
                Xs, ys, aud = resample(strategy, Xtr, ytr, disc, opts["seed"])
                pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
                clf = make_classifier(spec, opts["seed"], strategy == "class_weight", pos_weight, opts["lr_max_iter"])
                clf.fit(Xs, ys)
                pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
                if opts["float32_predictions"]:
                    pva, pte = pva.astype(np.float32), pte.astype(np.float32)
                thr = choose_threshold(yva, pva); p_tr = float(ytr.mean())
                row = {**group["meta"], "target": target, "strategy": strategy, "classifier": clf_name,
                       "family": spec["family"], "threshold": thr, "n_features": Xtr.shape[1], "n_discrete": len(disc),
                       "pca_explained_var": evr, "n_train": len(ytr), "pos_train": int(ytr.sum()), "n_train_after": len(ys), **aud}
                row.update(metrics(yva, pva, thr, "val", p_tr)); row.update(metrics(yte, pte, thr, "test", p_tr))
                if group["meta"]["split"] == "record":
                    s = idx["seen_te"]
                    row.update(metrics(yte[s], pte[s], thr, "seen", p_tr)); row.update(metrics(yte[~s], pte[~s], thr, "unseen", p_tr))
                    if opts.get("unseen_clean_threshold"):
                        u = idx["unseen_va"]
                        thr_u = choose_threshold(yva[u], pva[u]) if yva[u].sum() > 0 else thr
                        row["unseen_clean_threshold"] = thr_u
                        row.update(metrics(yte[~s], pte[~s], thr_u, "unseenclean", p_tr))
                if group["aligned_te"] is not None:
                    a = group["aligned_te"]
                    row.update(metrics(yte[a], pte[a], thr, "aligned", p_tr)); row.update(metrics(yte[~a], pte[~a], thr, "other", p_tr))
                row["fit_seconds"] = time.time() - t0
                rows.append(row)
                if opts.get("save_predictions"):
                    key = f"{target}__{strategy}__{clf_name}"
                    preds[f"{key}__p_va"] = pva; preds[f"{key}__p_te"] = pte
            except Exception as e:                       # keep going; failures are listed in errors.csv
                errors.append({"group": out.stem, "target": target, "strategy": strategy, "classifier": clf_name,
                               "error": repr(e)[:300], "trace": traceback.format_exc()[-1500:]})
    if preds:
        atomic_npz(out.with_suffix(".npz"), **preds)
    if errors:
        atomic_csv(pd.DataFrame(errors), out.with_name(out.stem + "__errors.csv"))
    atomic_csv(pd.DataFrame(rows), out)
    return len(rows), len(errors)


# ---------------------------------------------------------------- experiment
def label_columns(labels: str) -> dict:
    cols = C.LABEL_COLUMNS[labels]
    names = C.TARGETS if labels != "CX" else list(C.CHEXPERT_TARGETS)
    return dict(zip(names, cols))


def run_experiment(cfg: dict, name: str, folds=None, targets=None, n_jobs=None) -> pd.DataFrame:
    exp = cfg["experiments"][name]
    run_dir = cfg["dirs"]["runs"] / name
    (run_dir / "groups").mkdir(parents=True, exist_ok=True); (run_dir / "splits").mkdir(exist_ok=True)
    need_pat = any(p.get("table") == "patient" for p in exp["protocols"])
    T = load_tables(cfg, need_patient_table=need_pat)
    for tname in (["rec", "pat"] if need_pat else ["rec"]):
        pd.to_pickle(T[tname], run_dir / f"{tname}.pkl")
    opts = {"seed": cfg["random_state"], "pca_components": cfg["preprocessing"]["pca_components"],
            "knn_neighbors": cfg["preprocessing"]["knn_neighbors"], "lr_max_iter": exp.get("lr_max_iter", 2000),
            "float32_predictions": exp.get("float32_predictions", True), "save_predictions": exp.get("save_predictions", False),
            "smote_nc_without_discrete": exp.get("smote_nc_without_discrete", "skip"),
            "smote_nc_indicators_first": exp.get("smote_nc_indicators_first", False),
            "unseen_clean_threshold": exp.get("unseen_clean_threshold", False)}
    min_pos = cfg["min_positives"]
    groups, skipped, baselines = [], [], []
    for p in exp["protocols"]:
        tname = "pat" if p.get("table") == "patient" else "rec"
        tab = T[tname]
        lcols = label_columns(p["labels"])
        tgts = [t for t in (targets or lcols) if t in lcols]
        seeds = p.get("seeds", [cfg["cv"]["seeds"][0]]) if p["split"] != "temporal" else [0]
        jobs = [(s, c, cfg["classifiers"][c]) for s in p.get("strategies", ["none"]) for c in p["classifiers"]]
        jobs += [("native", c, cfg["classifiers"][c]) for c in p.get("ensembles", [])]
        for seed in seeds:
            man = manifest(cfg, p["split"], seed if seed else cfg["cv"]["seeds"][0])
            fold_ids = sorted(man.fold.unique())
            for k in [f for f in fold_ids if folds is None or f in folds]:
                roles = roles_for(tab, man, k)
                if p["split"] == "patient_aligned":
                    roles = np.where(tab.aligned.values == 1, roles, None)
                idx = split_indices(tab, roles, grouped=p["split"] != "record")
                sfile = run_dir / "splits" / f"{p['name']}__s{seed}__f{k}.npz"
                np.savez(sfile, **idx)
                ok_t = []
                for t in tgts:
                    y = tab[lcols[t]].values
                    n = {r: int(y[idx[r]].sum()) for r in ("train", "val", "test")}
                    if n["train"] < min_pos["train"] or n["val"] < min_pos["val"] or n["test"] < min_pos["test"]:
                        skipped.append({"protocol": p["name"], "seed": seed, "fold": k, "target": t, **n}); continue
                    ok_t.append(t)
                    baselines.append(baseline_row(tab, y, idx, p, seed, k, t))
                al = tab.aligned.values[idx["test"]].astype(bool) if p.get("aligned_metrics") else None
                meta = {"experiment": name, "protocol": p["name"], "split": p["split"], "labels": p["labels"], "seed": seed, "fold": k}
                for feat in p["features"]:
                    fs = C.resolve_feature_set(cfg["feature_sets"][feat], T["embeddings"])
                    if fs["emb"] == [] and cfg["feature_sets"][feat].get("emb"):
                        raise RuntimeError(f"{feat}: embedding '{cfg['feature_sets'][feat]['emb']}' not found (run 01_extract_embeddings.py)")
                    groups.append({"out": str(run_dir / "groups" / f"{p['name']}__s{seed}__f{k}__{feat}.csv"),
                                   "table_file": str(run_dir / f"{tname}.pkl"), "split_file": str(sfile), "feature_set": fs,
                                   "targets": ok_t, "label_cols": lcols, "jobs": jobs, "aligned_te": al,
                                   "meta": {**meta, "feature_set": feat}})
    atomic_csv(pd.DataFrame(skipped, columns=["protocol", "seed", "fold", "target", "train", "val", "test"]), run_dir / "skipped.csv")
    atomic_csv(pd.DataFrame(baselines), run_dir / "baselines.csv")
    todo = [g for g in groups if not Path(g["out"]).exists()]
    print(f"{name}: {len(groups)} groups, {len(todo)} to run, {len(skipped)} protocol/fold/target cells skipped")
    nj = n_jobs or cfg.get("n_jobs") or os.cpu_count()
    nj = os.cpu_count() if nj == -1 else nj
    res = Parallel(n_jobs=min(nj, max(1, len(todo))))(delayed(run_group)(g, opts) for g in tqdm(todo, desc=name))
    print(f"fits: {sum(r[0] for r in res)}, errors: {sum(r[1] for r in res)}")
    files = sorted((run_dir / "groups").glob("*.csv"))
    R = pd.concat([pd.read_csv(f) for f in files if not f.stem.endswith("__errors")], ignore_index=True)
    R = R[R.protocol.isin({p["name"] for p in exp["protocols"]})]
    atomic_csv(R, run_dir / "results.csv")
    return R


def baseline_row(tab, y, idx, p, seed, k, t) -> dict:
    """All-positive, no-learning EHR lookup (label of the first training row with the same EHR vector; fallback chosen
    on validation) and, for A/C/CX labels, the prior-diagnosis or prior-study rule."""
    keys = pd.util.hash_pandas_object(tab[C.EHR].fillna(-999), index=False).values
    tr, va, te = idx["train"], idx["val"], idx["test"]
    src_all = pd.Series(y[tr], index=keys[tr])
    src = src_all[~src_all.index.duplicated()]
    def lookup(rows, fb):
        m = pd.Series(keys[rows]).map(src)
        return np.where(m.notna(), m.fillna(fb), fb).astype(int), float(m.notna().mean())
    fb = max([0, 1], key=lambda b: f2_score(y[va], lookup(va, b)[0]))
    yhat, hit = lookup(te, fb)
    prev = y[te].mean()
    row = {"protocol": p["name"], "seed": seed, "fold": k, "target": t, "test_prev": prev, "trivial_f2": 5 * prev / (4 * prev + 1),
           "lookup_f2": f2_score(y[te], yhat), "lookup_hit_rate": hit, "lookup_fallback": fb}
    if p["labels"] in C.PRIOR_COLUMNS:
        prior = dict(zip(label_columns(p["labels"]), C.PRIOR_COLUMNS[p["labels"]]))[t]
        row["prior_rule_f2"] = f2_score(y[te], tab[prior].values[te])
    return row
