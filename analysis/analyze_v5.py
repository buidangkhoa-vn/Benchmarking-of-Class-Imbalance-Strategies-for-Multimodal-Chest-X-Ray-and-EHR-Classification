"""v5: CheXpert report labels (P1 record vs P2 patient split) and time-aligned EHR-history features with repaired labels
(P4_hist / P4C_hist, paired with the v4 P4 / P4-C results on the same folds)."""
import os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option("display.width", 250)
A5 = os.environ.get("MMMDA_V5_ANALYSIS", "v5_2026-10-19_chexpert/v5_analysis")
A4 = os.environ.get("MMMDA_V4_ANALYSIS", "v4_2026-10-19_admlabels/v4_analysis")
OUT = os.environ.get("MMMDA_V5_OUT", "out_v5"); os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(f"{A5}/v5_results_all.csv"); B = pd.read_csv(f"{A5}/v5_baselines.csv")
R4 = pd.read_csv(f"{A4}/v4_results_all.csv")
KEY = ["protocol", "feature_set", "target"]
def select(R):
    g = R.groupby(KEY + ["strategy", "classifier"]).agg(val_f2=("val_f2", "mean")).reset_index()
    best = g.sort_values("val_f2", ascending=False).groupby(KEY).head(1)[KEY + ["strategy", "classifier"]]
    return R.merge(best, on=KEY + ["strategy", "classifier"]), best
Rb, best = select(R); Rb4, _ = select(R4)
Rb.to_csv(f"{OUT}/v5_selected_rows.csv", index=False)
# ---- macro table with fold SD and paired test vs all-positive ----
fm = Rb.groupby(["protocol", "feature_set", "fold"]).agg(f2=("test_f2", "mean"), pr=("test_pr_auc", "mean"), au=("test_roc_auc", "mean"), tv=("test_trivial_f2", "mean")).reset_index()
TA = fm.groupby(["protocol", "feature_set"]).agg(f2=("f2", "mean"), f2_sd=("f2", "std"), pr=("pr", "mean"), au=("au", "mean"), tv=("tv", "mean"))
pv = []
for (p, fs), d in Rb.groupby(["protocol", "feature_set"]):
    diff = d.test_f2 - d.test_trivial_f2
    pv.append(dict(protocol=p, feature_set=fs, d_triv=diff.mean(), wins=int((diff > 0).sum()), n=len(diff), p=wilcoxon(diff).pvalue))
TA = TA.join(pd.DataFrame(pv).set_index(["protocol", "feature_set"]))
TA.round(4).to_csv(f"{OUT}/v5_tableA.csv")
print("TABLE A v5\n", TA.round(3).to_string())
print("\nBASELINES\n", B.groupby("protocol")[["trivial_f2", "lookup_f2", "lookup_any_f2", "lookup_hit_rate", "history_f2"]].mean().round(3).to_string())
# ---- CheXpert: P1 vs P2, seen/unseen ----
c = Rb[Rb.protocol == "P1_chexpert"].groupby("feature_set")[["test_f2", "seen_f2", "unseen_f2", "unseen_trivial_f2", "test_roc_auc", "unseen_roc_auc"]].mean()
print("\nP1_chexpert seen/unseen\n", c.round(3).to_string())
p1 = Rb[Rb.protocol == "P1_chexpert"].groupby(["feature_set"]).apply(lambda d: (d.test_f2 - d.test_trivial_f2).mean())
p2 = Rb[Rb.protocol == "P2_chexpert"].groupby(["feature_set"]).apply(lambda d: (d.test_f2 - d.test_trivial_f2).mean())
print("\nmargin over all-positive  P1 vs P2\n", pd.concat({"P1": p1, "P2": p2}, axis=1).round(3))
# per-target CheXpert P2
PT = Rb[Rb.protocol.str.endswith("chexpert")].groupby(["protocol", "feature_set", "target"])[["test_f2", "test_trivial_f2", "test_roc_auc"]].mean()
PT.round(4).to_csv(f"{OUT}/v5_per_target_chexpert.csv")
print("\nP2_chexpert AUROC per target\n", PT.xs("P2_chexpert").test_roc_auc.unstack(0).round(3).to_string())
# ---- paired modality tests ----
def paired(Ra, pa, fa, Rb_, pb, fb, metric="test_f2"):
    x = Ra[(Ra.protocol == pa) & (Ra.feature_set == fa)].set_index(["target", "fold"])[metric]
    y = Rb_[(Rb_.protocol == pb) & (Rb_.feature_set == fb)].set_index(["target", "fold"])[metric]
    i = x.index.intersection(y.index); d = (x.loc[i] - y.loc[i]).dropna()
    return d.mean(), (wilcoxon(d).pvalue if (d != 0).any() else 1.0), len(d)
rows = []
for m in ["test_f2", "test_roc_auc"]:
    for (pa, fa, pb, fb, src_b) in [("P2_chexpert", "img_raddino", "P2_chexpert", "ehr", "v5"), ("P2_chexpert", "img_raddino", "P2_chexpert", "img_xrv", "v5"),
                                    ("P2_chexpert", "fusion_raddino", "P2_chexpert", "img_raddino", "v5"),
                                    ("P4_hist", "ehrhist_a", "P4_admission", "ehr", "v4"), ("P4_hist", "fusionhist_a", "P4_admission", "fusion_raddino", "v4"),
                                    ("P4_hist", "fusionhist_a", "P4_hist", "ehrhist_a", "v5"),
                                    ("P4C_hist", "ehrhist_c", "P4_admission_clean", "ehr", "v4"), ("P4C_hist", "fusionhist_c", "P4_admission_clean", "fusion_raddino", "v4"),
                                    ("P4C_hist", "fusionhist_c", "P4C_hist", "ehrhist_c", "v5"), ("P4C_hist", "ehrhist_c", "P4_admission_clean", "img_raddino", "v4")]:
        mu, p, n = paired(Rb, pa, fa, Rb if src_b == "v5" else Rb4, pb, fb, m)
        rows.append(dict(metric=m, a=f"{pa}:{fa}", b=f"{pb}:{fb}", diff=mu, p=p, n=n))
MOD = pd.DataFrame(rows); MOD.round(4).to_csv(f"{OUT}/v5_modality.csv", index=False); print("\nPAIRED\n", MOD.round(4).to_string(index=False))
# prior-diagnosis rule vs EHR-history model (per target x fold)
B4 = pd.read_csv(f"{A4}/v4_baselines.csv")
for p4, p5, fs in [("P4_admission", "P4_hist", "ehrhist_a"), ("P4_admission_clean", "P4C_hist", "ehrhist_c")]:
    h = B4[B4.protocol == p4].set_index(["target", "fold"]).history_f2
    m = Rb[(Rb.protocol == p5) & (Rb.feature_set == fs)].set_index(["target", "fold"]).test_f2
    i = h.index.intersection(m.index); d = m.loc[i] - h.loc[i]
    print(f"{p5} {fs} minus prior-diagnosis rule: {d.mean():+.4f} (p={wilcoxon(d).pvalue:.4g}, n={len(d)})")
print("\nper-target F2 P4_hist / P4C_hist\n", Rb[Rb.protocol.str.endswith("hist")].groupby(["protocol", "feature_set", "target"]).test_f2.mean().unstack([0, 1]).round(3).to_string())
# ---- strategies vs none (pooled over classifiers), CI over classifier x fold ----
tb = []
for p in sorted(R.protocol.unique()):
    Rp = R[R.protocol == p]; b0 = Rp[Rp.strategy == "none"].set_index(["feature_set", "target", "classifier", "fold"])
    for (fs, st), d in Rp[Rp.strategy != "none"].groupby(["feature_set", "strategy"]):
        dd = d.set_index(["feature_set", "target", "classifier", "fold"]); i = dd.index.intersection(b0.index)
        diff = dd.loc[i, "test_f2"] - b0.loc[i, "test_f2"]; u = diff.groupby(level=["classifier", "fold"]).mean()
        tb.append(dict(protocol=p, feature_set=fs, strategy=st, dF2=diff.mean(), lo=u.mean() - 1.96 * u.std() / np.sqrt(len(u)),
                       hi=u.mean() + 1.96 * u.std() / np.sqrt(len(u)), p=wilcoxon(diff).pvalue if (diff != 0).any() else 1.0, n=len(diff)))
TB = pd.DataFrame(tb)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); r = 0
    for i, j in enumerate(o): r = max(r, (m - i) * p[j]); a[j] = min(1, r)
    return a
TB["holm"] = holm(TB.p.values); TB.round(5).to_csv(f"{OUT}/v5_strategies.csv", index=False)
print("\nSTRATEGIES\n", TB.round(4).to_string(index=False))
print("\nclassifier of selected configs\n", best.groupby(["protocol", "classifier"]).size().unstack(fill_value=0))
