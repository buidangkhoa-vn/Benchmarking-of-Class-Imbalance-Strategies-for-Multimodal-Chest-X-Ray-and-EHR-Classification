"""SD and 95% CIs for the protocol tables (paper Tables III-VI) from saved per-fold results; no refits.
F2/PR-AUC/AUROC: mean and SD over the 5 outer folds of the macro (8-target) value, F2 CI = t-interval over folds.
Delta (F2 minus all-positive F2): mean and t-interval over the 40 target x fold pairs; p = paired Wilcoxon."""
import os, pandas as pd, numpy as np
from scipy import stats
from scipy.stats import wilcoxon
V2 = os.environ.get('MMMDA_OUT', 'out_v2')                       # output folder of analyze_v2_merge.py
A4 = os.environ.get('MMMDA_V4_ANALYSIS', 'v4_2026-10-19_admlabels/v4_analysis')
A5 = os.environ.get('MMMDA_V5_ANALYSIS', 'v5_2026-10-19_chexpert/v5_analysis')
OUT = os.environ.get('MMMDA_TABLES_OUT', 'out_tables'); os.makedirs(OUT, exist_ok=True)
KEY = ['protocol', 'feature_set', 'target']
def select(R):
    g = R.groupby(KEY + ['strategy', 'classifier']).val_f2.mean().reset_index()
    b = g.sort_values('val_f2', ascending=False).groupby(KEY).head(1)[KEY + ['strategy', 'classifier']]
    return R.merge(b, on=KEY + ['strategy', 'classifier'])
S = pd.concat([pd.read_csv(f'{V2}/selected_rows.csv'), select(pd.read_csv(f'{A4}/v4_results_all.csv')),
               select(pd.read_csv(f'{A5}/v5_results_all.csv'))], ignore_index=True)
T4 = stats.t.ppf(0.975, 4); rows = []
for (p, fs), d in S.groupby(['protocol', 'feature_set']):
    fm = d.groupby('fold').agg(f2=('test_f2', 'mean'), tv=('test_trivial_f2', 'mean'), pr=('test_pr_auc', 'mean'), au=('test_roc_auc', 'mean'))
    n = len(fm); r = dict(protocol=p, feature_set=fs, folds=n)
    for c in ['f2', 'pr', 'au']:
        r[c] = fm[c].mean(); r[c + '_sd'] = fm[c].std(ddof=1) if n > 1 else np.nan
    h = T4 * r['f2_sd'] / np.sqrt(n) if n == 5 else np.nan; r['f2_lo'], r['f2_hi'] = r['f2'] - h, r['f2'] + h
    diff = (d.test_f2 - d.test_trivial_f2).values; m = len(diff); hd = stats.t.ppf(0.975, m - 1) * diff.std(ddof=1) / np.sqrt(m)
    r.update(delta=diff.mean(), delta_lo=diff.mean() - hd, delta_hi=diff.mean() + hd, pairs=m, p=wilcoxon(diff).pvalue if (diff != 0).any() else 1.0)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tables_sd_ci.csv', index=False)
pd.set_option('display.width', 250); print(T.round(4).to_string(index=False))
