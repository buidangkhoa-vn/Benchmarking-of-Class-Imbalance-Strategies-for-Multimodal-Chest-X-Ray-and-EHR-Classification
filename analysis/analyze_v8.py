"""v8: PCA sensitivity, age/history decomposition, repeated (2-seed) patient-grouped CV with the corrected repeated
k-fold t-test (Nadeau & Bengio, 2003), Brier scores (paper Sec. V-F)."""
import os, pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 250)
A = os.environ.get('MMMDA_V8_ANALYSIS', 'v8_2026-10-22_sensitivity/v8_analysis'); OUT = os.environ.get('MMMDA_V8_OUT', 'out_v8'); os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(f'{A}/v8_results_all.csv.gz')
print(R.shape, R.groupby('protocol').size().to_dict())
K, RATIO = 5, 0.2 / 0.64
def fold_macro(d, metric='test_f2'):
    return d.groupby(['seed', 'fold'])[metric].mean()
def nb_test(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = np.sqrt((1 / n + RATIO) * x.var(ddof=1)) if n > 1 else np.nan
    h = stats.t.ppf(0.975, n - 1) * se; return m, m - h, m + h, 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 1.0
N = R[R.strategy == 'none']
def macro_table(d):
    fm = d.groupby(['protocol', 'feature_set', 'seed', 'fold']).agg(f2=('test_f2', 'mean'), tv=('test_trivial_f2', 'mean'), au=('test_roc_auc', 'mean'),
                                                                  pr=('test_pr_auc', 'mean'), br=('test_brier', 'mean'), bp=('test_brier_prev', 'mean'),
                                                                  ev=('pca_explained_var', 'mean')).reset_index()
    t = fm.groupby(['protocol', 'feature_set']).agg(f2=('f2', 'mean'), f2_sd=('f2', 'std'), triv=('tv', 'mean'), auroc=('au', 'mean'), prauc=('pr', 'mean'),
                                                   brier=('br', 'mean'), brier_prev=('bp', 'mean'), pca_ev=('ev', 'mean')).reset_index()
    t['delta'] = t.f2 - t.triv; return t
T = macro_table(N); T.to_csv(f'{OUT}/v8_macro_none.csv', index=False)
print('\nMACRO (LR, no augmentation)'); print(T.round(4).to_string(index=False))
def paired(d, p, a, b, metric='test_f2'):
    x = fold_macro(d[(d.protocol == p) & (d.feature_set == a)], metric); y = fold_macro(d[(d.protocol == p) & (d.feature_set == b)], metric)
    i = x.index.intersection(y.index); m, lo, hi, pv = nb_test((x.loc[i] - y.loc[i]).values)
    return dict(protocol=p, a=a, b=b, metric=metric, diff=m, lo=lo, hi=hi, p=pv, n=len(i))
rows = []
for p in ['P2_rel', 'P4_adm', 'P4C_cln']:                                   # 1) PCA sensitivity vs 64 PCs
    for g in ['img', 'fus']:
        for d_ in ['pc32', 'pc128', 'full']:
            for met in ['test_f2', 'test_roc_auc']:
                rows.append(paired(N, p, f'{g}_{d_}', f'{g}_pc64', met))
for p, h in [('P4_adm', 'a'), ('P4C_cln', 'c')]:                            # 2) age / history decomposition
    for a, b in [('age_own', 'age_rel'), (f'hist_{h}', 'ehr'), (f'agehist_{h}', f'hist_{h}'), (f'agehist_{h}', 'ehr'), ('age_own', 'ehr')]:
        for met in ['test_f2', 'test_roc_auc']:
            rows.append(paired(N, p, a, b, met))
P = pd.DataFrame(rows); P.to_csv(f'{OUT}/v8_paired_seed42.csv', index=False)
print('\nPAIRED (seed 42, corrected resampled t-test over 5 folds)'); print(P.round(4).to_string(index=False))
# 3) repeated CV: both seeds
both = N.copy(); both['protocol'] = both.protocol.str.replace('_s43', '', regex=False)
rows = []
for p, comps in [('P2_rel', [('fus_pc64', 'ehr'), ('img_pc64', 'ehr')]), ('P4_adm', [('fus_pc64', 'ehr'), ('img_pc64', 'ehr'), ('agehist_a', 'ehr')]),
                 ('P4C_cln', [('fus_pc64', 'ehr'), ('img_pc64', 'ehr'), ('agehist_c', 'ehr')])]:
    for a, b in comps:
        for met in ['test_f2', 'test_roc_auc']:
            rows.append(paired(both, p, a, b, met))
S = R[R.protocol.str.startswith('P2_rel') & R.feature_set.isin(['ehr', 'fus_pc64'])].copy(); S['protocol'] = 'P2_rel'
for fs in ['ehr', 'fus_pc64']:
    for st in ['class_weight', 'ROS', 'SMOTE*']:
        x = fold_macro(S[(S.feature_set == fs) & (S.strategy == st)]); y = fold_macro(S[(S.feature_set == fs) & (S.strategy == 'none')])
        i = x.index.intersection(y.index); m, lo, hi, pv = nb_test((x.loc[i] - y.loc[i]).values)
        rows.append(dict(protocol='P2_rel', a=f'{fs}:{st}', b=f'{fs}:none', metric='test_f2', diff=m, lo=lo, hi=hi, p=pv, n=len(i)))
        xb = fold_macro(S[(S.feature_set == fs) & (S.strategy == st)], 'test_brier'); yb = fold_macro(S[(S.feature_set == fs) & (S.strategy == 'none')], 'test_brier')
        m, lo, hi, pv = nb_test((xb.loc[i] - yb.loc[i]).values)
        rows.append(dict(protocol='P2_rel', a=f'{fs}:{st}', b=f'{fs}:none', metric='test_brier', diff=m, lo=lo, hi=hi, p=pv, n=len(i)))
RC = pd.DataFrame(rows); RC.to_csv(f'{OUT}/v8_repeated_cv.csv', index=False)
print('\nREPEATED CV (2 seeds x 5 folds, corrected repeated k-fold t-test)'); print(RC.round(4).to_string(index=False))
# seed-to-seed variation of macro F2 for the same configuration
sv = N[N.protocol.isin(['P2_rel', 'P4_adm', 'P4C_cln', 'P2_rel_s43', 'P4_adm_s43', 'P4C_cln_s43'])].copy(); sv['base'] = sv.protocol.str.replace('_s43', '', regex=False)
mm = sv.groupby(['base', 'feature_set', 'seed']).test_f2.mean().unstack('seed').dropna(); mm['abs_diff'] = (mm[42] - mm[43]).abs()
mm.to_csv(f'{OUT}/v8_seed_variation.csv'); print('\nSEED 42 vs 43 macro F2'); print(mm.round(4).to_string())
# 4) Brier: strategies vs none (both seeds) and none vs prevalence
print('\nBRIER (LR): mean over folds and seeds'); print(R[R.protocol.str.startswith('P2_rel') & R.feature_set.isin(['ehr', 'fus_pc64'])].groupby(['feature_set', 'strategy'])[['test_brier', 'test_brier_prev']].mean().round(4).to_string())
