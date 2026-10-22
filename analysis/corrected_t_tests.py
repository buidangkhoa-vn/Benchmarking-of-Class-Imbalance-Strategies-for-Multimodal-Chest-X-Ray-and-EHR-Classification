"""Corrected resampled t-test (Nadeau & Bengio, 2003) on fold-level macro values for the main pairwise comparisons
(seed-42 runs; k=5, n_test/n_train = 0.2/0.64). Paper Sec. IV-D, V-C, V-F."""
import pandas as pd, numpy as np
from scipy import stats
import os
V2=os.environ.get('MMMDA_OUT','out_v2')
A4=os.environ.get('MMMDA_V4_ANALYSIS','v4_2026-10-19_admlabels/v4_analysis')
A5=os.environ.get('MMMDA_V5_ANALYSIS','v5_2026-10-19_chexpert/v5_analysis')
A7=os.environ.get('MMMDA_V7_ANALYSIS','v7_2026-10-22_ablation/v7_analysis')
S2=pd.read_csv(f'{V2}/selected_rows.csv')
R4=pd.read_csv(f'{A4}/v4_results_all.csv'); B4=pd.read_csv(f'{A4}/v4_baselines.csv')
R5=pd.read_csv(f'{A5}/v5_results_all.csv')
R7=pd.read_csv(f'{A7}/v7_results_all.csv.gz')
KEY=['protocol','feature_set','target']
def sel(R,cfg=['strategy','classifier']):
    g=R.groupby(KEY+cfg).val_f2.mean().reset_index(); b=g.sort_values('val_f2',ascending=False).groupby(KEY).head(1)[KEY+cfg]
    return R.merge(b,on=KEY+cfg)
S4=sel(R4); S5=sel(R5)
K=5; RATIO=0.2/0.64
def nb(x):  # x: fold-level differences (5)
    m=x.mean(); se=np.sqrt((1/K+RATIO)*x.var(ddof=1)); t=m/se; return m, 2*stats.t.sf(abs(t),K-1), m-stats.t.ppf(.975,K-1)*se, m+stats.t.ppf(.975,K-1)*se
def fold_macro(S,p,fs,metric='test_f2'):
    return S[(S.protocol==p)&(S.feature_set==fs)].groupby('fold')[metric].mean()
out=[]
def comp(name,a,b):
    m,p,lo,hi=nb((a-b).dropna()); out.append(dict(comparison=name,diff=m,lo=lo,hi=hi,p=p))
for met in ['test_f2','test_roc_auc']:
    comp(f'P3 fusion RAD-DINO - EHR [{met}]',fold_macro(S2,'P3_patient','fusion_raddino',met),fold_macro(S2,'P3_patient','ehr',met))
    comp(f'P3 image RAD-DINO - EHR [{met}]',fold_macro(S2,'P3_patient','img_raddino',met),fold_macro(S2,'P3_patient','ehr',met))
    comp(f'P3 image RAD-DINO - 11 scores [{met}]',fold_macro(S2,'P3_patient','img_raddino',met),fold_macro(S2,'P3_patient','img_xrv',met))
    comp(f'P4 image RAD-DINO - EHR [{met}]',fold_macro(S4,'P4_admission','img_raddino',met),fold_macro(S4,'P4_admission','ehr',met))
    comp(f'P4C image RAD-DINO - EHR [{met}]',fold_macro(S4,'P4_admission_clean','img_raddino',met),fold_macro(S4,'P4_admission_clean','ehr',met))
    comp(f'P2CX image RAD-DINO - EHR [{met}]',fold_macro(S5,'P2_chexpert','img_raddino',met),fold_macro(S5,'P2_chexpert','ehr',met))
    comp(f'P4H history - released EHR [{met}]',fold_macro(S5,'P4_hist','ehrhist_a',met),fold_macro(S4,'P4_admission','ehr',met))
    comp(f'P4CH history - released EHR [{met}]',fold_macro(S5,'P4C_hist','ehrhist_c',met),fold_macro(S4,'P4_admission_clean','ehr',met))
    comp(f'P4H fusion+hist - history [{met}]',fold_macro(S5,'P4_hist','fusionhist_a',met),fold_macro(S5,'P4_hist','ehrhist_a',met))
    comp(f'P4CH fusion+hist - history [{met}]',fold_macro(S5,'P4C_hist','fusionhist_c',met),fold_macro(S5,'P4C_hist','ehrhist_c',met))
rule=lambda p: B4[B4.protocol==p].groupby('fold').history_f2.mean()
comp('P4H history - prior-dx rule [F2]',fold_macro(S5,'P4_hist','ehrhist_a'),rule('P4_admission'))
comp('P4CH history - prior-dx rule [F2]',fold_macro(S5,'P4C_hist','ehrhist_c'),rule('P4_admission_clean'))
# all-positive margins of released EHR under P2/P3/P4/P4C
for S,p in [(S2,'P2_grouped'),(S2,'P3_patient'),(S4,'P4_admission'),(S4,'P4_admission_clean')]:
    d=S[(S.protocol==p)&(S.feature_set=='ehr')]; comp(f'{p} EHR - all-positive [F2]',d.groupby('fold').test_f2.mean(),d.groupby('fold').test_trivial_f2.mean())
# v7 tuned trees vs tuned LR
agg=R7.groupby(KEY+['family','config']).val_f2.mean().reset_index()
pick=lambda sub: sub.sort_values('val_f2',ascending=False).groupby(KEY).head(1)[KEY+['config']]
TT=R7.merge(pick(agg[agg.family.isin(['RF','ET','LGBM'])]),on=KEY+['config']); LT=R7.merge(pick(agg[agg.family=='LR']),on=KEY+['config'])
for p in ['P2_rel','P4_adm','P4C_cln','P2al_adm','P2al_cln']:
    for fs in ['ehr','fusion_raddino']:
        comp(f'v7 {p} {fs} tuned trees - tuned LR [F2]',fold_macro(TT,p,fs),fold_macro(LT,p,fs))
O=pd.DataFrame(out); O.to_csv(os.environ.get('MMMDA_TESTS_OUT','corrected_t_tests.csv'),index=False); pd.set_option('display.width',200); print(O.round(4).to_string(index=False))
