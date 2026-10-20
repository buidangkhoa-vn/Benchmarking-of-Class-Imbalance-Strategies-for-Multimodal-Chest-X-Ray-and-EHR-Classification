"""v4 (repaired labels: P1-A, P4, P4-C, P5): Table 4 (tab:repaired), modality and strategy tests, per-target results."""
import pandas as pd, numpy as np, json, os
from scipy.stats import wilcoxon
pd.set_option('display.width',250)
A=os.environ.get("MMMDA_V4_ANALYSIS","v4_2026-10-19_admlabels/v4_analysis")   # v4_analysis folder written by STEP5
V2OUT=os.environ.get("MMMDA_OUT","out_v2")                                     # outputs of analyze_v2_merge.py
OUT=os.environ.get("MMMDA_V4_OUT","out_v4"); os.makedirs(OUT,exist_ok=True)
R=pd.read_csv(f"{A}/v4_results_all.csv"); B=pd.read_csv(f"{A}/v4_baselines.csv")
R2=pd.read_csv(f"{V2OUT}/results_all_merged_15680.csv")
T=['pneumonia','tuberculosis','bronchitis','arthritis','fracture','lung_cancer','scoliosis','lung_infection']
KEY=['protocol','feature_set','target']
def select(R):
    g=R.groupby(KEY+['strategy','classifier']).agg(val_f2=('val_f2','mean')).reset_index()
    best=g.sort_values('val_f2',ascending=False).groupby(KEY).head(1)[KEY+['strategy','classifier']]
    return R.merge(best,on=KEY+['strategy','classifier']), best
Rb,best=select(R); Rb.to_csv(f"{OUT}/v4_selected_rows.csv",index=False)
# Table A with fold SD and paired test vs trivial
fm=Rb.groupby(['protocol','feature_set','fold']).agg(f2=('test_f2','mean'),pr=('test_pr_auc','mean'),au=('test_roc_auc','mean'),tv=('test_trivial_f2','mean')).reset_index()
TA=fm.groupby(['protocol','feature_set']).agg(f2=('f2','mean'),f2_sd=('f2','std'),pr=('pr','mean'),au=('au','mean'),tv=('tv','mean'))
pv=[]
for (p,fs),d in Rb.groupby(['protocol','feature_set']):
    diff=d.test_f2-d.test_trivial_f2
    pv.append(dict(protocol=p,feature_set=fs,d_triv=diff.mean(),wins=int((diff>0).sum()),n=len(diff),p=wilcoxon(diff).pvalue if (diff!=0).any() else 1))
TA=TA.join(pd.DataFrame(pv).set_index(['protocol','feature_set']))
LB=B.groupby('protocol')[['trivial_f2','lookup_f2','lookup_any_f2','lookup_hit_rate','history_f2']].mean()
TA.round(4).to_csv(f"{OUT}/v4_tableA.csv"); LB.round(4).to_csv(f"{OUT}/v4_baselines_macro.csv")
print("TABLE A v4\n",TA.round(3).to_string()); print("\nBASELINES\n",LB.round(3).to_string())
# per-target
PT=Rb.groupby(['protocol','feature_set','target'])[['test_f2','test_trivial_f2','test_pr_auc','test_roc_auc','test_prev']].mean()
PT.round(4).to_csv(f"{OUT}/v4_per_target.csv")
for p in ['P4_admission','P4_admission_clean']:
    x=PT.xs(p,level=0)
    print(f"\nPER TARGET {p}: F2 (trivial)")
    print(pd.concat([x.test_f2.unstack(0).round(3), x.test_trivial_f2.unstack(0).iloc[:,0].rename('trivial').round(3), x.test_prev.unstack(0).iloc[:,0].rename('prev').round(4)],axis=1).to_string())
    print(f"AUROC {p}\n", x.test_roc_auc.unstack(0).round(3).to_string())
# history baseline per target for adm protocols
print("\nHISTORY baseline per target\n", B[B.protocol.isin(['P4_admission','P4_admission_clean'])].groupby(['protocol','target'])[['history_f2','trivial_f2','history_pos_rate','test_prev']].mean().round(3).unstack(0).to_string())
# P1 vs P1_admission
P1old=select(R2[R2.protocol=='P1_record'])[0]
c1=P1old.groupby(['feature_set','target'])[['test_f2','seen_f2','unseen_f2','test_trivial_f2']].mean()
c2=Rb[Rb.protocol=='P1_admission'].groupby(['feature_set','target'])[['test_f2','seen_f2','unseen_f2','unseen_trivial_f2','test_trivial_f2']].mean()
print("\nP1 released labels vs P1 repaired labels (macro)")
print(pd.concat({'P1_released':c1.groupby('feature_set').mean(),'P1_repaired':c2.groupby('feature_set').mean()}).round(3).to_string())
print("\nP1 per target EHR:", pd.concat({'released':c1.xs('ehr').test_f2,'repaired':c2.xs('ehr').test_f2,'triv_rep':c2.xs('ehr').test_trivial_f2},axis=1).round(3).to_string())
B2=pd.read_csv(f"{V2OUT}/baselines.csv")
print("\nlookup: released P1", B2[B2.protocol=='P1_record'].lookup_f2.mean().round(3), "hit", B2[B2.protocol=='P1_record'].hit.mean().round(3))
# Modality paired tests (P4, P4_clean, target x fold, selected configs)
def paired(p,a,b,metric='test_f2'):
    x=Rb[(Rb.protocol==p)&(Rb.feature_set==a)].set_index(['target','fold'])[metric]
    y=Rb[(Rb.protocol==p)&(Rb.feature_set==b)].set_index(['target','fold'])[metric]
    c=x.index.intersection(y.index); d=(x.loc[c]-y.loc[c]).dropna()
    return d.mean(), wilcoxon(d).pvalue if (d!=0).any() else 1, len(d)
print("\nMODALITY paired (target x fold)")
rows=[]
for p in ['P4_admission','P4_admission_clean']:
    for a,b in [('img_raddino','img_xrv'),('img_raddino','ehr'),('fusion_raddino','ehr'),('fusion_raddino','img_raddino'),('ehr','img_xrv')]:
        for m in ['test_f2','test_roc_auc','test_pr_auc']:
            mu,pval,n=paired(p,a,b,m); rows.append(dict(protocol=p,a=a,b=b,metric=m,diff=mu,p=pval,n=n))
MOD=pd.DataFrame(rows); MOD.round(4).to_csv(f"{OUT}/v4_modality.csv",index=False); print(MOD.round(4).to_string(index=False))
# Strategy effects pooled over classifiers (P4, P4_clean)
tb=[]
for p in ['P4_admission','P4_admission_clean','P1_admission']:
    Rp=R[R.protocol==p]; base=Rp[Rp.strategy=='none'].set_index(['feature_set','target','classifier','fold'])
    for (fs,st),d in Rp[Rp.strategy!='none'].groupby(['feature_set','strategy']):
        dd=d.set_index(['feature_set','target','classifier','fold']); c=dd.index.intersection(base.index)
        diff=dd.loc[c,'test_f2']-base.loc[c,'test_f2']; dpr=dd.loc[c,'test_pr_auc']-base.loc[c,'test_pr_auc']
        se=diff.groupby(level=['classifier','fold']).mean()   # unit = classifier x fold
        tb.append(dict(protocol=p,feature_set=fs,strategy=st,dF2=diff.mean(),ci_lo=se.mean()-1.96*se.std()/np.sqrt(len(se)),ci_hi=se.mean()+1.96*se.std()/np.sqrt(len(se)),dPR=dpr.mean(),p=wilcoxon(diff).pvalue if (diff!=0).any() else 1,n=len(diff)))
TB=pd.DataFrame(tb)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
TB['holm']=holm(TB.p.values); TB.round(5).to_csv(f"{OUT}/v4_strategies.csv",index=False)
print("\nSTRATEGIES vs none\n",TB.round(4).to_string(index=False))
# strategy-selected macro: best clf per strategy by val
g=R.groupby(KEY+['strategy','classifier']).agg(val_f2=('val_f2','mean')).reset_index()
sel_s=g.sort_values('val_f2',ascending=False).groupby(KEY+['strategy']).head(1)
Rs=R.merge(sel_s[KEY+['strategy','classifier']],on=KEY+['strategy','classifier'])
SM=Rs.groupby(['protocol','feature_set','strategy','target']).test_f2.mean().groupby(['protocol','feature_set','strategy']).mean().unstack()
print("\nSTRATEGY macro\n",SM.round(3).to_string())
print("\nclassifier of selected configs\n", best.groupby(['protocol','classifier']).size().unstack(fill_value=0))
print("\nsampler audit (SMOTENC type-aware bad rows)\n", R[R.strategy=='SMOTENC_typeaware'].groupby('protocol').syn_bad_row_frac.mean())
P5=pd.read_csv(f"{A}/v4_P5_bootstrap.csv"); print("\nP5 bootstrap\n",P5.round(3).to_string(index=False))
