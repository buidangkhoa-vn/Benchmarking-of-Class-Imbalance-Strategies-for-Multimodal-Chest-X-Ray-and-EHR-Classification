"""v2 (P1/P2/P3, released labels): merge the FINAL results with the worker-0 rows (15,680 fits), recompute baselines
from the fold manifests, and produce Tables A/B/C, strategy summaries and the sampler audit (paper Sections IV-V)."""
import pandas as pd, numpy as np, json, sys
from scipy.stats import wilcoxon
import os
U=os.environ.get('MMMDA_DATA_DIR','data')            # x_ray_and_features_data.csv + analysis/ (FINAL outputs: results_all.csv, manifests/)
S=os.environ.get('MMMDA_WORKER0_DIR',U)               # contains results/results_worker0_5125.csv (worker-0 rows recovered from its own results_all.csv)
OUT=os.environ.get('MMMDA_OUT','out_v2'); os.makedirs(OUT,exist_ok=True)
R=pd.concat([pd.read_csv(f'{U}/analysis/results_all.csv'), pd.read_csv(f'{S}/results/results_worker0_5125.csv')]).drop_duplicates('job_key')
assert len(R)==15680
R.to_csv(f'{OUT}/results_all_merged_15680.csv',index=False)
T=['pneumonia','tuberculosis','bronchitis','arthritis','fracture','lung_cancer','scoliosis','lung_infection']
EHR=['age','oxygen','heart_rate','temperature','hemoglobin','rbc','wbc','diabete','hiv','smoke','cough']
df=pd.read_csv(f'{U}/x_ray_and_features_data.csv',low_memory=False)
rec=df; pat=df.groupby('subject_id')[T+EHR].first().reset_index()
# ---------- baselines ----------
def f2(y,yh):
    y=y.astype(bool); yh=yh.astype(bool); tp=(y&yh).sum(); fp=(~y&yh).sum(); fn=(y&~yh).sum(); d=5*tp+4*fn+fp
    return 5*tp/d if d else 0.0
keys={'rec':pd.util.hash_pandas_object(rec[EHR].fillna(-999),index=False).values,'pat':pd.util.hash_pandas_object(pat[EHR].fillna(-999),index=False).values}
rows=[]
for proto,tn in [('P1_record','rec'),('P2_grouped','rec'),('P3_patient','pat')]:
    tab=rec if tn=='rec' else pat; k=keys[tn]
    for f in range(5):
        sp=np.load(f'{U}/analysis/manifests/split_{proto}__s42__f{f}.npz')
        for t in T:
            y=tab[t].values; src=pd.Series(y[sp['idx_tr']],index=k[sp['idx_tr']]); src=src[~src.index.duplicated()]
            def lk(idx,fb):
                m=pd.Series(k[idx]).map(src); return np.where(m.notna(),m.fillna(fb),fb).astype(int), float(m.notna().mean())
            fb=max([0,1],key=lambda b: f2(y[sp['idx_va']],lk(sp['idx_va'],b)[0]))
            yh,hit=lk(sp['idx_te'],fb); yte=y[sp['idx_te']]; p=yte.mean()
            rows.append(dict(protocol=proto,fold=f,target=t,trivial_f2=5*p/(4*p+1),lookup_f2=f2(yte,yh),hit=hit,prev=p))
B=pd.DataFrame(rows); B.to_csv(f'{OUT}/baselines.csv',index=False)
# ---------- selection by inner-val F2 ----------
KEY=['protocol','feature_set','target']
g=R.groupby(KEY+['strategy','classifier']).agg(val_f2=('val_f2','mean'),n=('fold','size')).reset_index()
assert (g.n==5).all()
best=g.sort_values('val_f2',ascending=False).groupby(KEY).head(1)[KEY+['strategy','classifier']]
Rb=R.merge(best,on=KEY+['strategy','classifier'])
Rb.to_csv(f'{OUT}/selected_rows.csv',index=False)
# fold-level macro
fm=Rb.groupby(['protocol','feature_set','fold']).agg(f2=('test_f2','mean'),prauc=('test_pr_auc','mean'),auroc=('test_roc_auc','mean'),triv=('test_trivial_f2','mean'),prev=('test_prev','mean')).reset_index()
TA=fm.groupby(['protocol','feature_set']).agg(f2=('f2','mean'),f2_sd=('f2','std'),prauc=('prauc','mean'),prauc_sd=('prauc','std'),auroc=('auroc','mean'),auroc_sd=('auroc','std'),triv=('triv','mean'))
# PR-AUC lift macro: mean over targets of prauc/prev
lift=Rb.assign(l=Rb.test_pr_auc/Rb.test_prev).groupby(['protocol','feature_set']).l.mean()
TA['prauc_lift']=lift
# paired F2 - trivial over target x fold
pv=[]
for (p,fs),d in Rb.groupby(['protocol','feature_set']):
    diff=d.test_f2-d.test_trivial_f2
    pv.append(dict(protocol=p,feature_set=fs,d_triv=diff.mean(),wins=int((diff>0).sum()),n=len(diff),p=wilcoxon(diff).pvalue))
TA=TA.join(pd.DataFrame(pv).set_index(['protocol','feature_set']))
LB=B.groupby('protocol')[['trivial_f2','lookup_f2','hit']].mean()
TA.round(4).to_csv(f'{OUT}/tableA.csv')
pd.set_option('display.width',250)
print('TABLE A\n',TA.round(3).to_string()); print('\nBASELINES\n',LB.round(3))
# per-target selected (EHR) for P1/P2/P3
PT=Rb.groupby(['protocol','feature_set','target'])[['test_f2','test_trivial_f2','test_pr_auc','test_prev','test_roc_auc']].mean()
PT.round(4).to_csv(f'{OUT}/per_target_selected.csv')
print('\nPER-TARGET EHR\n',PT.xs('ehr',level=1).round(3).unstack(0)['test_f2'].to_string())
print('\nselected configs (counts of classifier by protocol)\n',best.groupby(['protocol','classifier']).size().unstack(fill_value=0))
print(best.groupby(['protocol','strategy']).size().unstack(fill_value=0))
# ---------- Table B ----------
tb=[]
for p in ['P2_grouped','P3_patient','P1_record']:
    Rp=R[(R.protocol==p)&(R.strategy!='native')]
    base=Rp[Rp.strategy=='none'].set_index(['feature_set','target','classifier','fold'])
    for (fs,st),d in Rp[Rp.strategy!='none'].groupby(['feature_set','strategy']):
        dd=d.set_index(['feature_set','target','classifier','fold']); c=dd.index.intersection(base.index)
        diff=dd.loc[c,'test_f2']-base.loc[c,'test_f2']; dpr=dd.loc[c,'test_pr_auc']-base.loc[c,'test_pr_auc']
        tb.append(dict(protocol=p,feature_set=fs,strategy=st,dF2=diff.mean(),dF2_sd=diff.std(),dPR=dpr.mean(),wins=int((diff>0).sum()),losses=int((diff<0).sum()),n=len(diff),p=wilcoxon(diff).pvalue if (diff!=0).any() else 1.0,maxabs=diff.abs().max()))
TB=pd.DataFrame(tb)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); a=np.empty(m); r=0
    for i,j in enumerate(o): r=max(r,(m-i)*p[j]); a[j]=min(1,r)
    return a
TB['holm']=holm(TB.p.values); TB.round(5).to_csv(f'{OUT}/tableB_pooled.csv',index=False)
print('\nTABLE B (pooled over classifiers, paired target x fold x clf)\n',TB[TB.protocol!='P1_record'].round(4).to_string(index=False))
# native ensembles vs best data-level per protocol/feature (macro of selected)
nat=R[R.strategy=='native'].groupby(['protocol','feature_set','classifier','target']).test_f2.mean().groupby(['protocol','feature_set','classifier']).mean().unstack()
print('\nNATIVE macro F2\n',nat.round(3))
# best-of-strategy macro (selection within strategy by val)
sel_s=g.sort_values('val_f2',ascending=False).groupby(KEY+['strategy']).head(1)
Rs=R.merge(sel_s[KEY+['strategy','classifier']],on=KEY+['strategy','classifier'])
SM=Rs.groupby(['protocol','feature_set','strategy','target']).test_f2.mean().groupby(['protocol','feature_set','strategy']).mean().unstack()
SM.round(4).to_csv(f'{OUT}/strategy_macro_selected.csv'); print('\nSTRATEGY macro F2 (best clf per strategy by val)\n',SM.round(3).to_string())
# ---------- sampler audit ----------
aud=R[R.strategy.isin(['SMOTE_naive','SMOTENC_typeaware'])].groupby(['protocol','feature_set','strategy'])[['syn_bad_row_frac','syn_bad_cell_frac']].mean()
aud.round(4).to_csv(f'{OUT}/sampler_audit.csv'); print('\nAUDIT\n',aud.round(3).unstack().to_string())
# ---------- Table C ----------
P1=Rb[Rb.protocol=='P1_record']
TC=P1.groupby('feature_set')[['test_f2','seen_f2','unseen_f2','unseenclean_f2','unseen_trivial_f2','unseen_pr_auc','unseen_prev','unseen_roc_auc','seen_n','unseen_n']].mean()
TC.round(4).to_csv(f'{OUT}/tableC.csv'); print('\nTABLE C\n',TC.round(3).to_string())
# classifier dominance P1 vs P3
print('\nmean test F2 by classifier (EHR, strategy none):')
print(R[(R.feature_set=='ehr')&(R.strategy=='none')].groupby(['protocol','classifier']).test_f2.mean().unstack().round(3))
# lookup per target P1
print('\nLOOKUP per target P1\n',B[B.protocol=='P1_record'].groupby('target')[['lookup_f2','trivial_f2','hit']].mean().round(3))
