"""v7: hyperparameter ablation (regularised trees, tuned LR) and fully aligned subset (paper Sec. V-F)."""
import pandas as pd, numpy as np, os, sys
from scipy.stats import wilcoxon
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
A=os.environ.get('MMMDA_V7_ANALYSIS','v7_2026-10-22_ablation/v7_analysis'); OUT=os.environ.get('MMMDA_V7_OUT','out_v7'); os.makedirs(OUT,exist_ok=True)
R=pd.read_csv(f'{A}/v7_results_all.csv.gz')
print(R.shape, R.protocol.value_counts().to_dict())
KEY=['protocol','feature_set','target']
agg=R.groupby(KEY+['family','config']).agg(val_f2=('val_f2','mean'),n=('fold','size')).reset_index()
assert (agg.n==5).all()
def pick(sub): return sub.sort_values('val_f2',ascending=False).groupby(KEY).head(1)[KEY+['config']]
SEL={'LR_default':agg[agg.config=='LR_C1_default'][KEY+['config']],'LR_tuned':pick(agg[agg.family=='LR'])}
for f in ['RF','ET','LGBM']:
    SEL[f+'_default']=agg[agg.config==f+'_default'][KEY+['config']]; SEL[f+'_tuned']=pick(agg[agg.family==f])
SEL['trees_default']=pick(agg[agg.config.isin(['RF_default','ET_default','LGBM_default'])])
SEL['trees_tuned']=pick(agg[agg.family.isin(['RF','ET','LGBM'])]); SEL['overall_tuned']=pick(agg)
rows=[]; SR={}
for s,sel in SEL.items():
    m=R.merge(sel,on=KEY+['config']); SR[s]=m
    fm=m.groupby(['protocol','feature_set','fold']).agg(f2=('test_f2','mean'),tv=('test_trivial_f2','mean'),au=('test_roc_auc','mean'),pr=('test_pr_auc','mean')).reset_index()
    t=fm.groupby(['protocol','feature_set']).agg(f2=('f2','mean'),sd=('f2','std'),tv=('tv','mean'),au=('au','mean'),pr=('pr','mean')).reset_index(); t['d']=t.f2-t.tv; t['sel']=s; rows.append(t)
SUM=pd.concat(rows); SUM.to_csv(f'{OUT}/v7_summary.csv',index=False)
print('\nMACRO F2 by selection'); print(SUM.pivot_table(index=['protocol','feature_set'],columns='sel',values='f2').round(3)[['LR_default','LR_tuned','RF_default','RF_tuned','ET_default','ET_tuned','LGBM_default','LGBM_tuned','trees_default','trees_tuned','overall_tuned']].to_string())
print('\nMACRO AUROC by selection'); print(SUM.pivot_table(index=['protocol','feature_set'],columns='sel',values='au').round(3)[['LR_default','LR_tuned','trees_default','trees_tuned','overall_tuned']].to_string())
print('\nDelta vs all-positive'); print(SUM.pivot_table(index=['protocol','feature_set'],columns='sel',values='d').round(3)[['LR_default','LR_tuned','trees_default','trees_tuned','overall_tuned']].to_string())
# paired tests
def paired(a,b,metric='test_f2'):
    x=SR[a].set_index(KEY+['fold'])[metric]; y=SR[b].set_index(KEY+['fold'])[metric]; i=x.index.intersection(y.index)
    out=[]
    for (p,fs),g in pd.DataFrame({'a':x.loc[i],'b':y.loc[i]}).groupby(level=[0,1]):
        d=g.a-g.b; out.append(dict(protocol=p,feature_set=fs,comp=f'{a}-{b}',metric=metric,diff=d.mean(),p=wilcoxon(d).pvalue if (d!=0).any() else 1,n=len(d)))
    return pd.DataFrame(out)
PT=pd.concat([paired('trees_tuned','LR_tuned'),paired('trees_tuned','trees_default'),paired('LR_tuned','LR_default'),paired('overall_tuned','LR_default'),
              paired('trees_tuned','LR_tuned','test_roc_auc'),paired('overall_tuned','LR_default','test_roc_auc')])
PT.to_csv(f'{OUT}/v7_paired.csv',index=False); print('\nPAIRED'); print(PT.round(4).to_string(index=False))
# family chosen when all compete
ch=SEL['overall_tuned'].merge(agg[KEY+['config','family']],on=KEY+['config'])
print('\nfamily chosen (overall)'); print(ch.groupby(['protocol','family']).size().unstack(fill_value=0).to_string())
print('\nconfig chosen within trees'); print(SEL['trees_tuned'].groupby(['protocol','config']).size().unstack(fill_value=0).to_string())
# P1 memorisation: per config (no selection), EHR and fusion
P1=R[R.protocol=='P1_rel']
pc=P1.groupby(['feature_set','config'])[['test_f2','seen_f2','unseen_f2','unseen_trivial_f2','test_trivial_f2']].mean()
print('\nP1 per config (macro over targets and folds)'); print(pc.round(3).to_string())
pc.to_csv(f'{OUT}/v7_P1_per_config.csv')
for s in ['LR_default','trees_default','trees_tuned','overall_tuned']:
    m=SR[s]; m=m[m.protocol=='P1_rel']
    print(s, m.groupby('feature_set')[['test_f2','seen_f2','unseen_f2','unseen_trivial_f2']].mean().round(3).to_dict('index'))
# aligned vs misaligned in P2-split protocols
for s in ['LR_default','overall_tuned']:
    m=SR[s]; m=m[m.split=='P2']
    am=m.groupby(['protocol','feature_set'])[['test_f2','test_trivial_f2','al_f2','al_trivial_f2','mis_f2','mis_trivial_f2','al_roc_auc','mis_roc_auc','al_n','mis_n']].mean()
    am['al_d']=am.al_f2-am.al_trivial_f2; am['mis_d']=am.mis_f2-am.mis_trivial_f2
    print(f'\naligned vs misaligned ({s})'); print(am.round(3).to_string()); am.to_csv(f'{OUT}/v7_aligned_{s}.csv')
# P2al protocols: modality comparison
for s in ['LR_default','overall_tuned']:
    t=SUM[(SUM.sel==s)&SUM.protocol.str.startswith('P2al')]
    print(f'\nP2al ({s})'); print(t[['protocol','feature_set','f2','sd','tv','d','au','pr']].round(3).to_string(index=False))
P2a=pd.concat([paired('overall_tuned','overall_tuned')]) if False else None
def modal(s,p,a,b,metric):
    m=SR[s]; m=m[m.protocol==p]
    x=m[m.feature_set==a].set_index(['target','fold'])[metric]; y=m[m.feature_set==b].set_index(['target','fold'])[metric]; i=x.index.intersection(y.index); d=x.loc[i]-y.loc[i]
    return dict(sel=s,protocol=p,a=a,b=b,metric=metric,diff=d.mean(),p=wilcoxon(d).pvalue,n=len(d))
mm=[]
for s in ['LR_default','overall_tuned']:
    for p in ['P2al_adm','P2al_cln']:
        for a,b in [('img_raddino','ehr'),('fusion_raddino','ehr'),('fusion_raddino','img_raddino')]:
            for met in ['test_f2','test_roc_auc']: mm.append(modal(s,p,a,b,met))
    for p in ['P2_rel','P4_adm','P4C_cln']:
        for met in ['test_f2','test_roc_auc']: mm.append(modal(s,p,'fusion_raddino','ehr',met))
MM=pd.DataFrame(mm); MM.to_csv(f'{OUT}/v7_modality.csv',index=False); print('\nMODALITY'); print(MM.round(4).to_string(index=False))
# trivial test of EHR vs all-positive on aligned subset
for s in ['LR_default','overall_tuned']:
    m=SR[s]
    for p in ['P2al_adm','P2al_cln']:
        for fs in ['ehr','img_raddino','fusion_raddino']:
            d=m[(m.protocol==p)&(m.feature_set==fs)]; diff=d.test_f2-d.test_trivial_f2
            print(s,p,fs,'delta %.4f p=%.4g n=%d'%(diff.mean(),wilcoxon(diff).pvalue,len(diff)))
