"""Paper figures 1 and 3 (and the original version of Fig. 2) from the outputs of analyze_v2_merge.py."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
O=os.environ.get('MMMDA_OUT','out_v2'); F=os.environ.get('MMMDA_FIG_DIR','figures')
os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e',
                     'axes.linewidth':0.6,'xtick.color':'#52514e','ytick.color':'#52514e','text.color':'#0b0b0b'})
C1,C2,C3='#2a78d6','#eb6834','#1baf7a'; INK='#0b0b0b'; MUTED='#8a8984'
TA=pd.read_csv(f'{O}/tableA.csv'); B=pd.read_csv(f'{O}/baselines.csv'); R=pd.read_csv(f'{O}/results_all_merged_15680.csv')
PT=pd.read_csv(f'{O}/per_target_selected.csv')
PN={'P1_record':'P1 record split','P2_grouped':'P2 patient-grouped','P3_patient':'P3 patient-level'}
# ---------------- Fig 1: protocol effect (3 base representations) ----------------
fig,ax=plt.subplots(figsize=(3.45,2.45))
reps=[('ehr','EHR only',C1,None),('fusion_xrv','Early fusion',C2,None),('img_xrv','Image only (11 scores)',C3,'////')]
protos=['P1_record','P2_grouped','P3_patient']; w=0.24
for i,(fs,lab,c,h) in enumerate(reps):
    vals=[TA[(TA.protocol==p)&(TA.feature_set==fs)].f2.values[0] for p in protos]
    sds=[TA[(TA.protocol==p)&(TA.feature_set==fs)].f2_sd.values[0] for p in protos]
    x=np.arange(3)+(i-1)*(w+0.02)
    ax.bar(x,vals,w,color=c,label=lab,hatch=h,edgecolor='white',linewidth=0.4,zorder=2)
    ax.errorbar(x,vals,yerr=sds,fmt='none',ecolor=INK,elinewidth=0.6,capsize=1.5,zorder=3)
    if fs=='ehr':
        for xi,v in zip(x,vals): ax.text(xi,v+0.025,f'{v:.2f}',ha='center',va='bottom',fontsize=6.5,color=INK)
LB=B.groupby('protocol')[['trivial_f2','lookup_f2']].mean()
for j,p in enumerate(protos):
    ax.hlines(LB.loc[p,'trivial_f2'],j-0.42,j+0.42,colors=INK,linestyles='--',lw=0.8,zorder=4,label='Predict all positive' if j==0 else None)
    ax.hlines(LB.loc[p,'lookup_f2'],j-0.42,j+0.42,colors='#d03b3b',linestyles=':',lw=1.1,zorder=4,label='EHR lookup table (no learning)' if j==0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([PN[p] for p in protos]); ax.set_ylabel('Macro test F2 (8 targets)')
ax.set_ylim(0,1.0); ax.yaxis.grid(True,color='#e6e5e0',lw=0.5,zorder=0); ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(f'{F}/fig1_protocol_effect.pdf'); fig.savefig(f'{F}/fig1_protocol_effect.png',dpi=250)
# ---------------- Fig 2: per-target EHR, P1 vs P3 ----------------
order=['pneumonia','lung_cancer','fracture','bronchitis','arthritis','lung_infection','tuberculosis','scoliosis']
nice={'pneumonia':'Pneumonia','lung_cancer':'Lung cancer','fracture':'Fracture','bronchitis':'Bronchitis','arthritis':'Arthritis',
      'lung_infection':'Lung infection','tuberculosis':'Tuberculosis','scoliosis':'Scoliosis$^*$'}
fig,ax=plt.subplots(figsize=(3.45,2.5))
e=PT[PT.feature_set=='ehr'].set_index(['protocol','target'])
for i,t in enumerate(order[::-1]):
    a1=e.loc[('P1_record',t),'test_f2']; a3=e.loc[('P3_patient',t),'test_f2']; tv=e.loc[('P3_patient',t),'test_trivial_f2']
    ax.plot([a3,a1],[i,i],'-',color='#c3c2b7',lw=1.2,zorder=1)
    ax.plot(a1,i,'o',ms=4.5,color=C2,mec='white',mew=0.6,zorder=3,label='P1 record split' if i==0 else None)
    ax.plot(a3,i,'o',ms=4.5,color=C1,mec='white',mew=0.6,zorder=3,label='P3 patient-level' if i==0 else None)
    ax.plot(tv,i,'|',ms=7,color=INK,mew=1.0,zorder=2,label='Predict all positive (P3)' if i==0 else None)
ax.set_yticks(range(8)); ax.set_yticklabels([nice[t] for t in order[::-1]]); ax.set_xlim(-0.02,1.0)
ax.set_xlabel('Test F2, selected EHR-only model'); ax.xaxis.grid(True,color='#e6e5e0',lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.42,1.0),ncol=3,handletextpad=0.2,columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(f'{F}/fig2_targets_P1_vs_P3.pdf'); fig.savefig(f'{F}/fig2_targets_P1_vs_P3.png',dpi=250)
# ---------------- Fig 3: strategy effect forest (paired vs none, same classifier) ----------------
Rd=R[R.strategy!='native'].copy()
Rd['strat']=Rd.strategy.replace({'SMOTE':'SMOTE_naive'})
base=Rd[Rd.strat=='none'].set_index(['protocol','feature_set','target','classifier','fold']).test_f2
rows=[]
for s in ['class_weight','ROS','SMOTE_naive','SMOTENC_typeaware']:
    d=Rd[Rd.strat==s].set_index(['protocol','feature_set','target','classifier','fold']).test_f2
    c=d.index.intersection(base.index); diff=(d.loc[c]-base.loc[c]).reset_index()
    for p in ['P2_grouped','P3_patient','P1_record']:
        dp=diff[diff.protocol==p]
        if not len(dp): continue
        agg=dp.groupby(['classifier','fold']).test_f2.mean()   # n = classifiers x folds
        m=agg.mean(); se=agg.std(ddof=1)/np.sqrt(len(agg)); tcrit=stats.t.ppf(0.975,len(agg)-1)
        rows.append(dict(protocol=p,strategy=s,mean=m,lo=m-tcrit*se,hi=m+tcrit*se,n_units=len(agg),n_pairs=len(dp),maxabs=dp.test_f2.abs().max()))
FS=pd.DataFrame(rows); FS.round(4).to_csv(f'{O}/fig3_strategy_forest.csv',index=False)
SN={'class_weight':'Class weighting','ROS':'Random oversampling','SMOTE_naive':'SMOTE (naive)','SMOTENC_typeaware':'SMOTE-NC (type-aware)'}
fig,ax=plt.subplots(figsize=(3.45,2.05))
strats=list(SN); off={'P1_record':0.24,'P2_grouped':0.0,'P3_patient':-0.24}; col={'P1_record':C2,'P2_grouped':C3,'P3_patient':C1}
mk={'P1_record':'s','P2_grouped':'^','P3_patient':'o'}
for p in ['P1_record','P2_grouped','P3_patient']:
    d=FS[FS.protocol==p].set_index('strategy')
    ys=[len(strats)-1-i+off[p] for i,s in enumerate(strats) if s in d.index]
    ss=[s for s in strats if s in d.index]
    ax.errorbar(d.loc[ss,'mean'],ys,xerr=[d.loc[ss,'mean']-d.loc[ss,'lo'],d.loc[ss,'hi']-d.loc[ss,'mean']],fmt=mk[p],ms=4,color=col[p],
                ecolor=col[p],elinewidth=0.9,capsize=1.5,mec='white',mew=0.5,label=PN[p])
ax.axvline(0,color=INK,lw=0.7)
ax.set_yticks(range(len(strats))); ax.set_yticklabels([SN[s] for s in strats[::-1]])
ax.set_xlabel(r'$\Delta$ test F2 vs. no augmentation'); ax.xaxis.grid(True,color='#e6e5e0',lw=0.5); ax.set_axisbelow(True)
lim=max(0.03,FS[['lo','hi']].abs().max().max()*1.1); ax.set_xlim(-lim,lim)
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.35,1.0),ncol=3,fontsize=6,handletextpad=0.1,columnspacing=0.6)
fig.tight_layout(pad=0.3); fig.savefig(f'{F}/fig3_strategy_effect.pdf'); fig.savefig(f'{F}/fig3_strategy_effect.png',dpi=250)
print(FS.round(4).to_string(index=False))
