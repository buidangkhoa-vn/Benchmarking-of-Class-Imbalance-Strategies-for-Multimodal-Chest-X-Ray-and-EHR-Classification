"""Paper Fig. 2: per-target F2 margin over all-positive for P1 (released labels), P1-A (admission-aligned) and P3."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
O=os.environ.get('MMMDA_OUT','out_v2'); F=os.environ.get('MMMDA_FIG_DIR','figures'); os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.3,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e',
                     'axes.linewidth':0.6,'xtick.color':'#52514e','ytick.color':'#52514e','text.color':'#0b0b0b'})
C1,C2,C3='#2a78d6','#eb6834','#1baf7a'; INK='#0b0b0b'
PT=pd.read_csv(f'{O}/per_target_selected.csv')
P4=pd.read_csv(os.environ.get('MMMDA_V4_OUT','out_v4')+'/v4_per_target.csv')
order=['pneumonia','lung_cancer','fracture','bronchitis','arthritis','lung_infection','tuberculosis','scoliosis']
nice={'pneumonia':'Pneumonia','lung_cancer':'Lung cancer','fracture':'Fracture','bronchitis':'Bronchitis','arthritis':'Arthritis',
      'lung_infection':'Lung infection','tuberculosis':'Tuberculosis','scoliosis':'Scoliosis$^*$'}
e=PT[PT.feature_set=='ehr'].set_index(['protocol','target'])
a=P4[(P4.protocol=='P1_admission')&(P4.feature_set=='ehr')].set_index('target')
fig,ax=plt.subplots(figsize=(3.45,2.55))
for i,t in enumerate(order[::-1]):
    a1=e.loc[('P1_record',t),'test_f2']-e.loc[('P1_record',t),'test_trivial_f2']; a3=e.loc[('P3_patient',t),'test_f2']-e.loc[('P3_patient',t),'test_trivial_f2']
    aa=a.loc[t,'test_f2']-a.loc[t,'test_trivial_f2']
    ax.plot([min(a3,aa),a1],[i,i],'-',color='#c3c2b7',lw=1.2,zorder=1)
    ax.plot(a1,i,'o',ms=4.5,color=C2,mec='white',mew=0.6,zorder=3,label='P1, released labels' if i==0 else None)
    ax.plot(aa,i,'D',ms=4.0,color=C3,mec='#0b0b0b',mew=0.5,zorder=4,label='P1-A, admission-aligned' if i==0 else None)
    ax.plot(a3,i,'o',ms=4.5,color=C1,mec='white',mew=0.6,zorder=3,label='P3 patient-level' if i==0 else None)
ax.axvline(0,color=INK,lw=0.8,zorder=2)
ax.set_yticks(range(8)); ax.set_yticklabels([nice[t] for t in order[::-1]]); ax.set_xlim(-0.06,0.9)
ax.set_xlabel('F2 margin over all-positive (EHR-only)'); ax.xaxis.grid(True,color='#e6e5e0',lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.40,1.0),ncol=3,handletextpad=0.2,columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(f'{F}/fig2_targets_P1_vs_P3.pdf'); fig.savefig(f'{F}/fig2_targets_P1_vs_P3.png',dpi=250)
print(pd.DataFrame({t:[e.loc[('P1_record',t),'test_f2']-e.loc[('P1_record',t),'test_trivial_f2'],a.loc[t,'test_f2']-a.loc[t,'test_trivial_f2'],e.loc[('P3_patient',t),'test_f2']-e.loc[('P3_patient',t),'test_trivial_f2']] for t in order},index=['P1','P1A','P3']).round(3).T)
