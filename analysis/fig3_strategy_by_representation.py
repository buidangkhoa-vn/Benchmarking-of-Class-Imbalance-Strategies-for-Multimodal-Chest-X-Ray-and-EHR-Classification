"""Fig. 3: paired change in test F2 vs. no augmentation, per representation group (EHR only / image only / early fusion),
for P1-P3. Unit = outer fold (mean over targets, classifiers and representations of the group); 95% CI from the
corrected resampled t-test (Nadeau & Bengio), variance x (1/k + n_test/n_train) with k=5 and n_test/n_train = 0.2/0.64."""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
O = os.environ.get('MMMDA_OUT', 'out_v2'); F = os.environ.get('MMMDA_FIG_DIR', 'figures'); os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.3,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.linewidth': 0.6, 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b'})
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'; INK = '#0b0b0b'
R = pd.read_csv(f'{O}/results_all_merged_15680.csv')
R = R[R.strategy != 'native'].copy(); R['strat'] = R.strategy.replace({'SMOTE': 'SMOTE_naive'})
G = {'ehr': 'EHR only', 'img_xrv': 'Image only', 'img_xrv1024': 'Image only', 'img_raddino': 'Image only',
     'fusion_xrv': 'Early fusion', 'fusion_xrv1024': 'Early fusion', 'fusion_raddino': 'Early fusion'}
idx = ['protocol', 'feature_set', 'target', 'classifier', 'fold']
base = R[R.strat == 'none'].set_index(idx).test_f2
K, RATIO = 5, 0.2 / 0.64; rows = []
for s in ['class_weight', 'ROS', 'SMOTE_naive', 'SMOTENC_typeaware']:
    d = R[R.strat == s].set_index(idx).test_f2; c = d.index.intersection(base.index)
    diff = (d.loc[c] - base.loc[c]).reset_index(); diff['grp'] = diff.feature_set.map(G)
    for (p, g), dp in diff.groupby(['protocol', 'grp']):
        f = dp.groupby('fold').test_f2.mean(); m = f.mean(); se = np.sqrt((1 / K + RATIO) * f.var(ddof=1))
        h = stats.t.ppf(0.975, K - 1) * se
        rows.append(dict(protocol=p, group=g, strategy=s, mean=m, lo=m - h, hi=m + h, p=2 * stats.t.sf(abs(m / se), K - 1) if se > 0 else 1.0, n_pairs=len(dp)))
FS = pd.DataFrame(rows); FS.round(4).to_csv(f'{O}/fig3_strategy_by_representation.csv', index=False)
SN = {'class_weight': 'Class weighting', 'ROS': 'Random oversampling', 'SMOTE_naive': 'SMOTE (naive)', 'SMOTENC_typeaware': 'SMOTE-NC (type-aware)'}
PN = {'P1_record': 'P1 record split', 'P2_grouped': 'P2 patient-grouped', 'P3_patient': 'P3 patient-level'}
off = {'P1_record': 0.24, 'P2_grouped': 0.0, 'P3_patient': -0.24}; col = {'P1_record': C2, 'P2_grouped': C3, 'P3_patient': C1}
mk = {'P1_record': 's', 'P2_grouped': '^', 'P3_patient': 'o'}; strats = list(SN)
fig, axes = plt.subplots(1, 3, figsize=(3.45, 2.15), sharey=True)
lim_lo = min(-0.03, FS.lo.min() * 1.08); lim_hi = max(0.03, FS.hi.max() * 1.1)
for ax, g in zip(axes, ['EHR only', 'Image only', 'Early fusion']):
    for p in ['P1_record', 'P2_grouped', 'P3_patient']:
        d = FS[(FS.protocol == p) & (FS.group == g)].set_index('strategy')
        ss = [s for s in strats if s in d.index]; ys = [len(strats) - 1 - strats.index(s) + off[p] for s in ss]
        if not ss: continue
        ax.errorbar(d.loc[ss, 'mean'], ys, xerr=[d.loc[ss, 'mean'] - d.loc[ss, 'lo'], d.loc[ss, 'hi'] - d.loc[ss, 'mean']], fmt=mk[p], ms=3.2,
                    color=col[p], ecolor=col[p], elinewidth=0.8, capsize=1.2, mec='white', mew=0.4, label=PN[p])
    ax.axvline(0, color=INK, lw=0.6); ax.set_xlim(lim_lo, lim_hi); ax.set_title(g, fontsize=7, pad=2)
    ax.xaxis.grid(True, color='#e6e5e0', lw=0.5); ax.set_axisbelow(True); ax.set_xticks([-0.08, -0.04, 0]); ax.set_xticks([0.03], minor=True)
    ax.set_xticklabels(['−.08', '−.04', '0'])
axes[0].set_yticks(range(len(strats))); axes[0].set_yticklabels([SN[s] for s in strats[::-1]])
axes[1].set_xlabel(r'$\Delta$ test F2 vs. no augmentation')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='upper center', ncol=3, fontsize=6, handletextpad=0.1, columnspacing=0.6, bbox_to_anchor=(0.6, 1.01))
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93), w_pad=0.4)
fig.savefig(f'{F}/fig3_strategy_effect.pdf'); fig.savefig(f'{F}/fig3_strategy_effect.png', dpi=250)
print(FS.round(4).to_string(index=False))
