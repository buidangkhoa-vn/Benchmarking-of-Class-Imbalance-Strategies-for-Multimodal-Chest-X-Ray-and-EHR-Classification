"""Robustness checks quoted in the paper (Sec. IV-E/F, V-F):
 1) configuration selected within each outer fold (strictly nested) vs. averaged over folds;
 2) macro results without the three rarest targets;
 3) P5: share of training radiographs that fall in the test period;
 4) prior-diagnosis flags: earlier admissions still open at the radiograph's admission."""
import os, pandas as pd, numpy as np
D = os.environ.get('MMMDA_DATA_DIR', 'data')                      # x_ray_and_features_data.csv
V2 = os.environ.get('MMMDA_OUT', 'out_v2')                        # results_all_merged_15680.csv (analyze_v2_merge.py)
A4 = os.environ.get('MMMDA_V4_ANALYSIS', 'v4_2026-10-19_admlabels/v4_analysis')
A5 = os.environ.get('MMMDA_V5_ANALYSIS', 'v5_2026-10-19_chexpert/v5_analysis')
A7 = os.environ.get('MMMDA_V7_ANALYSIS', 'v7_2026-10-22_ablation/v7_analysis')
BQ = os.environ.get('MMMDA_V3_BQ', 'v3_2026-10-19_P4P5/v3_bq')
P5M = os.environ.get('MMMDA_P5_MANIFEST', 'v3_2026-10-19_P4P5/v3_manifests/v3_folds_P5_temporal.csv')
KEY = ['protocol', 'feature_set', 'target']; RARE = ['scoliosis', 'tuberculosis', 'lung_infection']
runs = {'v2': (pd.read_csv(f'{V2}/results_all_merged_15680.csv'), ['strategy', 'classifier']),
        'v4': (pd.read_csv(f'{A4}/v4_results_all.csv'), ['strategy', 'classifier']),
        'v5': (pd.read_csv(f'{A5}/v5_results_all.csv'), ['strategy', 'classifier']),
        'v7': (pd.read_csv(f'{A7}/v7_results_all.csv.gz'), ['config'])}
def sel_avg(R, cfg):
    g = R.groupby(KEY + cfg).val_f2.mean().reset_index()
    return R.merge(g.sort_values('val_f2', ascending=False).groupby(KEY).head(1)[KEY + cfg], on=KEY + cfg)
print('1) per-fold vs fold-averaged selection (macro test F2)')
out = []
for v, (R, cfg) in runs.items():
    R = R[~R.protocol.str.startswith('P5')]
    a = sel_avg(R, cfg).groupby(['protocol', 'feature_set', 'fold']).test_f2.mean().groupby(['protocol', 'feature_set']).mean()
    b = R.sort_values('val_f2', ascending=False).groupby(KEY + ['fold']).head(1).groupby(['protocol', 'feature_set', 'fold']).test_f2.mean().groupby(['protocol', 'feature_set']).mean()
    out.append(pd.DataFrame({'run': v, 'avg_sel': a, 'per_fold': b, 'diff': b - a}))
O = pd.concat(out); print(O.round(4).to_string()); print('paper tables (v2,v4,v5): diff range', O[O.run != 'v7']['diff'].min().round(4), O[O.run != 'v7']['diff'].max().round(4), 'mean', O[O.run != 'v7']['diff'].mean().round(4))
print('\n2) margin over all-positive without the three rarest targets')
for v, (R, cfg) in runs.items():
    if v == 'v7': continue
    d = sel_avg(R, cfg); d = d[~d.target.isin(RARE)]
    fm = d.groupby(['protocol', 'feature_set', 'fold']).agg(f2=('test_f2', 'mean'), tv=('test_trivial_f2', 'mean')).groupby(['protocol', 'feature_set']).mean()
    print((fm.f2 - fm.tv).round(4).to_string())
print('\n3) P5: training radiographs in the test period')
df = pd.read_csv(f'{D}/x_ray_and_features_data.csv', usecols=['subject_id', 'hadm_id', 'StudyDate'])
pts = pd.read_csv(f'{BQ}/v3_bq_patients.csv.gz').drop_duplicates('subject_id').set_index('subject_id')
man = pd.read_csv(P5M)
est = pts.anchor_year_group.str.slice(0, 4).astype(int).reindex(df.subject_id).values + 1 + ((df.StudyDate // 10000).values - pts.anchor_year.reindex(df.subject_id).values)
role = df.subject_id.map(man.set_index('subject_id').role); t0 = man[man.role == 'test'].est_first_year.min()
tr = role == 'train'; print(f'test period starts {t0}; training radiographs in it: {int((est[tr] >= t0).sum())} of {int(tr.sum())} ({(est[tr] >= t0).mean():.1%})')
print('\n4) earlier admissions still open at the radiograph admission')
adm = pd.read_csv(f'{BQ}/v3_bq_admissions.csv.gz', parse_dates=['admittime', 'dischtime'])
cur = df.merge(adm[['hadm_id', 'admittime']].rename(columns={'admittime': 'a0'}), on='hadm_id')
m = cur[['subject_id', 'hadm_id', 'a0']].merge(adm[['subject_id', 'hadm_id', 'admittime', 'dischtime']], on='subject_id', suffixes=('', '_e'))
m = m[(m.admittime < m.a0) & (m.hadm_id_e != m.hadm_id)]
print('earlier admissions (pairs):', len(m), '| still open:', int((m.dischtime > m.a0).sum()))
