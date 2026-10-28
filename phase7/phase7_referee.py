"""Answers to a self-written referee report on docs/01 (see docs/01_referee_report.md). Each block is one point.

R3  light spread to the 'unsilenced' hemisphere: in VGAT-ChR2 mice light drives inhibitory (FS) cells UP. FS cells
    of B (identified by being driven up >= 2x by B's own light) should then also rise when only A is lit, if light
    reached B. Report the distribution of their rate ratio under A's light.
R4  behaviour/arousal: does unilateral silencing change early licking (a gross sign of movement during the delay)?
    Early-lick trials are excluded from all analyses; this checks that the exclusion does not select differently.
R5  baseline for the 'correlated subspace' result: fraction of the response in B's own top-3 PCs (Phase 2 output).
R10 'gates are not needed' is an absence of evidence; give the mouse-level 95% CI of (gate - fixed).
R11 individual differences were pair-level correlations; give within-mouse versions (pairs demeaned per mouse).

usage: python phase7_referee.py CACHE_DIR RESULTS_DIR
"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
CACHE, R = sys.argv[1:3]
rng = np.random.default_rng(0); out = {}

# R3
d = pd.read_csv(os.path.join(R, 'units_fr_all.csv'))
f = d[(d.celltype == 'FS') & (d.fr0 >= 2)].copy(); f['r1'] = f.fr1 / f.fr0; f['r2'] = f.fr2 / f.fr0
f = f[np.maximum(f.r1, f.r2) >= 2]; own = np.maximum(f.r1, f.r2); oth = np.minimum(f.r1, f.r2)
out['R3_light_spread'] = dict(n_FS=len(f), own_light_ratio_median=float(own.median()), other_light_ratio_median=float(oth.median()),
                              frac_other_ge_1p5=float((oth >= 1.5).mean()), frac_other_ge_2=float((oth >= 2).mean()),
                              frac_other_lt_0p8=float((oth < 0.8).mean()))
# R4
rows = []
for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
    z = np.load(p); g = z['GoodTrials'].astype(bool) & (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05)
    pt = z['PhotostimulationType']; le = z['LickEarly'].astype(bool)
    rows.append(dict(mouse=os.path.basename(p).split('__')[0], ctrl=le[g & (pt == 0)].mean(),
                     uni=np.r_[le[g & (pt == 1)], le[g & (pt == 2)]].mean()))
el = pd.DataFrame(rows); mm = el.groupby('mouse')[['ctrl', 'uni']].median()
out['R4_early_licks'] = dict(ctrl_mean=float(el.ctrl.mean()), unilateral_mean=float(el.uni.mean()),
                             p_mouse=float(wilcoxon(mm.ctrl, mm.uni).pvalue), n_mice=len(mm))
# R5
p2 = pd.read_csv(os.path.join(R, 'phase2', 'phase2_pairs.csv')); p2 = p2[p2.epoch == 'during']
out['R5_baselines_3dim'] = {k: float(p2[k].median()) for k in ('f_corr3', 'f_pc3', 'f_sig3', 'f_rand3')}
# R10
m3 = pd.read_csv(os.path.join(R, 'phase3', 'phase3_models.csv'))
m3 = m3[(m3.train == 'ctrl+stim') & (m3.L == 1) & m3.bits.isna() & (m3.vsrc == 'rrr') & (m3.d == 'full')]
m3['mouse'] = m3.session.str.split('/').str[0]
g = m3.groupby(['session', 'silenced', 'mouse', 'gate']).r2d_during.mean().unstack('gate')
for gate in ('2state', 'prec'):
    pm = (g[gate] - g['fixed']).dropna().groupby(level='mouse').median()
    bs = [np.median(rng.choice(pm.values, len(pm))) for _ in range(5000)]
    out[f'R10_{gate}_minus_fixed'] = dict(mouse_median=float(pm.median()), ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                                          fixed_median=float(g['fixed'].median()), n_mice=len(pm))
# R11
mo = pd.read_csv(os.path.join(R, 'phase4', 'phase4_modularity.csv')); be = pd.read_csv(os.path.join(R, 'phase5', 'phase5_behaviour.csv'))
x = mo.dropna(subset=['modularity', 'recovery']).copy()
w = lambda df, c: df[c] - df.groupby('mouse')[c].transform('mean')
r = spearmanr(w(x, 'modularity'), w(x, 'recovery')); out['R11_modularity_recovery_within_mouse'] = dict(rho=r.statistic, p=r.pvalue, n=len(x))
y = mo.merge(be, on=['session', 'mouse', 'silenced', 'receiving'])
y = y.merge(mo[['session', 'receiving', 'modularity']].rename(columns={'receiving': 'silenced', 'modularity': 'mod_sil'}), on=['session', 'silenced'])
y['drop'] = y.perf_ctrl - y.perf_stim; y['lead'] = y.mod_sil - y.modularity; y = y.dropna(subset=['modularity', 'drop', 'lead'])
r = spearmanr(w(y, 'modularity'), w(y, 'drop')); out['R11_modularity_drop_within_mouse'] = dict(rho=r.statistic, p=r.pvalue, n=len(y))
r = spearmanr(w(y, 'lead'), w(y, 'drop')); out['R11_leader_drop_within_mouse'] = dict(rho=r.statistic, p=r.pvalue, n=len(y))
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(R, 'phase7', 'phase7_referee.json'), 'w'), indent=1, default=float)
