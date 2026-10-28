"""Phase 5 figure (+ the FS/pyramidal comparison, computed here from Phase 0 rates).
usage: python phase5_plot.py RESULTS_DIR"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

R = sys.argv[1]
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.4))

# ---- A: FS vs pyramidal in the hemisphere that is NOT silenced
d = pd.read_csv(os.path.join(R, 'units_fr_all.csv'))
f = d[(d.celltype == 'FS') & (d.fr0 >= 2)].copy()
f['r1'] = f.fr1 / f.fr0; f['r2'] = f.fr2 / f.fr0
fs = f[np.maximum(f.r1, f.r2) >= 2]
fs_other = np.minimum(fs.r1, fs.r2).values                    # own hemisphere = the light that activates it
p = pd.read_csv(os.path.join(R, 'units_labeled.csv')); p = p[p.fr0 >= 2]
py_other = np.where(p.hemi == 'code1', p.S2, p.S1)
fs_other.tofile if False else None
pd.DataFrame({'celltype': ['FS'] * len(fs_other) + ['pyramidal'] * len(py_other),
              'ratio_other_side_silenced': np.r_[fs_other, py_other]}).to_csv(os.path.join(R, 'phase5', 'phase5_celltypes.csv'), index=False)
a = ax[0, 0]
for v, col, lab, lw in ((py_other, MUTED, f'pyramidal (n = {len(py_other)})', 2.5), (fs_other, ORANGE, f'fast-spiking (n = {len(fs_other)})', 2.5)):
    s = np.sort(np.clip(v, 0.011, 9.9)); a.plot(s, np.arange(1, len(s) + 1) / len(s), color=col, lw=lw, label=lab)
a.axvline(1, color=INK, lw=1, ls='--'); a.axhline(0.5, color=GRID, lw=1, zorder=0)
a.set_xscale('log'); a.set_xlabel('rate while the OTHER hemisphere is silenced / control')
a.set_ylabel('cumulative fraction of units'); a.legend(frameon=False, fontsize=7, loc='upper left')
a.set_title(f'A  Callosal input drives inhibitory cells\n(median FS {np.median(fs_other):.2f} vs pyramidal {np.median(py_other):.2f})')

# ---- B: rotation of the receiving direction
z = np.load(os.path.join(R, 'phase5', 'phase5_rotation.npz'), allow_pickle=True)
Sc, Sa = z['S_comm'], z['S_act']
a = ax[0, 1]
def curve(S, idx, n):
    return [np.nanmedian([np.nanmean(np.diag(s[np.ix_(idx, idx)], l)) for s in S]) for l in range(n)]
for idx, ls, lab in ((np.arange(0, 8), '-', 'laser on'), (np.arange(8, 17), '--', 'after')):
    n = len(idx)
    a.plot(np.arange(n) * 100, curve(Sc, idx, n), color=BLUE, lw=2, ls=ls, label=f'receiving direction, {lab}')
    a.plot(np.arange(n) * 100, curve(Sa, idx, n), color=MUTED, lw=2, ls=ls, label=f'activity direction, {lab}')
cross = np.nanmedian([np.nanmean(s[np.ix_(range(8), range(8, 17))]) for s in Sc])
a.axhline(cross, color=ORANGE, lw=1.5); a.text(790, cross + 0.03, f'laser-on vs after directions: {cross:.2f}', color=ORANGE, fontsize=7, ha='right')
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xlabel('time lag (ms)'); a.set_ylabel('cross-validated cosine similarity'); a.set_ylim(-0.15, 1.1)
a.legend(frameon=False, fontsize=6.5, loc='lower left', bbox_to_anchor=(0, 0.08)); a.set_title('B  The channel switches at laser offset')

# ---- C, D: behaviour
b = pd.read_csv(os.path.join(R, 'phase5', 'phase5_behaviour.csv')); b['drop'] = b.perf_ctrl - b.perf_stim
md = pd.read_csv(os.path.join(R, 'phase4', 'phase4_modularity.csv'))
md = md[(md.n_sel_rec >= 5) & (md.ctrl_sel_early >= 0.5) & (md.ctrl_sel_late >= 1)].copy(); md['mod_c'] = md.modularity.clip(0, 1)
x = b.merge(md[['session', 'silenced', 'mod_c']], on=['session', 'silenced'])
a = ax[0, 2]; r = spearmanr(x.mod_c, x['drop'])
a.scatter(x.mod_c, 100 * x['drop'], s=10, color=INK, alpha=0.5, linewidths=0)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xlabel('modularity of the hemisphere left running'); a.set_ylabel('drop in % correct when silencing')
a.set_title(f'C  Behaviour follows the choice axis…\n(ρ = {r[0]:.2f}, p = {r[1]:.0e})')
p2 = pd.read_csv(os.path.join(R, 'phase2', 'phase2_pairs.csv')); p2 = p2[p2.epoch == 'during']
y = b.merge(p2[['session', 'silenced', 'power']], on=['session', 'silenced'])
a = ax[1, 0]; r = spearmanr(y.power, y['drop'])
a.scatter(y.power.clip(lower=1e-3), 100 * y['drop'], s=10, color=BLUE, alpha=0.5, linewidths=0)
a.axhline(0, color=GRID, lw=1, zorder=0); a.set_xscale('log')
a.set_xlabel('size of the receiving response (z², Phase 2)'); a.set_ylabel('drop in % correct when silencing')
a.set_title(f'D  …not the size of the receiving response\n(ρ = {r[0]:.2f}, p = {r[1]:.2f})')

# ---- E: choice-axis models
c = pd.read_csv(os.path.join(R, 'phase5', 'phase5_choice_axis.csv')).groupby(['session', 'silenced']).mean(numeric_only=True)
a = ax[1, 1]
mods = [('obs', 'data', INK), ('alone_linear', 'B alone\nlinear', MUTED), ('alone_cubic', 'B alone\nattractor', MUTED),
        ('link_linear', 'fixed\nlink', BLUE), ('link_gate_linear', 'precision\ngate', ORANGE), ('link_gate_cubic', 'gate +\nattractor', ORANGE)]
for i, (m, lab, col) in enumerate(mods):
    for j, (k, mk) in enumerate((('c0', 'o'), ('c1', 's'))):
        v = c[f'{m}_{k}'].values
        a.errorbar(i + (j - 0.5) * 0.3, np.nanmedian(v), yerr=[[np.nanmedian(v) - np.nanpercentile(v, 25)], [np.nanpercentile(v, 75) - np.nanmedian(v)]],
                   fmt=mk, color=col, ms=5, capsize=2)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(len(mods))); a.set_xticklabels([l for _, l, _ in mods], fontsize=7)
a.scatter([], [], marker='o', color=INK, label='c0  mean effect'); a.scatter([], [], marker='s', color=INK, label='c1  × B sure')
a.legend(frameon=False, fontsize=7, loc='lower right'); a.set_ylabel('coefficient (held-out, median, IQR)')
a.set_title('E  Choice axis: no model gives c1 < 0')

# ---- F: AUC
a = ax[1, 2]
items = [('auc_B_laser', 'B\nlaser on', BLUE), ('auc_A_laser', 'A (silenced)\nlaser on', MUTED),
         ('auc_B_end', 'B\nend of delay', BLUE), ('auc_A_end', 'A\nend of delay', MUTED)]
for i, (k, lab, col) in enumerate(items):
    v = b[k].dropna().values
    a.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=6, color=col, alpha=0.4, linewidths=0)
    a.plot([i - 0.2, i + 0.2], [np.median(v)] * 2, color=INK, lw=2.5)
a.axhline(0.5, color=GRID, lw=1, zorder=0)
a.set_xticks(range(4)); a.set_xticklabels([l for _, l, _ in items], fontsize=7.5)
a.set_ylabel('AUC: choice confidence predicts a correct lick'); a.set_title('F  On silencing trials, the remaining\nhemisphere carries the choice')

fig.tight_layout()
fig.savefig(os.path.join(R, 'phase5', 'fig_phase5.png'), dpi=150)
print('saved')
