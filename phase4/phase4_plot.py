"""Phase 4 figure. usage: python phase4_plot.py RESULTS_DIR (the results/ folder)"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, f_oneway

R = sys.argv[1]
md = pd.read_csv(os.path.join(R, 'phase4', 'phase4_modularity.csv'))
gd = pd.read_csv(os.path.join(R, 'phase4', 'phase4_gating_test.csv'))
nl = pd.read_csv(os.path.join(R, 'phase4', 'phase4_nonlinear.csv'))
p3 = pd.read_csv(os.path.join(R, 'phase3', 'phase3_models.csv'))

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
md['incl'] = (md.n_sel_rec >= 5) & (md.ctrl_sel_early >= 0.5) & (md.ctrl_sel_late >= 1)
m = md[md.incl].copy()
m['mod_c'] = m.modularity.clip(0, 1); m['rec_c'] = m.recovery.clip(0, 1)
fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.4))


def strip(a, x, y, col, w=0.12, seed=0, lw=2.5):
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    a.scatter(x + np.random.default_rng(seed).uniform(-w, w, len(y)), y, s=6, color=col, alpha=0.35, linewidths=0)
    a.plot([x - w - 0.04, x + w + 0.04], [np.median(y)] * 2, color=col, lw=lw)


# A: modularity predicts recovery (Chen 2021 Fig. 4H)
a = ax[0, 0]
r = spearmanr(m.mod_c, m.rec_c)
a.scatter(m.mod_c, m.rec_c, s=10, color=INK, alpha=0.5, linewidths=0)
a.set_xlabel('modularity of the unperturbed hemisphere'); a.set_ylabel('recovery of selectivity')
a.set_title(f'A  Modular hemispheres recover better\n(ρ = {r[0]:.2f}, p = {r[1]:.0e}, {len(m)} pairs)')

# B: which hemisphere drives is a trait of the mouse
a = ax[0, 1]
pv = m.pivot_table(index=['mouse', 'session'], columns='receiving', values='mod_c').dropna()
pv['diff'] = pv['code1'] - pv['code2']
pv = pv.reset_index()
keep = pv.groupby('mouse').filter(lambda v: len(v) >= 3)
order = keep.groupby('mouse')['diff'].mean().sort_values().index
for i, mo in enumerate(order):
    y = keep[keep.mouse == mo]['diff'].values
    a.scatter(np.full(len(y), i), y, s=10, color=BLUE, alpha=0.6, linewidths=0)
    a.plot([i - 0.35, i + 0.35], [y.mean()] * 2, color=INK, lw=1.5)
a.axhline(0, color=GRID, lw=1, zorder=0)
F = f_oneway(*[keep[keep.mouse == mo]['diff'].values for mo in order])
a.set_xticks([]); a.set_xlabel(f'mice with ≥ 3 sessions (n = {len(order)}), sorted')
a.set_ylabel('modularity: code-1 minus code-2 hemisphere')
a.set_title(f'B  Which side drives is a trait of the mouse\n(one-way ANOVA across mice, p = {F.pvalue:.0e})')

# C: model-free gating test
a = ax[0, 2]
prop = gd.laser_c0_stim / gd.laser_b0 * gd.laser_b1_cB
items = [(gd.laser_c0_stim, INK, 'c0\nmean effect'), (gd.laser_c1_stim_x_cB, BLUE, 'c1\n× B sure'),
         (prop, MUTED, 'c1 if loss were\nproportional'), (gd.laser_c2_stim_x_cA, ORANGE, 'c2\n× A sure')]
for i, (y, col, lab) in enumerate(items):
    strip(a, i, y.clip(-1.2, 0.8), col, seed=i)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(4)); a.set_xticklabels([l for *_, l in items], fontsize=7.5); a.set_ylim(-1.25, 0.85)
a.set_ylabel("effect on B's choice confidence (laser on, z)")
a.set_title('C  Silencing A, by the state before the laser\n(H4 predicts c1 > 0, c2 < 0)')

# D: nonlinear models: held-out silencing response
a = ax[1, 0]
labs = [('nonlin_none_0', 'no\nlink', MUTED), ('linear_fixed_20', 'linear\nfix 20', INK),
        ('nonlin_fixed_20', 'nonlin\nfix 20', INK), ('nonlin_learned_20', 'learned\ngate 20', BLUE),
        ('nonlin_fixed_3', 'nonlin\nfix 3', INK), ('nonlin_learned_3', 'learned\ngate 3', BLUE),
        ('nonlin_precision_3', 'precision\ngate 3', ORANGE)]
for i, (mdl, lab, col) in enumerate(labs):
    strip(a, i, nl[nl.model == mdl].r2d_laser.clip(-0.8, 1.2), col, w=0.18, seed=i)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(len(labs))); a.set_xticklabels([l for _, l, _ in labs], fontsize=6.5)
a.set_ylabel('R² of the silencing response (laser on)'); a.set_ylim(-0.85, 1.25)
a.set_title('D  Nonlinear dynamics and gates add ~nothing')

# E: choice-axis state dependence: observed vs models
a = ax[1, 1]
mods = [('observed', 'data', INK), ('nonlin_fixed_20', 'fixed 20', MUTED), ('nonlin_learned_20', 'learned gate 20', BLUE),
        ('nonlin_precision_3', 'precision gate 3', ORANGE)]
for j, (coef, xl) in enumerate((('G_c0', 'c0  mean effect'), ('G_c1', 'c1  × B sure'))):
    for i, (mdl, lab, col) in enumerate(mods):
        y = nl[nl.model == mdl][coef].values
        x = j * 5 + i
        a.errorbar(x, np.median(y), yerr=[[np.median(y) - np.percentile(y, 25)], [np.percentile(y, 75) - np.median(y)]],
                   fmt='o', color=col, ms=6, capsize=2, label=lab if j == 0 else None)
    a.text(j * 5 + 1.5, 0.32, xl, ha='center', fontsize=8, color=INK)
a.axhline(0, color=GRID, lw=1, zorder=0); a.set_xticks([]); a.set_ylim(-0.55, 0.4)
a.set_ylabel('coefficient on held-out trials (median, IQR)')
a.legend(frameon=False, fontsize=7, loc='lower right')
a.set_title("E  No model reproduces the choice-axis effect")

# F: does gating make hemispheres modular?
a = ax[1, 2]
p3['bits'] = p3.bits.fillna(-1)
k = ['session', 'silenced', 'gate', 'd', 'bits', 'vsrc', 'train', 'L']
g = p3.groupby(k, dropna=False).r2d_during.mean().reset_index(); g['d'] = g.d.astype(str)
base = g[(g.train == 'ctrl+stim') & (g.L == 1) & (g.d == 'full') & (g.bits == -1) & (g.vsrc == 'rrr')]
fx = base[base.gate == 'fixed'].set_index(['session', 'silenced']).r2d_during
ts = base[base.gate == '2state'].set_index(['session', 'silenced']).r2d_during
gain = (ts - fx).rename('gain')
x = m.set_index(['session', 'silenced']).join(gain, how='inner')
x = x.dropna(subset=['mod_c', 'gain'])
r = spearmanr(x.mod_c, x.gain)
a.scatter(x.mod_c, x.gain.clip(-0.3, 0.3), s=10, color=ORANGE, alpha=0.5, linewidths=0)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xlabel('modularity of the receiving hemisphere'); a.set_ylabel('gain from two-state gate (R², linear model)')
a.set_title(f'F  Gating does not make a hemisphere modular\n(ρ = {r[0]:.2f}; Chen et al. predict ρ > 0)')

fig.tight_layout()
fig.savefig(os.path.join(R, 'phase4', 'fig_phase4.png'), dpi=150)
print('saved')
