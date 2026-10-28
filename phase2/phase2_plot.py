"""Phase 2 figure. usage: python phase2_plot.py RESULT_DIR"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

D = sys.argv[1]
df = pd.read_csv(os.path.join(D, 'phase2_pairs.csv'))
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
K = np.arange(1, 6)
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))


def curve(a, s, prefix, col, lab, ls='-'):
    v = np.array([s[f'{prefix}{k}'].values for k in K])
    med = np.nanmedian(v, 1); lo, hi = np.nanpercentile(v, 25, 1), np.nanpercentile(v, 75, 1)
    a.fill_between(K, lo, hi, color=col, alpha=0.12, linewidth=0)
    a.plot(K, med, color=col, lw=2, ls=ls, marker='o', ms=4, label=lab)
    return med


# A: reliability (negative control)
a = ax[0, 0]
for i, e in enumerate(['pre', 'during', 'after']):
    y = df[df.epoch == e].rel.values
    a.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(y)), y, s=7, color=INK, alpha=0.35, linewidths=0)
    a.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color=BLUE, lw=2.5)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(3)); a.set_xticklabels(['sample\n(laser off)', 'delay, laser on\n(0–0.8 s)', 'delay, after\n(0.8–1.7 s)'])
a.set_ylabel('split-half reliability of the response')
a.set_title(f'A  The other hemisphere responds\n(n = {df[df.epoch == "pre"].shape[0]} hemisphere pairs, 133 sessions)')

# B, C: captured fraction vs dimension
for a, e, title in ((ax[0, 1], 'during', 'B  Laser on: where does the response live?'),
                    (ax[0, 2], 'after', 'C  After the laser: recovery')):
    s = df[df.epoch == e]
    curve(a, s, 'g_rec', BLUE, 'receiving subspace (own, cross-validated)')
    curve(a, s, 'f_sig', YELLOW, 'task subspace (condition means)')
    curve(a, s, 'f_pc', AQUA, 'largest fluctuations of the hemisphere')
    curve(a, s, 'f_corr', ORANGE, 'correlated subspace (shared with other side)')
    if e == 'after':
        curve(a, s, 'g_recDuring', BLUE, 'receiving subspace of the laser-on epoch', ls='--')
    curve(a, s, 'f_rand', MUTED, 'random directions (chance)', ls=':')
    a.scatter([1], [s.f_cd.median()], marker='D', s=30, color=INK, zorder=5, label='choice decoder CD')
    a.set_xticks(K); a.set_xlabel('number of dimensions k'); a.set_ylabel('fraction of reliable response captured')
    a.set_ylim(-0.05, 1.3); a.set_title(title)
ax[0, 1].legend(frameon=False, fontsize=6.5, loc='upper left')
ax[0, 2].legend(frameon=False, fontsize=6.5, loc='upper left')

# D: per pair, k = 3
a = ax[1, 0]
s = df[df.epoch == 'during']
a.scatter(s.g_rec3, s.f_corr3, s=9, color=ORANGE, alpha=0.5, linewidths=0)
a.plot([-0.2, 1], [-0.2, 1], color=MUTED, lw=1, ls='--')
a.set_xlim(-0.2, 1.05); a.set_ylim(-0.2, 1.05)
p = wilcoxon(s.g_rec3, s.f_corr3).pvalue
a.set_xlabel('receiving subspace, 3 dims'); a.set_ylabel('correlated subspace, 3 dims')
a.set_title(f'D  Every pair: receiving ≫ correlated\n(laser on, Wilcoxon p = {p:.0e})')

# E: choice-specific part
a = ax[1, 1]
for i, e in enumerate(['during', 'after']):
    s = df[df.epoch == e]
    for j, (col, c, lab) in enumerate(((INK, 'frac_typediff', 'share of response that\ndiffers by trial type'),
                                       (BLUE, 'f_cd_typediff', 'of that share,\nfraction along CD'))):
        y = s[c].clip(-1, 1.35).values; x0 = i + (j - 0.5) * 0.4
        a.scatter(x0 + np.random.default_rng(10 * i + j).uniform(-0.08, 0.08, len(y)), y, s=6, color=col,
                  alpha=0.35, linewidths=0)
        a.plot([x0 - 0.14, x0 + 0.14], [np.nanmedian(y)] * 2, color=col, lw=2.5, label=lab if i == 0 else None)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks([0, 1]); a.set_xticklabels(['laser on', 'after']); a.set_ylim(-0.6, 1.95)
a.set_title('E  Most of the response is not about the choice'); a.legend(frameon=False, fontsize=7, loc='upper center', ncol=2)

# F: is the correlated subspace real? held-out R^2
a = ax[1, 2]
s = df[df.epoch == 'during'].drop_duplicates(['session', 'silenced'])
v = np.array([s[f'R2_corr{k}'].values for k in K])
a.fill_between(K, np.percentile(v, 25, 1), np.percentile(v, 75, 1), color=ORANGE, alpha=0.12, linewidth=0)
a.plot(K, np.median(v, 1), color=ORANGE, lw=2, marker='o', ms=4)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(K); a.set_xlabel('rank of the correlated subspace'); a.set_ylabel('held-out R² of B from A (control trials)')
a.set_title('F  The correlated subspace is real but small')

fig.tight_layout()
fig.savefig(os.path.join(D, 'fig_phase2_subspaces.png'), dpi=150)
print('saved')
