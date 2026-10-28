"""Phase 1, step 3: figure (Chen et al. 2021 Fig. 1F-H and S1H, plus error-trial agreement).

usage: python phase1_plot.py RESULT_DIR
"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

D = sys.argv[1]
df = pd.read_csv(os.path.join(D, 'phase1_sessions.csv'))
rt = np.load(os.path.join(D, 'phase1_rho_time.npy'))
t_end = np.load(os.path.join(D, 'phase1_t_end.npy'))
ex = np.load(os.path.join(D, 'phase1_example.npz'), allow_pickle=True)

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))

# A: example session endpoints (Fig. 1F)
a = ax[0, 0]
w = int(ex['end_w']); pA, pB = ex['projA'][:, w], ex['projB'][:, w]
instR = ex['instR']
for trials, filled in ((ex['corr_tr'], True), (ex['err_tr'], False)):
    for side, col, lab in ((True, BLUE, '"lick right"'), (False, ORANGE, '"lick left"')):
        t = trials[instR[trials] == side]
        a.scatter(pA[t], pB[t], s=16, facecolors=col if filled else 'none', edgecolors=col, linewidths=1,
                  label=f'{lab} {"correct" if filled else "error"}')
a.set_xlabel('CD projection, hemisphere code 1 (Hz)'); a.set_ylabel('CD projection, hemisphere code 2 (Hz)')
a.set_title('A  Example session: last 400 ms of delay')
a.legend(frameon=False, fontsize=7, loc='upper left')

# B: rank correlation per session (Fig. 1G)
a = ax[0, 1]
cols = [('rho_R', BLUE, '"lick right"'), ('rho_L', ORANGE, '"lick left"'), ('rho_both', INK, 'both types')]
for i, (c, col, lab) in enumerate(cols):
    y = df[c].values; x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(y))
    a.scatter(x, y, s=10, color=col, alpha=0.5, linewidths=0)
    a.plot([i - 0.25, i + 0.25], [np.nanmean(y)] * 2, color=col, lw=2)
    ys = df[c + '_shuf'].values
    a.plot([i - 0.25, i + 0.25], [np.nanmean(ys)] * 2, color=MUTED, lw=2, ls=':')
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(3)); a.set_xticklabels([l for _, _, l in cols])
a.set_ylabel("Spearman's ρ between hemispheres")
a.set_title(f'B  Endpoint correlation (n={len(df)} sessions)')
a.text(2.35, np.nanmean(df.rho_both_shuf), 'shuffled', color=MUTED, fontsize=7, va='center')

# C: correlation over time (Fig. S1H)
a = ax[0, 2]
tc = t_end - 0.2
m, s = np.nanmean(rt, 0), np.nanstd(rt, 0) / np.sqrt(np.sum(~np.isnan(rt), 0))
a.fill_between(tc, m - s, m + s, color=INK, alpha=0.15, linewidth=0)
a.plot(tc, m, color=INK, lw=2)
for x, lab in ((-3.0, 'sample'), (-1.7, 'delay'), (0, 'go')):
    a.axvline(x, color=GRID, lw=1, zorder=0); a.text(x + 0.05, a.get_ylim()[1] if False else 0.0, '', fontsize=7)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xlabel('time from go cue (s), 400-ms window centre'); a.set_ylabel('within-type ρ (mean ± s.e.m.)')
a.set_title('C  Correlation develops over the trial')
ymax = np.nanmax(m + s)
for x, lab in ((-3.0, 'sample'), (-1.7, 'delay'), (0, 'go')):
    a.text(x + 0.04, ymax, lab, color=MUTED, fontsize=7, va='bottom')

# D: agreement and choice prediction (Fig. 1H), correct vs error trials
a = ax[1, 0]
items = [('agree', 'two hemispheres\nagree'), ('predA', 'code 1 predicts\nthe lick'), ('predB', 'code 2 predicts\nthe lick')]
for i, (k, lab) in enumerate(items):
    for j, (suf, col, nm) in enumerate((('_corr', INK, 'correct trials'), ('_err', ORANGE, 'error trials'))):
        y = df[k + suf].dropna().values; x0 = i + (j - 0.5) * 0.36
        a.scatter(x0 + np.random.default_rng(10 * i + j).uniform(-0.08, 0.08, len(y)), y, s=8, color=col,
                  alpha=0.45, linewidths=0)
        a.plot([x0 - 0.13, x0 + 0.13], [y.mean()] * 2, color=col, lw=2, label=nm if i == 0 else None)
a.axhline(0.5, color=GRID, lw=1, zorder=0)
a.set_xticks(range(3)); a.set_xticklabels([l for _, l in items]); a.set_ylim(0, 1.02)
a.set_ylabel('fraction of trials'); a.set_title('D  Decoding the choice (held-out trials)')
a.legend(frameon=False, fontsize=7, loc='lower right')

# E: error trials, observed agreement vs independence
a = ax[1, 1]
e = df.dropna(subset=['agree_err'])
e = e[e.n_err >= 5]
a.scatter(e.agree_err_indep, e.agree_err, s=np.clip(e.n_err, 5, 80), color=ORANGE, alpha=0.6, linewidths=0)
a.plot([0.3, 1], [0.3, 1], color=MUTED, lw=1, ls='--')
st = wilcoxon(e.agree_err, e.agree_err_indep)
a.set_xlabel('agreement expected if independent'); a.set_ylabel('observed agreement')
a.set_title(f'E  Error trials: hemispheres err together\n(n={len(e)} sessions with ≥5 errors, Wilcoxon p={st.pvalue:.1e})')

# F: correlation vs number of labeled units (sanity)
a = ax[1, 2]
n = np.minimum(df.nA, df.nB)
r = df[['rho_R', 'rho_L']].mean(1)
a.scatter(n, r, s=12, color=INK, alpha=0.55, linewidths=0)
a.set_xlabel('labeled units in the smaller hemisphere'); a.set_ylabel('within-type ρ (mean of R, L)')
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_title('F  More neurons, higher correlation')

fig.tight_layout()
fig.savefig(os.path.join(D, 'fig_phase1_cd.png'), dpi=150)
print('saved')
