"""Phase 3 figure. usage: python phase3_plot.py RESULT_DIR"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1]
d = pd.read_csv(os.path.join(D, 'phase3_models.csv'))
d['bits'] = d.bits.fillna(-1)
key = ['session', 'silenced', 'train', 'L', 'gate', 'd', 'bits', 'vsrc']
g = d.groupby(key, dropna=False)[['r2_ctrl', 'r2d_during', 'r2d_after']].mean().reset_index()
g['d'] = g.d.astype(str)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, INK, MUTED, GRID = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                                                         '#0b0b0b', '#8a8984', '#e4e3df')
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def sel(train='ctrl+stim', L=1, gate='fixed', dd='full', bits=-1, vsrc='rrr'):
    return g[(g.train == train) & (g.L == L) & (g.gate == gate) & (g.d == str(dd)) & (g.bits == bits) & (g.vsrc == vsrc)]


DS = ['1', '2', '3', '5', '10']
fig, ax = plt.subplots(2, 3, figsize=(12, 7.2))

for a, col, title in ((ax[0, 0], 'r2d_during', 'A  Laser on: which link reproduces the response?'),
                      (ax[0, 1], 'r2d_after', 'B  After the laser')):
    x = np.arange(len(DS))
    for vsrc, train, c, lab in (('delta', 'ctrl+stim', BLUE, 'link along the receiving subspace'),
                                ('rrr', 'ctrl+stim', ORANGE, 'link along best-predicting directions'),
                                ('rrr', 'ctrl', MUTED, 'link learned from control trials only')):
        ys = [sel(train=train, dd=k, vsrc=vsrc)[col] for k in DS]
        med = [np.nanmedian(y) for y in ys]
        lo = [np.nanpercentile(y, 25) for y in ys]; hi = [np.nanpercentile(y, 75) for y in ys]
        a.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
        a.plot(x, med, color=c, lw=2, marker='o', ms=4, label=lab)
    full = np.nanmedian(sel()[col])
    a.axhline(full, color=INK, lw=1, ls='--')
    a.text(len(DS) - 1, full + 0.015, 'full-rank link', ha='right', fontsize=7, color=INK)
    a.axhline(0, color=GRID, lw=1, zorder=0)
    a.set_xticks(x); a.set_xticklabels(DS); a.set_xlabel('link dimensions d')
    a.set_ylabel('R² of the silencing response\n(held-out trials, noise-corrected)')
    a.set_ylim(-0.45, 0.85); a.set_title(title)
ax[0, 0].legend(frameon=False, fontsize=7, loc='upper left')

# C: delay
a = ax[0, 2]
for col, c, lab in (('r2d_during', BLUE, 'laser on'), ('r2d_after', AQUA, 'after')):
    ys = [sel(L=L)[col] for L in (1, 2, 3)]
    a.plot([100, 200, 300], [np.nanmedian(y) for y in ys], color=c, lw=2, marker='o', ms=4, label=lab)
    a.fill_between([100, 200, 300], [np.nanpercentile(y, 25) for y in ys], [np.nanpercentile(y, 75) for y in ys],
                   color=c, alpha=0.12, linewidth=0)
a.set_xticks([100, 200, 300]); a.set_xlabel('link delay L (ms)'); a.set_ylabel('R² of the silencing response')
a.set_ylim(0, 0.6); a.legend(frameon=False, fontsize=7); a.set_title('C  The link must be fast (full rank)')

# D: gates
a = ax[1, 0]
labs = [('fixed', 'fixed\n(g = 1)', INK), ('2state', 'two-state\n(Chen 2021)', ORANGE), ('prec', 'precision\n(H4)', BLUE)]
for i, (gt, lab, c) in enumerate(labs):
    for j, (dd, mk) in enumerate((('full', 'full rank'), ('5', 'd = 5'))):
        y = sel(gate=gt, dd=dd).r2d_during.values; x0 = i + (j - 0.5) * 0.36
        a.scatter(x0 + np.random.default_rng(i * 10 + j).uniform(-0.07, 0.07, len(y)), y.clip(-0.5, 1), s=5,
                  color=c, alpha=0.3, linewidths=0)
        a.plot([x0 - 0.13, x0 + 0.13], [np.nanmedian(y)] * 2, color=c, lw=2.5)
        a.text(x0, -0.62, mk, ha='center', fontsize=6.5, color=MUTED)
a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in labs]); a.set_ylim(-0.7, 1.05)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_ylabel('R² of the silencing response (laser on)'); a.set_title('D  Gating does not help here')

# E: bits
a = ax[1, 1]
for dd, c in (('1', MUTED), ('2', AQUA), ('3', BLUE), ('5', YELLOW)):
    xs, ys = [], []
    for b in (1, 2, 3, 4, 6):
        xs.append(int(dd) * b * 10); ys.append(np.nanmedian(sel(dd=dd, bits=b, vsrc='delta').r2d_during))
    a.plot(xs, ys, color=c, lw=2, marker='o', ms=4, label=f'd = {dd}')
a.axvline(90, color=GRID, lw=1, zorder=0)
a.text(92, 0.05, '90 bit/s', color=MUTED, fontsize=7)
a.set_xscale('log'); a.set_xlabel('link bit rate (bit/s) = d × bits × 10 Hz')
a.set_ylabel('R² of the silencing response (laser on)'); a.set_ylim(0, 0.55)
a.legend(frameon=False, fontsize=7, loc='lower right'); a.set_title('E  About 90 bit/s is enough')

# F: invisible in control trials
a = ax[1, 2]
base = sel(dd=0).set_index(['session', 'silenced']).r2_ctrl
gains, ds = [], ['1', '2', '3', '5', '10', 'full']
for dd in ds:
    s = sel(dd=dd).set_index(['session', 'silenced']).r2_ctrl
    gains.append((s - base).dropna().values)
a.plot(range(len(ds)), [np.median(v) for v in gains], color=ORANGE, lw=2, marker='o', ms=4,
       label='control trials: gain in one-step R²')
s0 = sel(dd=0).r2d_during
a.plot(range(len(ds)), [np.nanmedian(sel(dd=dd).r2d_during) - np.nanmedian(s0) for dd in ds], color=BLUE, lw=2,
       marker='o', ms=4, label='silencing: gain in R² of the response')
a.set_xticks(range(len(ds))); a.set_xticklabels(ds); a.set_xlabel('link dimensions d (best-predicting directions)')
a.set_ylabel('gain over a model without link'); a.axhline(0, color=GRID, lw=1, zorder=0)
a.legend(frameon=False, fontsize=7, loc='upper left'); a.set_title('F  The link is nearly invisible in normal trials')

fig.tight_layout()
fig.savefig(os.path.join(D, 'fig_phase3_link.png'), dpi=150)
print('saved')
