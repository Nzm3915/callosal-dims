"""Figure for T1-T4. usage: python h_plot.py RESULTS_DIR"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(sys.argv[1], 'hypotheses')
BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
t1 = pd.read_csv(os.path.join(R, 't1_committee.csv')).groupby('session').mean(numeric_only=True)
t2 = pd.read_csv(os.path.join(R, 't2_gain.csv')).groupby(['session', 'silenced']).mean(numeric_only=True)
t3 = pd.read_csv(os.path.join(R, 't3_sampling.csv'))
t4 = pd.read_csv(os.path.join(R, 't4_classical.csv'))
fig, ax = plt.subplots(1, 4, figsize=(15, 3.9))


def strip(a, x, y, col, seed=0, w=0.12):
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    a.scatter(x + np.random.default_rng(seed).uniform(-w, w, len(y)), y, s=6, color=col, alpha=0.35, linewidths=0)
    a.plot([x - w - 0.05, x + w + 0.05], [np.median(y)] * 2, color=INK, lw=2.5)


a = ax[0]
items = [('accdis_single', 'best single\nhemisphere', MUTED), ('accdis_linear', 'weighted\nsum', BLUE),
         ('accdis_precision', 'precision-\nweighted', ORANGE), ('p_dominant_wins', 'dominant side\nwins', MUTED),
         ('p_moreconf_wins', 'more confident\nside wins', BLUE)]
for i, (k, lab, col) in enumerate(items):
    strip(a, i, t1[k], col, i)
a.axhline(0.5, color=GRID, lw=1, zorder=0); a.set_ylim(0.2, 1.02)
a.set_xticks(range(len(items))); a.set_xticklabels([l for _, l, _ in items], fontsize=6.8)
a.set_ylabel('accuracy on disagreement trials'); a.set_title('T1  When the hemispheres disagree')

a = ax[1]
items = [('obs_c1', 'data', INK), ('additive_c1', 'additive\nlink', MUTED), ('gain_c1', 'gain\nlink', ORANGE)]
for i, (k, lab, col) in enumerate(items):
    strip(a, i, t2[k].clip(-0.6, 0.4), col, i)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in items]); a.set_ylabel('c1 (state dependence, held-out)')
a.set_title('T2  A gain link does not\nreproduce c1')

a = ax[2]
items = [('excess_highconf', 'confident\ntrials', BLUE), ('excess_lowconf', 'unsure\ntrials', ORANGE),
         ('excess_correct', 'correct', BLUE), ('excess_error', 'error', ORANGE)]
for i, (k, lab, col) in enumerate(items):
    strip(a, i, t3[k].clip(0, 2), col, i)
a.axhline(1, color=GRID, lw=1, zorder=0)
a.set_xticks(range(4)); a.set_xticklabels([l for _, l, _ in items]); a.set_ylabel('within-trial variability / Poisson')
a.set_title('T3  More variability when unsure\n(sampling or attractor)')

a = ax[3]
a.scatter(t4.r_ceiling_cross, t4.r_model_cross, s=10, color=AQUA, alpha=0.6, linewidths=0, label='across hemispheres')
a.scatter(t4.r_ceiling_within, t4.r_model_within, s=10, color=INK, alpha=0.4, linewidths=0, label='within a hemisphere')
a.plot([-0.2, 1], [-0.2, 1], color=MUTED, lw=1, ls='--')
a.set_xlabel('split-half ceiling (data vs data)'); a.set_ylabel('classical model vs held-out data')
a.legend(frameon=False, fontsize=7, loc='upper left'); a.set_title('T4  A classical model matches\nthe joint covariance')
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig_hypotheses.png'), dpi=150)
print('saved')
