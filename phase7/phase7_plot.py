"""Phase 7 figure (self-improvement loops). usage: python phase7_plot.py RESULTS_DIR"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = os.path.join(sys.argv[1], 'phase7')
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
def strip(a, x, y, col, seed=0, w=0.13):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    a.scatter(x + np.random.default_rng(seed).uniform(-w, w, len(y)), y, s=5, color=col, alpha=0.3, linewidths=0)
    a.plot([x - w - 0.05, x + w + 0.05], [np.median(y)] * 2, color=INK, lw=2.5)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
lp = pd.read_csv(os.path.join(R, 'phase7_loop_ratio.csv')); lm = pd.read_csv(os.path.join(R, 'phase7_loop_model.csv'))
a = ax[0]
items = [(lp.c1, 'observed c1', INK), (lp.c1_loop_c0, 'round trip\n(w from c0;\nupper bound)', ORANGE),
         (lp.c1_loop_c2, 'round trip\n(w from c2)', ORANGE), (lm.full_c1, 'coupled model\n(full)', BLUE), (lm.noBtoA_c1, 'coupled model\n(no B->A)', BLUE)]
for i, (v, lab, col) in enumerate(items): strip(a, i, np.clip(v, -0.6, 0.4), col, i)
a.axhline(0, color=GRID, lw=1, zorder=0); a.set_xticks(range(len(items))); a.set_xticklabels([l for _, l, _ in items], fontsize=7)
a.set_ylabel('state dependence c1'); a.set_title('A  The interhemispheric round trip\ndoes not produce c1')
a = ax[1]; g = json.load(open(os.path.join(R, 'phase7_gain_pooled.json'))); au = json.load(open(os.path.join(R, 'phase7_audit.json')))['A1']
labs = [('own gain', 'phase7-2: fraction of c1 by own (bootstrap over mice)'), ('own gain\n+ one recurrent pass', 'phase7-2: fraction of c1 by loop1 (bootstrap over mice)')]
for i, (lab, k) in enumerate(labs):
    v = au[k]; a.errorbar(i, v['fraction'], yerr=[[v['fraction'] - v['ci'][0]], [v['ci'][1] - v['fraction']]], fmt='o', color=YELLOW, ms=8, capsize=6, lw=2)
a.axhline(1, color=INK, lw=1, ls='--'); a.axhline(0, color=GRID, lw=1)
a.set_xticks(range(2)); a.set_xticklabels([l for l, _ in labs]); a.set_xlim(-0.5, 1.5); a.set_ylim(-0.1, 2.7)
a.set_ylabel('fraction of c1 explained (pooled)'); a.set_title('B  Single-neuron gain: pooled estimate\n(95% CI, bootstrap over 39 mice)')
a = ax[2]; ct = pd.read_csv(os.path.join(R, 'phase7_committee_test.csv'))
a.scatter(ct.null_conf, ct.obs_conf, s=12, color=BLUE, alpha=0.6, linewidths=0, label='P(more confident side wins)')
a.scatter(ct.null_dom, ct.obs_dom, s=12, color=MUTED, alpha=0.6, linewidths=0, label='P(dominant side wins)')
a.plot([0.3, 1], [0.3, 1], color=INK, lw=1, ls='--'); a.set_xlim(0.3, 1); a.set_ylim(0.3, 1)
a.set_xlabel('single-driver null (calibrated per session)'); a.set_ylabel('observed (disagreement trials)')
a.legend(frameon=False, fontsize=7, loc='upper left'); a.set_title('C  Committee readout vs a single-driver\nnull with measurement noise')
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig_phase7.png'), dpi=150); print('saved')
