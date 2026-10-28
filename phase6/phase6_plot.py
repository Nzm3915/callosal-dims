"""Phase 6 figure. usage: python phase6_plot.py RESULTS_DIR"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = os.path.join(sys.argv[1], 'phase6')
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def strip(a, x, y, col, seed=0, w=0.13):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    a.scatter(x + np.random.default_rng(seed).uniform(-w, w, len(y)), y, s=5, color=col, alpha=0.3, linewidths=0)
    a.plot([x - w - 0.05, x + w + 0.05], [np.median(y)] * 2, color=INK, lw=2.5)


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
# A modes
a = ax[0, 0]; m = pd.read_csv(os.path.join(R, 'phase6_modes.csv'))
items = [('during_f_CD', 'choice\nmode', BLUE), ('during_f_RM_early', 'ramp\n(laser win.)', ORANGE),
         ('during_f_RM_late', 'ramp\n(late)', ORANGE), ('during_f_SM', 'sample\nonset', ORANGE),
         ('during_f_NS3', 'all 3\nnon-sel.', ORANGE), ('during_f_CDe', 'CD\nearly', BLUE), ('during_f_CDl', 'CD\nlate', BLUE),
         ('during_f_CDgo', 'CD\ngo', BLUE), ('during_f_rec1', 'top recv.\ndirection', AQUA), ('f_rand1', 'random\n1-D', MUTED)]
for i, (k, lab, col) in enumerate(items):
    strip(a, i, m[k].clip(-0.2, 1.05), col, i)
a.set_xticks(range(len(items))); a.set_xticklabels([l for _, l, _ in items], fontsize=6.5)
a.set_ylabel('fraction of reliable response'); a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_title("A  The callosal response lies outside\nthe receiver's task modes")

# B what explains c1
a = ax[0, 1]
p4 = pd.read_csv(os.path.join(R, 'phase6_ramp.csv')); pt = pd.read_csv(os.path.join(R, 'phase6_partner.csv'))
fg = pd.read_csv(os.path.join(R, 'phase6_fsgain.csv')); ng = pd.read_csv(os.path.join(R, 'phase6_neurongain_pairs.csv'))
best_partner = pt[(pt.winB == 0.4) & (pt.winA == '1.2') & (pt.Aunits == 'all')]
items = [(p4.p4_c1, 'observed c1\n(Phase 4 model)', INK), (p4.r_c1, '+ ramp\nterms', ORANGE),
         (best_partner.c1, 'A measured\nbetter (1.2 s)', BLUE), (fg.F_med_c1, '+ FS gain\nterms', AQUA),
         (fg.F_c1pred, 'predicted from\nFS gain', AQUA), (ng.c1_obs, 'observed\n(neuron sum)', INK),
         (ng['c1_pred_1.3'], 'predicted from\nneuron gain', YELLOW)]
for i, (v, lab, col) in enumerate(items):
    strip(a, i, np.clip(v, -0.6, 0.4), col, i)
a.axhline(0, color=GRID, lw=1, zorder=0)
a.set_xticks(range(len(items))); a.set_xticklabels([l for _, l, _ in items], fontsize=6.5)
a.set_ylabel('state dependence c1'); a.axvline(4.5, color=GRID, lw=1)
a.set_title('B  Candidate carriers of the state dependence\n(left: regression units; right: neuron-sum units)')

# C single-neuron gain
a = ax[0, 2]; un = pd.read_csv(os.path.join(R, 'phase6_neurongain_units.csv'))
un['lg'] = np.log2(un.g); bins = pd.qcut(un.lg, 8); gb = un.groupby(bins, observed=True)
g = gb.g.median().values; rr = (gb.u_stim.sum() / gb.u_ctrl.sum()).values
# bootstrap CI over units
boot = []
for b in range(200):
    s = un.sample(frac=1, replace=True, random_state=b); gbb = s.groupby(pd.cut(s.lg, bins.cat.categories), observed=True)
    boot.append((gbb.u_stim.sum() / gbb.u_ctrl.sum()).values)
lo, hi = np.nanpercentile(np.array(boot), [2.5, 97.5], axis=0)
a.fill_between(g, lo, hi, color=YELLOW, alpha=0.25, lw=0)
a.plot(g, rr, 'o-', color=YELLOW, lw=2, ms=4, label='data (cross-validated)')
gg = np.linspace(0.3, 2.0, 100)
for al, ls in ((0, ':'), (1, '--'), (2, '-.')):
    a.plot(gg, gg ** al, color=MUTED, ls=ls, lw=1, label=f'g^{al}')
a.set_xlabel('rate ratio g (silence A / control)'); a.set_ylabel("slope ratio (state carried, stim / control)")
a.set_xlim(0.3, 2.0); a.set_ylim(0, 2.2); a.legend(frameon=False, fontsize=7)
a.set_title('C  Neurons pushed down lose the\nstate they carry, faster than their rate')

# D rewrite
a = ax[1, 0]; rw = pd.read_csv(os.path.join(R, 'phase6_rewrite.csv'))
items = [(rw.d1, "A's own memory\n(stim x cA_pre)", BLUE), (rw.d2, "copy from B\n(stim x cB_laser)", ORANGE),
         (rw.agree_stim - rw.agree_ctrl, 'end-point agreement\nA with B (stim - ctrl)', MUTED)]
for i, (v, lab, col) in enumerate(items):
    strip(a, i, np.clip(v, -0.6, 0.6), col, i)
a.axhline(0, color=GRID, lw=1, zorder=0); a.set_xticks(range(3)); a.set_xticklabels([l for _, l, _ in items], fontsize=7)
a.set_title('D  After silencing, A recovers its OWN plan;\nit is not rewritten from B')

# E DLAG overlap
a = ax[1, 1]
ov = os.path.join(R, 'phase6_dlag_overlap.csv')
if os.path.exists(ov):
    o = pd.read_csv(ov); o = o[(o.rel_den > 0) & (o.pa > 0)]
    items = [('f_dlag_across', 'DLAG\nacross', BLUE), ('f_dlag_AtoB', 'DLAG\nA leads', BLUE), ('f_dlag_BtoA', 'DLAG\nB leads', BLUE),
             ('f_dlag_within', 'DLAG\nwithin B', MUTED), ('f_rec_pa', 'receiving\n(same k)', AQUA), ('f_rand_pa', 'random\n(same k)', MUTED)]
    for i, (k, lab, col) in enumerate(items):
        strip(a, i, o[k].clip(-0.3, 1.05), col, i)
    a.set_xticks(range(len(items))); a.set_xticklabels([l for _, l, _ in items], fontsize=7)
    a.axhline(0, color=GRID, lw=1, zorder=0); a.set_ylabel('fraction of reliable response')
    a.set_title(f'E  DLAG on normal trials misses most\nof the callosal response (n = {len(o)})')
# F DLAG direction vs causal asymmetry
a = ax[1, 2]
sp = os.path.join(R, 'phase6_dlag_sessions.csv')
if os.path.exists(sp):
    ss = pd.read_csv(sp)
    c0 = p4.pivot_table(index='session', columns='silenced', values='p4_c0'); c0['causal'] = c0['code2'] - c0['code1']
    mm = ss.merge(c0[['causal']], left_on='session', right_index=True).dropna()
    rho, p = spearmanr(mm.flow, mm.causal)
    a.scatter(mm.flow, mm.causal, s=14, color=BLUE, alpha=0.7, linewidths=0)
    a.axhline(0, color=GRID, lw=1); a.axvline(0, color=GRID, lw=1)
    a.set_xlabel('DLAG flow index (+ = code1 leads)'); a.set_ylabel('causal asymmetry (silencing; + = code1 drives)')
    a.set_title(f'F  Does the DLAG direction match the\ncausal direction? rho = {rho:+.2f}, p = {p:.2f}')
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig_phase6.png'), dpi=150)
print('saved')
