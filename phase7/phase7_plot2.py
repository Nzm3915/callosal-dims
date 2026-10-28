"""Second Phase 7 figure: the referee-report controls and the gain account in time.
A  hemisphere labels checked without any laser trial (fast co-fluctuations vs channel distance)
B  light spread: FS cells under their own light vs under the other hemisphere's light
C  the three timescales at the end of the laser (rate ratio, state dependence, mean loss)

usage: python phase7_plot2.py RESULTS_DIR
"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = sys.argv[1]
P = os.path.join(R, 'phase7')
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#0b0b0b', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

lv = json.load(open(os.path.join(P, 'phase7_label_validation.json')))
tab = pd.DataFrame(lv['table']); order = ['0', '1-2', '3-8', '>8']
a = ax[0]; w = 0.36
for k, (same, col, lab) in enumerate(((True, BLUE, 'same hemisphere label'), (False, MUTED, 'different label'))):
    v = [100 * float(tab[(tab.dist == d) & (tab.same == same)].frac_strong.iloc[0]) for d in order]
    a.bar(np.arange(len(order)) + (k - 0.5) * w, v, width=w, color=col, label=lab)
a.set_xticks(range(len(order))); a.set_xticklabels(['same\nchannel', '1-2', '3-8', '>8'])
a.set_xlabel('distance between channel numbers'); a.set_ylabel('strongly co-fluctuating pairs (%)')
a.legend(frameon=False, fontsize=7.5)
a.set_title('A  Labels checked with no laser trial:\nonly same-label pairs fall off with distance')

ref = json.load(open(os.path.join(P, 'phase7_referee.json')))['R3_light_spread']
d = pd.read_csv(os.path.join(R, 'units_fr_all.csv'))
f = d[(d.celltype == 'FS') & (d.fr0 >= 2)].copy(); f['r1'] = f.fr1 / f.fr0; f['r2'] = f.fr2 / f.fr0
f = f[np.maximum(f.r1, f.r2) >= 2]
own = np.maximum(f.r1, f.r2); oth = np.minimum(f.r1, f.r2)
a = ax[1]; bins = np.geomspace(0.05, 30, 36)
a.hist(np.clip(own, 0.05, 30), bins=bins, color=ORANGE, alpha=0.75, label=f'own light (n = {len(f)})')
a.hist(np.clip(oth, 0.05, 30), bins=bins, color=BLUE, alpha=0.75, label="other hemisphere's light")
a.set_xscale('log'); a.axvline(1, color=INK, lw=1, ls='--'); a.axvline(2, color=MUTED, lw=1, ls=':')
a.set_xlabel('rate ratio of FS cells (laser / control)'); a.set_ylabel('cells')
a.legend(frameon=False, fontsize=7.5)
a.set_title("B  No light spread: FS cells go down,\nnot up, under the other side's light")

rd = json.load(open(os.path.join(P, 'phase7_rampdown.json')))
rm = json.load(open(os.path.join(P, 'phase7_rampdown_mean.json')))
wins = [k for k in rd if k != 'within_pair_change']
mid = [np.mean([float(x) for x in w.split('..')]) * 1000 for w in wins]
a = ax[2]
series = [([1 - rd[w]['g_wmean'] for w in wins], ORANGE, 'rate loss (1 - gain)'),
          ([-rd[w]['obs'] for w in wins], BLUE, 'state dependence (-c1 / b1)'),
          ([rm[w]['L0'] for w in wins], AQUA, 'mean loss of the choice signal')]
for v, col, lab in series:
    a.plot(mid, np.array(v) / v[0], 'o-', color=col, label=f'{lab}  [{v[0]:.2f}]')
a.fill_between([-1700, -900], -0.3, 1.2, color=BLUE, alpha=0.05, lw=0)
a.axhline(0, color=GRID, lw=1); a.set_ylim(-0.3, 1.15); a.set_xlabel('centre of the response window (ms, go cue = 0)')
a.set_ylabel('relative to the end of the laser'); a.legend(frameon=False, fontsize=7, title='[value in the first window]', title_fontsize=6.5)
a.set_title('C  The state dependence follows the rates,\nnot the slow loss of the choice signal')
fig.tight_layout(); fig.savefig(os.path.join(P, 'fig_phase7b.png'), dpi=150); print('saved', os.path.join(P, 'fig_phase7b.png'))
