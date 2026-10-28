"""Phase 5c: time course of the contralateral-silencing effect, fast-spiking vs pyramidal units of the hemisphere
that is NOT silenced (50-ms bins, normalised change, aligned to laser onset).
usage: python phase5_latency.py CACHE_DIR RESULTS_DIR"""
import sys, os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE, R = sys.argv[1:3]
d = pd.read_csv(os.path.join(R, 'units_fr_all.csv')); lab = pd.read_csv(os.path.join(R, 'units_labeled.csv'))
f = d[(d.celltype == 'FS') & (d.fr0 >= 2)].copy(); f['r1'] = f.fr1 / f.fr0; f['r2'] = f.fr2 / f.fr0
f = f[np.maximum(f.r1, f.r2) >= 2]; f['hemi'] = np.where(f.r1 >= f.r2, 'code1', 'code2')
lab = lab[lab.fr0 >= 2]
cur = {k: [] for k in ('silenced side, pyramidal', 'other side, fast-spiking', 'other side, pyramidal (decrease)',
                       'other side, pyramidal (increase)')}
for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
    z = np.load(p); sess = os.path.basename(p)[:-4].replace('__', '/')
    uid = list(z['unit_ids']); c = z['counts'].astype(np.float32); t = z['edges'][:-1]
    ok = z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) & (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05)
    pt = z['PhotostimulationType']
    for kind, df in (('FS', f[f.session == sess]), ('pyr', lab[lab.session == sess])):
        for _, u in df.iterrows():
            i = uid.index(u.unit); own = 1 if u.hemi == 'code1' else 2; other = 3 - own
            ctrl = c[ok & (pt == 0), i].mean(0) / 0.05
            base = ctrl[(t >= -2.2) & (t < -1.7)].mean() + 0.5
            st = (c[ok & (pt == other), i].mean(0) / 0.05 - ctrl) / base
            if kind == 'FS':
                cur['other side, fast-spiking'].append(st); continue
            cur['silenced side, pyramidal'].append((c[ok & (pt == own), i].mean(0) / 0.05 - ctrl) / base)
            r = u.S2 if own == 1 else u.S1
            if r < 0.8: cur['other side, pyramidal (decrease)'].append(st)
            if r > 1.2: cur['other side, pyramidal (increase)'].append(st)
tc = (t + 0.025 + 1.7) * 1000
np.savez(os.path.join(R, 'phase5', 'phase5_latency.npz'), t_ms=tc, **{k.replace(' ', '_').replace(',', ''): np.array(v) for k, v in cur.items()})
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
fig, a = plt.subplots(figsize=(6.4, 3.8))
cols = {'silenced side, pyramidal': '#8a8984', 'other side, fast-spiking': '#eb6834',
        'other side, pyramidal (decrease)': '#2a78d6', 'other side, pyramidal (increase)': '#1baf7a'}
w = (tc > -300) & (tc < 1100)
for k, v in cur.items():
    v = np.array(v); m = v.mean(0); se = v.std(0) / np.sqrt(len(v))
    a.fill_between(tc[w], (m - 2 * se)[w], (m + 2 * se)[w], color=cols[k], alpha=0.15, linewidth=0)
    a.plot(tc[w], m[w], color=cols[k], lw=2, label=f'{k} (n = {len(v)})')
a.axvspan(0, 800, color='#2a78d6', alpha=0.06, lw=0); a.axhline(0, color='#e4e3df', lw=1)
a.set_xlabel('time from laser onset (ms), 50-ms bins'); a.set_ylabel('change in rate / baseline')
a.legend(frameon=False, fontsize=7, loc='lower right')
a.set_title('Inhibitory cells on the other side drop first; some pyramidal cells rise later')
fig.tight_layout(); fig.savefig(os.path.join(R, 'phase5', 'fig_phase5_latency.png'), dpi=150)
print('saved')
