"""Phase 0: recover hemisphere labels of units and meaning of PhotostimulationType codes
from DANDI 000894 (Chen, Kang et al., Cell 2021) using the data itself.
Idea: unilateral photoinhibition silences pyramidal neurons in the illuminated hemisphere.
"""
import sys, json, re, h5py, remfile, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dandi.dandiapi import DandiAPIClient

WIN = (0.10, 0.70)   # s after pole-out (= delay onset). Laser runs pole-out to +0.8 s, last 0.2 s a ramp-down.
# Pole-out is 1.87 s after trial start in some sessions and 2.87 s in others, so we align to it per trial.
OUT = sys.argv[1]  # output directory
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 999

def dec(x): return x.decode() if isinstance(x, bytes) else str(x)

def process(asset):
    f = h5py.File(remfile.File(asset.get_content_url(follow_redirects=1, strip_query=True)), 'r')
    tr = f['intervals/trials']
    good = tr['GoodTrials'][()].astype(bool)
    pt = tr['PhotostimulationType'][()]
    t0 = tr['start_time'][()]
    poleout = tr['PoleOutTime'][()]
    # unit metadata from analysis/unitNSS (cell type, electrode)
    meta = {}
    for k in f['analysis']:
        g = f['analysis'][k]
        if 'control' not in g: continue
        uid = int(g['control'][0])
        meta[uid] = dict(celltype=dec(g.attrs.get('comments', '')).replace('cellType: ', ''),
                         electrode=int(np.median(g['electrodes'][:50])))
    ids = f['units/id'][()]
    idx = f['units/spike_times_index'][()]
    st_all = f['units/spike_times'][()]
    rows = []
    typical = np.nanmedian(poleout[good])
    sel_trials = {c: np.where(good & (pt == c) & np.isclose(poleout, typical, atol=0.05))[0] for c in [0, 1, 2, 3]}
    lo = 0
    for i, uid in enumerate(ids):
        st = st_all[lo:idx[i]]; lo = idx[i]
        fr = {}
        for c, trs in sel_trials.items():
            if len(trs) < 10: fr[c] = np.nan; continue
            a = t0[trs] + poleout[trs] + WIN[0]; b = t0[trs] + poleout[trs] + WIN[1]
            n = np.searchsorted(st, b) - np.searchsorted(st, a)
            fr[c] = n.mean() / (WIN[1] - WIN[0])
        m = meta.get(int(uid), {})
        rows.append(dict(session=asset.path, unit=int(uid), celltype=m.get('celltype', ''),
                         electrode=m.get('electrode', -1), fr0=fr[0], fr1=fr[1], fr2=fr[2], fr3=fr[3],
                         ntr0=len(sel_trials[0]), ntr1=len(sel_trials[1]), ntr2=len(sel_trials[2]), ntr3=len(sel_trials[3])))
    return pd.DataFrame(rows)

import os, time
def safe(a):
    out = os.path.join(OUT, a.path.replace('/', '__') + '.csv')
    if os.path.exists(out): return out
    t = time.time()
    try:
        d = process(a); d.to_csv(out, index=False)
        print('done', a.path, len(d), f'{time.time()-t:.0f}s', flush=True); return out
    except Exception as e:
        print('ERR', a.path, e, flush=True); return None

if __name__ == '__main__':
    assets = sorted(DandiAPIClient().get_dandiset('000894').get_assets(), key=lambda a: a.path)[:NMAX]
    os.makedirs(OUT, exist_ok=True)
    print('assets', len(assets), flush=True)
    with ThreadPoolExecutor(8) as ex:
        outs = [o for o in ex.map(safe, assets) if o]
    df = pd.concat([pd.read_csv(o) for o in outs]); df.to_csv(os.path.join(OUT, 'ALL.csv'), index=False)
    print('sessions', df.session.nunique(), 'units', len(df), flush=True)
