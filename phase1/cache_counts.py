"""Phase 1, step 1: stream spike times from DANDI 000894 and cache binned spike counts.

For every session and every trial we count spikes of every unit in 50-ms bins aligned to the
go cue (the end of the delay epoch). Saved per session as a compressed .npz so that the
analyses (this phase and later ones) never need to touch the 270 GB of raw files again.

usage: python cache_counts.py OUTDIR [NMAX]
"""
import sys, os, time, h5py, remfile, numpy as np
from concurrent.futures import ThreadPoolExecutor
from dandi.dandiapi import DandiAPIClient

BIN = 0.05                       # s
T0, T1 = -3.6, 1.0               # s relative to go cue: covers sample (-3.0..-1.7) and delay (-1.7..0)
EDGES = np.round(np.arange(T0, T1 + 1e-9, BIN), 3)
OUT = sys.argv[1]
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 999
FIELDS = ['GoodTrials', 'PhotostimulationType', 'HitL', 'HitR', 'ErrL', 'ErrR', 'NoLickL', 'NoLickR',
          'LickEarly', 'PoleOutTime', 'start_time']

def go_cue(cue):
    """CueTime is (n, 2); the go cue sits in the second column (first is NaN in this dataset)."""
    c = cue[:, 1].copy()
    bad = np.isnan(c)
    c[bad] = np.nanmax(cue[bad], axis=1) if bad.any() else c[bad]
    return c

def process(asset):
    f = h5py.File(remfile.File(asset.get_content_url(follow_redirects=1, strip_query=True)), 'r')
    tr = f['intervals/trials']
    info = {k: tr[k][()] for k in FIELDS}
    cue = go_cue(tr['CueTime'][()])
    t_go = info['start_time'] + cue                      # absolute time of go cue per trial
    ids = f['units/id'][()]
    idx = f['units/spike_times_index'][()]
    st_all = f['units/spike_times'][()]
    ntr, nu, nb = len(t_go), len(ids), len(EDGES) - 1
    counts = np.zeros((ntr, nu, nb), dtype=np.uint8)
    ok = ~np.isnan(t_go)
    lo = 0
    for i in range(nu):
        st = st_all[lo:idx[i]]; lo = idx[i]
        for k in np.where(ok)[0]:
            e = t_go[k] + EDGES
            j = np.searchsorted(st, e)
            counts[k, i] = np.minimum(np.diff(j), 255)
    np.savez_compressed(os.path.join(OUT, asset.path.replace('/', '__') + '.npz'),
                        counts=counts, unit_ids=ids, edges=EDGES, cue=cue, **info)
    return nu, ntr

def safe(a):
    out = os.path.join(OUT, a.path.replace('/', '__') + '.npz')
    if os.path.exists(out): return out
    t = time.time()
    try:
        nu, ntr = process(a)
        print('done', a.path, nu, 'units', ntr, 'trials', f'{time.time()-t:.0f}s', flush=True); return out
    except Exception as e:
        print('ERR', a.path, repr(e), flush=True); return None

if __name__ == '__main__':
    assets = sorted(DandiAPIClient().get_dandiset('000894').get_assets(), key=lambda a: a.path)[:NMAX]
    os.makedirs(OUT, exist_ok=True)
    print('assets', len(assets), flush=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(safe, assets))
    print('finished', sum(r is not None for r in res), 'of', len(assets), flush=True)
