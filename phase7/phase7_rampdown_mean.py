"""Companion to phase7_rampdown.py: is the state dependence that outlasts the rates (about 0.1 s) just the displaced
state of B recovering? For the same windows, the mean loss of B's choice signal on stimulation trials,

    L0(w) = 1 - <y_stim>/<y_ctrl>,  y = trial-type-signed projection of B's rates in window w on B's choice axis,

pooled over pairs (sums). If the late state dependence is the tail of a recovering displacement, c1_obs(w)/b1 and
L0(w) fall off together (their ratio stays constant across windows). If it is a separate slow mechanism, the ratio
changes.

usage: python phase7_rampdown_mean.py CACHE_DIR LABELS_CSV OUTDIR   (after phase7_rampdown.py)
"""
import sys, os, glob, json, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
CACHE, LABELS, OUT = sys.argv[1:4]
sys.path.insert(0, os.path.join(here, '..', 'phase6'))
sys.argv = [sys.argv[0], CACHE, LABELS, OUT]
from phase6_ramp import load, rate
WINDOWS = [(-1.3, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.4)]
rng = np.random.default_rng(3)

if __name__ == '__main__':
    lab = pd.read_csv(LABELS); rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]; stim = np.where(d['ok'] & (d['pt'] == code))[0]
            ub = d['idx'][rec]
            if len(ub) < 8 or len(stim) < 20: continue
            fold = rng.permutation(len(ctrl)) % 2
            fit = ctrl[(fold == 0) & d['correct'][ctrl]]; ev = ctrl[fold == 1]
            R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
            cd = rate(d, ub, R, -1.7, 0).mean(0) - rate(d, ub, L, -1.7, 0).mean(0); cd /= np.linalg.norm(cd) + 1e-12
            for w in WINDOWS:
                bnd = (rate(d, ub, R, *w) @ cd).mean() / 2 + (rate(d, ub, L, *w) @ cd).mean() / 2
                y = lambda tr: np.where(d['instR'][tr], 1.0, -1.0) * (rate(d, ub, tr, *w) @ cd - bnd)
                rows.append(dict(session=d['sess'], mouse=d['mouse'], silenced=sil, win=f'{w[0]}..{w[1]}',
                                 y_ctrl=y(ev).mean(), y_stim=y(stim).mean()))
    df = pd.DataFrame(rows)
    rd = pd.read_csv(os.path.join(OUT, 'phase7_rampdown_pairs.csv'))
    out = {}
    mice = df.mouse.unique(); brng = np.random.default_rng(9)
    for w in [f'{a}..{b}' for a, b in WINDOWS]:
        s, r = df[df.win == w], rd[rd.win == w]
        L0 = 1 - s.y_stim.sum() / s.y_ctrl.sum(); c1 = r.c1_obs.sum() / r.b1_sum.sum()
        bs = []
        for _ in range(1000):
            mm = brng.choice(mice, len(mice))
            s2 = pd.concat([s[s.mouse == m] for m in mm]); r2 = pd.concat([r[r.mouse == m] for m in mm])
            l0 = 1 - s2.y_stim.sum() / s2.y_ctrl.sum(); bs.append((l0, (r2.c1_obs.sum() / r2.b1_sum.sum()) / l0))
        bs = np.array(bs)
        out[w] = dict(L0=L0, L0_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), c1_rel=c1, c1_over_L0=c1 / L0,
                      c1_over_L0_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())
        print(w, {k: np.round(v, 3).tolist() for k, v in out[w].items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, 'phase7_rampdown_mean.json'), 'w'), indent=1)
