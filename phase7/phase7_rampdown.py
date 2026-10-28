"""Next question 1 of docs/01: does the state dependence follow the single-neuron gain while the silencing strength
changes continuously - within the same neurons?

The between-neuron test of Phase 6-7 (neurons silenced more lose more of B's state) cannot exclude that "strongly
silenced" neurons are a different kind of neuron. During the laser ramp-down (-0.9 to about -0.7 s) the same
neurons' rate ratios g_i(t) climb back toward 1. If the gain account is right, the loss of state information must
climb back with them, along the prediction fixed beforehand (alpha = 1, no fit to the stimulation slopes):

    c1_pred(t) = sum_i u_ctrl,i(t) (g_i(t) - 1),        c1_obs(t) = sum_i (u_stim,i(t) - u_ctrl,i(t))

computed exactly as in phase6_neurongain.py (leave-one-out confidence before the laser, g_i from the other half of
the stimulation trials), but with the response window moved through the end of the laser.
Pooled over pairs (sums, relative to sum b1) with a mouse bootstrap.

usage: python phase7_rampdown.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, json, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
CACHE, LABELS, OUT = sys.argv[1:4]
sys.path.insert(0, os.path.join(here, '..', 'phase6'))
sys.argv = [sys.argv[0], CACHE, LABELS, OUT]
import phase6_neurongain as ng
from phase6_ramp import load

WINDOWS = [(-1.3, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.4)]


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS); rows = []
    files = sorted(glob.glob(os.path.join(CACHE, '*.npz')))
    for p in files:
        d = load(p, lab)
        for w in WINDOWS:
            ng.LAS = w; ng.ALPHAS = np.array([1.0]); ng.rng = np.random.default_rng(65)
            for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
                r, u = ng.pair_rows(d, sil, rec, code)
                if not r: continue
                r = r[0]; r['win'] = f'{w[0]}..{w[1]}'
                r['g_med'] = float(np.median(u.g)); r['g_wmean'] = r['g_weighted']
                rows.append({k: r[k] for k in ('session', 'mouse', 'silenced', 'win', 'b1_sum', 'c1_obs', 'c1_pred_1.0',
                                                'g_med', 'g_wmean', 'frac_down')})
        print(os.path.basename(p), len(rows), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, 'phase7_rampdown_pairs.csv'), index=False)
    rng = np.random.default_rng(7); mice = df.mouse.unique()
    def pooled(sub):
        b = sub.b1_sum.sum()
        return sub.c1_obs.sum() / b, sub['c1_pred_1.0'].sum() / b
    out = {}
    for w in df.win.unique():
        sub = df[df.win == w]; o, pr = pooled(sub)
        bs = []
        for _ in range(1000):
            mm = rng.choice(mice, len(mice)); s2 = pd.concat([sub[sub.mouse == m] for m in mm])
            bs.append(pooled(s2))
        bs = np.array(bs)
        out[w] = dict(obs=o, pred=pr, obs_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                      pred_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                      ratio=o / pr if pr else np.nan, ratio_ci=np.percentile(bs[:, 0] / bs[:, 1], [2.5, 97.5]).tolist(),
                      g_med=float(sub.g_med.median()), g_wmean=float(sub.g_wmean.median()), n_pairs=len(sub))
        print(w, {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in out[w].items()}, flush=True)
    # within-pair test across windows: does the change of c1_obs from the plateau follow the change of c1_pred?
    piv_o = df.pivot_table(index=['session', 'silenced'], columns='win', values='c1_obs')
    piv_p = df.pivot_table(index=['session', 'silenced'], columns='win', values='c1_pred_1.0')
    base = f'{WINDOWS[0][0]}..{WINDOWS[0][1]}'
    do = (piv_o.sub(piv_o[base], axis=0)).drop(columns=base).values.ravel()
    dp = (piv_p.sub(piv_p[base], axis=0)).drop(columns=base).values.ravel()
    ok = np.isfinite(do) & np.isfinite(dp)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(dp[ok], do[ok])
    out['within_pair_change'] = dict(spearman_rho=float(rho), p=float(pval), n=int(ok.sum()),
                                     slope_obs_on_pred=float(np.polyfit(dp[ok], do[ok], 1)[0]))
    print(out['within_pair_change'])
    json.dump(out, open(os.path.join(OUT, 'phase7_rampdown.json'), 'w'), indent=1)
