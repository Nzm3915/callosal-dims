"""Phase 6-4: a better hypothesis for the state dependence -- each neuron's input-output curve.

Every gain carrier tried at the population level failed (gate, attractor, ramp, partner loss, FS rate, mean rate).
Alternative: the gain change is per neuron and comes from the curvature of single-neuron input-output functions.
For a power-law neuron r = k [I]_+^n (supralinear, n > 1; e.g. Priebe & Ferster 2008, the stabilised
supralinear network), the response to a small extra input dI is dr = n k^(1/n) r^((n-1)/n) dI. So when silencing A
lowers neuron i's rate from r_i to g_i r_i, the slope with which it carries B's own trial-to-trial state scales as

    slope_stim,i / slope_ctrl,i = g_i ** alpha,      alpha = (n - 1) / n

alpha = 0: purely additive (threshold-linear above threshold; no state dependence)
alpha = 1: purely multiplicative (rates and their fluctuations scale together)

Because g_i differs widely between neurons (some up, many down), averaging rates into one "carrier" cancels the
effect, which would explain why every population carrier failed.

Per neuron i of B (pair: silence A):
  slope_i = cov(x_i, cB^-i) / var(cB^-i),  x_i = rate during the laser (-1.7..-0.9 s), cB^-i = B's pre-laser
            confidence computed WITHOUT neuron i (no shared spike noise); separately on control and on stim trials
  g_i     = mean rate on stim trials / control trials, laser window
  u_i     = cd_i * slope_i / sd  -> the population slope is b1 = sum_i u_i
Prediction without any fit to the stimulation slopes:  c1_pred(alpha) = sum_i u_ctrl,i (g_i^alpha - 1)
Observed (same decomposition):                          c1_obs        = sum_i (u_stim,i - u_ctrl,i)
alpha is then estimated per pair and pooled, and checked in rate-ratio bins.

usage: python phase6_neurongain.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from phase6_ramp import load, rate

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(65)
PRE, LAS = (-2.1, -1.7), (-1.7, -0.9)
ALPHAS = np.round(np.arange(0, 3.01, 0.05), 2)


def pair_rows(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ub = d['idx'][rec]
    if len(ub) < 8 or len(stim) < 20:
        return [], None
    fold = rng.permutation(len(ctrl_all)) % 2
    U = {k: np.zeros(len(ub)) for k in ('u_ctrl', 'u_stim')}
    G = np.zeros(len(ub)); RC = np.zeros(len(ub)); nf = 0
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        ev = ctrl_all[fold == f]
        R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
        cd = rate(d, ub, R, -1.7, 0).mean(0) - rate(d, ub, L, -1.7, 0).mean(0); cd /= np.linalg.norm(cd) + 1e-12
        preR, preL = rate(d, ub, R, *PRE), rate(d, ub, L, *PRE)
        sp = rng.permutation(stim); sA, sB = (sp[:len(sp) // 2], sp[len(sp) // 2:]) if f == 0 else (sp[len(sp) // 2:], sp[:len(sp) // 2])
        Xpre = {'c': rate(d, ub, ev, *PRE), 's': rate(d, ub, sB, *PRE)}
        Xlas = {'c': rate(d, ub, ev, *LAS), 's': rate(d, ub, sB, *LAS)}
        XlasG = rate(d, ub, sA, *LAS)                              # rate ratio from the OTHER half of stim trials
        sg = {'c': np.where(d['instR'][ev], 1.0, -1.0), 's': np.where(d['instR'][sB], 1.0, -1.0)}
        # y uses the full CD in the laser window: its sd (control, fit trials) normalises the slopes
        lasR, lasL = rate(d, ub, R, *LAS) @ cd, rate(d, ub, L, *LAS) @ cd
        sd_y = np.sqrt((lasR.var() + lasL.var()) / 2) + 1e-6
        for i in range(len(ub)):
            m = np.ones(len(ub), bool); m[i] = False
            c = cd[m] / (np.linalg.norm(cd[m]) + 1e-12)
            pR, pL = preR[:, m] @ c, preL[:, m] @ c
            bnd = (pR.mean() + pL.mean()) / 2; sdp = np.sqrt((pR.var() + pL.var()) / 2) + 1e-6
            cb = {k: sg[k] * (Xpre[k][:, m] @ c - bnd) / sdp for k in ('c', 's')}
            mu0, s0 = cb['c'].mean(), cb['c'].std() + 1e-9
            for k, nm in (('c', 'u_ctrl'), ('s', 'u_stim')):
                z = (cb[k] - mu0) / s0
                yk = sg[k] * Xlas[k][:, i]                         # signed so that the CD sign convention holds
                slope = np.cov(yk, z)[0, 1] / (np.var(z, ddof=1) + 1e-9)
                U[nm][i] += cd[i] * slope / sd_y / 2
            G[i] += (XlasG[:, i].mean() + 0.5) / (Xlas['c'][:, i].mean() + 0.5) / 2
            RC[i] += Xlas['c'][:, i].mean() / 2
        nf += 1
    c1_obs = np.sum(U['u_stim'] - U['u_ctrl']); b1 = np.sum(U['u_ctrl'])
    pred = {a: np.sum(U['u_ctrl'] * (G ** a - 1)) for a in ALPHAS}
    a_best = min(ALPHAS, key=lambda a: (pred[a] - c1_obs) ** 2)
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, nB=len(ub), b1_sum=b1,
               c1_obs=c1_obs, alpha_best=a_best, frac_down=np.mean(G < 0.8), frac_up=np.mean(G > 1.2),
               g_weighted=np.sum(np.abs(U['u_ctrl']) * G) / (np.sum(np.abs(U['u_ctrl'])) + 1e-12))
    row.update({f'c1_pred_{a}': pred[a] for a in ALPHAS})
    units = pd.DataFrame(dict(session=d['sess'], silenced=sil, unit=ub, g=G, rate_ctrl=RC, u_ctrl=U['u_ctrl'], u_stim=U['u_stim']))
    return [row], units


if __name__ == '__main__':
    lab = pd.read_csv(LABELS)
    rows, units = [], []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            r, u = pair_rows(d, sil, rec, code)
            rows += r
            if u is not None: units.append(u)
    df = pd.DataFrame(rows); un = pd.concat(units)
    df.to_csv(os.path.join(OUT, 'phase6_neurongain_pairs.csv'), index=False)
    un.to_csv(os.path.join(OUT, 'phase6_neurongain_units.csv'), index=False)
    print(len(df), 'pairs', len(un), 'units')
    print('b1 (sum) median', df.b1_sum.median().round(3), ' c1_obs median', df.c1_obs.median().round(3),
          ' relative', (df.c1_obs / df.b1_sum).median().round(3))
    for a in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        print(f'alpha {a}: c1_pred median {df[f"c1_pred_{a}"].median():+.3f}')
    # pooled alpha: minimise squared error across pairs
    err = {a: np.mean((df[f'c1_pred_{a}'] - df.c1_obs) ** 2) for a in ALPHAS}
    print('pooled best alpha', min(err, key=err.get), ' per-pair alpha median', df.alpha_best.median())
    # binned check
    un['lg'] = np.log2(un.g)
    bins = pd.qcut(un.lg, 8)
    gb = un.groupby(bins, observed=True)
    print(pd.DataFrame(dict(g=gb.g.median(), ratio=gb.u_stim.sum() / gb.u_ctrl.sum(), n=gb.size())).round(3))
