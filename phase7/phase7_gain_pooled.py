"""Phase 7-2: how much of the state dependence does single-neuron gain explain? (pooled, with bootstrap CIs)

Phase 6-4 compared medians of per-pair sums, which are very noisy (a pair's c1 in neuron-sum units is a small
difference of two sums). Here the predictions and observations are summed over all pairs first (ratio of sums),
and the uncertainty comes from resampling pairs.

Predictions (no parameter fitted to the silencing slopes):
  own gain           u_stim,i = u_ctrl,i * g_i^alpha
  + recurrent pass   u_stim,i = u_ctrl,i * g_i^alpha * S_i^k,  S_i = |u|-weighted mean of g_j^alpha over the other
                     neurons of the pair (the state signal a neuron receives from a population that is itself scaled),
                     k = number of recurrent passes (k = 1: one pass)
Also: do neurons whose own rate is unchanged (0.9 < g < 1.1) still lose the state they carry?

usage: python phase7_gain_pooled.py RESULTS_DIR
"""
import sys, os, json, numpy as np, pandas as pd

R = sys.argv[1]
u = pd.read_csv(os.path.join(R, 'phase6', 'phase6_neurongain_units.csv'))
rows, units = [], []
for (s, sil), d in u.groupby(['session', 'silenced']):
    uc, us, G = d.u_ctrl.values, d.u_stim.values, d.g.values
    W = np.abs(uc)
    r = dict(session=s, silenced=sil, obs=np.sum(us - uc), b1=uc.sum())
    for a in (1.0, 1.3):
        own = G ** a; tot = np.sum(W * own); S = (tot - W * own) / (W.sum() - W + 1e-12)
        for k in (0, 1, 2):
            r[f'a{a}_k{k}'] = np.sum(uc * (own * S ** k - 1))
        if a == 1.0:
            units.append(d.assign(S=S))
    rows.append(r)
P = pd.DataFrame(rows); U = pd.concat(units)
res = dict(n_pairs=len(P), pooled_relative_loss=P.obs.sum() / P.b1.sum(),
           median_obs=P.obs.median())
for c in [x for x in P.columns if x.startswith('a')]:
    bs = [(lambda q: q[c].sum() / q.obs.sum())(P.sample(frac=1, replace=True, random_state=i)) for i in range(2000)]
    res[c] = dict(fraction=P[c].sum() / P.obs.sum(), ci=list(np.percentile(bs, [2.5, 97.5])),
                  median_ratio=P[c].median() / P.obs.median())
m = (U.g > 0.9) & (U.g < 1.1)
ratio = lambda q: q.u_stim.sum() / q.u_ctrl.sum()
bs = [ratio(U[m].sample(frac=1, replace=True, random_state=i)) for i in range(2000)]
res['unchanged_rate'] = dict(n=int(m.sum()), slope_ratio=ratio(U[m]), ci=list(np.percentile(bs, [2.5, 97.5])),
                             predicted_S=float(np.average(U[m].S, weights=np.abs(U[m].u_ctrl))))
os.makedirs(os.path.join(R, 'phase7'), exist_ok=True)
json.dump(res, open(os.path.join(R, 'phase7', 'phase7_gain_pooled.json'), 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
