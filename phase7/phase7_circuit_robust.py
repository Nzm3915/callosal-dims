"""Phase 7-5: refit the heterogeneous circuit with a loss averaged over 3 network seeds (fit seeds 0-2), then test the
held-out quantities on 5 other seeds (3-7). Removes the dependence of B5' on one random network."""
import sys, json, numpy as np
sys.argv = ['h5b', 'results']
exec(open('hypotheses/h5b_circuit.py').read().replace("if __name__ == '__main__':", "if False:"))

def loss_multi(p, het, seeds=(0, 1, 2)):
    tot = 0.0
    for s in seeds:
        tb, out, _ = curves(p, het, seed=s)
        tot += sum(np.mean((out[k] - np.interp(tb, t_ms[W], target[k][W])) ** 2) for k in keys)
    return tot / len(seeds)

rs = np.random.default_rng(11)
res = {}
for variant in ('het', 'both'):
    lo = np.array([0.0, 0.0, 2.0, 0.3, 0.0, 0.0]); hi = np.array([1.5, 3.0, 30.0, 3.0, 0.5 if variant == 'both' else 0.0, 2.5])
    J = json.load(open('results/hypotheses/t5b_circuit.json')); f = {x['variant']: x for x in J['fits']}[variant]
    best = (loss_multi(np.array([f['cE'], f['cI'], f['L'], f['gI'], f['beta'], f['sigma_c']]), True),
            np.array([f['cE'], f['cI'], f['L'], f['gI'], f['beta'], f['sigma_c']]))
    for i in range(60):
        p = lo + (hi - lo) * rs.random(6) if i < 20 else np.clip(best[1] + rs.normal(0, 0.06, 6) * (hi - lo), lo, hi)
        e = loss_multi(p, True)
        if np.isfinite(e) and e < best[0]: best = (e, p)
    e, p = best
    test = [curves(p, True, seed=s)[2] for s in range(3, 8)]
    res[variant] = dict(loss=e, params=dict(zip(['cE', 'cI', 'L', 'gI', 'beta', 'sigma_c'], map(float, p))),
                        **{k: [float(np.mean([t[k] for t in test])), float(np.std([t[k] for t in test]))] for k in test[0]})
    print(variant, json.dumps(res[variant]), flush=True)
json.dump(res, open('results/phase7/phase7_circuit_robust.json', 'w'), indent=1)
