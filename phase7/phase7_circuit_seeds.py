"""Phase 7-4: are the B5' circuit results robust to the random network (connectivity, drives)?
Re-evaluates the fitted parameters of each variant on 5 new network seeds (no refit)."""
import sys, json, numpy as np
sys.argv = ['h5b', sys.argv[1] if len(sys.argv) > 1 else 'results']
exec(open('hypotheses/h5b_circuit.py').read().replace("if __name__ == '__main__':", "if False:"))
J = json.load(open('results/hypotheses/t5b_circuit.json'))
out = {}
for f in J['fits']:
    p = np.array([f['cE'], f['cI'], f['L'], f['gI'], f['beta'], f['sigma_c']]); het = f['variant'] in ('het', 'both')
    vals = []
    for seed in range(1, 6):
        tb, o, ex = curves(p, het, seed=seed); vals.append(ex)
    out[f['variant']] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))] for k in vals[0]}
    print(f['variant'], {k: [round(x, 3) for x in v] for k, v in out[f['variant']].items()}, flush=True)
json.dump(out, open('results/phase7/phase7_circuit_seeds.json', 'w'), indent=1)
