"""Phase 7-6: could 'the more confident hemisphere wins' (B1) arise without any combination of the two hemispheres?

Null model: behaviour follows ONE hemisphere's true state (the dominant one); the other is irrelevant. Both true states
are correlated (shared fluctuations, r_true), and the decoders measure them with noise (reliability rel). Simulated
over a grid of r_true and rel, and of how strongly the dominant hemisphere drives behaviour; reported is
P(more confident wins | disagreement) - P(dominant wins | disagreement). The data gave +0.07 (0.68 vs 0.61).
Alternative: behaviour follows the SUM of the true states. If the null can produce a positive difference as large as
the data, B1 is not supported by this statistic.
"""
import numpy as np, json
rng = np.random.default_rng(0)
out = []
for r_true in (0.2, 0.4, 0.6, 0.8):
    for rel in (0.3, 0.5, 0.7, 0.9):
        for model in ('dominant_only', 'sum'):
            n = 200000
            z = rng.normal(size=(n, 2)); s = np.where(rng.random(n) < 0.5, 1.0, -1.0)
            # true states: trial-type signal + correlated fluctuation
            cov = np.array([[1, r_true], [r_true, 1]]); L = np.linalg.cholesky(cov)
            true = 0.8 * s[:, None] + z @ L.T
            # measurement noise sets reliability: var_true / (var_true + var_noise) = rel
            vt = true.var(0); noise = rng.normal(size=(n, 2)) * np.sqrt(vt * (1 - rel) / rel)
            meas = true + noise
            if model == 'dominant_only':
                lick = np.sign(true[:, 0] + rng.normal(0, 0.3, n))          # hemisphere 0 decides
            else:
                lick = np.sign(true.sum(1) + rng.normal(0, 0.3, n))
            dis = np.sign(meas[:, 0]) != np.sign(meas[:, 1])
            conf = np.where(np.abs(meas[:, 0]) >= np.abs(meas[:, 1]), np.sign(meas[:, 0]), np.sign(meas[:, 1]))
            # 'dominant' as identified from data: the hemisphere whose measured sign agrees more with the lick
            acc = [np.mean(np.sign(meas[:, i]) == lick) for i in (0, 1)]; dom = int(np.argmax(acc))
            pc = np.mean(conf[dis] == lick[dis]); pdm = np.mean(np.sign(meas[dis, dom]) == lick[dis])
            out.append(dict(r_true=r_true, rel=rel, model=model, p_conf=pc, p_dom=pdm, diff=pc - pdm))
for model in ('dominant_only', 'sum'):
    d = [o['diff'] for o in out if o['model'] == model]
    print(model, 'diff range', round(min(d), 3), round(max(d), 3))
json.dump(out, open('results/phase7/phase7_committee_null.json', 'w'), indent=1)
