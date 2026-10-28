"""Phase 7-7: B1 (committee readout) against a single-driver null, calibrated per session.

Observed per session (held-out control trials, end of delay): x_A, x_B = signed-by-choice confidence of each
hemisphere (CD from training trials); on disagreement trials, P(more confident side matches the lick) and
P(dominant side matches the lick). Per session we also measure, by splitting each hemisphere's units in halves,
the reliability of x_A and x_B (Spearman-Brown), and the correlation of x_A and x_B within trial type.

Null (single driver): the lick follows the dominant hemisphere's TRUE state only. True states have the
disattenuated correlation r_true = r_meas / sqrt(rel_A rel_B); measurements add noise with the measured
reliabilities; trial-type separation and lick noise are set so that each hemisphere's measured accuracy matches
the data. Alternative: the lick follows w_dom true_dom + w_other true_other; w_other is fitted per session by
matching P(more confident wins).

usage: python phase7_committee_test.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase6'))
from phase6_ramp import load, rate

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(77)


def conf(d, units, fit, trials, a=-0.4, b=0.0):
    R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
    cd = rate(d, units, R, -1.7, 0).mean(0) - rate(d, units, L, -1.7, 0).mean(0); cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = rate(d, units, R, a, b) @ cd, rate(d, units, L, a, b) @ cd
    bnd = (pR.mean() + pL.mean()) / 2; sd = np.sqrt((pR.var() + pL.var()) / 2) + 1e-6
    return (rate(d, units, trials, a, b) @ cd - bnd) / sd          # + = lick right side


def simulate(sep, rel, r_true, lick_noise, w_other, dom, n=60000, seed=0):
    r = np.random.default_rng(seed)
    s = np.where(r.random(n) < 0.5, 1.0, -1.0)
    L = np.linalg.cholesky(np.array([[1, r_true], [r_true, 1]]))
    true = s[:, None] * np.asarray(sep)[None, :] + r.normal(size=(n, 2)) @ L.T
    vt = true.var(0)
    meas = true + r.normal(size=(n, 2)) * np.sqrt(vt * (1 - np.asarray(rel)) / np.asarray(rel))
    drive = true[:, dom] + w_other * true[:, 1 - dom]
    lick = np.sign(drive + r.normal(0, lick_noise, n))
    dis = np.sign(meas[:, 0]) != np.sign(meas[:, 1])
    cf = np.where(np.abs(meas[:, 0]) >= np.abs(meas[:, 1]), np.sign(meas[:, 0]), np.sign(meas[:, 1]))
    return dict(p_conf=np.mean(cf[dis] == lick[dis]), p_dom=np.mean(np.sign(meas[dis, dom]) == lick[dis]),
                acc=[np.mean(np.sign(meas[:, i]) == lick) for i in (0, 1)])


rows = []
lab = pd.read_csv(LABELS)
for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
    d = load(p, lab)
    ua, ub = d['idx']['code1'], d['idx']['code2']
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    if min(len(ua), len(ub)) < 8 or len(ctrl) < 80: continue
    fold = rng.permutation(len(ctrl)) % 2
    X, Hh, lick, inst = [], [], [], []
    for f in (0, 1):
        tr, te = ctrl[fold != f], ctrl[fold == f]
        fit = tr[d['correct'][tr]]
        xa, xb = conf(d, ua, fit, te), conf(d, ub, fit, te)
        pa, pb = rng.permutation(len(ua)), rng.permutation(len(ub))
        halves = [conf(d, ua[pa[:len(ua) // 2]], fit, te), conf(d, ua[pa[len(ua) // 2:]], fit, te),
                  conf(d, ub[pb[:len(ub) // 2]], fit, te), conf(d, ub[pb[len(ub) // 2:]], fit, te)]
        X.append(np.c_[xa, xb]); Hh.append(np.c_[tuple(halves)]); lick.append(d['lickR'][te] if 'lickR' in d else None)
        inst.append(d['instR'][te])
    X = np.concatenate(X); Hh = np.concatenate(Hh); inst = np.concatenate(inst)
    z = np.load(p); lk = (((z['HitR'] + z['ErrL']) > 0))
    lick = np.concatenate([lk[ctrl[fold == f]] for f in (0, 1)])
    ls = np.where(lick, 1.0, -1.0)
    # observed statistics
    dis = np.sign(X[:, 0]) != np.sign(X[:, 1])
    acc = [np.mean(np.sign(X[:, i]) == ls) for i in (0, 1)]; dom = int(np.argmax(acc))
    cf = np.where(np.abs(X[:, 0]) >= np.abs(X[:, 1]), np.sign(X[:, 0]), np.sign(X[:, 1]))
    if dis.sum() < 10: continue
    obs_conf, obs_dom = np.mean(cf[dis] == ls[dis]), np.mean(np.sign(X[dis, dom]) == ls[dis])
    # reliabilities (split-half units, Spearman-Brown), within trial type
    def within(a):
        out = a.copy()
        for v in (True, False): out[inst == v] -= a[inst == v].mean(0)
        return out
    Hw = within(Hh); Xw = within(X)
    rh = [np.corrcoef(Hw[:, 0], Hw[:, 1])[0, 1], np.corrcoef(Hw[:, 2], Hw[:, 3])[0, 1]]
    rel = [float(np.clip(2 * r / (1 + r), 0.05, 0.98)) for r in rh]
    r_meas = np.corrcoef(Xw[:, 0], Xw[:, 1])[0, 1]
    r_true = float(np.clip(r_meas / np.sqrt(rel[0] * rel[1]), -0.95, 0.95))
    # trial-type separation in true-state SD units: mean |x| difference / within-type sd of the TRUE part
    sep = [abs(X[inst, i].mean() - X[~inst, i].mean()) / 2 / np.sqrt(Xw[:, i].var() * rel[i]) for i in (0, 1)]
    # lick noise chosen so that the null reproduces the dominant hemisphere's measured accuracy
    best = None
    for ln in (0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0):
        sim = simulate(sep, rel, r_true, ln, 0.0, dom)
        e = abs(sim['acc'][dom] - acc[dom])
        if best is None or e < best[0]: best = (e, ln, sim)
    _, ln, null = best
    # alternative: fit w_other to match the observed P(conf wins)
    alt = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        sim = simulate(sep, rel, r_true, ln, w, dom, seed=1)
        e = abs(sim['p_conf'] - obs_conf)
        if alt is None or e < alt[0]: alt = (e, w, sim)
    rows.append(dict(session=d['sess'], mouse=d['mouse'], n_dis=int(dis.sum()), rel_A=rel[0], rel_B=rel[1], r_meas=r_meas,
                     r_true=r_true, obs_conf=obs_conf, obs_dom=obs_dom, null_conf=null['p_conf'], null_dom=null['p_dom'],
                     w_other_fit=alt[1]))
df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True)
df.to_csv(os.path.join(OUT, 'phase7_committee_test.csv'), index=False)
from scipy.stats import wilcoxon
m = df.assign(ex_obs=df.obs_conf - df.obs_dom, ex_null=df.null_conf - df.null_dom, gap=df.obs_conf - df.null_conf)
g = m.groupby('mouse')[['ex_obs', 'ex_null', 'gap', 'w_other_fit', 'rel_A', 'rel_B', 'r_true']].median()
res = {c: dict(median_sessions=float(m[c].median()), median_mice=float(g[c].median()),
               p_mice=float(wilcoxon(g[c]).pvalue) if c in ('ex_obs', 'ex_null', 'gap') else None)
       for c in ('ex_obs', 'ex_null', 'gap', 'w_other_fit', 'rel_A', 'rel_B', 'r_true')}
res['p_mice_obs_vs_null'] = float(wilcoxon(g.ex_obs - g.ex_null).pvalue); res['n_sessions'] = len(df); res['n_mice'] = len(g)
json.dump(res, open(os.path.join(OUT, 'phase7_committee_test.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
