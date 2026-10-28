"""Phase 6-5 (B4 upgraded): fit DLAG to normal (control) trials of both hemispheres.

B4 asked whether a classical latent-variable model reproduces the joint statistics of the two hemispheres (static
factor analysis: yes). DLAG is the strongest classical, *dynamic and directional* version: it splits shared activity
into within-hemisphere latents and across-hemisphere latents, each across latent with its own continuous delay
(sign = direction of flow). If correlations could reveal the callosal channel anywhere, it should be here.

Per session (code1 = area 1, code2 = area 2): control trials, delay epoch (-1.7..0 s), 50-ms bins, sqrt counts,
condition means (instruction x lick) removed (trial-to-trial fluctuations, as in Gokcen et al.).
Model selection as in the paper: FA dimensionality per area (cross-validated), then across dimensionality pa
(0..min(p1, p2, 4)) by 2-fold cross-validated likelihood, pw_i = p_i - pa. Each delay is called "directional" when
fixing it to 0 lowers the cross-validated likelihood.
Saved: delays, timescales, loadings, unit indices -> results/phase6/dlag/<session>.npz and a summary CSV.

usage: python phase6_dlag.py CACHE_DIR LABELS_CSV OUTDIR [i0 i1]
"""
import sys, os, glob, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from dlag import DLAG
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

torch.set_num_threads(1)
CACHE, LABELS, OUT = sys.argv[1:4]
I0, I1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 999)
MAXU = 45
rng = np.random.default_rng(66)


def prep(path, lab):
    z = np.load(path)
    sess = os.path.basename(path)[:-4].replace('__', '/')
    l = lab[lab.session == sess]; hemi = dict(zip(l.unit, l.hemi)); uid = z['unit_ids']
    ok = (z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) &
          (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05) & ((z['HitR'] + z['HitL'] + z['ErrR'] + z['ErrL']) > 0))
    t = z['edges'][:-1]; D = (t >= -1.7 - 1e-6) & (t < -1e-6)
    ctrl = np.where(ok & (z['PhotostimulationType'] == 0))[0]
    c = z['counts'][ctrl][:, :, D].astype(float)
    instR = ((z['HitR'] + z['ErrR']) > 0)[ctrl]; lickR = ((z['HitR'] + z['ErrL']) > 0)[ctrl]
    units = {}
    for h in ('code1', 'code2'):
        ix = np.array([i for i, u in enumerate(uid) if hemi.get(u) == h])
        if len(ix) == 0: units[h] = ix; continue
        fr = c[:, ix].mean((0, 2)) / 0.05
        ix = ix[fr >= 1.0]; fr = fr[fr >= 1.0]
        units[h] = ix[np.argsort(-fr)[:MAXU]]
    if min(len(units['code1']), len(units['code2'])) < 8 or len(ctrl) < 80:
        return None
    u = np.r_[units['code1'], units['code2']]
    Y = np.sqrt(c[:, u])                                        # trials x units x T
    for a in (True, False):
        for b in (True, False):
            m = (instR == a) & (lickR == b)
            if m.sum(): Y[m] -= Y[m].mean(0)
    if not os.environ.get('KEEP_TRIAL_MEAN'):
        Y -= Y.mean(2, keepdims=True)                           # remove slow trial-to-trial offsets (as in the paper)
    return dict(sess=sess, Y=Y.transpose(0, 2, 1).copy(), q1=len(units['code1']), q2=len(units['code2']),
                units=units, n=len(ctrl))


def fa_dim(Y, kmax=8):
    """cross-validated FA dimensionality, folds over trials (Y: trials x T x units)"""
    f = rng.permutation(len(Y)) % 3
    best, bk = -np.inf, 0
    for k in range(0, min(kmax, Y.shape[2] - 1) + 1):
        s = 0.0
        for j in range(3):
            tr = Y[f != j].reshape(-1, Y.shape[2]); te = Y[f == j].reshape(-1, Y.shape[2])
            if k == 0:
                mu, v = tr.mean(0), tr.var(0) + 1e-6
                s += np.sum(-0.5 * (np.log(2 * np.pi * v).sum() + ((te - mu) ** 2 / v).sum(1)))
            else:
                s += FactorAnalysis(k, random_state=0).fit(tr).score(te) * len(te)
        if s > best: best, bk = s, k
    return bk


def cv_ll(Y, q1, q2, pa, pw1, pw2, folds, iters=150, fixed=None, warm=None):
    tot = 0.0
    for k in range(2):
        tr, te = Y[folds != k], Y[folds == k]
        m = DLAG(q1, q2, pa, pw1, pw2, Y.shape[1], 0.05, seed=k)
        if fixed is not None:
            m.dmask[fixed] = 0.0
        if warm is not None:
            m.copy_from(warm[k]); m.fit(tr, iters=60, init=False)
        else:
            m.fit(tr, iters=iters)
        tot += m.score(te) * len(te)
        if warm is None and fixed is None:
            cv_ll.models[k] = m
    return tot / len(Y)


cv_ll.models = {}


def run(d):
    Y, q1, q2 = d['Y'], d['q1'], d['q2']
    X = Y.reshape(-1, q1 + q2)
    p1, p2 = fa_dim(Y[:, :, :q1]), fa_dim(Y[:, :, q1:])
    folds = rng.permutation(len(Y)) % 2
    res, models = {}, {}
    for pa in range(0, min(p1, p2, 4) + 1):
        cv_ll.models = {}
        res[pa] = cv_ll(Y, q1, q2, pa, p1 - pa, p2 - pa, folds)
        models[pa] = dict(cv_ll.models)
    pa = max(res, key=res.get)
    row = dict(session=d['sess'], mouse=d['sess'].split('/')[0], q1=q1, q2=q2, n_trials=len(Y), p1=p1, p2=p2, pa=pa,
               **{f'cvll_pa{k}': v for k, v in res.items()})
    final = DLAG(q1, q2, pa, p1 - pa, p2 - pa, Y.shape[1], 0.05).fit(Y, iters=200)
    s = final.summary()
    sig = np.zeros(pa, bool)
    for j in range(pa):
        ll0 = cv_ll(Y, q1, q2, pa, p1 - pa, p2 - pa, folds, fixed=j, warm=models[pa])
        sig[j] = ll0 < res[pa]
    row.update(across_strength1=s['across_strength1'], across_strength2=s['across_strength2'],
               delays_ms=';'.join(f'{x * 1000:.1f}' for x in s['delays']),
               directional=';'.join(str(int(x)) for x in sig),
               taus_ms=';'.join(f'{x * 1000:.0f}' for x in s['tau'][:pa]))
    np.savez(os.path.join(OUT, 'dlag', d['sess'].replace('/', '__') + '.npz'), units1=d['units']['code1'],
             units2=d['units']['code2'], delays=s['delays'], tau=s['tau'], Ca1=s['Ca1'], Ca2=s['Ca2'], Cw1=s['Cw1'],
             Cw2=s['Cw2'], R=s['R'], directional=sig, var_across1=s['var_across1'], var_across2=s['var_across2'])
    return row


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT, 'dlag'), exist_ok=True)
    lab = pd.read_csv(LABELS)
    out = os.path.join(OUT, f'phase6_dlag_{I0:03d}_{I1:03d}.csv')
    from dlag_summary import read_summary
    _s = read_summary(os.path.dirname(OUT.rstrip('/')) if os.path.basename(OUT.rstrip('/')) == 'phase6' else OUT)
    done = set(_s.session) if len(_s) else set()
    step = int(os.environ.get('SUBSET_STEP', '1'))
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz')))[I0:I1:step]:
        d = prep(p, lab)
        if d is None or d['sess'] in done: continue
        t = time.time(); row = run(d); row['sec'] = time.time() - t
        pd.DataFrame([row]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(row['session'], 'pa', row['pa'], 'delays', row['delays_ms'], f"{row['sec']:.0f}s", flush=True)
