"""Bold hypotheses turned into algorithms and tested on DANDI 000894 (see docs/03_bold_hypotheses.md).

T1 Committee readout (who decides when the hemispheres disagree?)
   Each hemisphere's choice decoder gives x_A, x_B at the end of the delay (positive = "lick right").
   Readout algorithms compared on held-out control trials (correct + error), predicting the actual lick:
     single   best single hemisphere (chosen on training trials)
     sum      sign(x_A + x_B)
     linear   logistic(w_A x_A + w_B x_B + b)
     maxconf  follow the hemisphere with the larger |x| (winner-take-all by confidence)
     precision logistic(w_A x_A |x_A|^g + w_B x_B |x_B|^g + b), g fitted (g = 0 -> linear; g > 0 -> confident side
              weighs more, a precision-weighted vote)
   Disagreement trials (sign x_A != sign x_B) are scored separately.

T2 Gain link (callosal input sets the gain of the other side's choice representation)
   Choice-axis model x_B(t+1) = x_B + a(t) + k x_B + p u(t) + m u(t) x_B, u(t) = A's mean z-scored population rate
   (drops when A is silenced). m != 0 is a multiplicative (gain) effect. Compared with the additive model (m = 0)
   by the held-out state dependence c1 (data: c1 < 0).

T3 Sampling hypothesis (neural variability = samples from a posterior over the choice)
   Within-trial fluctuation of x_B over the late delay (100-ms windows, trial-type mean removed) relative to the
   fluctuation expected from Poisson spiking along the decoder. Predictions: excess variability is larger when the
   trial's mean confidence is low and on error trials; the two hemispheres' within-trial fluctuations co-vary.

T4 Can a classical generative model reproduce the joint statistics of both hemispheres?
   Factor analysis (a classical latent-variable model) fitted to spike counts (late delay, 400 ms) of both
   hemispheres on half of the control trials of one trial type; compared with the held-out covariance, within and
   across hemispheres, relative to the split-half ceiling.

usage: python h_tests.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(7)


def load(path, lab):
    z = np.load(path)
    sess = os.path.basename(path)[:-4].replace('__', '/')
    l = lab[lab.session == sess]
    hemi = dict(zip(l.unit, l.hemi))
    uid = z['unit_ids']
    idx = {h: np.array([i for i, u in enumerate(uid) if hemi.get(u) == h]) for h in ('code1', 'code2')}
    ok = (z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) &
          (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05) &
          ((z['HitR'] + z['HitL'] + z['ErrR'] + z['ErrL']) > 0))
    c = z['counts'].astype(np.float32)
    return dict(sess=sess, mouse=sess.split('/')[0], c=c, t=z['edges'][:-1], idx=idx, ok=ok,
                instR=(z['HitR'] + z['ErrR']) > 0, lickR=(z['HitR'] + z['ErrL']) > 0,
                correct=(z['HitR'] + z['HitL']) > 0, pt=z['PhotostimulationType'])


def rate(d, units, trials, a, b):
    w = (d['t'] >= a - 1e-6) & (d['t'] < b - 1e-6)
    return d['c'][np.ix_(trials, units)][:, :, w].sum(-1) / (w.sum() * 0.05)


def cd_fit(d, units, fit):
    R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
    cd = rate(d, units, R, -1.7, 0).mean(0) - rate(d, units, L, -1.7, 0).mean(0)
    cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = rate(d, units, R, -0.4, 0) @ cd, rate(d, units, L, -0.4, 0) @ cd
    bnd = (pR.mean() + pL.mean()) / 2; sd = np.sqrt((pR.var() + pL.var()) / 2) + 1e-6
    return cd, bnd, sd


def loglik(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def t1_committee(d):
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    if min(len(d['idx']['code1']), len(d['idx']['code2'])) < 5: return None
    fold = rng.permutation(len(ctrl)) % 2
    rows = []
    for f in (0, 1):
        tr, te = ctrl[fold != f], ctrl[fold == f]
        fit = tr[d['correct'][tr]]
        X = {}
        for h in ('code1', 'code2'):
            cd, bnd, sd = cd_fit(d, d['idx'][h], fit)
            X[h] = ((rate(d, d['idx'][h], tr, -0.4, 0) @ cd - bnd) / sd, (rate(d, d['idx'][h], te, -0.4, 0) @ cd - bnd) / sd)
        ytr, yte = d['lickR'][tr].astype(float), d['lickR'][te].astype(float)
        a_tr, b_tr = X['code1'][0], X['code2'][0]; a_te, b_te = X['code1'][1], X['code2'][1]

        def fitlog(feat_tr):
            def nll(w): return -loglik(1 / (1 + np.exp(-(feat_tr @ w[:-1] + w[-1]))), ytr) + 1e-3 * np.sum(w[:-1] ** 2)
            return minimize(nll, np.zeros(feat_tr.shape[1] + 1), method='L-BFGS-B').x

        out = {}
        # single best hemisphere
        accs = {h: np.mean((X[h][0] > 0) == ytr) for h in X}
        hb = max(accs, key=accs.get)
        out['single'] = (X[hb][1] > 0).astype(float)
        out['sum'] = ((a_te + b_te) > 0).astype(float)
        w = fitlog(np.c_[a_tr, b_tr]); out['linear'] = 1 / (1 + np.exp(-(np.c_[a_te, b_te] @ w[:-1] + w[-1])))
        out['maxconf'] = np.where(np.abs(a_te) >= np.abs(b_te), a_te > 0, b_te > 0).astype(float)
        best = None
        for g in (0.0, 0.5, 1.0, 2.0):
            ftr = np.c_[a_tr * np.abs(a_tr) ** g, b_tr * np.abs(b_tr) ** g]
            w = fitlog(ftr); ll = loglik(1 / (1 + np.exp(-(ftr @ w[:-1] + w[-1]))), ytr)
            if best is None or ll > best[0]: best = (ll, g, w)
        _, g, w = best
        out['precision'] = 1 / (1 + np.exp(-(np.c_[a_te * np.abs(a_te) ** g, b_te * np.abs(b_te) ** g] @ w[:-1] + w[-1])))
        dis = np.sign(a_te) != np.sign(b_te)
        row = dict(session=d['sess'], mouse=d['mouse'], fold=f, n_te=len(te), n_dis=int(dis.sum()), g=g,
                   dominant=hb, w_ratio=np.nan)
        for k, p in out.items():
            row[f'acc_{k}'] = np.mean((p > 0.5) == yte)
            row[f'accdis_{k}'] = np.mean((p[dis] > 0.5) == yte[dis]) if dis.sum() else np.nan
        # on disagreement trials: does the more confident hemisphere win? does the dominant one win?
        if dis.sum():
            conf_win = np.where(np.abs(a_te) >= np.abs(b_te), a_te > 0, b_te > 0)[dis] == yte[dis]
            dom = (X[hb][1] > 0)[dis] == yte[dis]
            row.update(p_moreconf_wins=conf_win.mean(), p_dominant_wins=dom.mean())
        rows.append(row)
    return rows


def popz(d, units, ctrl):
    """mean z-scored rate of a population per 100-ms bin (z from control delay)"""
    c = d['c'][:, units, :]
    nb = c.shape[-1] // 2
    r = (c[..., :2 * nb:2] + c[..., 1:2 * nb:2]) / 0.1
    t = d['t'][:2 * nb:2]
    D = (t >= -1.7) & (t < 0)
    mu = r[ctrl][:, :, D].mean((0, 2)); sd = r[ctrl][:, :, D].std((0, 2)) + 1.0
    return ((r - mu[None, :, None]) / sd[None, :, None]).mean(1), t


def t2_gain(d, sil, rec, code):
    """choice-axis model with additive vs multiplicative input from A's population rate"""
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]; stim = np.where(d['ok'] & (d['pt'] == code))[0]
    if len(stim) < 20: return []
    uA, t = popz(d, d['idx'][sil], ctrl)
    # choice confidence of B, 400-ms windows stepped by 100 ms
    c = d['c'][:, d['idx'][rec], :]
    cs = np.cumsum(np.concatenate([np.zeros_like(c[..., :1]), c], -1), -1)
    r = ((cs[..., 8:] - cs[..., :-8]) / 0.4)[..., ::2]; tw = d['t'][8:][::2] + 0.05   # window end times
    D = np.where((tw > -1.3 - 1e-6) & (tw <= 1e-6))[0]
    pre = int(np.argmin(np.abs(tw - (-1.7)))); steps = np.arange(pre, pre + 8)
    # map uA (100-ms bins starting at t) to window ends: use the bin that ends at each window end
    ub = np.array([int(np.argmin(np.abs(t + 0.1 - te))) for te in tw])
    rows = []
    for f in (0, 1):
        sets = {}
        for name, pool in (('c', ctrl), ('s', stim)):
            a_, b_ = [], []
            for side in (True, False):
                p = np.random.default_rng(30).permutation(pool[d['instR'][pool] == side]); h = len(p) // 2
                a_.append(p[:h]); b_.append(p[h:])
            A_, B_ = np.concatenate(a_), np.concatenate(b_)
            sets[name] = (A_, B_) if f == 0 else (B_, A_)
        tr = np.r_[sets['c'][0], sets['s'][0]]; te = np.r_[sets['c'][1], sets['s'][1]]
        st_te = np.r_[np.zeros(len(sets['c'][1])), np.ones(len(sets['s'][1]))]
        fit = sets['c'][0][d['correct'][sets['c'][0]]]
        R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
        cd = r[R][:, :, D].mean((0, 2)) - r[L][:, :, D].mean((0, 2)); cd /= np.linalg.norm(cd) + 1e-12
        pR, pL = np.einsum('tub,u->tb', r[R], cd), np.einsum('tub,u->tb', r[L], cd)
        bnd = (pR.mean(0) + pL.mean(0)) / 2; sd = np.sqrt((pR.var(0) + pL.var(0)) / 2).mean() + 1e-6

        def X(trials):
            s = np.where(d['instR'][trials], 1.0, -1.0)[:, None]
            return s * (np.einsum('tub,u->tb', r[trials], cd) - bnd) / sd
        xtr, xte = X(tr), X(te)
        u_tr, u_te = uA[tr][:, ub], uA[te][:, ub]
        out = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, fold=f)

        def reg(y, x0, st):
            m = st == 0; xs = (x0 - x0[m].mean()) / (x0[m].std() + 1e-9)
            b = np.linalg.lstsq(np.c_[np.ones_like(xs), xs, st, st * xs], y, rcond=None)[0]
            return b
        bo = reg(xte[:, steps + 1].mean(1), xte[:, pre], st_te)
        out.update(obs_c0=bo[2], obs_c1=bo[3], obs_b1=bo[1], obs_b0=bo[0])
        for model in ('additive', 'gain'):
            rowsX, Y = [], []
            for j, tt in enumerate(steps):
                T = np.zeros((len(tr), len(steps))); T[:, j] = 1
                cols = [T, xtr[:, tt:tt + 1], u_tr[:, tt:tt + 1]]
                if model == 'gain': cols.append(u_tr[:, tt:tt + 1] * xtr[:, tt:tt + 1])
                rowsX.append(np.hstack(cols)); Y.append(xtr[:, tt + 1] - xtr[:, tt])
            Xm, Ym = np.vstack(rowsX), np.concatenate(Y)
            w = np.linalg.lstsq(Xm, Ym, rcond=None)[0]
            x = xte[:, pre].copy(); traj = []
            for j, tt in enumerate(steps):
                T = np.zeros((len(te), len(steps))); T[:, j] = 1
                cols = [T, x[:, None], u_te[:, tt:tt + 1]]
                if model == 'gain': cols.append(u_te[:, tt:tt + 1] * x[:, None])
                x = x + np.hstack(cols) @ w; traj.append(x)
            bp = reg(np.stack(traj, 1).mean(1), xte[:, pre], st_te)
            out.update({f'{model}_c0': bp[2], f'{model}_c1': bp[3], f'{model}_b1': bp[1], f'{model}_b0': bp[0]})
            if model == 'gain': out['gain_m'] = w[-1]; out['gain_p'] = w[-2]
            else: out['add_p'] = w[-1]
        rows.append(out)
    return rows


def t3_sampling(d, h):
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    units = d['idx'][h]
    if len(units) < 5 or len(ctrl) < 60: return None
    fold = rng.permutation(len(ctrl)) % 2
    fit = ctrl[(fold == 0) & d['correct'][ctrl]]; te = ctrl[fold == 1]
    cd, bnd, sd = cd_fit(d, units, fit)
    # 100-ms bins over the last 1.0 s of the delay
    edges = np.arange(-1.0, 0.0001, 0.1)
    proj, expv = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        rr = rate(d, units, te, a, b)                       # Hz
        proj.append(rr @ cd)
        expv.append((rr / 0.1) @ (cd ** 2))                  # Poisson variance of the projection: sum cd^2 * rate / dt
    P = np.stack(proj, 1); E = np.stack(expv, 1)
    s = np.where(d['instR'][te], 1.0, -1.0)
    conf = s * (P.mean(1) - bnd) / sd
    # within-trial fluctuation after removing the trial-type mean trajectory
    res = P.copy()
    for side in (True, False):
        m = d['instR'][te] == side; res[m] -= P[m].mean(0)
    within = res.var(1, ddof=1)
    excess = within / E.mean(1)                              # 1 = Poisson
    cor = d['correct'][te]
    lo = conf < np.median(conf)
    return dict(session=d['sess'], mouse=d['mouse'], hemi=h, n=len(te),
                excess_median=np.median(excess),
                excess_lowconf=np.median(excess[lo]), excess_highconf=np.median(excess[~lo]),
                excess_error=np.median(excess[~cor]) if (~cor).sum() >= 5 else np.nan,
                excess_correct=np.median(excess[cor]),
                rho_conf_excess=pd.Series(conf).corr(pd.Series(np.log(excess)), method='spearman'), res=res)


def t4_classical(d):
    ctrl = np.where(d['ok'] & (d['pt'] == 0) & d['correct'])[0]
    ia, ib = d['idx']['code1'], d['idx']['code2']
    if min(len(ia), len(ib)) < 8: return []
    rows = []
    for side in (True, False):
        tr = ctrl[d['instR'][ctrl] == side]
        if len(tr) < 60: continue
        u = np.r_[ia, ib]
        X = rate(d, u, tr, -0.4, 0) * 0.4                   # counts
        keep = X.mean(0) > 0.5
        X = X[:, keep]; isA = np.r_[np.ones(len(ia)), np.zeros(len(ib))][keep].astype(bool)
        if isA.sum() < 5 or (~isA).sum() < 5: continue
        X = np.sqrt(X + 3 / 8)                               # variance-stabilising (Anscombe)
        p = rng.permutation(len(X)); h = len(p) // 2
        Xtr, Xte1, Xte2 = X[p[:h]], X[p[h:h + (len(p) - h) // 2]], X[p[h + (len(p) - h) // 2:]]
        Xte = np.r_[Xte1, Xte2]
        best = None
        for k in (1, 2, 3, 5, 8):
            if k >= X.shape[1] - 1: break
            fa = FactorAnalysis(k, random_state=0).fit(Xtr)
            ll = fa.score(Xte1)
            if best is None or ll > best[0]: best = (ll, k, fa)
        _, k, fa = best
        Cm = fa.get_covariance(); Ct = np.cov(Xte2.T); Ctr = np.cov(Xtr.T); Ch1 = np.cov(Xte1.T)
        off = ~np.eye(len(Cm), dtype=bool)
        cross = np.outer(isA, ~isA) | np.outer(~isA, isA)
        def r(a, b, m): return np.corrcoef(a[m], b[m])[0, 1]
        rows.append(dict(session=d['sess'], mouse=d['mouse'], side='R' if side else 'L', k=k, n_units=X.shape[1],
                         r_model_within=r(Cm, Ct, off & ~cross), r_ceiling_within=r(Ch1, Ct, off & ~cross),
                         r_model_cross=r(Cm, Ct, cross), r_ceiling_cross=r(Ch1, Ct, cross),
                         r_trainemp_cross=r(Ctr, Ct, cross)))
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    t1, t2, t3, t4, xcorr = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        r = t1_committee(d)
        if r: t1 += r
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            if min(len(d['idx'][sil]), len(d['idx'][rec])) >= 5:
                t2 += t2_gain(d, sil, rec, code)
        s = {h: t3_sampling(d, h) for h in ('code1', 'code2')}
        for h, v in s.items():
            if v is not None:
                res = v.pop('res'); t3.append(v)
        if all(v is not None for v in s.values()):
            pass
        t4 += t4_classical(d)
    for name, rows in (('t1_committee', t1), ('t2_gain', t2), ('t3_sampling', t3), ('t4_classical', t4)):
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'{name}.csv'), index=False)
        print(name, len(rows))
