"""Phase 5: mechanisms behind the Phase 2-4 results, with approaches taken from the literature.

5a  Attractor or gate? (Inagaki et al. 2019 drift analysis; Finkelstein et al. 2021 logic)
    B's signed choice confidence x(t) (CD projection, 100-ms bins) is modelled as a 1-D dynamical system
        x(t+1) = x(t) + f(x(t), t) + stim(t) * delta
    f is fitted on control trials only (drift analysis), either linear in x or cubic in x (a cubic with negative
    slope near the endpoints and positive slope near the boundary is a pair of attractors with a saddle).
    delta (one number: the average loss of A's input during the laser) is fitted on the training silencing trials.
    On held-out trials the model is run from the state before the laser, and the Phase 4a regression
        y = b0 + b1 x0 + stim * (c0 + c1 x0)
    is applied to observed and predicted laser-epoch confidence. If B's own attractor explains the state
    dependence found in Phase 4, the cubic model with a constant (ungated) loss reproduces c1 and the relative
    protection G = c1/b1 - c0/b0; the linear model does not.

5b  Is the channel static or rotating? (Javadzadeh & Hofer 2022)
    Receiving direction per 100-ms bin of the delay, from two disjoint halves of the trials; reliability-
    normalised cross-validated cosine similarity between bins i and j:
        s_ij = <d_i^(1), d_j^(2)> / sqrt(<d_i^(1), d_i^(2)> <d_j^(1), d_j^(2)>)
    compared with the same quantity for B's own activity direction (control-trial mean minus pre-sample mean).

5e  Behaviour: does the hemisphere that is left behave as the readout? B's laser-epoch confidence predicts
    whether the mouse licks correctly on silencing trials (AUC), and session-level behavioural deficits.

usage: python phase5_mechanisms.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(5)


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
    nb = c.shape[-1] // 2
    r = (c[..., :2 * nb:2] + c[..., 1:2 * nb:2]) / 0.1
    t0 = z['edges'][:-1][:2 * nb:2]
    return dict(sess=sess, mouse=sess.split('/')[0], r=r, t0=t0, idx=idx, ok=ok,
                instR=(z['HitR'] + z['ErrR']) > 0, correct=(z['HitR'] + z['HitL']) > 0,
                pt=z['PhotostimulationType'])


def bins(t0, a, b):
    return np.where((t0 >= a - 1e-6) & (t0 < b - 1e-6))[0]


def conf_series(d, units, fit, trials):
    """signed CD confidence per 100-ms bin, CD/boundary/sd from correct control trials `fit`"""
    t0 = d['t0']; D = bins(t0, -1.7, 0)
    R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
    rr = d['r'][:, units, :]
    cd = rr[R][:, :, D].mean((0, 2)) - rr[L][:, :, D].mean((0, 2)); cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = np.einsum('tub,u->tb', rr[R], cd), np.einsum('tub,u->tb', rr[L], cd)
    bnd = (pR.mean(0) + pL.mean(0)) / 2
    sd = np.sqrt((pR.var(0) + pL.var(0)) / 2) + 1e-6
    # smooth boundary/sd over 3 bins to reduce noise
    k = np.ones(3) / 3
    bnd = np.convolve(np.pad(bnd, 1, mode='edge'), k, 'valid'); sd = np.convolve(np.pad(sd, 1, mode='edge'), k, 'valid')
    s = np.where(d['instR'][trials], 1.0, -1.0)[:, None]
    p = np.einsum('tub,u->tb', rr[trials], cd)
    return s * (p - bnd[None]) / sd[None]


def reg(y, x0, stim):
    m = stim == 0
    xs = (x0 - x0[m].mean()) / (x0[m].std() + 1e-9)
    X = np.c_[np.ones_like(xs), xs, stim, stim * xs]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(b0=b[0], b1=b[1], c0=b[2], c1=b[3], G=b[3] / b[1] - b[2] / b[0] if b[1] != 0 and b[0] != 0 else np.nan)


def attractor_test(d, sil, rec, code):
    t0 = d['t0']
    pre = bins(t0, -1.8, -1.7)[0]                                  # last 100-ms bin before the laser
    L8 = bins(t0, -1.7, -0.9)                                       # laser epoch (8 bins)
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    if len(stim) < 20: return []
    rows = []
    for fold in range(2):
        tr_c, te_c, tr_s, te_s = [], [], [], []
        for pool, a_, b_ in ((ctrl, tr_c, te_c), (stim, tr_s, te_s)):
            for side in (True, False):
                p = np.random.default_rng(10 + fold // 2).permutation(pool[d['instR'][pool] == side]); h = len(p) // 2
                (a_ if fold == 0 else b_).append(p[:h]); (b_ if fold == 0 else a_).append(p[h:])
        tr_c, te_c, tr_s, te_s = (np.concatenate(v) for v in (tr_c, te_c, tr_s, te_s))
        fit = tr_c[d['correct'][tr_c]]
        units = d['idx'][rec]
        X_trc = conf_series(d, units, fit, tr_c)
        X_trs = conf_series(d, units, fit, tr_s)
        X_te = conf_series(d, units, fit, np.concatenate([te_c, te_s]))
        stim_te = np.r_[np.zeros(len(te_c)), np.ones(len(te_s))]
        # drift analysis on control trials: dx = x(t+1) - x(t) over the window covering laser bins
        steps = np.r_[pre, L8[:-1]]
        x = X_trc[:, steps].ravel(); dx = (X_trc[:, steps + 1] - X_trc[:, steps]).ravel()
        tt = np.tile(np.arange(len(steps)), len(X_trc))
        T = np.eye(len(steps))[tt]                                  # time-specific offsets (ramping input)
        out = {}
        for name, deg in (('linear', 1), ('cubic', 3)):
            P = np.column_stack([T] + [x ** k for k in range(1, deg + 1)])
            w = np.linalg.lstsq(P, dx, rcond=None)[0]

            def run(x0, delta):
                xs = x0.copy(); traj = []
                for j in range(len(steps)):
                    Pj = np.column_stack([np.tile(np.eye(len(steps))[j], (len(xs), 1))] + [xs ** k for k in range(1, deg + 1)])
                    xs = xs + Pj @ w + delta
                    traj.append(xs)
                return np.stack(traj, 1)                            # trials x 8 (laser bins)
            # fit delta on training silencing trials (grid search on the mean laser-epoch confidence)
            grid = np.linspace(-1.5, 0.5, 81)
            obs_mean = X_trs[:, L8].mean()
            err = [abs(run(X_trs[:, pre], g).mean() - obs_mean) for g in grid]
            delta = grid[int(np.argmin(err))]
            pred = np.r_[run(X_te[stim_te == 0, pre], 0.0).mean(1), run(X_te[stim_te == 1, pre], delta).mean(1)]
            out[name] = (reg(pred, X_te[:, pre], stim_te), delta, w)
        obs = reg(X_te[:, L8].mean(1), X_te[:, pre], stim_te)
        row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, fold=fold,
                   **{f'obs_{k}': v for k, v in obs.items()})
        for name, (rr, delta, w) in out.items():
            row.update({f'{name}_{k}': v for k, v in rr.items()}); row[f'{name}_delta'] = delta
        wc = out['cubic'][2]; n = len(steps)
        row.update(cub_a1=wc[n], cub_a2=wc[n + 1], cub_a3=wc[n + 2])
        rows.append(row)
    return rows


def rotation(d, sil, rec, code):
    t0 = d['t0']; D = bins(t0, -1.7, 0); P0 = bins(t0, -3.6, -3.0)
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    if len(stim) < 20: return None
    u = d['idx'][rec]
    rr = d['r'][:, u, :]
    mu = rr[ctrl][:, :, D].mean((0, 2)); sd = rr[ctrl][:, :, D].std((0, 2)) + 1.0
    z = (rr - mu[None, :, None]) / sd[None, :, None]
    S_comm, S_act = np.zeros((len(D), len(D))), np.zeros((len(D), len(D)))
    nrep = 20
    for _ in range(nrep):
        dd, aa = [], []
        for h in range(2):
            pass
        halves_c, halves_s = [{}, {}], [{}, {}]
        for side in (True, False):
            pc = rng.permutation(ctrl[d['instR'][ctrl] == side]); ps = rng.permutation(stim[d['instR'][stim] == side])
            halves_c[0][side], halves_c[1][side] = pc[:len(pc) // 2], pc[len(pc) // 2:]
            halves_s[0][side], halves_s[1][side] = ps[:len(ps) // 2], ps[len(ps) // 2:]
        comm, act = [], []
        for h in range(2):
            # direction per bin: concatenate both trial types
            c_ = np.concatenate([z[halves_s[h][s]][:, :, D].mean(0) - z[halves_c[h][s]][:, :, D].mean(0) for s in (True, False)], 0)
            a_ = np.concatenate([z[halves_c[h][s]][:, :, D].mean(0) - z[halves_c[h][s]][:, :, P0].mean((0, 2))[:, None]
                                 for s in (True, False)], 0)
            comm.append(c_); act.append(a_)
        S_comm += comm[0].T @ comm[1] / nrep; S_act += act[0].T @ act[1] / nrep
    def norm(S):
        S = (S + S.T) / 2; dg = np.clip(np.diag(S), 1e-9, None)
        return S / np.sqrt(np.outer(dg, dg))
    return norm(S_comm), norm(S_act)


def behaviour(d, sil, rec, code):
    t0 = d['t0']; L8 = bins(t0, -1.7, -0.9); E = bins(t0, -0.4, 0)
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    if len(stim) < 20: return None
    fit = ctrl[d['correct'][ctrl]]
    out = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec,
               perf_ctrl=d['correct'][ctrl].mean(), perf_stim=d['correct'][stim].mean())
    for h, nm in ((rec, 'B'), (sil, 'A')):
        x = conf_series(d, d['idx'][h], fit, stim)
        for ep, b in (('laser', L8), ('end', E)):
            v = x[:, b].mean(1); cor = d['correct'][stim]
            if cor.all() or (~cor).all(): out[f'auc_{nm}_{ep}'] = np.nan; continue
            out[f'auc_{nm}_{ep}'] = mannwhitneyu(v[cor], v[~cor]).statistic / (cor.sum() * (~cor).sum())
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    att, beh, Sc, Sa, keys = [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            if min(len(d['idx'][sil]), len(d['idx'][rec])) < 5: continue
            att += attractor_test(d, sil, rec, code)
            r = rotation(d, sil, rec, code)
            if r is not None:
                Sc.append(r[0]); Sa.append(r[1]); keys.append((d['sess'], sil))
            b = behaviour(d, sil, rec, code)
            if b is not None: beh.append(b)
        print(d['sess'], flush=True)
    pd.DataFrame(att).to_csv(os.path.join(OUT, 'phase5_attractor.csv'), index=False)
    pd.DataFrame(beh).to_csv(os.path.join(OUT, 'phase5_behaviour.csv'), index=False)
    np.savez(os.path.join(OUT, 'phase5_rotation.npz'), S_comm=np.array(Sc), S_act=np.array(Sa),
             keys=np.array(keys, dtype=object))
    print('done', len(att), len(beh), len(Sc))
