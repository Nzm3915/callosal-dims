"""Phase 5b: the choice-axis state dependence (Phase 4: c1 < 0, relative protection G > 0) — attractor, gate, or
simply a fixed link from a correlated partner?

Two-hemisphere model on the choice axis (400-ms windows stepped by 100 ms, signed by trial type):
    xB(t+1) = xB(t) + a(t) + f(xB(t)) + w * uA(t) [* gate]
    uA(t)   = signed CD projection of A (Hz), observed: normal on control trials, ~0 while A is silenced
    f       = 0 (linear decay is in a(t)+k xB)  | cubic in xB (attractor-like)
    gate    = none | precision: sigmoid(kappa (|xA| - |xB|) + beta) fitted by grid
Fitted on half of the control and silencing trials (one-step regression), run forward on the other half from the
state before the laser using A's observed activity, and scored with the Phase 4a regression
    y = b0 + b1 xB0 + stim (c0 + c1 xB0)   (y = mean xB over the laser epoch)
A model without the link term, but with a constant stim offset, is the null 'B alone'.

usage: python phase5_choice_axis.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd

CACHE, LABELS, OUT = sys.argv[1:4]


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
    cs = np.cumsum(np.concatenate([np.zeros_like(c[..., :1]), c], -1), -1)
    r = (cs[..., 8:] - cs[..., :-8]) / 0.4                               # 400-ms windows, 50-ms steps
    t_end = z['edges'][8:]
    keep = np.arange(0, r.shape[-1], 2)                                  # 100-ms steps
    return dict(sess=sess, mouse=sess.split('/')[0], r=r[..., keep], t=t_end[keep], idx=idx, ok=ok,
                instR=(z['HitR'] + z['ErrR']) > 0, correct=(z['HitR'] + z['HitL']) > 0, pt=z['PhotostimulationType'])


def proj(d, units, fit, trials, D):
    rr = d['r'][:, units, :]
    R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
    cd = rr[R][:, :, D].mean((0, 2)) - rr[L][:, :, D].mean((0, 2)); cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = np.einsum('tub,u->tb', rr[R], cd), np.einsum('tub,u->tb', rr[L], cd)
    bnd = (pR.mean(0) + pL.mean(0)) / 2; sd = np.sqrt((pR.var(0) + pL.var(0)) / 2).mean() + 1e-6
    s = np.where(d['instR'][trials], 1.0, -1.0)[:, None]
    p = np.einsum('tub,u->tb', rr[trials], cd)
    return s * (p - bnd) / sd, s * p / sd                                # centred confidence, raw signed drive


def reg(y, x0, stim):
    m = stim == 0
    xs = (x0 - x0[m].mean()) / (x0[m].std() + 1e-9)
    b = np.linalg.lstsq(np.c_[np.ones_like(xs), xs, stim, stim * xs], y, rcond=None)[0]
    return dict(b0=b[0], b1=b[1], c0=b[2], c1=b[3], G=b[3] / b[1] - b[2] / b[0])


def design(xB, uA, xA, stim, steps, model, kappa=0.0, beta=0.0):
    n = len(steps)
    cols, Y = [], []
    for j, t in enumerate(steps):
        x = xB[:, t]; T = np.zeros((len(x), n)); T[:, j] = 1
        parts = [T, x[:, None]]
        if 'cubic' in model: parts += [x[:, None] ** 2, x[:, None] ** 3]
        if 'link' in model:
            g = 1.0
            if 'gate' in model:
                g = 1 / (1 + np.exp(-(kappa * (np.abs(xA[:, t]) - np.abs(x)) + beta)))
            parts.append((g * uA[:, t])[:, None])
        else:
            parts.append(stim[:, None] * 1.0)                            # null: constant offset while A is silenced
        cols.append(np.hstack(parts)); Y.append(xB[:, t + 1] - x)
    return np.vstack(cols), np.concatenate(Y)


def simulate(w, xB0, uA, xA, stim, steps, model, kappa=0.0, beta=0.0):
    n = len(steps); x = xB0.copy(); traj = []
    for j, t in enumerate(steps):
        T = np.zeros((len(x), n)); T[:, j] = 1
        parts = [T, x[:, None]]
        if 'cubic' in model: parts += [x[:, None] ** 2, x[:, None] ** 3]
        if 'link' in model:
            g = 1.0
            if 'gate' in model:
                g = 1 / (1 + np.exp(-(kappa * (np.abs(xA[:, t]) - np.abs(x)) + beta)))
            parts.append((g * uA[:, t])[:, None])
        else:
            parts.append(stim[:, None] * 1.0)
        x = x + np.hstack(parts) @ w; traj.append(x)
    return np.stack(traj, 1)


MODELS = ['alone_linear', 'alone_cubic', 'link_linear', 'link_cubic', 'link_gate_linear', 'link_gate_cubic']


def run(d, sil, rec, code):
    t = d['t']
    D = np.where((t > -1.7 + 0.4 - 1e-6) & (t <= 1e-6))[0]                # windows fully inside the delay (for CD)
    pre = int(np.argmin(np.abs(t - (-1.7))))                            # window ending at laser onset
    steps = np.arange(pre, pre + 8)                                      # predict windows ending -1.6 ... -0.9
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]; stim = np.where(d['ok'] & (d['pt'] == code))[0]
    if len(stim) < 20: return []
    rows = []
    for fold in range(2):
        sets = {}
        for name, pool in (('c', ctrl), ('s', stim)):
            a_, b_ = [], []
            for side in (True, False):
                p = np.random.default_rng(20).permutation(pool[d['instR'][pool] == side]); h = len(p) // 2
                a_.append(p[:h]); b_.append(p[h:])
            A_, B_ = np.concatenate(a_), np.concatenate(b_)
            sets[name] = (A_, B_) if fold == 0 else (B_, A_)
        tr = np.r_[sets['c'][0], sets['s'][0]]; te = np.r_[sets['c'][1], sets['s'][1]]
        st_tr = np.r_[np.zeros(len(sets['c'][0])), np.ones(len(sets['s'][0]))]
        st_te = np.r_[np.zeros(len(sets['c'][1])), np.ones(len(sets['s'][1]))]
        fit = sets['c'][0][d['correct'][sets['c'][0]]]
        xB_tr, _ = proj(d, d['idx'][rec], fit, tr, D); xA_tr, uA_tr = proj(d, d['idx'][sil], fit, tr, D)
        xB_te, _ = proj(d, d['idx'][rec], fit, te, D); xA_te, uA_te = proj(d, d['idx'][sil], fit, te, D)
        obs = reg(xB_te[:, steps + 1].mean(1), xB_te[:, pre], st_te)
        row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, fold=fold,
                   **{f'obs_{k}': v for k, v in obs.items()})
        for m in MODELS:
            best = None
            grid = [(0.0, 0.0)] if 'gate' not in m else [(k, b) for k in (-2, -1, -0.5, 0.5, 1, 2) for b in (-1, 0, 1, 2)]
            for kappa, beta in grid:
                X, Y = design(xB_tr, uA_tr, xA_tr, st_tr, steps, m, kappa, beta)
                w = np.linalg.lstsq(X, Y, rcond=None)[0]
                sse = np.sum((Y - X @ w) ** 2)
                if best is None or sse < best[0]: best = (sse, w, kappa, beta)
            _, w, kappa, beta = best
            sim = simulate(w, xB_te[:, pre], uA_te, xA_te, st_te, steps, m, kappa, beta)
            pr = reg(sim.mean(1), xB_te[:, pre], st_te)
            # held-out accuracy of the silencing effect on the choice axis (trial-type-signed mean trajectory)
            eff_obs = xB_te[st_te == 1][:, steps + 1].mean(0) - xB_te[st_te == 0][:, steps + 1].mean(0)
            eff_pred = sim[st_te == 1].mean(0) - sim[st_te == 0].mean(0)
            row.update({f'{m}_{k}': v for k, v in pr.items()})
            row[f'{m}_efferr'] = np.sqrt(np.mean((eff_obs - eff_pred) ** 2)); row[f'{m}_kappa'] = kappa
            if 'link' in m: row[f'{m}_w'] = w[len(steps) + (3 if 'cubic' in m else 1)]
        rows.append(row)
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            if min(len(d['idx'][sil]), len(d['idx'][rec])) < 5: continue
            rows += run(d, sil, rec, code)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'phase5_choice_axis.csv'), index=False)
    print('done', len(rows))
