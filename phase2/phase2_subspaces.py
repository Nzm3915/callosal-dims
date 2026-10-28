"""Phase 2: does the other hemisphere's response to unilateral silencing ("receiving subspace") live in the
directions along which the two hemispheres co-fluctuate in control trials ("correlated subspace")?

For every session and both directions (silence hemisphere A, look at hemisphere B):

  Delta_B(t, c) = mean rate of B on "silence A" trials - mean rate of B on control trials,
                  per 100-ms bin t of the delay epoch and per trial type c ("lick right"/"lick left")

Rates are z-scored per unit with control statistics. Two epochs: during the laser (first 0.8 s of the delay)
and after it (last 0.9 s); the sample epoch (laser off) is a negative control.

Candidate subspaces of B, all estimated on a separate third of the control trials:
  CD        choice decoder of B (1-D)
  corr-k    correlated subspace: top-k directions of B predicted from A by ridge regression of
            trial-to-trial residuals (reduced-rank regression, Semedo et al. 2019)
  pc-k      top-k principal components of B's own residual fluctuations
  sig-k     top-k principal components of B's condition-averaged activity (task subspace)
  uniform   all units up or down together (1-D)
  random-k  expected k / n_B

Noise-robust measure of how much of the response lies in a subspace S (projector P):
  f(S) = <P Delta1, P Delta2> / <Delta1, Delta2>
with Delta1, Delta2 from disjoint halves of the trials. Noise is independent between halves, so it cancels in
expectation; f is the fraction of the *reliable* response inside S (1 = all of it, k/n = chance).
The receiving subspace's own ceiling uses cross-validated PCA (Stringer et al. 2019): top-k directions of
Delta1, variance measured as <u.Delta1, u.Delta2>.

usage: python phase2_subspaces.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd

CACHE, LABELS, OUT = sys.argv[1:4]
NSPLIT = 30
KMAX = 5
MIN_UNITS = 5
EPOCHS = {'pre': (-3.0, -1.7), 'during': (-1.7, -0.9), 'after': (-0.9, 0.0)}   # s from go cue
rng = np.random.default_rng(1)


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
    instR = (z['HitR'] + z['ErrR']) > 0
    correct = (z['HitR'] + z['HitL']) > 0
    edges = z['edges']
    c = z['counts'].astype(np.float32)
    # 100-ms bins (pairs of 50-ms bins) from -3.6 s
    nb = c.shape[-1] // 2
    r = (c[..., :2 * nb:2] + c[..., 1:2 * nb:2]) / 0.1                  # Hz
    t0 = edges[:-1][:2 * nb:2]                                           # bin starts
    return dict(sess=sess, r=r, t0=t0, idx=idx, ok=ok, instR=instR, correct=correct,
                pt=z['PhotostimulationType'])


def proj(V):
    """orthonormal basis (n x k) -> projector"""
    return V @ V.T


def orth(M, k):
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return u[:, :k]


def analyse_pair(d, silenced, receiving):
    ia, ib = d['idx'][silenced], d['idx'][receiving]
    if min(len(ia), len(ib)) < MIN_UNITS:
        return []
    code = 1 if silenced == 'code1' else 2                               # PhotostimulationType for silencing it
    t0 = d['t0']
    delay = (t0 >= -1.7 - 1e-6) & (t0 < -1e-6)
    ctrl = {s: np.where(d['ok'] & (d['pt'] == 0) & (d['instR'] == s))[0] for s in (True, False)}
    stim = {s: np.where(d['ok'] & (d['pt'] == code) & (d['instR'] == s))[0] for s in (True, False)}
    if min(len(v) for v in stim.values()) < 6 or min(len(v) for v in ctrl.values()) < 30:
        return []
    nB = len(ib)
    acc = {e: {} for e in EPOCHS}
    accR2 = {}
    rel = {e: [] for e in EPOCHS}

    def add(e, key, val):
        acc[e][key] = acc[e].get(key, 0.0) + val

    for _ in range(NSPLIT):
        est, c1, c2, s1, s2 = {}, {}, {}, {}, {}
        u1, u2, u3, v1_, v2_, v3_ = {}, {}, {}, {}, {}, {}             # three-way split for unbiased checks
        for s in (True, False):
            p = rng.permutation(ctrl[s]); n3 = len(p) // 3
            est[s], c1[s], c2[s] = p[:n3], p[n3:2 * n3], p[2 * n3:]
            q = rng.permutation(stim[s]); h = len(q) // 2
            s1[s], s2[s] = q[:h], q[h:2 * h]
            rest = p[n3:]; m = len(rest) // 3
            v1_[s], v2_[s], v3_[s] = rest[:m], rest[m:2 * m], rest[2 * m:3 * m]
            g = len(q) // 3
            u1[s], u2[s], u3[s] = q[:g], q[g:2 * g], q[2 * g:3 * g]
        E = np.concatenate([est[True], est[False]])
        rB, rA = d['r'][:, ib, :], d['r'][:, ia, :]
        # z-scoring from the estimation set (delay bins)
        mu = rB[E][:, :, delay].mean((0, 2)); sd = rB[E][:, :, delay].std((0, 2)) + 1.0
        zB = (rB - mu[None, :, None]) / sd[None, :, None]
        muA = rA[E][:, :, delay].mean((0, 2)); sdA = rA[E][:, :, delay].std((0, 2)) + 1.0
        zA = (rA - muA[None, :, None]) / sdA[None, :, None]
        # --- candidate subspaces from the estimation set (delay epoch) ---
        mR, mL = zB[est[True]][:, :, delay].mean(0), zB[est[False]][:, :, delay].mean(0)   # n x T
        cd = (mR - mL).mean(1); cd = cd / (np.linalg.norm(cd) + 1e-12)
        sig = np.concatenate([mR, mL], 1); sig = sig - sig.mean(1, keepdims=True)
        V_sig = orth(sig, KMAX)
        resB = np.concatenate([zB[est[s]][:, :, delay] - zB[est[s]][:, :, delay].mean(0) for s in (True, False)])
        resA = np.concatenate([zA[est[s]][:, :, delay] - zA[est[s]][:, :, delay].mean(0) for s in (True, False)])
        Y = resB.transpose(0, 2, 1).reshape(-1, nB)                       # samples x nB
        X = resA.transpose(0, 2, 1).reshape(-1, len(ia))
        V_pc = np.linalg.svd(Y, full_matrices=False)[2][:KMAX].T
        lam = 1e-1 * np.trace(X.T @ X) / X.shape[1]
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        V_corr = np.linalg.svd(X @ W, full_matrices=False)[2][:KMAX].T
        uni = np.ones(nB) / np.sqrt(nB)
        # held-out quality of the correlated subspace: R^2 of B residuals predicted from A (rank k), on c1
        rB1 = np.concatenate([zB[c1[s]][:, :, delay] - zB[c1[s]][:, :, delay].mean(0) for s in (True, False)])
        rA1 = np.concatenate([zA[c1[s]][:, :, delay] - zA[c1[s]][:, :, delay].mean(0) for s in (True, False)])
        Y1 = rB1.transpose(0, 2, 1).reshape(-1, nB); X1 = rA1.transpose(0, 2, 1).reshape(-1, len(ia))
        sst = np.sum(Y1 ** 2)
        for k in range(1, KMAX + 1):
            Vk = V_corr[:, :k]
            accR2[k] = accR2.get(k, 0.0) + (1 - np.sum((Y1 - X1 @ W @ Vk @ Vk.T) ** 2) / sst) / NSPLIT
        accR2['full'] = accR2.get('full', 0.0) + (1 - np.sum((Y1 - X1 @ W) ** 2) / sst) / NSPLIT
        # three-way split: receiving subspace from part 3, evaluated on parts 1 and 2 (no shared noise)
        def delta(stim_part, ctrl_part, w):
            return np.concatenate([zB[stim_part[s]][:, :, w].mean(0) - zB[ctrl_part[s]][:, :, w].mean(0)
                                   for s in (True, False)], 1)
        wd = (t0 >= -1.7 - 1e-6) & (t0 < -0.9 - 1e-6); wa = (t0 >= -0.9 - 1e-6) & (t0 < -1e-6)
        U3d = orth(delta(u3, v3_, wd), KMAX)
        for e, w in (('during', wd), ('after', wa)):
            E1, E2 = delta(u1, v1_, w), delta(u2, v2_, w)
            add(e, 'den3', np.sum(E1 * E2))
            U3 = orth(delta(u3, v3_, w), KMAX)
            for k in range(1, KMAX + 1):
                P = proj(U3[:, :k]); add(e, f'g_rec{k}', np.sum((P @ E1) * (P @ E2)))
                P = proj(U3d[:, :k]); add(e, f'g_recDuring{k}', np.sum((P @ E1) * (P @ E2)))
                P = proj(V_corr[:, :k]); add(e, f'g_corr{k}', np.sum((P @ E1) * (P @ E2)))
        for e, (a, b) in EPOCHS.items():
            w = (t0 >= a - 1e-6) & (t0 < b - 1e-6)
            D1 = np.concatenate([zB[s1[s]][:, :, w].mean(0) - zB[c1[s]][:, :, w].mean(0) for s in (True, False)], 1)
            D2 = np.concatenate([zB[s2[s]][:, :, w].mean(0) - zB[c2[s]][:, :, w].mean(0) for s in (True, False)], 1)
            den = np.sum(D1 * D2)
            rel[e].append(den / (np.linalg.norm(D1) * np.linalg.norm(D2) + 1e-12))
            add(e, 'den', den)
            add(e, 'f_cd', np.sum((cd @ D1) * (cd @ D2)))
            add(e, 'f_uniform', np.sum((uni @ D1) * (uni @ D2)))
            # receiving subspace, cross-validated PCA (both directions, averaged)
            U1 = orth(D1, KMAX); U2 = orth(D2, KMAX)
            for k in range(1, KMAX + 1):
                for V, nm in ((V_corr, 'corr'), (V_pc, 'pc'), (V_sig, 'sig')):
                    P = proj(V[:, :k])
                    add(e, f'f_{nm}{k}', np.sum((P @ D1) * (P @ D2)))
                v1 = 0.5 * (np.sum((U1[:, :k].T @ D1) * (U1[:, :k].T @ D2)) +
                            np.sum((U2[:, :k].T @ D2) * (U2[:, :k].T @ D1)))
                add(e, f'f_rec{k}', v1)
            # how much of the response is choice-specific (differs between trial types) vs common
            T = D1.shape[1] // 2
            add(e, 'den_typediff', np.sum((D1[:, :T] - D1[:, T:]) * (D2[:, :T] - D2[:, T:])) / 2)
            add(e, 'f_cd_typediff', np.sum((cd @ (D1[:, :T] - D1[:, T:])) * (cd @ (D2[:, :T] - D2[:, T:]))) / 2)
            # effect size: reliable response power per unit per column, relative to z-units
            add(e, 'power', den / D1.size)
    # shape of the main receiving direction (all trials, during the laser): is it carried by a few units?
    rB = d['r'][:, ib, :]
    allc = np.concatenate([ctrl[True], ctrl[False]])
    mu = rB[allc][:, :, delay].mean((0, 2)); sd = rB[allc][:, :, delay].std((0, 2)) + 1.0
    zB = (rB - mu[None, :, None]) / sd[None, :, None]
    wd = (t0 >= -1.7 - 1e-6) & (t0 < -0.9 - 1e-6)
    Dfull = np.concatenate([zB[stim[s]][:, :, wd].mean(0) - zB[ctrl[s]][:, :, wd].mean(0) for s in (True, False)], 1)
    u = orth(Dfull, 1)[:, 0]
    pr = 1.0 / np.sum(u ** 4) / nB                                     # participation ratio / n (1 = all units equal)
    unit_eff = Dfull.mean(1)                                           # mean z-change per unit
    shape = dict(pr_u1=pr, top10_share=np.sort(u ** 2)[::-1][:max(1, nB // 10)].sum(),
                 frac_units_down=np.mean(unit_eff < 0), mean_unit_change=unit_eff.mean())
    rows = []
    for e in EPOCHS:
        a = acc[e]
        row = dict(session=d['sess'], silenced=silenced, receiving=receiving, epoch=e, nA=len(ia), nB=nB,
                   n_stimR=len(stim[True]), n_stimL=len(stim[False]), rel=np.mean(rel[e]),
                   power=a['power'] / NSPLIT, frac_typediff=a['den_typediff'] / a['den'],
                   f_cd_typediff=a['f_cd_typediff'] / a['den_typediff'] if a['den_typediff'] > 0 else np.nan)
        for key, v in a.items():
            if key.startswith('f_') and key != 'f_cd_typediff':
                row[key] = v / a['den']
            if key.startswith('g_'):
                row[key] = v / a['den3']
        for k, v in accR2.items():
            row[f'R2_corr{k}'] = v
        for k in range(1, KMAX + 1):
            row[f'f_rand{k}'] = k / nB
        row.update(shape)
        rows.append(row)
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    rows = []
    NMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 999
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz')))[:NMAX]:
        d = load(p, lab)
        rows += analyse_pair(d, 'code1', 'code2')
        rows += analyse_pair(d, 'code2', 'code1')
        print(d['sess'], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase2_pairs.csv'), index=False)
    for e in EPOCHS:
        s = df[df.epoch == e]
        print(e, len(s), 'pairs; reliability median', s.rel.median().round(3))
