"""Phase 3: a virtual callosal link. What is the smallest link from hemisphere A that lets a model of
hemisphere B reproduce what B does when A is silenced?

Model of B (one 100-ms step, z-scored rates in a basis of B's top principal components):

    y_B(t) = M y_B(t-1) + o(t, c) + g(t) * W y_A(t-L)          rank(W) <= d

  M        B's own dynamics
  o(t, c)  time- and trial-type-specific drive (stimulus, task timing)
  W        the link, rank d, delay L (100-ms steps)
  g(t)     gate:  fixed   g = 1
                  2state  Chen et al. 2021: separate link when B is "highly selective" (CD projection
                          beyond the median, on the correct side) vs not
                  prec    precision gate g = sigmoid(kappa * (conf_A(t-L) - conf_B(t-1))),
                          conf = |CD projection - boundary| / sd   (H4: listen more when you are unsure
                          and the sender is sure)

Training on half of the trials (all control + half of the silencing trials, or control trials only);
testing on the other half. Test of the link: start each held-out trial at delay onset from its observed
state, run the model forward using only the observed activity of A (silenced or not), and compare the
predicted silencing response Delta_B = mean(silence A) - mean(control) with the observed one:

    R2_Delta = (2 <Delta_obs, Delta_pred> - ||Delta_pred||^2) / <Delta_obs1, Delta_obs2>

(the denominator uses two disjoint halves of the test trials, so measurement noise does not count against
the model; 1 = the reproducible response is fully predicted, 0 = no better than predicting no response).

Bit rate of the link: the d link signals s = (W y_A) projected on the link's d output directions are
quantised to b bits each (uniform over +-3 sd) and sent every 100 ms: R = d * b * 10 bit/s.

usage: python phase3_link.py CACHE_DIR LABELS_CSV PHASE2_CSV OUTDIR [NMAX]
"""
import sys, os, glob, numpy as np, pandas as pd

CACHE, LABELS, P2, OUT = sys.argv[1:5]
NMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 999
NPC = 20
RIDGE = 1.0
DS = [0, 1, 2, 3, 5, 10, 'full']
LAGS = [1, 2, 3]
BITS = [1, 2, 3, 4, 6, None]
KAPPAS = [0.5, 1.0, 2.0]
rng = np.random.default_rng(3)


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
    return dict(sess=sess, r=r, t0=t0, idx=idx, ok=ok, instR=(z['HitR'] + z['ErrR']) > 0,
                correct=(z['HitR'] + z['HitL']) > 0, pt=z['PhotostimulationType'])


def ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


class Pair:
    """Everything for one direction (silence A, model B) and one train/test split."""

    def __init__(self, d, ia, ib, code, train, test):
        self.t0 = d['t0']
        self.D = np.where((self.t0 >= -1.7 - 1e-6) & (self.t0 < -1e-6))[0]     # 17 delay bins
        self.instR = d['instR']; self.correct = d['correct']
        tr_ctrl = train['ctrl']
        rA, rB = d['r'][:, ia, :], d['r'][:, ib, :]
        self.yA = self._basis(rA, tr_ctrl)
        self.yB = self._basis(rB, tr_ctrl)
        self.train, self.test = train, test
        # choice decoder and confidence statistics in the basis, from correct control training trials
        self.stats = {h: self._cd_stats(y, tr_ctrl) for h, y in (('A', self.yA), ('B', self.yB))}

    def _basis(self, r, tr):
        D = (self.t0 >= -1.7 - 1e-6) & (self.t0 < -1e-6)
        mu = r[tr][:, :, D].mean((0, 2)); sd = r[tr][:, :, D].std((0, 2)) + 1.0
        z = (r - mu[None, :, None]) / sd[None, :, None]
        X = z[tr][:, :, D].transpose(0, 2, 1).reshape(-1, z.shape[1])
        k = min(NPC, z.shape[1] - 1)
        Q = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T          # units x k
        self.Q = Q if not hasattr(self, 'QA') else Q
        return np.einsum('tub,uk->tbk', z, Q)                                   # trials x bins x k

    def _cd_stats(self, y, tr):
        trc = tr[self.correct[tr]]
        R, L = trc[self.instR[trc]], trc[~self.instR[trc]]
        cd = (y[R][:, self.D].mean(0) - y[L][:, self.D].mean(0)).mean(0)
        cd /= np.linalg.norm(cd) + 1e-12
        p = y[trc] @ cd                                                         # trials x bins
        pR, pL = (y[R] @ cd), (y[L] @ cd)
        bnd = (pR.mean(0) + pL.mean(0)) / 2
        sd = np.sqrt((pR.var(0) + pL.var(0)) / 2) + 1e-6
        late = self.D[-4:]
        thrR = np.median(pR[:, late]); thrL = np.median(pL[:, late])
        return dict(cd=cd, bnd=bnd, sd=sd, thrR=thrR, thrL=thrL)

    def conf(self, h, y_t, b):
        s = self.stats[h]
        return np.abs(y_t @ s['cd'] - s['bnd'][b]) / s['sd'][b]

    def selective(self, y_t, b, isR):
        s = self.stats['B']; p = y_t @ s['cd']
        return np.where(isR, p > s['thrR'], p < s['thrL'])

    def link_input(self, gate, kappa, yB_prev, yA_lag, b, bL, isR):
        if gate == 'fixed':
            return yA_lag
        if gate == '2state':
            s = self.selective(yB_prev, b - 1, isR)[:, None]
            return np.concatenate([s * yA_lag, (~s) * yA_lag], 1)
        g = 1 / (1 + np.exp(-kappa * (self.conf('A', yA_lag, bL) - self.conf('B', yB_prev, b - 1))))
        return g[:, None] * yA_lag

    def design(self, trials, L, gate, kappa):
        """teacher-forced regression design over delay bins"""
        Xs, Zs, Ys = [], [], []
        nD = len(self.D)
        for j, b in enumerate(self.D):
            yp = self.yB[trials, b - 1]; ya = self.yA[trials, b - L]
            isR = self.instR[trials]
            off = np.zeros((len(trials), 2 * nD)); off[np.arange(len(trials)), j + nD * (~isR)] = 1
            Xs.append(np.concatenate([yp, off], 1))
            Zs.append(self.link_input(gate, kappa, yp, ya, b, b - L, isR))
            Ys.append(self.yB[trials, b])
        return np.concatenate(Xs), np.concatenate(Zs), np.concatenate(Ys)

    def fit(self, trials, L, gate, kappa, d, vsrc='rrr'):
        X, Z, Y = self.design(trials, L, gate, kappa)
        nx = X.shape[1]
        if d == 0:
            B0 = ridge(X, Y, RIDGE); return dict(B0=B0, W=np.zeros((Z.shape[1], Y.shape[1])), V=None, nx=nx)
        # residualise Y and Z on X (own history + drive), reduced-rank ridge for the link, refit B0
        H = ridge(X, np.concatenate([Y, Z], 1), RIDGE)
        Yr = Y - X @ H[:, :Y.shape[1]]; Zr = Z - X @ H[:, Y.shape[1]:]
        W = ridge(Zr, Yr, RIDGE * 10)
        V = None
        if d != 'full' and d < Y.shape[1]:
            if vsrc == 'rrr':      # directions that best predict B one step ahead (dominated by control trials)
                V = np.linalg.svd(Zr @ W, full_matrices=False)[2][:d].T
            else:                  # receiving subspace measured on the training trials (silence - control)
                V = np.linalg.svd(self.train_delta(), full_matrices=False)[2][:d].T
            W = W @ V @ V.T
        B0 = ridge(X, Y - Z @ W, RIDGE)
        sig = (Zr @ W @ V).std(0) + 1e-9 if V is not None else None
        return dict(B0=B0, W=W, V=V, nx=nx, sig=sig)

    def train_delta(self):
        tr = self.train
        rows = []
        for side in (True, False):
            s = tr['stim'][self.instR[tr['stim']] == side]; c = tr['ctrl'][self.instR[tr['ctrl']] == side]
            rows.append(self.yB[s][:, self.D].mean(0) - self.yB[c][:, self.D].mean(0))
        return np.concatenate(rows)                                             # (2 x bins) x k

    def one_step_r2(self, m, trials, L, gate, kappa):
        X, Z, Y = self.design(trials, L, gate, kappa)
        P = X @ m['B0'] + Z @ m['W']
        return 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(0)) ** 2)

    def simulate(self, m, trials, L, gate, kappa, bits=None):
        nD = len(self.D)
        y = self.yB[trials, self.D[0] - 1].copy()
        isR = self.instR[trials]
        out = np.zeros((len(trials), nD, y.shape[1]))
        Mt, Off = m['B0'][:y.shape[1]], m['B0'][y.shape[1]:]
        for j, b in enumerate(self.D):
            ya = self.yA[trials, b - L]
            z = self.link_input(gate, kappa, y, ya, b, b - L, isR)
            link = z @ m['W']
            if bits is not None and m['V'] is not None:
                s = link @ m['V']
                lim = 3 * m['sig']; step = 2 * lim / (2 ** bits)
                s = np.clip(np.floor((s + lim) / step), 0, 2 ** bits - 1) * step - lim + step / 2
                link = s @ m['V'].T
            y = y @ Mt + Off[j + nD * (~isR)] + link
            out[:, j] = y
        return out

    def delta_obs(self, stim, ctrl, epoch):
        e = self.D[epoch]
        def dl(s, c):
            return np.concatenate([self.yB[s[k]][:, e].mean(0) - self.yB[c[k]][:, e].mean(0) for k in (True, False)])
        return dl

    def r2_delta(self, m, L, gate, kappa, bits=None):
        te = self.test
        res = {}
        sims = {k: self.simulate(m, te[k], L, gate, kappa, bits) for k in ('stim', 'ctrl')}
        for ename, sl in (('during', slice(0, 8)), ('after', slice(8, 17))):
            bins = self.D[sl]
            obs, pred, h1, h2 = [], [], [], []
            for side in (True, False):
                s = te['stim'][self.instR[te['stim']] == side]; c = te['ctrl'][self.instR[te['ctrl']] == side]
                ss = sims['stim'][self.instR[te['stim']] == side][:, sl]
                cs = sims['ctrl'][self.instR[te['ctrl']] == side][:, sl]
                obs.append(self.yB[s][:, bins].mean(0) - self.yB[c][:, bins].mean(0))
                pred.append(ss.mean(0) - cs.mean(0))
                hs, hc = len(s) // 2, len(c) // 2
                h1.append(self.yB[s[:hs]][:, bins].mean(0) - self.yB[c[:hc]][:, bins].mean(0))
                h2.append(self.yB[s[hs:2 * hs]][:, bins].mean(0) - self.yB[c[hc:2 * hc]][:, bins].mean(0))
            O, P, H1, H2 = (np.concatenate(v) for v in (obs, pred, h1, h2))
            den = np.sum(H1 * H2)
            res[ename] = ((2 * np.sum(O * P) - np.sum(P * P)) / den if den > 0 else np.nan, den)
        return res


def run_pair(d, sil, rec, rel):
    ia, ib = d['idx'][sil], d['idx'][rec]
    code = 1 if sil == 'code1' else 2
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    rows = []
    for fold in range(2):
        tr, te = {}, {}
        for name, pool in (('ctrl', ctrl), ('stim', stim)):
            a_, b_ = [], []
            for side in (True, False):
                p = pool[d['instR'][pool] == side]
                p = np.random.default_rng(100 + fold // 2).permutation(p)
                h = len(p) // 2
                (a_, b_)[0].append(p[:h]); (a_, b_)[1].append(p[h:])
            A_, B_ = np.concatenate(a_), np.concatenate(b_)
            tr[name], te[name] = (A_, B_) if fold == 0 else (B_, A_)
        pr = Pair(d, ia, ib, code, tr, te)
        both = np.concatenate([tr['ctrl'], tr['stim']])
        base = dict(session=d['sess'], silenced=sil, receiving=rec, fold=fold, nA=len(ia), nB=len(ib), rel=rel)

        def evaluate(train_set, L, gate, kappa, dd, bits=None, tag='', vsrc='rrr'):
            m = pr.fit(train_set, L, gate, kappa, dd, vsrc)
            r2c = pr.one_step_r2(m, te['ctrl'], L, gate, kappa)
            rd = pr.r2_delta(m, L, gate, kappa, bits)
            rows.append(dict(base, train=tag or ('ctrl+stim' if len(train_set) > len(tr['ctrl']) else 'ctrl'),
                             L=L, gate=gate, kappa=kappa, d=str(dd), bits=str(bits), vsrc=vsrc, r2_ctrl=r2c,
                             r2d_during=rd['during'][0], r2d_after=rd['after'][0],
                             den_during=rd['during'][1], den_after=rd['after'][1]))

        for L in LAGS:
            for dd in DS:
                evaluate(both, L, 'fixed', None, dd)
        for dd in DS:
            evaluate(tr['ctrl'], 1, 'fixed', None, dd)
        # gates at L = 1; kappa for the precision gate chosen by training one-step fit (full rank)
        best_k = max(KAPPAS, key=lambda k: pr.one_step_r2(pr.fit(both, 1, 'prec', k, 'full'), both, 1, 'prec', k))
        for dd in DS:
            evaluate(both, 1, '2state', None, dd)
            evaluate(both, 1, 'prec', best_k, dd)
        for dd in (1, 2, 3, 5, 10):
            evaluate(both, 1, 'fixed', None, dd, vsrc='delta')
        for dd in (1, 2, 3, 5):
            for bits in BITS[:-1]:
                evaluate(both, 1, 'fixed', None, dd, bits=bits)
                evaluate(both, 1, 'fixed', None, dd, bits=bits, vsrc='delta')
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    p2 = pd.read_csv(P2)
    p2 = p2[p2.epoch == 'during'].set_index(['session', 'silenced']).rel
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz')))[:NMAX]:
        d = load(p, lab)
        for sil, rec in (('code1', 'code2'), ('code2', 'code1')):
            if min(len(d['idx'][sil]), len(d['idx'][rec])) < 5: continue
            rel = p2.get((d['sess'], sil), np.nan)
            rows += run_pair(d, sil, rec, rel)
        print(d['sess'], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'phase3_models.csv'), index=False)
    print('done', len(rows))
