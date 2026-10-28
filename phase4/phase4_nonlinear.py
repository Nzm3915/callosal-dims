"""Phase 4b: nonlinear model of hemisphere B with a link from A, and the gating question revisited.

B is a gated recurrent network (GRU, 32 units) that starts from B's observed state at delay onset and runs
forward through the delay (17 steps of 100 ms) driven by task timing (time x trial type) and by a link from A:

    u(t) = g(t) * V y_A(t-1),        V: 20 -> d   (rank-d link)
    h(t) = GRU(h(t-1), [u(t), drive(t)]),   y_B(t) = C h(t)

Link variants:  none | fixed d=3 | fixed d=20 | learned gate d=3 (g = sigmoid(w.[h, y_A] + b)) |
                precision gate d=3 (g = sigmoid(kappa (conf_A - conf_B) + beta), kappa and beta learned)

Trained free-running on half of the trials (control + silencing), tested on the other half:
  R2_Delta     silencing response of B (noise-corrected, as in Phase 3), laser epoch and after
  G            state-dependence index from the Phase 4a regression, computed on held-out observed trials
               and on the model's simulations of the same trials, in the same PC / 100-ms representation:
               G = c1/b1 - c0/b0  (> 0: a hemisphere that is already sure loses a smaller fraction of its
               state when the other side is silenced)

usage: python phase4_nonlinear.py CACHE_DIR LABELS_CSV OUTDIR [START STOP]
"""
import sys, os, glob, numpy as np, pandas as pd, torch

torch.set_num_threads(1)
CACHE, LABELS, OUT = sys.argv[1:4]
START, STOP = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 999)
NPC, STEPS, LR = 20, 600, 1e-2
VARIANTS = [('fixed', 20, False), ('none', 0, True), ('fixed', 3, True), ('fixed', 20, True),
            ('learned', 3, True), ('precision', 3, True), ('learned', 20, True)]


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


def basis(r, tr, D):
    mu = r[tr][:, :, D].mean((0, 2)); sd = r[tr][:, :, D].std((0, 2)) + 1.0
    z = (r - mu[None, :, None]) / sd[None, :, None]
    X = z[tr][:, :, D].transpose(0, 2, 1).reshape(-1, z.shape[1])
    k = min(NPC, z.shape[1] - 1)
    Q = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T
    y = np.einsum('tub,uk->tbk', z, Q)
    if k < NPC:
        y = np.concatenate([y, np.zeros(y.shape[:2] + (NPC - k,))], -1)
    return y.astype(np.float32)


def cd_stats(y, trc, instR, D, PRE):
    R, L = trc[instR[trc]], trc[~instR[trc]]
    cd = (y[R][:, D].mean(0) - y[L][:, D].mean(0)).mean(0); cd /= np.linalg.norm(cd) + 1e-12
    out = {}
    for name, bins in (('pre', PRE), ('delay', D)):
        pR, pL = y[R][:, bins] @ cd, y[L][:, bins] @ cd                 # trials x bins
        out[name] = ((pR.mean(0) + pL.mean(0)) / 2, np.sqrt((pR.var(0) + pL.var(0)) / 2) + 1e-6)
    return cd.astype(np.float32), out


class Model(torch.nn.Module):
    """y(t) = y(t-1) M + o(j, type) + nonlin * phi(y(t-1)) + g(t) * U V y_A(t-1)

    phi: small MLP (20 -> 16 -> 20, tanh) giving B its own nonlinear dynamics (attractors, saturation).
    Trained teacher-forced (one step ahead, like Phase 3) with early stopping; tested free-running."""

    def __init__(self, kind, d, nD, nonlin, cdB, bndB, sdB, cdA, bndA, sdA):
        super().__init__()
        self.kind, self.d, self.nD, self.nonlin = kind, d, nD, nonlin
        self.M = torch.nn.Linear(NPC, NPC, bias=False); torch.nn.init.zeros_(self.M.weight)
        self.o = torch.nn.Parameter(torch.zeros(2 * nD, NPC))
        if nonlin:
            self.phi = torch.nn.Sequential(torch.nn.Linear(NPC, 16), torch.nn.Tanh(), torch.nn.Linear(16, NPC))
            torch.nn.init.zeros_(self.phi[2].weight); torch.nn.init.zeros_(self.phi[2].bias)
        if d > 0:
            self.V = torch.nn.Linear(NPC, d, bias=False); self.U = torch.nn.Linear(d, NPC, bias=False)
            torch.nn.init.zeros_(self.U.weight)
        if kind == 'learned':
            self.gw = torch.nn.Linear(2 * NPC, 1); torch.nn.init.zeros_(self.gw.weight); torch.nn.init.constant_(self.gw.bias, 2.0)
        if kind == 'precision':
            self.kappa = torch.nn.Parameter(torch.tensor(0.0)); self.beta = torch.nn.Parameter(torch.tensor(2.0))
        for n, v in (('cdB', cdB), ('bndB', bndB), ('sdB', sdB), ('cdA', cdA), ('bndA', bndA), ('sdA', sdA)):
            self.register_buffer(n, torch.tensor(v, dtype=torch.float32))

    def step(self, y, a, j, isR):
        n = y.shape[0]
        idx = j + self.nD * (~isR).long()
        out = y + self.M(y) + self.o[idx]
        if self.nonlin:
            out = out + self.phi(y)
        g = None
        if self.d > 0:
            u = self.U(self.V(a))
            if self.kind == 'learned':
                g = torch.sigmoid(self.gw(torch.cat([y, a], 1)))
            elif self.kind == 'precision':
                s = torch.where(isR, 1.0, -1.0); jj = max(j - 1, 0)
                cB = (s * (y @ self.cdB - self.bndB[jj]) / self.sdB[jj]).abs()
                cA = (s * (a @ self.cdA - self.bndA[jj]) / self.sdA[jj]).abs()
                g = torch.sigmoid(self.kappa * (cA - cB) + self.beta)[:, None]
            out = out + (u if g is None else g * u)
        return out, g

    def teacher(self, yprev, yA, isR):
        """yprev: n x nD x NPC observed B at t-1; returns one-step predictions n x nD x NPC"""
        return torch.stack([self.step(yprev[:, j], yA[:, j], j, isR)[0] for j in range(self.nD)], 1)

    def forward(self, y0, yA, isR):
        y = y0; outs, gates = [], []
        for j in range(self.nD):
            y, g = self.step(y, yA[:, j], j, isR)
            outs.append(y)
            if g is not None: gates.append(g)
        return torch.stack(outs, 1), (torch.stack(gates, 1) if gates else None)


def r2_delta(yB, sim, te, instR, sl):
    obs, pred, h1, h2 = [], [], [], []
    for side in (True, False):
        s = te['stim'][instR[te['stim']] == side]; c = te['ctrl'][instR[te['ctrl']] == side]
        si = np.isin(te['all'], s); ci = np.isin(te['all'], c)
        obs.append(yB[s][:, sl].mean(0) - yB[c][:, sl].mean(0))
        pred.append(sim[si][:, sl].mean(0) - sim[ci][:, sl].mean(0))
        hs, hc = len(s) // 2, len(c) // 2
        h1.append(yB[s[:hs]][:, sl].mean(0) - yB[c[:hc]][:, sl].mean(0))
        h2.append(yB[s[hs:2 * hs]][:, sl].mean(0) - yB[c[hc:2 * hc]][:, sl].mean(0))
    O, P, H1, H2 = (np.concatenate(v) for v in (obs, pred, h1, h2))
    den = np.sum(H1 * H2)
    return (2 * np.sum(O * P) - np.sum(P * P)) / den if den > 0 else np.nan


def g_index(yB_delay, yB_pre, yA_pre, stim_ind, isR, cdB, stB, cdA, stA):
    """state-dependence index G from the Phase 4a regression in PC space"""
    s = np.where(isR, 1.0, -1.0)
    cB = s * ((yB_pre @ cdB) - stB['pre'][0].mean()) / stB['pre'][1].mean()
    cA = s * ((yA_pre @ cdA) - stA['pre'][0].mean()) / stA['pre'][1].mean()
    y = s * ((yB_delay[:, :8] @ cdB).mean(1) - stB['delay'][0][:8].mean()) / stB['delay'][1][:8].mean()
    m = stim_ind == 0
    cB = (cB - cB[m].mean()) / (cB[m].std() + 1e-9); cA = (cA - cA[m].mean()) / (cA[m].std() + 1e-9)
    X = np.c_[np.ones_like(cB), cB, cA, stim_ind, stim_ind * cB, stim_ind * cA]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(b0=b[0], b1=b[1], c0=b[3], c1=b[4], c2=b[5], G=b[4] / b[1] - b[3] / b[0])


def run_pair(d, sil, rec, seed=0):
    ia, ib = d['idx'][sil], d['idx'][rec]
    code = 1 if sil == 'code1' else 2
    t0 = d['t0']
    D = np.where((t0 >= -1.7 - 1e-6) & (t0 < -1e-6))[0]
    PRE = D[0] - 4 + np.arange(4)
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    rs = np.random.default_rng(seed)
    tr, te = {}, {}
    for name, pool in (('ctrl', ctrl), ('stim', stim)):
        a_, b_ = [], []
        for side in (True, False):
            p = rs.permutation(pool[d['instR'][pool] == side]); h = len(p) // 2
            a_.append(p[:h]); b_.append(p[h:])
        tr[name], te[name] = np.concatenate(a_), np.concatenate(b_)
    tr['all'] = np.concatenate([tr['ctrl'], tr['stim']]); te['all'] = np.concatenate([te['ctrl'], te['stim']])
    yA = basis(d['r'][:, ia, :], tr['ctrl'], D); yB = basis(d['r'][:, ib, :], tr['ctrl'], D)
    trc = tr['ctrl'][d['correct'][tr['ctrl']]]
    cdB, stB = cd_stats(yB, trc, d['instR'], D, PRE); cdA, stA = cd_stats(yA, trc, d['instR'], D, PRE)

    def tensors(trials):
        return (torch.tensor(yB[trials, D[0] - 1]), torch.tensor(yA[trials][:, D - 1]),
                torch.tensor(d['instR'][trials]), torch.tensor(yB[trials][:, D]))

    Xtr, Xte = tensors(tr['all']), tensors(te['all'])
    stim_te = np.isin(te['all'], te['stim']).astype(float)
    obsG = g_index(yB[te['all']][:, D], yB[te['all']][:, PRE].mean(1), yA[te['all']][:, PRE].mean(1),
                   stim_te, d['instR'][te['all']], cdB, stB, cdA, stA)
    rows = [dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, model='observed',
                 **{f'G_{k}': v for k, v in obsG.items()})]
    # 20% of the training trials for early stopping
    nv = len(tr['all']) // 5
    perm = rs.permutation(len(tr['all'])); vi, ti = perm[:nv], perm[nv:]
    yprev_tr = torch.tensor(yB[tr['all']][:, D - 1])
    for kind, dd, nonlin in VARIANTS:
        torch.manual_seed(seed)
        m = Model(kind, dd, len(D), nonlin, cdB, stB['delay'][0], stB['delay'][1], cdA, stA['delay'][0], stA['delay'][1])
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-3)
        best, best_state, wait = np.inf, None, 0
        for it in range(STEPS):
            opt.zero_grad()
            pred = m.teacher(yprev_tr[ti], Xtr[1][ti], Xtr[2][ti])
            loss = ((pred - Xtr[3][ti]) ** 2).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                vl = ((m.teacher(yprev_tr[vi], Xtr[1][vi], Xtr[2][vi]) - Xtr[3][vi]) ** 2).mean().item()
            if vl < best - 1e-5:
                best, wait = vl, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                wait += 1
                if wait > 40: break
        m.load_state_dict(best_state)
        with torch.no_grad():
            sim, gates = m(*Xte[:3])
            yprev_te = torch.tensor(yB[te['all']][:, D - 1])
            one = m.teacher(yprev_te, Xte[1], Xte[2])
            cm = torch.tensor(stim_te == 0)
            tgt = Xte[3][cm]
            r2_ctrl = 1 - ((one[cm] - tgt) ** 2).sum().item() / ((tgt - tgt.mean((0, 1))) ** 2).sum().item()
            te_loss = ((sim - Xte[3]) ** 2).mean().item()
            kind_tag = f"{'nonlin' if nonlin else 'linear'}_{kind}_{dd}"
        sim = sim.numpy()
        full_sim = yB.copy(); full_sim[np.ix_(te['all'], D)] = sim
        simG = g_index(sim, yB[te['all']][:, PRE].mean(1), yA[te['all']][:, PRE].mean(1),
                       stim_te, d['instR'][te['all']], cdB, stB, cdA, stA)
        row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, model=kind_tag,
                   train_loss=best, iters=it, test_loss=te_loss, r2_ctrl=r2_ctrl,
                   r2d_laser=r2_delta(yB[:, D], sim, te, d['instR'], slice(0, 8)),
                   r2d_after=r2_delta(yB[:, D], sim, te, d['instR'], slice(8, 17)),
                   **{f'G_{k}': v for k, v in simG.items()})
        if gates is not None:
            g = gates.numpy()[..., 0]
            row.update(gate_mean_ctrl=g[stim_te == 0].mean(), gate_mean_stim=g[stim_te == 1].mean(),
                       gate_sd_ctrl=g[stim_te == 0].std())
        if kind == 'precision':
            row.update(kappa=m.kappa.item(), beta=m.beta.item())
        rows.append(row)
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    files = sorted(glob.glob(os.path.join(CACHE, '*.npz')))[START:STOP]
    out = os.path.join(OUT, f'phase4_nonlinear_{START:03d}_{STOP:03d}.csv')
    rows = []
    for p in files:
        d = load(p, lab)
        for sil, rec in (('code1', 'code2'), ('code2', 'code1')):
            code = 1 if sil == 'code1' else 2
            if min(len(d['idx'][sil]), len(d['idx'][rec])) < 5: continue
            if (d['ok'] & (d['pt'] == code)).sum() < 20: continue
            rows += run_pair(d, sil, rec)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(d['sess'], flush=True)
    print('done', len(rows))
