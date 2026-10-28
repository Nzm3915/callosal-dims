"""DLAG (delayed latents across groups; Gokcen et al., Nat Comput Sci 2022), re-implemented in PyTorch.

Model for one trial, area i in {1, 2}, time bin t:
    y_i,t = C^a_i x^a_i,t + C^w_i x^w_i,t + d_i + eps_i,t ,       eps ~ N(0, R_i), R_i diagonal
Each within-area latent is a Gaussian process with a squared-exponential kernel (timescale tau). Each across-area
latent j is ONE Gaussian process seen by both areas, by area 2 with a delay D_j:
    cov(x^a_1,j(t1), x^a_2,j(t2)) = (1 - s^2) exp(-(t2 - D_j - t1)^2 / (2 tau_j^2))
D_j > 0: area 1 leads (signal flows 1 -> 2); D_j < 0: area 2 leads.

Instead of the original EM, the exact marginal likelihood is maximised directly with L-BFGS (all parameters,
including continuous delays, get exact gradients through the kernel). With latents ordered time-major, the
observation matrix is I_T (x) C, so the Woodbury identity reduces every trial's likelihood to PT x PT algebra:
    Sigma = Chat K Chat' + Rhat,  log|Sigma| = log|R| + log|K| + log|M|,  M = K^-1 + Chat' R^-1 Chat
"""
import numpy as np, torch

torch.set_default_dtype(torch.float64)
S2 = 1e-3                       # GP noise variance (fixed, as in DLAG)


class DLAG:
    def __init__(self, q1, q2, pa, pw1, pw2, T, binw, dmax=0.15, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.q1, self.q2, self.pa, self.pw1, self.pw2, self.T, self.binw, self.dmax = q1, q2, pa, pw1, pw2, T, binw, dmax
        self.P = 2 * pa + pw1 + pw2
        q = q1 + q2
        self.C = torch.randn(q, self.P, generator=g) * 0.1
        mask = torch.zeros(q, self.P)
        # latent order: [across area1 (pa), across area2 (pa), within1 (pw1), within2 (pw2)]
        mask[:q1, :pa] = 1; mask[q1:, pa:2 * pa] = 1
        mask[:q1, 2 * pa:2 * pa + pw1] = 1; mask[q1:, 2 * pa + pw1:] = 1
        self.mask = mask
        self.logR = torch.zeros(q)
        self.logtau = torch.full((pa + pw1 + pw2,), float(np.log(0.1)))
        self.draw = torch.zeros(pa)                                   # delay = dmax * tanh(draw)
        self.dmask = torch.ones(pa)                                   # 0 = delay of that latent fixed at 0
        self.d = torch.zeros(q)
        self.params = [self.C, self.logR, self.logtau, self.draw]
        for p in self.params:
            p.requires_grad_(True)

    # ---------- covariance pieces ----------
    def delays(self):
        return self.dmax * torch.tanh(self.draw) * self.dmask

    def K(self):
        T, pa, pw1, pw2, P = self.T, self.pa, self.pw1, self.pw2, self.P
        t = torch.arange(T, dtype=torch.float64) * self.binw
        dt = t[None, :] - t[:, None]                                  # t2 - t1
        tau = torch.exp(self.logtau)
        K4 = torch.zeros(P, P, T, T)
        eye = torch.eye(T)
        D = self.delays()

        def se(delta, tau_):
            return (1 - S2) * torch.exp(-delta ** 2 / (2 * tau_ ** 2))
        idx_tau = {}
        for j in range(pa):
            a1, a2 = j, pa + j
            k11 = se(dt, tau[j]) + S2 * eye
            k12 = se(dt - D[j], tau[j])                              # area1 at t1, area2 at t2
            K4 = K4.index_put((torch.tensor([a1]), torch.tensor([a1])), k11[None])
            K4 = K4.index_put((torch.tensor([a2]), torch.tensor([a2])), k11[None])
            K4 = K4.index_put((torch.tensor([a1]), torch.tensor([a2])), k12[None])
            K4 = K4.index_put((torch.tensor([a2]), torch.tensor([a1])), k12.T[None])
        for j in range(pw1 + pw2):
            p = 2 * pa + j
            K4 = K4.index_put((torch.tensor([p]), torch.tensor([p])), (se(dt, tau[pa + j]) + S2 * eye)[None])
        return K4.permute(2, 0, 3, 1).reshape(T * P, T * P)            # time-major

    def Cm(self):
        return self.C * self.mask

    # ---------- likelihood ----------
    def nll(self, Y):
        """Y: (N, T, q) mean-removed observations. Returns negative log-likelihood per trial."""
        N, T, q = Y.shape
        C = self.Cm(); Rinv = torch.exp(-self.logR)
        K = self.K()
        Lk = torch.linalg.cholesky(K)
        Kinv = torch.cholesky_inverse(Lk)
        CtRC = C.T @ (C * Rinv[:, None])                               # P x P
        M = Kinv + torch.kron(torch.eye(T), CtRC)
        Lm = torch.linalg.cholesky(M)
        b = (Y * Rinv) @ C                                             # N x T x P
        b = b.reshape(N, -1)
        sol = torch.cholesky_solve(b.T, Lm)                            # PT x N
        quad = (Y ** 2 * Rinv).sum((1, 2)) - (b * sol.T).sum(1)
        logdet = T * self.logR.sum() + 2 * torch.log(torch.diagonal(Lk)).sum() + 2 * torch.log(torch.diagonal(Lm)).sum()
        return 0.5 * (quad + logdet + T * q * np.log(2 * np.pi))

    def copy_from(self, other):
        with torch.no_grad():
            for a, b in zip(self.params, other.params):
                a.copy_(b)
        return self

    def fit(self, Y, iters=150, init=True):
        Y = torch.as_tensor(Y)
        self.d = Y.mean((0, 1))
        Yc = Y - self.d
        if init:
            with torch.no_grad():
                # initialise loadings from the top principal components of each area and of the pooled data
                X = Yc.reshape(-1, Yc.shape[-1])
                v = X.var(0) + 1e-3
                self.logR.copy_(torch.log(v * 0.7))
                U, S, Vt = torch.linalg.svd(X / X.shape[0] ** 0.5, full_matrices=False)
                q1 = self.q1; pa = self.pa
                if pa:
                    top = (Vt[:pa].T * S[:pa])
                    self.C[:, :pa] = top; self.C[:, pa:2 * pa] = top
                for (lo, hi, off, k) in ((0, q1, 2 * pa, self.pw1), (q1, X.shape[1], 2 * pa + self.pw1, self.pw2)):
                    if k:
                        _, s_, vt_ = torch.linalg.svd(X[:, lo:hi] / X.shape[0] ** 0.5, full_matrices=False)
                        self.C[lo:hi, off:off + k] = (vt_[:k].T * s_[:k]) * 0.7
                self.draw.copy_(torch.randn(self.pa) * 0.1)
        opt = torch.optim.LBFGS(self.params, lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = self.nll(Yc).mean()
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            pass
        return self

    def score(self, Y):
        """mean log-likelihood per trial of held-out data"""
        with torch.no_grad():
            return float(-self.nll(torch.as_tensor(Y) - self.d).mean())

    def summary(self):
        with torch.no_grad():
            C = self.Cm().numpy(); R = np.exp(self.logR.numpy()); q1, pa = self.q1, self.pa
            out = dict(delays=self.delays().numpy().copy(), tau=np.exp(self.logtau.numpy()),
                       Ca1=C[:q1, :pa], Ca2=C[q1:, pa:2 * pa], Cw1=C[:q1, 2 * pa:2 * pa + self.pw1],
                       Cw2=C[q1:, 2 * pa + self.pw1:], R=R)
            # shared variance explained by each latent in each area (latents have unit variance)
            for a, ca, cw in ((1, out['Ca1'], out['Cw1']), (2, out['Ca2'], out['Cw2'])):
                va = (ca ** 2).sum(0); vw = (cw ** 2).sum(0)
                out[f'across_strength{a}'] = va.sum() / (va.sum() + vw.sum() + 1e-12)
                out[f'var_across{a}'] = va
        return out


def simulate(q1, q2, pa, pw1, pw2, T, binw, N, delays, taus, snr=0.3, seed=0):
    """draw a synthetic data set from the DLAG generative model (for validation)"""
    r = np.random.default_rng(seed)
    m = DLAG(q1, q2, pa, pw1, pw2, T, binw, seed=seed)
    with torch.no_grad():
        m.draw.copy_(torch.atanh(torch.tensor(delays) / m.dmax))
        m.logtau.copy_(torch.log(torch.tensor(taus)))
        C = torch.tensor(r.normal(size=(q1 + q2, m.P))) * m.mask
        m.C.copy_(C)
        sig = (C ** 2).sum(1)
        R = sig / snr
        m.logR.copy_(torch.log(R))
        K = m.K().numpy()
    L = np.linalg.cholesky(K)
    X = (L @ r.normal(size=(K.shape[0], N))).T.reshape(N, T, m.P)
    Y = X @ C.numpy().T + r.normal(size=(N, T, q1 + q2)) * np.sqrt(R.numpy())
    return Y, m
