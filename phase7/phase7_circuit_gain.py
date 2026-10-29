"""Item 1 of the review log's next list: check the recurrent-pass account of docs/01 section 5.5 against a circuit.

Section 5.5 explains the state dependence with two factors: a neuron's own gain (g_i^alpha) and the shrinkage of
the state signal it *receives*, estimated algebraically as S_i = the |u|-weighted mean of g_j^alpha over the other
neurons of the pair. S_i is an assumption: it supposes that what a neuron receives shrinks in the same proportion as
its partners' rates shrink. The mouse data cannot test it, because the input to a neuron is never measured.

A model can. Two hemispheres of rate neurons, each cell with its own input-output curve, joined by a rank-3 link;
one hemisphere is silenced by a current for 1 s. The free parameters are fitted to the three timescales measured at
laser offset (rate recovery, state dependence, mean loss: phase7_rampdown*.json) and to nothing else. Then, inside
the fitted model, the true shrinkage of each neuron's received state signal is measured by finite differences and
compared with the algebraic S_i that section 5.5 uses, and the pooled predictions of phase7_gain_pooled.py are
recomputed in a system whose answer is known.

    tau_r dv_i/dt = -v_i + b_i + (W r)_i + rho_s u_i m - w_inh <r>_hemi + s p_i - lambda 1[light, i in A] + noise
    tau_s dm/dt   = -m + u.r                                   (the slow loop)
    r_i = x^n_i/(1 + x^n_i),   x = k_i [v_i]_+

Three choices worth stating. The curve is expansive near threshold - the power law that section 5.5's gain argument
rests on (Priebe & Ferster 2008) - and saturates above it. The slow mode is an explicit slow loop rather than a
near-unity eigenvalue of the fast weights: in ALM the seconds-long preparatory activity is sustained by the loop
through thalamus (Guo et al. 2017, Nature 545:181), and a loop with its own time constant is both the documented
mechanism and the stable one; it is what makes the mean loss outlast the rates, since a displacement along it
relaxes in tau_s/(1 - loop gain) while the rates follow their input with tau_r. And the tonic drive b_i is not a
free parameter: it is solved from the fixed-point equation so that the control condition sits exactly at a
prescribed operating point v*, spread across the sensitive part of the curve. Without that the network drifts to
the saturated end, where nothing responds to anything.

Rather than assuming the exponent alpha of section 5.5, the model's own alpha_i = dlog(dr/dv)/dlog r is measured at
each cell's operating point, so the comparison is between the data analysis's algebra and the circuit's truth.

usage: python phase7_circuit_gain.py OUTDIR [--quick]
"""
import sys, os, json, time, numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/phase7'
QUICK = '--quick' in sys.argv
N = 70                      # neurons per hemisphere
DT, TAU_R = 0.005, 0.03     # s
T0, T1 = -2.4, -0.4         # trial window, cue at 0
BURN = 0.6                  # s of settling before T0
LIGHT = (-1.9, -0.9)
WINDOWS = [(-1.3, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.4)]
PRE = (-2.2, -1.9)          # pre-light window the confidence is read from
WKEY = [f'{a}..{b}' for a, b in WINDOWS]
# measured targets (results/phase7/phase7_rampdown.json, phase7_rampdown_mean.json)
TGT_G = [0.874, 0.982, 1.009, 1.012, 1.020]
TGT_C1 = [-0.357, -0.208, -0.043, 0.009, 0.027]
TGT_L0 = [0.613, 0.541, 0.438, 0.371, 0.312]
PNAMES = ['lam', 'w_rand', 'c_s', 'rho_s', 'tau_s', 'p_amp', 'noise', 'p_ratio', 'w_inh']
LO = np.array([0.05, 0.02, 0.02, 0.00, 0.08, 0.01, 0.002, 0.2, 0.0])
HI = np.array([2.50, 0.50, 0.60, 0.95, 1.20, 0.50, 0.080, 6.0, 0.6])


def make_net(seed=0):
    """fixed structure; only the nine fitted scalars change between evaluations"""
    g = np.random.default_rng(seed)
    n = g.uniform(1.3, 2.5, 2 * N)                         # per-cell curvature of the input-output curve
    k = np.exp(g.normal(0, 0.3, 2 * N))                    # per-cell scale
    vstar = g.uniform(0.45, 1.45, 2 * N)                   # prescribed operating point, across the sensitive range
    M = g.normal(0, 1, (N, N)) / np.sqrt(N); np.fill_diagonal(M, 0)
    u = None     # the slow (preparatory) mode: set below, along the choice-coding direction
    # callosal link: rank 3, as measured. The loadings are positively biased because callosal axons are excitatory,
    # so removing one hemisphere takes net drive away from the other - which is why B's mean rate falls at all.
    P = g.normal(0.7, 1, (3, N)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    Q = g.normal(0.7, 1, (3, N)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    # ALM cells mostly prefer the contralateral lick, so the preference gets a positive bias: it is what makes the
    # population slope (the sum of the per-neuron slopes) solidly non-zero, as b1 > 0 in the data.
    pA = g.normal(0.6, 1, N); pB = g.normal(0.6, 1, N)
    # The slow loop runs along the choice-coding direction, not along the mean rate. In ALM the activity that
    # outlasts a unilateral silencing is the coding direction, displaced and slow to come back, while the mean rate
    # follows its input within about 0.1 s (Li et al. 2016, Nature 532:459). A slow mode along the mean would tie
    # those two together and the three measured timescales could not come apart, which is the point of the fit.
    u = np.r_[pA, pB]; u = u - u.mean(); u /= np.linalg.norm(u)
    return dict(n=n, k=k, vstar=vstar, M=M, u=u, P=P, Q=Q, pA=pA, pB=pB)


def act(v, net):
    """expansive near threshold, saturating above: r = x^n/(1+x^n) with x = k [v]_+"""
    x = (net['k'] * np.clip(v, 0, None)) ** net['n']
    return x / (1.0 + x)


def dact(v, net):
    vv = np.clip(v, 1e-9, None)
    x = (net['k'] * vv) ** net['n']
    return net['n'] * x / (vv * (1.0 + x) ** 2)


def setup(net, par):
    """the fast weights and the tonic drive that puts the control fixed point exactly at v*.

    Each weight block is scaled by the *share of a cell's drive* it supplies at that fixed point, not by a raw
    weight: w_rand, c_s, rho_s and w_inh below are those shares. Parameterised by raw weights instead, the
    callosal input ends up a per-cent of the drive whatever the number, because its rank-3 structure and the
    1/sqrt(N) scaling fight each other - and then silencing one hemisphere cannot move the other, which is the one
    thing the data says it does."""
    w_rand, c_s, rho_s, w_inh = par[1], par[2], par[3], par[8]
    v = net['vstar']; r = act(v, net); vn = np.linalg.norm(v)
    sc = lambda X, frac: X * (frac * vn / (np.linalg.norm(X @ r) + 1e-12))
    W = np.zeros((2 * N, 2 * N))
    Wo = np.zeros((2 * N, 2 * N)); Wo[:N, :N] = net['M']; Wo[N:, N:] = net['M'].T
    Cc = np.zeros((2 * N, 2 * N))
    Cc[N:, :N] = net['P'].T @ net['Q']       # A -> B
    Cc[:N, N:] = net['Q'].T @ net['P']       # B -> A
    W += sc(Wo, w_rand) + sc(Cc, c_s)
    m = r @ net['u']
    # the slow loop is scaled by its gain, not by its value at the fixed point: along the coding direction that
    # value is nearly zero by construction. rho_s is then the loop gain, so the mode relaxes in tau_s/(1 - rho_s).
    us = net['u'] * (rho_s / (net['u'] @ (dact(v, net) * net['u']) + 1e-12))
    inh = np.r_[np.full(N, r[:N].mean()), np.full(N, r[N:].mean())]
    inh_w = w_inh * vn / (np.linalg.norm(inh) + 1e-12)
    b = v - W @ r - m * us + inh_w * inh
    return dict(W=W, b=b, us=us, inh_w=inh_w)


def simulate(net, par, n_tr, stim, seed, record_input=False, state=None, pre=None, probe=None):
    """window-averaged rates; optionally the input each cell receives from the other cells, and optionally a small
    constant extra current `probe`, used to read the circuit's own gains"""
    lam, w_rand, c_s, rho_s, tau_s, p_amp, noise, p_ratio, w_inh = par
    st = setup(net, par) if pre is None else pre
    W, b, us, inh_w = st['W'], st['b'], st['us'], st['inh_w']
    g = np.random.default_rng(seed)
    s = g.normal(0, 1, n_tr) if state is None else np.asarray(state, float)
    n_tr = len(s)
    p = np.r_[net['pA'] * p_amp * p_ratio, net['pB'] * p_amp]
    u = net['u']
    v = np.tile(net['vstar'], (n_tr, 1)); m = np.full(n_tr, act(net['vstar'], net) @ u)
    light = np.zeros(2 * N); light[:N] = 1.0
    ext = 0.0 if probe is None else probe
    keys = WKEY + ['pre']
    acc = {w: [np.zeros((n_tr, 2 * N)), 0] for w in keys}
    inp = {w: np.zeros((n_tr, 2 * N)) for w in WKEY} if record_input else None
    sd = np.sqrt(DT / TAU_R) * noise
    steps = int(round((T1 - T0 + BURN) / DT))
    for t_i in range(steps):
        t = T0 - BURN + t_i * DT
        r = act(v, net)
        inh = np.c_[np.tile(r[:, :N].mean(1, keepdims=True), N), np.tile(r[:, N:].mean(1, keepdims=True), N)]
        rec = r @ W.T + np.outer(m, us) - inh_w * inh
        I = b + rec + s[:, None] * p + ext
        if stim and LIGHT[0] <= t < LIGHT[1]:
            I = I - lam * np.linalg.norm(net['vstar']) / np.sqrt(2 * N) * light
        v = np.clip(v + (DT / TAU_R) * (-v + I) + sd * g.normal(0, 1, (n_tr, 2 * N)), 0, 20.0)
        m = m + (DT / tau_s) * (-m + r @ u)
        if t < T0:
            continue
        for (a, b2), w in zip(WINDOWS, WKEY):
            if a <= t < b2:
                acc[w][0] += r; acc[w][1] += 1
                if record_input:
                    inp[w] += rec
        if PRE[0] <= t < PRE[1]:
            acc['pre'][0] += r; acc['pre'][1] += 1
    out = {w: acc[w][0] / max(acc[w][1], 1) for w in keys}
    if record_input:
        out['_inp'] = {w: inp[w] / max(acc[w][1], 1) for w in WKEY}
    out['_s'] = s; out['_pre'] = st
    return out


def decoder(X, s):
    """linear read-out of the trial's state from a window of rates, fitted on control trials"""
    Xc = X - X.mean(0); A = Xc.T @ Xc + 1e-3 * np.trace(Xc.T @ Xc) / X.shape[1] * np.eye(X.shape[1])
    return np.linalg.solve(A, Xc.T @ (s - s.mean())), X.mean(0)


def slopes(Y, c):
    cc = c - c.mean()
    return (Y - Y.mean(0)).T @ cc / (cc @ cc + 1e-12)


def analyse(net, par, n_tr, seed, pre=None, record_input=False):
    """the data analysis, run on the model. B is the receiving side; A is silenced."""
    ctrl = simulate(net, par, n_tr, False, seed, pre=pre, record_input=record_input)
    stim = simulate(net, par, n_tr, True, seed + 5000, pre=ctrl['_pre'], record_input=record_input)
    B = slice(N, 2 * N)
    wB, muB = decoder(ctrl['pre'][:, B], ctrl['_s'])          # confidence from B's own pre-light window
    cB_c = (ctrl['pre'][:, B] - muB) @ wB; cB_s = (stim['pre'][:, B] - muB) @ wB
    res = {}
    half = np.arange(n_tr) % 2
    for w in WKEY:
        Yc, Ys = ctrl[w][:, B], stim[w][:, B]
        uc = slopes(Yc, cB_c)
        us = slopes(Ys[half == 0], cB_s[half == 0])           # slopes and rate ratios from separate halves,
        gi = Ys[half == 1].mean(0) / (Yc.mean(0) + 1e-12)     # so they share no noise (as in the data)
        cd = Yc.T @ (ctrl['_s'] - ctrl['_s'].mean()); cd /= np.linalg.norm(cd) + 1e-12
        bnd = (Yc @ cd).mean()
        y = lambda Y, s_: (np.sign(s_) * (Y @ cd - bnd)).mean()
        res[w] = dict(u_ctrl=uc, u_stim=us, g=gi, b1=float(uc.sum()),
                      c1_rel=float((us - uc).sum() / (uc.sum() + 1e-12)),
                      g_wmean=float(np.average(gi, weights=Yc.mean(0))),
                      L0=float(1 - y(Ys, stim['_s']) / (y(Yc, ctrl['_s']) + 1e-12)))
    return res, ctrl, stim, (wB, muB)


def score(par, net, n_tr, seed, pre=None):
    try:
        res, ctrl, *_ = analyse(net, par, n_tr, seed, pre=pre)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return 1e6
    # reject a parameter set whose control condition does not stay near the prescribed operating point
    drift = np.abs(ctrl[WKEY[-1]].mean(0) - act(net['vstar'], net)).mean()
    if not np.isfinite(drift) or drift > 0.12:
        return 1e6
    e = 0.0
    for w, tg, tc, tl in zip(WKEY, TGT_G, TGT_C1, TGT_L0):
        r = res[w]
        if not np.isfinite([r['g_wmean'], r['c1_rel'], r['L0']]).all() or r['b1'] <= 1e-6:
            return 1e6
        e += ((r['g_wmean'] - tg) / 0.04) ** 2 + ((r['c1_rel'] - tc) / 0.08) ** 2 + ((r['L0'] - tl) / 0.08) ** 2
    return e / (3 * len(WKEY))


def fit(net, n_tr, n_rand, n_ref, seed=100, verbose=True, rng_seed=1):
    g = np.random.default_rng(rng_seed); best, bs = None, np.inf
    t0 = time.time()
    for i in range(n_rand):
        p = LO + (HI - LO) * g.random(len(LO))
        s = score(p, net, n_tr, seed)
        if s < bs:
            bs, best = s, p
            if verbose:
                print(f'  [{i:3d}] score {s:9.3f}  {dict(zip(PNAMES, np.round(p, 3)))}  [{round(time.time()-t0)} s]', flush=True)
    if best is None:
        raise RuntimeError('no parameter set in range produced a usable network')
    step = 0.25 * (HI - LO)
    for it in range(n_ref):
        improved = False
        for j in range(len(LO)):
            for sgn in (1, -1):
                p = best.copy(); p[j] = float(np.clip(p[j] + sgn * step[j], LO[j], HI[j]))
                s = score(p, net, n_tr, seed)
                if s < bs - 1e-6:
                    bs, best, improved = s, p, True
        if not improved:
            step *= 0.5
        if verbose:
            print(f'  refine {it:2d}: score {bs:9.3f}  {dict(zip(PNAMES, np.round(best, 4)))}  [{round(time.time()-t0)} s]', flush=True)
    return best, bs


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    net = make_net(0)
    # the fit runs at dt = 10 ms, which gives numbers indistinguishable from dt = 5 ms at half the cost; the final
    # table is recomputed at dt = 5 ms, so the reported fit is not a product of the coarse step
    globals()['DT'] = 0.010
    best, bs = fit(net, 300, 60 if QUICK else 260, 6 if QUICK else 22)
    print(f'fitted: score {bs:.3f}  {dict(zip(PNAMES, np.round(best, 4)))}', flush=True)
    globals()['DT'] = 0.005
    bs = score(best, net, 900, 7)
    print(f'at dt = 5 ms, 900 trials: score {bs:.3f}', flush=True)
    res, *_ = analyse(net, best, 1500, 7)
    tbl = []
    for w, tg, tc, tl in zip(WKEY, TGT_G, TGT_C1, TGT_L0):
        r = res[w]
        tbl.append(dict(win=w, g_model=r['g_wmean'], g_data=tg, c1_model=r['c1_rel'], c1_data=tc,
                        L0_model=r['L0'], L0_data=tl))
        print(f"  {w:12s} g {r['g_wmean']:.3f}/{tg:.3f}   c1 {r['c1_rel']:+.3f}/{tc:+.3f}   L0 {r['L0']:.3f}/{tl:.3f}", flush=True)
    json.dump(dict(par=dict(zip(PNAMES, best.tolist())), score=float(bs), fit_table=tbl),
              open(os.path.join(OUT, 'phase7_circuit_gain_fit.json'), 'w'), indent=1, default=float)
    print('fit done', round(time.time() - t0), 's', flush=True)
