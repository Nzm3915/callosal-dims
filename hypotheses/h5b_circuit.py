"""T5b (B5'): the E/I circuit model of T5 with two additions the data asked for.

T5 fitted the mean time courses but missed two things it was not fitted to: far too many other-side pyramidal
units went down (90% vs 42% in the data) and almost none went up (1% vs 20%), and there was no post-laser
overshoot of the other side's FS cells (data +40%). Two candidate fixes, each switchable:

  heterogeneity  each pyramidal cell of B gets its own callosal excitation, cE_i = cE * lognormal(0, sigma_c),
                 while the disynaptic inhibition through B's FS cells is shared. Cells with weak direct input then
                 lose more inhibition than excitation when A is silenced and go UP (disinhibition).
  adaptation     each excitatory unit carries a slow adaptation current, tau_a da/dt = -a + beta * r, subtracted
                 from its input (spike-frequency adaptation). A silenced pyramidal cell recovers from adaptation
                 during the laser and overshoots afterwards; the overshoot is relayed across the callosum to the
                 other side's FS cells.

Four variants (base, +heterogeneity, +adaptation, +both) are fitted by the same random search to the same four mean
curves (results/phase5/phase5_latency.npz). The held-out tests are unchanged: fraction of other-side pyramidal
units down (> 20%) / up (> 20%), and the FS overshoot 850-1050 ms after laser onset.

usage: python h5b_circuit.py RESULTS_DIR [NEVAL]
"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
NEVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 200
lat = np.load(os.path.join(R, 'phase5', 'phase5_latency.npz'))
t_ms = lat['t_ms']
keys = ['silenced_side_pyramidal', 'other_side_fast-spiking', 'other_side_pyramidal_(decrease)', 'other_side_pyramidal_(increase)']
target = {k: lat[k].mean(0) for k in keys}
W = (t_ms > -300) & (t_ms < 1100)
DATA_TESTS = dict(frac_down=0.42, frac_up=0.20, fs_rebound=0.40)
NE, NI, DT = 400, 80, 0.001


def build(p, het, seed=0):
    cE, cI, L, gI, beta, sig = p
    r = np.random.default_rng(seed)
    N = 2 * (NE + NI); J = np.zeros((N, N)); H = []

    def block(post, pre, prob, scale, rowscale=None):
        m = (r.random((len(post), len(pre))) < prob) * scale * r.lognormal(0, 0.5, (len(post), len(pre)))
        if rowscale is not None:
            m = m * rowscale[:, None]
        J[np.ix_(post, pre)] = m
    for h in range(2):
        o = h * (NE + NI); E = np.arange(o, o + NE); I = np.arange(o + NE, o + NE + NI); H.append((E, I))
        block(E, E, 0.1, 0.5 / (0.1 * NE)); block(I, E, 0.2, 1.0 / (0.2 * NE))
        block(E, I, 0.3, -gI / (0.3 * NI)); block(I, I, 0.3, -0.5 / (0.3 * NI))
    for a, b in ((0, 1), (1, 0)):
        Ea, _ = H[a]; Eb, Ib = H[b]
        rs = r.lognormal(-sig ** 2 / 2, sig, NE) if het else None          # mean 1, heterogeneous per cell
        block(Eb, Ea, 0.05, cE / (0.05 * NE), rs)
        block(Ib, Ea, 0.05, cI / (0.05 * NE))
    hE = r.lognormal(0, 0.5, NE); hI = r.lognormal(0, 0.3, NI)
    h0 = np.concatenate([hE, hI * 1.5, hE[::-1], hI * 1.5])
    isE = np.zeros(N, bool); isE[np.r_[H[0][0], H[1][0]]] = True
    return J, h0, H, isE


def simulate(J, h0, H, isE, L, beta, silence=0, T=(-0.6, 1.2), tau_a=0.3):
    n = J.shape[0]; tau = np.where(isE, 0.02, 0.01)
    ts = np.arange(T[0], T[1], DT); r = np.zeros(n); a = np.zeros(n); out = np.zeros((len(ts), n))
    Isil = H[silence][1]
    for i, t in enumerate(ts):
        drive = h0.copy()
        if 0 <= t < 0.6: drive[Isil] += L
        elif 0.6 <= t < 0.8: drive[Isil] += L * (0.8 - t) / 0.2
        r = r + DT / tau * (-r + np.clip(J @ r + drive - a, 0, 200))
        a = a + DT / tau_a * (-a + beta * r * isE)
        out[i] = r
    return ts, out


def bin50(ts, out):
    edges = np.arange(-0.3, 1.1001, 0.05)
    return np.array([out[(ts >= a) & (ts < b)].mean(0) for a, b in zip(edges[:-1], edges[1:])]), edges[:-1] * 1000 + 25


def curves(p, het, seed=0):
    J, h0, H, isE = build(p, het, seed)
    beta = p[4]
    ts, ctrl = simulate(J, h0, H, isE, 0.0, beta); _, sil = simulate(J, h0, H, isE, p[2], beta, silence=0)
    bc, tb = bin50(ts, ctrl); bs, _ = bin50(ts, sil)
    pre = tb < 0; base = bc[pre].mean(0) + 0.5; ch = (bs - bc) / base
    (EA, IA), (EB, IB) = H
    win = (tb > 100) & (tb < 700)
    ratio = (bs[win].mean(0) + 1e-6) / (bc[win].mean(0) + 1e-6)
    active = bc[pre].mean(0) > 0.1
    down = EB[(ratio[EB] < 0.8) & active[EB]]; up = EB[(ratio[EB] > 1.2) & active[EB]]
    out = {keys[0]: ch[:, EA[active[EA]]].mean(1), keys[1]: ch[:, IB[active[IB]]].mean(1),
           keys[2]: ch[:, down].mean(1) if len(down) else np.zeros(len(tb)),
           keys[3]: ch[:, up].mean(1) if len(up) else np.zeros(len(tb))}
    nB = max(active[EB].sum(), 1)
    extra = dict(frac_down=len(down) / nB, frac_up=len(up) / nB,
                 fs_rebound=float(ch[(tb > 850) & (tb < 1050)][:, IB[active[IB]]].mean()))
    return tb, out, extra


def loss(p, het):
    tb, out, _ = curves(p, het)
    return sum(np.mean((out[k] - np.interp(tb, t_ms[W], target[k][W])) ** 2) for k in keys)


def fit(variant, rs):
    het, adapt = variant in ('het', 'both'), variant in ('adapt', 'both')
    lo = np.array([0.0, 0.0, 2.0, 0.3, 0.0, 0.0]); hi = np.array([1.5, 3.0, 30.0, 3.0, 2.0 if adapt else 0.0, 1.5 if het else 0.0])
    best = None
    for i in range(NEVAL):
        p = lo + (hi - lo) * rs.random(6) if (best is None or i < NEVAL // 2) else \
            np.clip(best[1] + rs.normal(0, 0.08, 6) * (hi - lo), lo, hi)
        try:
            e = loss(p, het)
        except Exception:
            continue
        if np.isfinite(e) and (best is None or e < best[0]):
            best = (e, p)
    e, p = best
    tb, out, extra = curves(p, het)
    return dict(variant=variant, loss=e, cE=p[0], cI=p[1], L=p[2], gI=p[3], beta=p[4], sigma_c=p[5], **extra), tb, out


if __name__ == '__main__':
    os.makedirs(os.path.join(R, 'hypotheses'), exist_ok=True)
    res, curvs = [], {}
    for v in ('base', 'het', 'adapt', 'both'):
        row, tb, out = fit(v, np.random.default_rng(5))
        res.append(row); curvs[v] = out
        print(json.dumps(row, default=float), flush=True)
    json.dump(dict(fits=res, data_tests=DATA_TESTS), open(os.path.join(R, 'hypotheses', 't5b_circuit.json'), 'w'),
              indent=1, default=float)
    cols = dict(zip(keys, ['#8a8984', '#eb6834', '#2a78d6', '#1baf7a']))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    names = dict(base='T5 (no additions)', het='+ heterogeneous callosal input', adapt='+ adaptation', both='+ both')
    for a, row in zip(ax, res):
        out = curvs[row['variant']]
        for k in keys:
            a.plot(t_ms[W], target[k][W], color=cols[k], lw=1.2, ls='--')
            a.plot(tb, out[k], color=cols[k], lw=2, label=k.replace('_', ' '))
        a.axvspan(0, 800, color='#2a78d6', alpha=0.06, lw=0); a.axhline(0, color='#e4e3df', lw=1)
        a.set_title(f"{names[row['variant']]}\ndown {row['frac_down']:.0%} / up {row['frac_up']:.0%} / FS rebound {row['fs_rebound']:+.2f}",
                    fontsize=9, loc='left', fontweight='bold')
        a.set_xlabel('ms from laser onset'); a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    ax[0].set_ylabel('change / baseline'); ax[0].legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.suptitle('Circuit model (solid) vs data (dashed). Data: down 42% / up 20% / FS rebound +0.40', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'hypotheses', 'fig_t5b_circuit.png'), dpi=150)


def fit_both_warm(het_row, adapt_row, rs, n=80):
    """'both' started from the best heterogeneous fit plus a grid of adaptation strengths, then local search"""
    lo = np.array([0.0, 0.0, 2.0, 0.3, 0.0, 0.0]); hi = np.array([1.5, 3.0, 30.0, 3.0, 2.0, 2.5])
    base = np.array([het_row['cE'], het_row['cI'], het_row['L'], het_row['gI'], 0.0, het_row['sigma_c']])
    best = None
    for b in (0.05, 0.1, 0.16, 0.25, 0.4):
        p = base.copy(); p[4] = b
        e = loss(p, True)
        if best is None or e < best[0]: best = (e, p)
    for i in range(n):
        p = np.clip(best[1] + rs.normal(0, 0.05, 6) * (hi - lo), lo, hi)
        e = loss(p, True)
        if np.isfinite(e) and e < best[0]: best = (e, p)
    e, p = best
    tb, out, extra = curves(p, True)
    return dict(variant='both', loss=e, cE=p[0], cI=p[1], L=p[2], gI=p[3], beta=p[4], sigma_c=p[5], **extra), tb, out
