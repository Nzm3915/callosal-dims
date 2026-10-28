"""T5: a mechanistic two-hemisphere E/I circuit model of the callosal link (hypothesis: the callosal input lands on
both pyramidal cells and fast-spiking interneurons of the other side; losing it removes excitation from some cells
and inhibition from others).

Each hemisphere: 400 excitatory (E) and 80 inhibitory (I) rate units, threshold-linear, sparse random connectivity,
heterogeneous baseline drive. Callosal projections: E_A -> E_B (strength cE) and E_A -> I_B (strength cI), and the
same from B to A. Silencing a hemisphere = extra drive L to its I units for 0.8 s with a 0.2-s linear ramp-down
(VGAT-ChR2 activates interneurons), as in the experiment.

Four free parameters (cE, cI, L, gI = strength of local inhibition) are fitted by random search to the four
measured mean time courses (silenced-side pyramidal; other-side FS; other-side pyramidal that decrease / increase;
results/phase5/phase5_latency.npz). Not fitted, used as tests: the fraction of other-side pyramidal units that go
down (> 20%) or up (> 20%), and the overshoot of other-side FS after the laser.

usage: python h5_circuit.py RESULTS_DIR
"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = sys.argv[1]
lat = np.load(os.path.join(R, 'phase5', 'phase5_latency.npz'))
t_ms = lat['t_ms']
keys = ['silenced_side_pyramidal', 'other_side_fast-spiking', 'other_side_pyramidal_(decrease)', 'other_side_pyramidal_(increase)']
target = {k: lat[k].mean(0) for k in keys}
W = (t_ms > -300) & (t_ms < 1100)
NE, NI, DT = 400, 80, 0.001
rs = np.random.default_rng(0)


def build(cE, cI, gI, seed=0):
    r = np.random.default_rng(seed)
    N = 2 * (NE + NI)
    J = np.zeros((N, N))
    def block(post, pre, p, scale):
        m = (r.random((len(post), len(pre))) < p) * scale * r.lognormal(0, 0.5, (len(post), len(pre)))
        J[np.ix_(post, pre)] = m
    H = []
    for h in range(2):
        o = h * (NE + NI); E = np.arange(o, o + NE); I = np.arange(o + NE, o + NE + NI)
        H.append((E, I))
        block(E, E, 0.1, 0.5 / (0.1 * NE))
        block(I, E, 0.2, 1.0 / (0.2 * NE))
        block(E, I, 0.3, -gI / (0.3 * NI))
        block(I, I, 0.3, -0.5 / (0.3 * NI))
    for a, b in ((0, 1), (1, 0)):
        Ea, _ = H[a]; Eb, Ib = H[b]
        block(Eb, Ea, 0.05, cE / (0.05 * NE))
        block(Ib, Ea, 0.05, cI / (0.05 * NE))
    hE = r.lognormal(0, 0.5, NE); hI = r.lognormal(0, 0.3, NI)
    h0 = np.concatenate([hE, hI * 1.5, hE[::-1], hI * 1.5])
    return J, h0, H


def simulate(J, h0, H, L, silence=0, T=(-0.6, 1.2)):
    n = J.shape[0]; tau = np.where(np.isin(np.arange(n), np.r_[H[0][1], H[1][1]]), 0.01, 0.02)
    ts = np.arange(T[0], T[1], DT); r = np.zeros(n); out = np.zeros((len(ts), n))
    Isil = H[silence][1]
    for i, t in enumerate(ts):
        drive = h0.copy()
        if 0 <= t < 0.6: drive[Isil] += L
        elif 0.6 <= t < 0.8: drive[Isil] += L * (0.8 - t) / 0.2
        r = r + DT / tau * (-r + np.clip(J @ r + drive, 0, 200))
        out[i] = r
    return ts, out


def bin50(ts, out):
    edges = np.arange(-0.3, 1.1001, 0.05)
    return np.array([out[(ts >= a) & (ts < b)].mean(0) for a, b in zip(edges[:-1], edges[1:])]), edges[:-1] * 1000 + 25


def curves(params, seed=0):
    cE, cI, L, gI = params
    J, h0, H = build(cE, cI, gI, seed)
    ts, ctrl = simulate(J, h0, H, 0.0)
    _, sil = simulate(J, h0, H, L, silence=0)
    bc, tb = bin50(ts, ctrl); bs, _ = bin50(ts, sil)
    pre = (tb < 0)
    base = bc[pre].mean(0) + 0.5
    ch = (bs - bc) / base                                       # bins x units
    (EA, IA), (EB, IB) = H
    win = (tb > 100) & (tb < 700)
    ratio = (bs[win].mean(0) + 1e-6) / (bc[win].mean(0) + 1e-6)
    active = bc[pre].mean(0) > 0.1
    down = EB[(ratio[EB] < 0.8) & active[EB]]; up = EB[(ratio[EB] > 1.2) & active[EB]]
    out = {'silenced_side_pyramidal': ch[:, EA[active[EA]]].mean(1), 'other_side_fast-spiking': ch[:, IB[active[IB]]].mean(1),
           'other_side_pyramidal_(decrease)': ch[:, down].mean(1) if len(down) else np.zeros(len(tb)),
           'other_side_pyramidal_(increase)': ch[:, up].mean(1) if len(up) else np.zeros(len(tb))}
    nB = active[EB].sum()
    extra = dict(frac_down=len(down) / max(nB, 1), frac_up=len(up) / max(nB, 1),
                 fs_rebound=ch[(tb > 850) & (tb < 1050), :][:, IB[active[IB]]].mean())
    return tb, out, extra


def loss(params):
    tb, out, _ = curves(params)
    tgt_t = t_ms[W]
    err = 0
    for k in keys:
        y = np.interp(tb, tgt_t, target[k][W])
        err += np.mean((out[k] - y) ** 2)
    return err


if __name__ == '__main__':
    best = None
    lo, hi = np.array([0.0, 0.0, 2.0, 0.3]), np.array([1.5, 3.0, 30.0, 3.0])
    hist = []
    for i in range(160):
        p = lo + (hi - lo) * rs.random(4) if (best is None or i < 80) else np.clip(best[1] + rs.normal(0, 0.08, 4) * (hi - lo), lo, hi)
        try:
            e = loss(p)
        except Exception:
            continue
        hist.append((e, *p))
        if best is None or e < best[0]: best = (e, p)
    e, p = best
    tb, out, extra = curves(p)
    # control: same model with callosal input only onto E (cI = 0) and only onto I (cE = 0), refit L/gI kept
    _, outE, exE = curves([p[0], 0.0, p[2], p[3]])
    _, outI, exI = curves([0.0, p[1], p[2], p[3]])
    res = dict(loss=e, cE=p[0], cI=p[1], L=p[2], gI=p[3], **extra,
               lossE_only=float(sum(np.mean((outE[k] - np.interp(tb, t_ms[W], target[k][W])) ** 2) for k in keys)),
               lossI_only=float(sum(np.mean((outI[k] - np.interp(tb, t_ms[W], target[k][W])) ** 2) for k in keys)),
               extraE_only=exE, extraI_only=exI)
    os.makedirs(os.path.join(R, 'hypotheses'), exist_ok=True)
    json.dump(res, open(os.path.join(R, 'hypotheses', 't5_circuit.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    cols = dict(zip(keys, ['#8a8984', '#eb6834', '#2a78d6', '#1baf7a']))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, (o, title) in zip(ax, ((out, f'fitted: callosal → E and I'), (outE, 'callosal → E only'), (outI, 'callosal → I only'))):
        for k in keys:
            a.plot(t_ms[W], target[k][W], color=cols[k], lw=1.2, ls='--')
            a.plot(tb, o[k], color=cols[k], lw=2, label=k.replace('_', ' '))
        a.axvspan(0, 800, color='#2a78d6', alpha=0.06, lw=0); a.axhline(0, color='#e4e3df', lw=1)
        a.set_title(title, fontsize=10, loc='left', fontweight='bold'); a.set_xlabel('ms from laser onset')
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    ax[0].set_ylabel('change / baseline'); ax[0].legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.suptitle('Circuit model (solid) vs data (dashed)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(R, 'hypotheses', 'fig_t5_circuit.png'), dpi=150)
