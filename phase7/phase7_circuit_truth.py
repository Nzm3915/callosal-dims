"""The test that phase7_circuit_gain.py's fit was for: inside the fitted circuit, is the algebra of docs/01
section 5.5 right?

Section 5.5 predicts a neuron's state-dependence slope under silencing as

    u_stim,i / u_ctrl,i  ~  g_i^alpha * S_i^k,      S_i = |u|-weighted mean of g_j^alpha over the other neurons

Both factors are guesses about things the mouse data cannot see. In the model they are exact and separable, because
each cell's membrane variable and total input are recorded:

    u_i = dr_i/dc_B = (dr_i/dv_i) * (dI_i/dc_B)

so  u_stim/u_ctrl = G_i * T_i  exactly, with
    G_i  the ratio of the cell's own gain, which g_i^alpha is meant to approximate
    T_i  the ratio of the state amplitude of its total input, which S_i^k is meant to approximate.

G_i is known in closed form for this curve: r = x/(1+x) with x = (k v)^n gives dr/dv = (n/v) r (1-r), and in the
expansive regime (r << 1) that is proportional to r^((n-1)/n) - the exponent alpha = (n-1)/n that section 5.5
derives. So the model also says what alpha each cell *should* have, and whether the single pooled alpha fitted to
the data is the right summary.

Reported:
  1. G_i against g_i^alpha, per cell and pooled - does the gain factor hold?
  2. T_i against S_i^k for k = 0, 1, 2 - does the receiving factor hold, and at which k?
  3. the pooled table of phase7_gain_pooled.py recomputed in the model, next to the mouse numbers
  4. the cells whose own rate barely moved (0.9 < g < 1.1): their slope ratio and what S_i predicts
  5. how much of B's state signal arrives through A at all, by cutting the link

usage: python phase7_circuit_truth.py OUTDIR [n_trials]
"""
import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/phase7'
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
_src = open(os.path.join(here, 'phase7_circuit_gain.py')).read()
exec(_src[:_src.index("\nif __name__ == '__main__':")])
W0 = WKEY[0]        # the window during the light, where section 5.5's numbers are quoted


def dact_exact(v, net):
    """dr/dv for r = x/(1+x), x = (k v)^n: (n/v) r (1-r)"""
    r = act(v, net); vv = np.clip(v, 1e-9, None)
    return net['n'] * r * (1 - r) / vv


def algebraic_S(u_ctrl, g, alpha):
    """S_i exactly as phase7_gain_pooled.py builds it"""
    Wt = np.abs(u_ctrl); own = g ** alpha; tot = np.sum(Wt * own)
    return (tot - Wt * own) / (Wt.sum() - Wt + 1e-12)


def run(net, par, n_tr, seed):
    ctrl = simulate(net, par, n_tr, False, seed, record_input=True)
    stim = simulate(net, par, n_tr, True, seed + 5000, pre=ctrl['_pre'], record_input=True)
    B = slice(N, 2 * N)
    wB, muB = decoder(ctrl['pre'][:, B], ctrl['_s'])
    cB_c = (ctrl['pre'][:, B] - muB) @ wB; cB_s = (stim['pre'][:, B] - muB) @ wB
    half = np.arange(n_tr) % 2
    Yc, Ys = ctrl[W0][:, B], stim[W0][:, B]
    d = dict(u_ctrl=slopes(Yc, cB_c), u_stim=slopes(Ys[half == 0], cB_s[half == 0]),
             g=Ys[half == 1].mean(0) / (Yc.mean(0) + 1e-12),
             # the exact split: own gain, and the state amplitude of the total input
             v_ctrl=ctrl['_inp'][W0]['v'][:, B].mean(0), v_stim=stim['_inp'][W0]['v'][:, B].mean(0),
             dI_ctrl=slopes(ctrl['_inp'][W0]['I'][:, B], cB_c), dI_stim=slopes(stim['_inp'][W0]['I'][:, B], cB_s),
             dR_ctrl=slopes(ctrl['_inp'][W0]['rec'][:, B], cB_c), dR_stim=slopes(stim['_inp'][W0]['rec'][:, B], cB_s))
    d['G'] = dact_exact(d['v_stim'], dict(net, n=net['n'][B], k=net['k'][B])) / \
             (dact_exact(d['v_ctrl'], dict(net, n=net['n'][B], k=net['k'][B])) + 1e-12)
    d['T'] = d['dI_stim'] / (d['dI_ctrl'] + np.sign(d['dI_ctrl'] + 1e-30) * 1e-9)
    d['T_rec'] = d['dR_stim'] / (d['dR_ctrl'] + np.sign(d['dR_ctrl'] + 1e-30) * 1e-9)
    # the cell's own exact local exponent of gain against rate, at its control operating point; (n-1)/n is its
    # limit at low rate, which is what section 5.5 assumes
    d['alpha_true'] = alpha_local(net['r0'][B], net['n'][B])
    d['alpha_lowrate'] = (net['n'][B] - 1) / net['n'][B]
    return d


def cut_link(net, par, n_tr, seed):
    """how much of B's state signal comes through A: the population slope with the callosal link removed"""
    p2 = np.array(par, float); p2[2] = 1e-9
    out = {}
    for name, pp in (('intact', np.array(par, float)), ('cut', p2)):
        c = simulate(net, pp, n_tr, False, seed)
        B = slice(N, 2 * N)
        wB, muB = decoder(c['pre'][:, B], c['_s'])
        out[name] = float(slopes(c[W0][:, B], (c['pre'][:, B] - muB) @ wB).sum())
    return out


def report(regime, par, n_tr):
    net = make_net(0, regime)
    d = run(net, par, n_tr, 11)
    uc, us, g = d['u_ctrl'], d['u_stim'], d['g']
    obs = (us - uc).sum(); b1 = uc.sum()
    res = dict(regime=regime, n_trials=n_tr, b1=float(b1), obs_rel=float(obs / b1), window=W0,
               operating_rate=[float(net['r0'].min()), float(net['r0'].max())],
               alpha_true=dict(mean=float(d['alpha_true'].mean()), lo=float(d['alpha_true'].min()),
                               hi=float(d['alpha_true'].max()),
                               lowrate_limit_mean=float(d['alpha_lowrate'].mean())))
    # --- 1. the gain factor
    cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    res['gain_factor'] = {f'alpha={a}': dict(corr_with_G=cor(g ** a, d['G']),
                                             mean_ratio=float(np.average(g ** a / (d['G'] + 1e-12), weights=np.abs(uc))))
                          for a in (0.5, 1.0, 1.3)}
    res['gain_factor']['alpha=per-cell true'] = dict(
        corr_with_G=cor(g ** d['alpha_true'], d['G']),
        mean_ratio=float(np.average(g ** d['alpha_true'] / (d['G'] + 1e-12), weights=np.abs(uc))))
    res['G'] = dict(wmean=float(np.average(d['G'], weights=np.abs(uc))), median=float(np.median(d['G'])))
    # --- 2. the receiving factor
    res['T'] = dict(wmean=float(np.average(d['T'], weights=np.abs(uc))), median=float(np.median(d['T'])),
                    wmean_from_other_cells_only=float(np.average(d['T_rec'], weights=np.abs(uc))))
    res['recv_factor'] = {}
    for a in (0.5, 1.0, 1.3):
        S = algebraic_S(uc, g, a)
        for k in (1, 2):
            res['recv_factor'][f'alpha={a},k={k}'] = dict(
                corr_with_T=cor(S ** k, d['T']), wmean_S=float(np.average(S ** k, weights=np.abs(uc))))
    # --- 3. the pooled table, as phase7_gain_pooled.py computes it
    res['pooled'] = {}
    for a in (0.5, 1.0, 1.3):
        S = algebraic_S(uc, g, a); own = g ** a
        for k in (0, 1, 2):
            res['pooled'][f'a{a}_k{k}'] = float(np.sum(uc * (own * S ** k - 1)) / obs)
    res['pooled']['exact_G_times_T'] = float(np.sum(uc * (d['G'] * d['T'] - 1)) / obs)
    res['pooled']['exact_G_only'] = float(np.sum(uc * (d['G'] - 1)) / obs)
    res['pooled']['exact_T_only'] = float(np.sum(uc * (d['T'] - 1)) / obs)
    # --- 4. the cells whose own rate barely moved
    m = (g > 0.9) & (g < 1.1)
    if m.sum() >= 5:
        S1 = algebraic_S(uc, g, 1.0)
        res['unchanged_rate'] = dict(n=int(m.sum()), slope_ratio=float(us[m].sum() / uc[m].sum()),
                                     predicted_S=float(np.average(S1[m], weights=np.abs(uc[m]))),
                                     true_T=float(np.average(d['T'][m], weights=np.abs(uc[m]))),
                                     true_G=float(np.average(d['G'][m], weights=np.abs(uc[m]))))
    # --- 5. how much of B's state signal needs A
    res['link'] = cut_link(net, par, max(n_tr // 3, 600), 23)
    res['link']['fraction_via_A'] = 1 - res['link']['cut'] / res['link']['intact']
    return res


if __name__ == '__main__':
    t0 = time.time()
    fits = json.load(open(os.path.join(OUT, 'phase7_circuit_gain_fit.json')))
    out = {}
    for regime, f in fits.items():
        par = np.array([f['par'][k] for k in PNAMES], float)
        print(f"\n=== {regime}: {{k: round(v, 4) for k, v in zip(PNAMES, par)}}".replace('{k: round(v, 4) for k, v in zip(PNAMES, par)}',
              str({k: round(v, 4) for k, v in zip(PNAMES, par)})), flush=True)
        r = report(regime, par, NTR)
        out[regime] = r
        print(f"  observed loss {r['obs_rel']:+.3f} of b1", flush=True)
        print(f"  own gain, exact            {r['pooled']['exact_G_only']:+.3f} of the loss", flush=True)
        print(f"  received signal, exact     {r['pooled']['exact_T_only']:+.3f}", flush=True)
        print(f"  both, exact (check ~ 1)    {r['pooled']['exact_G_times_T']:+.3f}", flush=True)
        for key in ('a0.5_k0', 'a1.0_k0', 'a1.3_k0', 'a1.0_k1', 'a1.0_k2'):
            print(f"  section 5.5 estimator {key:9s} {r['pooled'][key]:+.3f}", flush=True)
        print(f"  S_i vs the true input shrink: corr {r['recv_factor']['alpha=1.0,k=1']['corr_with_T']:+.3f}, "
              f"S {r['recv_factor']['alpha=1.0,k=1']['wmean_S']:.3f} vs true {r['T']['wmean']:.3f}", flush=True)
        json.dump(out, open(os.path.join(OUT, 'phase7_circuit_truth.json'), 'w'), indent=1, default=float)
    print('done', round(time.time() - t0), 's', flush=True)
