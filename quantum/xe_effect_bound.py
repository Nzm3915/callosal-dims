"""The prediction docs/03 registers before any experiment: how large can a xenon nuclear-spin effect be, given
(i) the hyperfine coupling computed for Xe next to superoxide (xe_superoxide_curve.py) and (ii) the electron-spin
relaxation of the superoxide electron, which is the free parameter nobody has measured in a protein pocket?

For a grid of relaxation rates r and recombination rates k, the singlet-yield difference between 129Xe/131Xe and a
spin-0 isotope is computed with radical_pair.py at the DFT coupling, and converted into the quantity the cell
experiment measures: the isotope difference in the NMDA inhibition at 50% xenon,

    Delta_I = I_132 * |rel|,   I_132 ~ 0.48 (Franks 1998 scaled to 50%),

assuming the singlet yield feeds through proportionally into the effect of xenon (the most favourable assumption,
so the result is an upper bound). The 2-point equivalence margin of the preregistered experiment is drawn on top:
this says which (r, k) the experiment can rule out, and which it cannot.

usage: python xe_effect_bound.py OUTDIR
"""
import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
exec(open(os.path.join(here, 'radical_pair.py')).read().split("if __name__")[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

I132 = 0.48            # inhibition at 50% xenon from Franks 1998 (60% at 80%, Hill 1 -> IC50 ~ 53%)
MARGIN = 2.0           # equivalence margin of the cell experiment, percentage points of inhibition
A_DFT = 0.76           # mT, 129Xe isotropic coupling at the Xe...O2*- minimum (3.4 A), xe_superoxide_curve.py
A_SMITH = 1.896        # mT, value fitted by Smith et al. 2021
A_BG = 0.347           # mT, partner radical's own nucleus (indole 14N)

if __name__ == '__main__':
    ks = np.array([1e7, 3e7, 5.5e7, 1e8, 3e8, 1e9])
    rs = np.array([0.0, 1e6, 1e7, 1e8, 3e8, 1e9, 3e9, 1e10])
    out = {}
    for name, a in (('dft_0.76mT', A_DFT), ('smith_1.9mT', A_SMITH)):
        M = np.zeros((len(ks), len(rs)))
        for i, k in enumerate(ks):
            for j, r in enumerate(rs):
                e = effects(a, k, r, a_bg=A_BG)
                M[i, j] = 100 * I132 * abs(0.5 * (e['rel129'] + e['rel131']))     # points of inhibition
        out[name] = dict(k=ks.tolist(), r=rs.tolist(), delta_points=M.tolist())
        print(f'--- a = {a} mT: predicted isotope difference in inhibition (percentage points)')
        print('k \\ r   ' + '  '.join(f'{r:8.0e}' for r in rs))
        for i, k in enumerate(ks):
            print(f'{k:7.1e} ' + '  '.join(f'{v:8.2f}' for v in M[i]))
        # the largest relaxation rate at which the effect still exceeds the experiment's margin
        lim = {}
        for i, k in enumerate(ks):
            ok = rs[M[i] > MARGIN]
            lim[f'{k:.1e}'] = float(ok.max()) if len(ok) else 0.0
        out[name]['max_r_detectable'] = lim
        print('largest relaxation rate (1/s) still giving > 2 points:', lim)
    json.dump(dict(I132=I132, margin_points=MARGIN, a_bg_mT=A_BG, **out), open(os.path.join(OUT, 'xe_effect_bound.json'), 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, col in (('dft_0.76mT', '#2a78d6'), ('smith_1.9mT', '#eb6834')):
        M = np.array(out[name]['delta_points']); i = list(ks).index(5.5e7)
        ax.plot(rs[1:], M[i][1:], color=col, marker='o', label=f"{name.replace('_', ', ')} (k = 5.5e7 /s)")
    ax.axhline(MARGIN, color='#8a8984', ls='--', lw=1)
    ax.text(1.2e6, MARGIN * 1.15, "experiment's 2-point margin", fontsize=7, color='#8a8984')
    ax.axvspan(1e9, 1e10, color='#eb6834', alpha=0.07, lw=0)
    ax.text(1.3e9, 0.02, 'free superoxide\n(relaxation < 1 ns)', fontsize=7, color='#8a8984')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('electron-spin relaxation rate of the superoxide electron (1/s)')
    ax.set_ylabel('predicted isotope difference (points of inhibition)')
    ax.set_title('What the cell experiment can rule out', loc='left', fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7.5); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_xe_effect_bound.png'), dpi=150)
