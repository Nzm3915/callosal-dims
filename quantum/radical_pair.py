"""Radical-pair model for docs/03 section 6: how large a nuclear-spin isotope effect and magnetic-field effect could a
xenon nucleus produce, if xenon acted through a radical pair?

System: two electron spins S1, S2 (the radical pair, born as a singlet) and one xenon nucleus I coupled to S1 by an
isotropic hyperfine interaction a (in mT). Hamiltonian (angular frequency units):
    H = gamma_e B (S1z + S2z) + a_rad * I . S1
Spin-selective recombination with rate k (Haberkorn model, equal singlet and triplet rates). Singlet yield
    Phi_S = (1 / Z) sum_{m,n} |<m|P_S|n>|^2 k^2 / (k^2 + w_mn^2),   Z = 2I + 1,
where |m> are eigenstates of H, w_mn their energy differences and P_S the singlet projector.
Isotopes: 129Xe (I = 1/2), 131Xe (I = 3/2, gyromagnetic ratio -0.297 x that of 129Xe, so a_131 = -0.297 a_129),
spin-0 isotopes (I = 0, no hyperfine term).
The hyperfine constant of xenon with a protein/oxygen radical is not known; it is scanned (0.05-5 mT), as is k.

usage: python radical_pair.py OUTDIR
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
GAMMA_E = 1.76085963e8        # rad s^-1 T^-1 (electron, g ~ 2)
sx = np.array([[0, 1], [1, 0]]) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.array([[1, 0], [0, -1]]) / 2


def spin_ops(I):
    d = int(round(2 * I + 1)); m = I - np.arange(d)
    Iz = np.diag(m); Ip = np.zeros((d, d))
    for i in range(1, d):
        Ip[i - 1, i] = np.sqrt(I * (I + 1) - m[i] * (m[i] + 1))
    return (Ip + Ip.T) / 2, (Ip - Ip.T) / (2j), Iz


def singlet_yield(I, a_mT, B_mT, k, a_bg_mT=0.0):
    """a_bg_mT: isotropic hyperfine of one spin-1/2 nucleus (e.g. a proton) of the partner radical on S2 - the
    radical's own nuclei mix singlet and triplet even without xenon"""
    if a_bg_mT:
        return _yield_bg(I, a_mT, B_mT, k, a_bg_mT)
    e2 = np.eye(2)
    if I == 0:
        dI = 1; Ix = Iy = Iz = np.zeros((1, 1))
    else:
        Ix, Iy, Iz = spin_ops(I); dI = Ix.shape[0]
    eI = np.eye(dI)
    kron = lambda a, b, c: np.kron(np.kron(a, b), c)
    S1 = [kron(s, e2, eI) for s in (sx, sy, sz)]; S2 = [kron(e2, s, eI) for s in (sx, sy, sz)]
    Iv = [kron(e2, e2, o) for o in (Ix, Iy, Iz)]
    a = a_mT * 1e-3 * GAMMA_E; w = B_mT * 1e-3 * GAMMA_E
    H = w * (S1[2] + S2[2]) + (a * sum(S1[i] @ Iv[i] for i in range(3)) if I else 0)
    S1S2 = sum(S1[i] @ S2[i] for i in range(3))
    PS = 0.25 * np.eye(4 * dI) - S1S2
    E, V = np.linalg.eigh(H)
    P = V.conj().T @ PS @ V
    dw = E[:, None] - E[None, :]
    return float(np.real(np.sum(np.abs(P) ** 2 * k ** 2 / (k ** 2 + dw ** 2))) / dI)


def _yield_bg(I, a_mT, B_mT, k, a_bg_mT):
    e2 = np.eye(2)
    ops = [spin_ops(I) if I else (np.zeros((1, 1)),) * 3, spin_ops(0.5)]
    dI = ops[0][0].shape[0]; eI = np.eye(dI)
    def K(*m):
        out = m[0]
        for x in m[1:]: out = np.kron(out, x)
        return out
    S1 = [K(s, e2, eI, e2) for s in (sx, sy, sz)]; S2 = [K(e2, s, eI, e2) for s in (sx, sy, sz)]
    Ix = [K(e2, e2, o, e2) for o in ops[0]]; Jn = [K(e2, e2, eI, o) for o in ops[1]]
    a = a_mT * 1e-3 * GAMMA_E; ab = a_bg_mT * 1e-3 * GAMMA_E; w = B_mT * 1e-3 * GAMMA_E
    H = w * (S1[2] + S2[2]) + ab * sum(S2[i] @ Jn[i] for i in range(3))
    if I: H = H + a * sum(S1[i] @ Ix[i] for i in range(3))
    PS = 0.25 * np.eye(H.shape[0]) - sum(S1[i] @ S2[i] for i in range(3))
    E, V = np.linalg.eigh(H); P = V.conj().T @ PS @ V; dw = E[:, None] - E[None, :]
    return float(np.real(np.sum(np.abs(P) ** 2 * k ** 2 / (k ** 2 + dw ** 2))) / (2 * dI))


def bg_table(a_bg=0.5, B=0.05):
    rows = []
    for k in (1e5, 1e6, 1e7):
        for a in (0.01, 0.05, 0.2, 0.5, 1.0, 2.0):
            y0 = singlet_yield(0, 0, B, k, a_bg); y129 = singlet_yield(0.5, a, B, k, a_bg); y131 = singlet_yield(1.5, -0.297 * a, B, k, a_bg)
            rows.append(dict(k=k, a_mT=a, a_bg_mT=a_bg, B_mT=B, phi0=y0, d129=y129 - y0, d131=y131 - y0,
                             rel129=(y129 - y0) / y0, rel131=(y131 - y0) / y0))
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = []
    for k in (1e5, 1e6, 1e7):
        for a in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
            for B in (0.0, 0.05, 0.5, 1.0):
                y0 = singlet_yield(0, 0, B, k); y129 = singlet_yield(0.5, a, B, k); y131 = singlet_yield(1.5, -0.297 * a, B, k)
                res.append(dict(k=k, a_mT=a, B_mT=B, phi0=y0, phi129=y129, phi131=y131, d129=y129 - y0, d131=y131 - y0))
    bg = bg_table()
    json.dump(dict(no_background=res, background_proton_0p5mT=bg), open(os.path.join(OUT, 'radical_pair.json'), 'w'), indent=1)
    print('--- with a partner-radical proton (0.5 mT), geomagnetic field')
    for r in bg:
        print(f"k={r['k']:.0e} a={r['a_mT']:>4} mT  phi0={r['phi0']:.3f}  rel129={100 * r['rel129']:+.1f}%  rel131={100 * r['rel131']:+.1f}%")
    # summary at the geomagnetic field
    for r in res:
        if r['B_mT'] == 0.05:
            print(f"k={r['k']:.0e} a={r['a_mT']:>4} mT  dPhi129={r['d129']:+.3f}  dPhi131={r['d131']:+.3f}  ratio={r['d129'] / r['d131'] if abs(r['d131']) > 1e-6 else np.nan:+.2f}")
    # figure: isotope effect vs hyperfine (geomagnetic field), and field dependence for a = 1 mT
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, ls in ((1e5, ':'), (1e6, '-'), (1e7, '--')):
        sub = [r for r in res if r['k'] == k and r['B_mT'] == 0.05]
        ax[0].plot([r['a_mT'] for r in sub], [r['d129'] for r in sub], color='#2a78d6', ls=ls, label=f'129Xe, k={k:.0e}/s')
        ax[0].plot([r['a_mT'] for r in sub], [r['d131'] for r in sub], color='#eb6834', ls=ls, label=f'131Xe, k={k:.0e}/s')
    ax[0].set_xscale('log'); ax[0].axhline(0, color='#e4e3df'); ax[0].set_xlabel('hyperfine coupling of 129Xe (mT)')
    ax[0].set_ylabel('change in singlet yield vs spin-0 Xe'); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title('Isotope effect at the geomagnetic field (50 µT)', loc='left', fontsize=10, fontweight='bold')
    Bs = np.linspace(0, 2, 41)
    for I, a, col, nm in ((0, 0, '#8a8984', 'spin 0'), (0.5, 1.0, '#2a78d6', '129Xe (a = 1 mT)'), (1.5, -0.297, '#eb6834', '131Xe')):
        ax[1].plot(Bs, [singlet_yield(I, a, B, 1e6) for B in Bs], color=col, label=nm)
    ax[1].set_xlabel('static field (mT)'); ax[1].set_ylabel('singlet yield'); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title('Field dependence (k = 1e6 /s)', loc='left', fontsize=10, fontweight='bold')
    for a_ in ax: a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_radical_pair.png'), dpi=150)
