"""Radical-pair model for docs/03 section 6: how large a nuclear-spin isotope effect could a xenon nucleus produce,
if xenon acted through a radical pair?

Correction (2026-10-29, second pass): the first version used the electron gyromagnetic ratio 1.76e8 rad/s/T
instead of 1.76e11 rad/s/T, which made every hyperfine and Zeeman frequency 1000x too small, so the isotope effects
only appeared at absurdly long lifetimes (10 us). Fixed here; all numbers in docs/03 were recomputed.

System: electrons S1, S2 (born as a singlet); a xenon nucleus I on S1 (isotropic hyperfine a); one spin-1/2 nucleus
of the partner radical on S2 (a_bg, the partner's own hyperfine field, which mixes singlet and triplet even without
xenon). H = gamma_e B (S1z + S2z) + a I.S1 + a_bg J.S2   (frequencies: a [mT] * gamma_e).
Recombination: Haberkorn, equal singlet/triplet rate k -> d rho/dt = -i[H, rho] - k rho + relaxation.
Relaxation: random-field relaxation of electron 1 at rate r (Lindblad operators sqrt(r) * S1x, S1y, S1z); r = 0 gives
the coherent limit. Singlet yield Phi_S = k Tr(P_S (k - L)^-1 rho0), rho0 = P_S / Z (nuclei unpolarised).
Isotopes: 129Xe (I = 1/2, gamma -7.452e7), 131Xe (I = 3/2, gamma +2.209e7 -> a_131 = -0.2964 a_129), spin-0 isotopes.

usage: python radical_pair.py OUTDIR
"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
GAMMA_E = 1.76085963e11       # rad s^-1 T^-1 (free electron)
R131 = 2.209e7 / -7.452e7     # a_131 / a_129
sx = np.array([[0, 1], [1, 0]]) / 2; sy = np.array([[0, -1j], [1j, 0]]) / 2; sz = np.array([[1, 0], [0, -1]]) / 2


def spin_ops(I):
    if I == 0:
        return [np.zeros((1, 1))] * 3
    d = int(round(2 * I + 1)); m = I - np.arange(d)
    Iz = np.diag(m).astype(complex); Ip = np.zeros((d, d), complex)
    for i in range(1, d):
        Ip[i - 1, i] = np.sqrt(I * (I + 1) - m[i] * (m[i] + 1))
    return [(Ip + Ip.T) / 2, (Ip - Ip.T) / (2j), Iz]


def kron(*m):
    out = m[0]
    for x in m[1:]:
        out = np.kron(out, x)
    return out


def system(I, a_mT, B_mT, a_bg_mT):
    Ix = spin_ops(I); dI = Ix[0].shape[0]; e2, eI = np.eye(2), np.eye(dI)
    S1 = [kron(s, e2, eI, e2) for s in (sx, sy, sz)]; S2 = [kron(e2, s, eI, e2) for s in (sx, sy, sz)]
    Iv = [kron(e2, e2, o, e2) for o in Ix]; Jv = [kron(e2, e2, eI, s) for s in (sx, sy, sz)]
    w = B_mT * 1e-3 * GAMMA_E; a = a_mT * 1e-3 * GAMMA_E; ab = a_bg_mT * 1e-3 * GAMMA_E
    H = w * (S1[2] + S2[2]) + ab * sum(S2[i] @ Jv[i] for i in range(3))
    if I:
        H = H + a * sum(S1[i] @ Iv[i] for i in range(3))
    PS = 0.25 * np.eye(H.shape[0]) - sum(S1[i] @ S2[i] for i in range(3))
    return H, PS, S1, 2 * dI


def singlet_yield(I, a_mT, B_mT, k, a_bg_mT=0.5, r=0.0):
    H, PS, S1, Z = system(I, a_mT, B_mT, a_bg_mT)
    n = H.shape[0]
    if r == 0:                                                # coherent: closed form in the eigenbasis
        E, V = np.linalg.eigh(H); P = V.conj().T @ PS @ V; dw = E[:, None] - E[None, :]
        return float(np.real(np.sum(np.abs(P) ** 2 * k ** 2 / (k ** 2 + dw ** 2))) / Z)
    rho0 = PS / Z
    if n > 48:
        # For a large nuclear spin the Liouvillian is n^2 x n^2 - 6400 x 6400 for I = 9/2, several GB once the
        # Kronecker products are formed. The superoperator is never built: -L is applied to a matrix directly and
        # the equation is solved iteratively, which costs a few n x n products per step.
        from scipy.sparse.linalg import LinearOperator, gmres
        SS = [Sop @ Sop for Sop in S1]

        def mv(x):
            R = x.reshape(n, n)
            out = 1j * (H @ R - R @ H) + k * R
            for Sop, S2op in zip(S1, SS):
                out -= r * (Sop @ R @ Sop - 0.5 * (S2op @ R + R @ S2op))
            return out.reshape(-1)

        A = LinearOperator((n * n, n * n), matvec=mv, dtype=complex)
        x, info = gmres(A, rho0.reshape(-1).astype(complex), rtol=1e-10, atol=0.0, restart=200, maxiter=40)
        if info != 0:
            raise RuntimeError(f'gmres did not converge (info={info}, n={n})')
        return float(np.real(k * np.trace(PS @ x.reshape(n, n))))
    Id = np.eye(n)
    # Liouvillian on vec(rho) (column stacking): vec(A rho B) = (B^T kron A) vec(rho)
    Lv = -1j * (np.kron(Id, H) - np.kron(H.T, Id)) - k * np.eye(n * n)
    for Sop in S1:                                            # Lindblad: r (S rho S - 1/2 {S S, rho}), S Hermitian
        SS = Sop @ Sop
        Lv += r * (np.kron(Sop.T, Sop) - 0.5 * np.kron(Id, SS) - 0.5 * np.kron(SS.T, Id))
    x = np.linalg.solve(-Lv, rho0.reshape(-1, order='F'))     # integral of rho(t) dt
    return float(np.real(k * np.trace(PS @ x.reshape(n, n, order='F'))))


def effects(a, k, r, B=0.05, a_bg=0.5):
    y0 = singlet_yield(0, 0, B, k, a_bg, r); y9 = singlet_yield(0.5, a, B, k, a_bg, r); y1 = singlet_yield(1.5, R131 * a, B, k, a_bg, r)
    return dict(phi0=y0, rel129=(y9 - y0) / y0, rel131=(y1 - y0) / y0)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    # check: the coherent closed form and the Liouville solver agree at tiny r
    c1 = singlet_yield(0.5, 1.0, 0.05, 5e7, 0.5, 0.0); c2 = singlet_yield(0.5, 1.0, 0.05, 5e7, 0.5, 1.0)
    print('closed form vs Liouville (r ~ 0):', round(c1, 6), round(c2, 6))
    rows = []
    for k in (1e6, 1e7, 5.5e7, 1e8, 1e9):
        for a in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            for r in (0.0, 1e7, 1e8, 1e9):
                e = effects(a, k, r); rows.append(dict(k=k, a_mT=a, r=r, **e))
    json.dump(dict(gamma_e=GAMMA_E, a_bg_mT=0.5, B_mT=0.05, rows=rows), open(os.path.join(OUT, 'radical_pair.json'), 'w'), indent=1)
    print('k (1/s)   a(mT)  r=0: 129 / 131      r=1e8: 129 / 131    r=1e9: 129 / 131')
    for k in (1e6, 1e7, 5.5e7, 1e8, 1e9):
        for a in (0.1, 0.2, 0.5, 1.0, 2.0):
            g = {x['r']: x for x in rows if x['k'] == k and x['a_mT'] == a}
            print(f'{k:8.1e} {a:5.2f}  {100*g[0.0]["rel129"]:+6.1f}% {100*g[0.0]["rel131"]:+6.1f}%   '
                  f'{100*g[1e8]["rel129"]:+6.1f}% {100*g[1e8]["rel131"]:+6.1f}%   {100*g[1e9]["rel129"]:+6.1f}% {100*g[1e9]["rel131"]:+6.1f}%')
    # Smith et al. 2021 parameters (O2*- / TrpH*+ pair): a' = 1.896 mT, k = 5.5e7, r = 5e6, partner 14N-like 0.347 mT
    sm = effects(1.896, 5.5e7, 5e6, a_bg=0.347)
    print('Smith 2021 parameter set:', {k_: round(v, 4) for k_, v in sm.items()})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for r, ls in ((0.0, '-'), (1e8, '--'), (1e9, ':')):
        for col, key in (('#2a78d6', 'rel129'), ('#eb6834', 'rel131')):
            sub = [x for x in rows if x['k'] == 5.5e7 and x['r'] == r]
            ax[0].plot([x['a_mT'] for x in sub], [100 * x[key] for x in sub], color=col, ls=ls,
                       label=f"{key[3:]}Xe, r={r:.0e}/s")
    ax[0].set_xscale('log'); ax[0].axhline(0, color='#e4e3df'); ax[0].set_xlabel('hyperfine coupling of 129Xe (mT)')
    ax[0].set_ylabel('change in singlet yield vs spin-0 Xe (%)'); ax[0].legend(frameon=False, fontsize=6.5)
    ax[0].set_title('Lifetime 18 ns (k = 5.5e7 /s), 50 uT', loc='left', fontsize=10, fontweight='bold')
    for a, col in ((0.1, '#8a8984'), (0.2, '#2a78d6'), (1.0, '#eb6834')):
        sub = [x for x in rows if x['a_mT'] == a and x['r'] == 0.0]
        ax[1].plot([x['k'] for x in sub], [100 * x['rel129'] for x in sub], color=col, marker='o', label=f'a = {a} mT')
    ax[1].set_xscale('log'); ax[1].axhline(0, color='#e4e3df'); ax[1].set_xlabel('recombination rate k (1/s)')
    ax[1].set_ylabel('129Xe effect (%)'); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title('129Xe effect vs lifetime (no relaxation)', loc='left', fontsize=10, fontweight='bold')
    for a_ in ax: a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_radical_pair.png'), dpi=150)
