"""DFT estimate of the isotropic (Fermi-contact) hyperfine coupling of a 129Xe nucleus in van der Waals contact with
small radicals - the number docs/03 section 6 said decides whether a radical-pair route for xenon is possible.

Models (doublets): methyl CH3* (carbon-centred, stands in for C-centred protein radicals such as tyrosyl/tryptophanyl
ring spin), hydroxyl HO*, hydroperoxyl HOO* (protonated superoxide, the oxygen partner proposed in radical-pair
schemes). Xe is placed along the axis of the singly-occupied p orbital (the lobe that points out of the radical), at
distances R from the spin-carrying atom, scanned across the van der Waals contact (~3.6-4.2 A).

Method: unrestricted PBE0, all-electron Xe basis (DGauss DZVP) - an effective core potential would delete the core
s density that the contact term samples. Spin density at the Xe nucleus rho_s(R_Xe) from the density matrices;
A_iso = (2 mu0 / 3h) g_e mu_B g_N mu_N rho_s / (2S); a (mT) = A / 28.025 MHz. Checks: H atom (1420 MHz) and the
methyl proton (exp. -2.3 mT). Non-relativistic; scalar-relativistic enhancement of the Xe s contact density is roughly
x1.3-1.5 (reported as a range, not applied). Basis-set dependence is checked with uncontracted Dyall v2z at one R.

usage: python xe_hyperfine.py OUTDIR
"""
import sys, os, json, numpy as np
from pyscf import gto, dft
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
MU0 = 4e-7 * np.pi; H_ = 6.62607015e-34; MUB = 9.2740100783e-24; MUN = 5.0507837461e-27; GE = 2.00231930436; A0 = 0.529177210903e-10
GN = {'H': 5.5856946893, 'C13': 1.404824, 'Xe129': -0.77797 / 0.5}


def a_iso(mol, mf, atom_idx, gN):
    """isotropic hyperfine (MHz, mT) at nucleus atom_idx"""
    dma, dmb = mf.make_rdm1()
    r = mol.atom_coord(atom_idx)[None]
    ao = mol.eval_gto('GTOval_sph', r)[0]
    rho_s = float(ao @ (dma - dmb) @ ao) / A0 ** 3          # bohr^-3 -> m^-3
    S = mol.spin / 2
    A = 2 * MU0 / (3 * H_) * GE * MUB * gN * MUN * rho_s / (2 * S)
    return A / 1e6, A / 1e6 / 28.02495


def p_spin(mol, mf, ia):
    """spin population of atom ia's p shells by axis - checks which orbital is singly occupied"""
    labs = mol.ao_labels(); dma, dmb = mf.make_rdm1(); sp = np.diag((dma - dmb) @ mol.intor('int1e_ovlp'))
    return {ax: float(sum(sp[i] for i, l in enumerate(labs) if int(l.split()[0]) == ia and l.strip().endswith(ax)))
            for ax in ('px', 'py', 'pz')}


def run(atoms, basis, spin=1, idx=0, gN=GN['Xe129'], xc='pbe0'):
    """Initial guess = isolated radical (its ground state, SOMO = pz) + closed-shell Xe atom, so the SCF does not
    drop into the excited state whose SOMO lies in the radical plane (happened for HOO without this guess)."""
    mol = gto.M(atom=atoms, basis=basis, spin=spin, verbose=0, unit='Angstrom')
    dm0 = None
    if mol.atom_symbol(0) == 'Xe' and mol.natm > 1:
        lines = [l for l in atoms.split(';')]
        rad = gto.M(atom=';'.join(lines[1:]), basis=basis, spin=spin, verbose=0, unit='Angstrom')
        mr = dft.UKS(rad); mr.xc = xc; mr.kernel()
        xe = gto.M(atom=lines[0], basis=basis, spin=0, verbose=0, unit='Angstrom')
        mx = dft.RKS(xe); mx.xc = xc; mx.kernel()
        nx = xe.nao; dx = mx.make_rdm1() / 2
        dm0 = np.zeros((2, mol.nao, mol.nao))
        for s_, d in enumerate(mr.make_rdm1()):
            dm0[s_, :nx, :nx] = dx; dm0[s_, nx:, nx:] = d
    mf = dft.UKS(mol); mf.xc = xc; mf.conv_tol = 1e-9; mf.max_cycle = 200; mf.level_shift = 0.2; mf.kernel(dm0)
    if not mf.converged:
        mf = mf.newton(); mf.kernel(mf.make_rdm1())
    s2 = mf.spin_square()[0]
    run.last = (p_spin(mol, mf, 1) if mol.natm > 1 else {}, mf.e_tot)
    return a_iso(mol, mf, idx, gN), mf.converged, s2, mol


def geoms(name, R, off=False):
    """Xe first (index 0). Spin-carrying atom at origin; p lobe along +z, Xe at (0, 0, R)."""
    if name == 'CH3':                                        # planar methyl, C-H 1.079 A, p orbital along z
        return f"Xe 0 0 {R}; C 0 0 0; H 1.079 0 0; H -0.5395 0.93444 0; H -0.5395 -0.93444 0"
    if name == 'OH':                                         # O-H along x; SOMO (one of the two pi) along z
        return f"Xe 0 {R} 0; O 0 0 0; H 0.970 0 0" if off else f"Xe 0 0 {R}; O 0 0 0; H 0.970 0 0"
    if name == 'H':                                          # H atom next to one Xe (substitutional site in solid
        return f"Xe 0 0 {R}; H 0 0 0"                        # Xe: H-Xe 4.33 A, 12 neighbours) - for calibration
    if name == 'HOO':                                        # terminal O at origin, molecule in the xy plane;
        return f"Xe 0 0 {R}; O 0 0 0; O 1.331 0 0; H 1.565 0.954 0"   # SOMO is the out-of-plane pi* (z)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); res = dict(checks={}, scan=[], basis_check={})
    (A, a), *_ = run("H 0 0 0", 'dzvp', idx=0, gN=GN['H']); res['checks']['H_atom_MHz'] = A; print('H atom', A, 'MHz')
    (A, a), ok, s2, _ = run(geoms('CH3', 50.0), 'dzvp', idx=2, gN=GN['H']); res['checks']['CH3_proton_mT'] = a
    print('CH3 proton', a, 'mT (exp -2.3)', flush=True)
    for name, off in (('CH3', False), ('OH', False), ('OH', True), ('HOO', False), ('H', False)):
        for R in ((3.4, 3.8, 4.33, 4.8) if name == 'H' else (3.0, 3.4, 3.8, 4.2, 4.6)):
            (A, a), ok, s2, _ = run(geoms(name, R, off), 'dzvp')
            ps, E = run.last
            res['scan'].append(dict(radical=name + ('_offlobe' if off else ''), R_A=R, A_MHz=A, a_mT=a, converged=bool(ok),
                                    S2=float(s2), E=E, spin_p_on_radical_atom=ps))
            print(name, 'off' if off else 'on', R, f'A={A:+.3f} MHz a={a:+.4f} mT conv={ok} S2={s2:.3f} p={ps}', flush=True)
            json.dump(res, open(os.path.join(OUT, 'xe_hyperfine.json'), 'w'), indent=1)
    for b in ('dyall-v2z',):
        mol = gto.M(atom=geoms('CH3', 3.8), basis={'Xe': gto.uncontract(gto.basis.load(b, 'Xe')), 'C': 'dzvp', 'H': 'dzvp'},
                    spin=1, verbose=0)
        mf = dft.UKS(mol); mf.xc = 'pbe0'; mf.level_shift = 0.2; mf.kernel()
        A, a = a_iso(mol, mf, 0, GN['Xe129']); res['basis_check'][f'CH3_3.8_{b}_uncontracted'] = dict(A_MHz=A, a_mT=a)
        print('basis check', b, A, a, flush=True)
    json.dump(res, open(os.path.join(OUT, 'xe_hyperfine.json'), 'w'), indent=1)
