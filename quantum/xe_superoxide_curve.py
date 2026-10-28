"""Where does xenon sit next to superoxide, and what is the hyperfine coupling there?

xe_hyperfine_ext.py placed Xe at 3.6-4.2 A from superoxide, the neutral van der Waals range. Smith et al. 2021
(Sci Rep 11:6287) found the energy minimum of Xe...O2*- near 3 A (an anion polarises xenon strongly) and a coupling of
~1.8 mT there. This script computes the binding curve (PBE0-D3(BJ), counterpoise-corrected, all-electron DZVP)
along the pi* lobe (Xe above one O, perpendicular to the O-O bond) and end-on (Xe on the O-O axis), and the 129Xe
isotropic coupling at each distance.

usage: python xe_superoxide_curve.py OUTDIR
"""
import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
exec(open(os.path.join(here, 'xe_hyperfine.py')).read().split("if __name__")[0])
HARTREE_KCAL = 627.509


def energy(atoms, charge, spin, basis='dzvp', dm0=None):
    mol = gto.M(atom=atoms, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = (dft.UKS if spin else dft.RKS)(mol); mf.xc = 'pbe0'; mf.disp = 'd3bj'; mf.conv_tol = 1e-9; mf.level_shift = 0.2
    mf.kernel(dm0)
    return mf, mol


def point(geom, dm0=None):
    """geom(R) returns (xe_line, o2_lines); counterpoise binding energy and the Xe coupling of the complex.
    dm0: density of the neighbouring point on the same branch (state following)"""
    xe, o2 = geom
    mf, mol = energy(f'{xe}; {o2}', -1, 1, dm0=dm0)
    point.dm = mf.make_rdm1()
    E_c = mf.e_tot; A, a = a_iso(mol, mf, 0, GN['Xe129'])
    ghost_xe = 'ghost-' + xe; ghost_o2 = '; '.join('ghost-' + l.strip() for l in o2.split(';'))
    E_o2 = energy(f'{ghost_xe}; {o2}', -1, 1)[0].e_tot
    E_xe = energy(f'{xe}; {ghost_o2}', 0, 0)[0].e_tot
    ps = p_spin(mol, mf, 1)                               # which pi* is singly occupied (z = toward Xe for 'lobe')
    return dict(E_bind_kcal=(E_c - E_o2 - E_xe) * HARTREE_KCAL, A_MHz=A, a_mT=a, converged=bool(mf.converged), spin_p_O1=ps)


if __name__ == '__main__':
    res = []; fn = os.path.join(OUT, 'xe_superoxide_curve.json')
    # Xe above one O: the singly occupied pi* either points at Xe (z) or lies perpendicular (y); the two states are
    # followed separately along R (each point starts from the density of its neighbour on the same branch).
    Rs = [2.8, 3.0, 3.2, 3.4, 3.6, 3.9, 4.2]
    for branch, order in (('somo_toward_Xe', Rs), ('somo_perpendicular', Rs[::-1])):
        dm = None
        for R in order:
            r = dict(mode='lobe', branch=branch, R_A=R, **point((f'Xe 0 0 {R}', 'O 0 0 0; O 1.33 0 0'), dm)); dm = point.dm
            res.append(r); print(r, flush=True); json.dump(res, open(fn, 'w'), indent=1)
    for R in (3.0, 3.4, 3.9):
        r = dict(mode='end_on', branch='', R_A=R, **point((f'Xe {-R} 0 0', 'O 0 0 0; O 1.33 0 0'))); res.append(r)
        print(r, flush=True); json.dump(res, open(fn, 'w'), indent=1)
