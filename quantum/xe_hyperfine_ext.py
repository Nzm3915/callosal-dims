"""Extension of xe_hyperfine.py to radicals closer to the ones invoked in biology: superoxide O2*- (the usual
partner in radical-pair schemes) and phenoxyl C6H5O* (model of the tyrosyl radical, spin delocalised over O and the
ring: roughly O 0.3, C1 -0.1, C2/C6 0.25, C4 0.4). Xe is placed above (along the ring normal / pi* lobe) one atom at
a time, at R = 3.6, 3.9, 4.2 A. Same method as xe_hyperfine.py (PBE0, all-electron DZVP, fragment initial guess).

usage: python xe_hyperfine_ext.py OUTDIR
"""
import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
exec(open(os.path.join(here, 'xe_hyperfine.py')).read().split("if __name__")[0])


def run_charged(atoms, charge, xc='pbe0', basis='dzvp'):
    mol = gto.M(atom=atoms, basis=basis, spin=1, charge=charge, verbose=0)
    lines = atoms.split(';')
    rad = gto.M(atom=';'.join(lines[1:]), basis=basis, spin=1, charge=charge, verbose=0)
    mr = dft.UKS(rad); mr.xc = xc; mr.kernel()
    xe = gto.M(atom=lines[0], basis=basis, verbose=0); mx = dft.RKS(xe); mx.xc = xc; mx.kernel()
    nx = xe.nao; dm0 = np.zeros((2, mol.nao, mol.nao))
    for s_, d in enumerate(mr.make_rdm1()):
        dm0[s_, :nx, :nx] = mx.make_rdm1() / 2; dm0[s_, nx:, nx:] = d
    mf = dft.UKS(mol); mf.xc = xc; mf.conv_tol = 1e-9; mf.max_cycle = 200; mf.level_shift = 0.2; mf.kernel(dm0)
    # Mulliken spin of the radical atoms (to report how much spin sits under the xenon)
    labs = mol.ao_labels(); dma, dmb = mf.make_rdm1(); sp = np.diag((dma - dmb) @ mol.intor('int1e_ovlp'))
    atom_spin = [float(sum(sp[i] for i, l in enumerate(labs) if int(l.split()[0]) == ia)) for ia in range(mol.natm)]
    return a_iso(mol, mf, 0, GN['Xe129']), bool(mf.converged), atom_spin


def phenoxyl(site, R):
    r = 1.40; ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    C = [(r * np.cos(a), r * np.sin(a)) for a in ang]                 # C1 at angle 0 bears the O
    Hs = [(2.48 * np.cos(a), 2.48 * np.sin(a)) for a in ang[1:]]
    pos = {'O': (r + 1.25, 0.0), 'C4': C[3], 'C2': C[1]}[site]
    s = f"Xe {pos[0]:.4f} {pos[1]:.4f} {R}; O {r + 1.25:.4f} 0 0; " + '; '.join(f"C {x:.4f} {y:.4f} 0" for x, y in C)
    return s + '; ' + '; '.join(f"H {x:.4f} {y:.4f} 0" for x, y in Hs)


if __name__ == '__main__':
    res = []
    fn = os.path.join(OUT, 'xe_hyperfine_ext.json')
    for nm, build, ch in (('superoxide_above_O', lambda R: f"Xe 0 0 {R}; O 0 0 0; O 1.33 0 0", -1),
                          ('phenoxyl_above_O', lambda R: phenoxyl('O', R), 0),
                          ('phenoxyl_above_C4', lambda R: phenoxyl('C4', R), 0),
                          ('phenoxyl_above_C2', lambda R: phenoxyl('C2', R), 0)):
        for R in (3.6, 3.9, 4.2):
            (A, a), ok, sp = run_charged(build(R), ch)
            res.append(dict(case=nm, R_A=R, A_MHz=A, a_mT=a, converged=ok, atom_spin=sp))
            print(nm, R, f'A={A:+.3f} MHz a={a:+.4f} mT conv={ok} spin(radical atoms)={np.round(sp[1:8], 2).tolist()}', flush=True)
            json.dump(res, open(fn, 'w'), indent=1)
