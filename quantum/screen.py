"""The candidate table of docs/03 section 7.1: for every (guest nucleus, radical, placement), compute the hyperfine
coupling by DFT and then the isotope effect it could produce, so that combinations can be ruled in or out before any
experiment.

Guests: 129Xe/131Xe (I = 1/2, 3/2), 83Kr (I = 9/2), 3He (I = 1/2). Lithium is excluded on purpose: a neutral Li
atom is itself a radical, and in cells lithium is a cation whose isotope effects are dominated by mass and by
transport (Poet et al. 2023, up to ~2% faster 6Li uptake), which this calculation does not address.
Radicals: methyl (localised C), hydroxyl (localised O), superoxide (pi, anion), phenoxyl (tyrosyl model).
Placement: 'lobe' (above the singly occupied orbital) and 'node' (in its nodal plane) at the van der Waals contact.

For each combination the script reports a (mT) for the spin-1/2-equivalent, the effect on the singlet yield at a
lifetime of 18 ns with no relaxation (the optimistic case) and with relaxation 1e8 /s, and the relaxation rate above
which the effect falls below the 2-point detection margin of the cell experiment.

usage: python screen.py OUTDIR [--quick]
"""
import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
QUICK = '--quick' in sys.argv
exec(open(os.path.join(here, 'xe_hyperfine.py')).read().split("if __name__")[0])
_rp = open(os.path.join(here, 'radical_pair.py')).read()
exec(_rp[:_rp.index("\nif __name__ == '__main__'")].split('"""', 2)[2])

I132, MARGIN, A_BG, K_RP = 0.48, 2.0, 0.347, 5.5e7
# nuclear g factors (mu/mu_N / I), from the standard tables
GN = dict(GN, **{'Kr83': -0.970669 / 4.5, 'He3': -2.127625 / 0.5, 'Li6': 0.822047 / 1.0, 'Li7': 3.256427 / 1.5})
# Lithium is left out: as a neutral atom it is an open-shell radical itself, and in cells it is a cation whose
# isotope effects are dominated by mass and by transport (Poet et al. 2023), not by a van der Waals contact.
GUESTS = {'129Xe': (0.5, GN['Xe129']), '131Xe': (1.5, GN['Xe129'] * (2.209e7 / -7.452e7)),
          '83Kr': (4.5, GN['Kr83']), '3He': (0.5, GN['He3'])}
RADS = {
    'methyl':     ("C 0 0 0; H 1.079 0 0; H -0.5395 0.93444 0; H -0.5395 -0.93444 0", 0),
    'hydroxyl':   ("O 0 0 0; H 0.970 0 0", 0),
    'superoxide': ("O 0 0 0; O 1.33 0 0", -1),
    'phenoxyl':   (None, 0),
    'indolyl':    (None, 0),      # tryptophan radical model: 3-methylindole minus a hydrogen, spin on the ring
    'flavin_iso': (None, 0),      # FAD model: the isoalloxazine's central ring reduced to a lumiflavin-like core
}
R_CONTACT = {'129Xe': 3.6, '131Xe': 3.6, '83Kr': 3.4, '3He': 3.0}   # van der Waals contact distances, A


def ring_geom(R, placement, extra='', spin_atom=3):
    """a benzene-like ring with a substituent; Xe above the ring plane (lobe) or in the plane beside it (node)"""
    r = 1.40; ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    C = [(r * np.cos(a), r * np.sin(a)) for a in ang]
    Hs = [(2.48 * np.cos(a), 2.48 * np.sin(a)) for a in ang[1:]]
    body = extra + '; '.join(f"C {x:.4f} {y:.4f} 0" for x, y in C) + '; ' + \
           '; '.join(f"H {x:.4f} {y:.4f} 0" for x, y in Hs)
    pos = (C[spin_atom][0], C[spin_atom][1], R) if placement == 'lobe' else (C[spin_atom][0], C[spin_atom][1] + R, 0.0)
    return pos, body


def phenoxyl_geom(R, placement):
    r = 1.40; ang = np.deg2rad([0, 60, 120, 180, 240, 300])
    C = [(r * np.cos(a), r * np.sin(a)) for a in ang]
    Hs = [(2.48 * np.cos(a), 2.48 * np.sin(a)) for a in ang[1:]]
    body = f"O {r + 1.25:.4f} 0 0; " + '; '.join(f"C {x:.4f} {y:.4f} 0" for x, y in C) + '; ' + \
           '; '.join(f"H {x:.4f} {y:.4f} 0" for x, y in Hs)
    pos = (C[3][0], C[3][1], R) if placement == 'lobe' else (C[3][0], C[3][1] + R, 0.0)   # above C4, or in the plane
    return pos, body


def coupling(guest_elem, rad, placement, R):
    """129Xe-equivalent coupling: computed with a xenon-like all-electron basis for the guest actually named"""
    if rad in ('phenoxyl', 'indolyl', 'flavin_iso'):
        if rad == 'phenoxyl':
            (x, y, z), body = phenoxyl_geom(R, placement)
        elif rad == 'indolyl':
            # pyrrole N fused at C1-C2 of the ring, reduced to an N-H bridge: a compact tryptophan-radical stand-in
            (x, y, z), body = ring_geom(R, placement, extra="N 2.10 1.21 0; H 2.80 1.95 0; ", spin_atom=4)
        else:
            # two carbonyl oxygens on opposite ring carbons: a lumiflavin-like pi system
            (x, y, z), body = ring_geom(R, placement, extra="O 2.65 0 0; O -2.65 0 0; ", spin_atom=1)
        atoms = f'{guest_elem} {x:.4f} {y:.4f} {z:.4f}; ' + body
    else:
        body, _ = RADS[rad]
        atoms = (f'{guest_elem} 0 0 {R}; ' if placement == 'lobe' else f'{guest_elem} 0 {R} 0; ') + body
    charge = RADS[rad][1]
    mol = gto.M(atom=atoms, basis='dzvp', spin=1, charge=charge, verbose=0)
    lines = atoms.split(';')
    rr = gto.M(atom=';'.join(lines[1:]), basis='dzvp', spin=1, charge=charge, verbose=0)
    mr = dft.UKS(rr); mr.xc = 'pbe0'; mr.kernel()
    gg = gto.M(atom=lines[0], basis='dzvp', verbose=0)              # closed-shell guest (Xe, Kr, He)
    mg = dft.RKS(gg); mg.xc = 'pbe0'; mg.kernel()
    nx = gg.nao; dm0 = np.zeros((2, mol.nao, mol.nao))
    for s_, dd in enumerate(mr.make_rdm1()):
        dm0[s_, :nx, :nx] = mg.make_rdm1() / 2; dm0[s_, nx:, nx:] = dd
    mf = dft.UKS(mol); mf.xc = 'pbe0'; mf.conv_tol = 1e-9; mf.max_cycle = 200; mf.level_shift = 0.2; mf.kernel(dm0)
    A, a = a_iso(mol, mf, 0, 1.0)                     # per unit g_N: scale per isotope below
    return A, a, bool(mf.converged)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); rows = []; t0 = time.time()
    fn = os.path.join(OUT, os.environ.get('SCREEN_OUT', 'screen.json'))
    elems = {'129Xe': 'Xe', '131Xe': 'Xe', '83Kr': 'Kr', '3He': 'He'}
    rs = np.r_[0.0, np.geomspace(1e6, 1e10, 9)]
    only = os.environ.get('RADS_ONLY')
    for rad in (only.split(',') if only else (['methyl', 'superoxide'] if QUICK else list(RADS))):
        for placement in ('lobe', 'node'):
            cache = {}
            for g, (I, gN) in GUESTS.items():
                el = elems[g]
                if el not in cache:
                    try:
                        cache[el] = coupling(el, rad, placement, R_CONTACT[g])
                    except Exception as e:
                        print('FAIL', rad, placement, el, str(e)[:70], flush=True); cache[el] = (np.nan,) * 3
                A_unit, a_unit, conv = cache[el]
                a = a_unit * gN                                  # mT, for this isotope's g_N
                if not np.isfinite(a):
                    continue
                y0 = singlet_yield(0, 0, 0.05, K_RP, A_BG, 0.0); y = singlet_yield(I, a, 0.05, K_RP, A_BG, 0.0)
                rel0 = 100 * (y - y0) / y0
                # the Liouville solve with relaxation costs (4*(2I+1)*2)^2 squared; skip it for the big nuclear spins
                big = (4 * (2 * I + 1) * 2) > 40      # I = 9/2 only: the Liouville matrix would be 6400 x 6400
                rel8 = np.nan; rmax = np.nan
                if not big:
                    y0r = singlet_yield(0, 0, 0.05, K_RP, A_BG, 1e8); yr = singlet_yield(I, a, 0.05, K_RP, A_BG, 1e8)
                    rel8 = 100 * (yr - y0r) / y0r
                    rmax = 0.0
                    for r in rs:
                        z0 = singlet_yield(0, 0, 0.05, K_RP, A_BG, r); z = singlet_yield(I, a, 0.05, K_RP, A_BG, r)
                        if I132 * abs(100 * (z - z0) / z0) > MARGIN:
                            rmax = r
                rows.append(dict(radical=rad, placement=placement, guest=g, I=I, R_A=R_CONTACT[g],
                                 a_mT=a, rel_norelax=rel0, rel_r1e8=rel8, r_max_detectable=rmax, converged=conv))
                print(f'{rad:11s} {placement:5s} {g:5s} a={a:+7.3f} mT  effect {rel0:+7.2f}% (r=0), '
                      f'{rel8:+6.2f}% (r=1e8), detectable up to r = {rmax:.0e} /s   [{round(time.time()-t0)} s]', flush=True)
                rows[-1]['relaxation_skipped'] = bool(big)
                json.dump(rows, open(fn, 'w'), indent=1)
    print('done', round(time.time() - t0), 's')
