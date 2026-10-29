"""The krypton arm of docs/03 section 5: if 83Kr is used instead of 129Xe, what does the experiment have to look
like, and is it worth it?

83Kr is the one nucleus that lets the mass confound be removed *within one element*: 84Kr has no nuclear spin and
is one mass unit away, where xenon's spin-0 control (132Xe) is three away from 129Xe. Against that, krypton is a
weaker anaesthetic - mouse loss of the righting reflex needs about 4.5 atm against xenon's 0.95 atm (Dickinson &
Franks 2010, Crit Care 14:229) - so a matched block of the NMDA current needs a pressurised chamber.

The quantities computed here:

  1. the partial pressure of krypton needed to reach the same block as 50% xenon, and the block reachable at 1 atm
  2. the predicted 83Kr-vs-84Kr difference in points of inhibition, for each candidate radical of the screen, as a
     function of the electron spin relaxation rate, next to 129Xe-vs-132Xe for the same radical
  3. the cells needed per condition for 90% power at those sizes
  4. what has to be bought: 83Kr is 11.5% of natural krypton, so the spin arm needs enrichment; 84Kr is 57% and
     the control arm does not
  5. the two confounds that change sign between the elements: the mass difference (smaller for krypton) and
     pressure reversal of anaesthesia (present for krypton, and cancelled by the design)

The 83Kr spin dynamics with relaxation only became computable when radical_pair.py stopped building the Liouvillian
(I = 9/2 gives a 6400 x 6400 superoperator) and started applying it matrix-free.

usage: python krypton_design.py OUTDIR
"""
import sys, os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
_rp = open(os.path.join(here, 'radical_pair.py')).read()
exec(_rp[:_rp.index("\nif __name__ == '__main__':")].split('"""', 2)[2])

# --- fixed inputs, all from the documents cited in docs/03
C50_XE = 53.0          # % atm of xenon for half block of the NMDA current (docs/03 section 5.2, from Franks 1998)
MAC_XE, MAC_KR = 0.95, 4.5      # atm, mouse loss of righting reflex (Dickinson & Franks 2010)
P_XE_WORK = 50.0       # % atm, the working xenon concentration of the planned experiment
K_RP, A_BG, B_MT = 5.5e7, 0.347, 0.05        # radical-pair lifetime 18 ns, partner hyperfine, 50 uT
ABUND = {'83Kr': 11.49, '84Kr': 56.99, '86Kr': 17.28, '129Xe': 26.40, '131Xe': 21.23, '132Xe': 26.91}
MASS = {'83Kr': 82.914, '84Kr': 83.912, '129Xe': 128.905, '132Xe': 131.904}
GN = {'129Xe': -7.452e7 / 2.6752219e8 * 5.5857, '83Kr': -0.970669 / 4.5}   # only the ratio matters below
# the couplings of the corrected screen (quantum/screen.py), in mT, per radical and placement
ROWS = [('methyl', 'lobe', -3.78, -0.34), ('hydroxyl', 'lobe', -1.18, -0.085),
        ('superoxide', 'node', 0.53, 0.029), ('phenoxyl', 'lobe', -0.237, -0.029)]
RS = [0.0, 1e7, 3e7, 1e8, 3e8, 1e9]


def inhibition(c, c50):
    return c / (c + c50)


def points(rel, I_work):
    """a fractional change in the singlet yield taken through to points of inhibition at the working block"""
    return 100.0 * I_work * abs(rel)


def n_cells(delta, sigma, power=0.90):
    from math import erf, sqrt
    z = {0.90: 1.2816, 0.80: 0.8416}[power]
    return int(np.ceil(((1.95996 + z) * sigma / max(delta, 1e-9)) ** 2))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {}

    # --- 1. pressure
    c50_kr = C50_XE * MAC_KR / MAC_XE
    I_xe = inhibition(P_XE_WORK, C50_XE)
    p_match = c50_kr * I_xe / (1 - I_xe)                     # krypton partial pressure for the same block
    I_kr_1atm = inhibition(100.0, c50_kr)
    res['pressure'] = dict(c50_xe_pct_atm=C50_XE, potency_ratio=MAC_KR / MAC_XE, c50_kr_pct_atm=c50_kr,
                           I_xe_at_50pct=I_xe, kr_partial_pressure_for_match_atm=p_match / 100,
                           total_absolute_atm_with_20pct_O2=p_match / 100 + 0.2,
                           I_kr_at_1atm=I_kr_1atm)
    print(f"krypton half-block at {c50_kr:.0f}% atm; to match xenon's block of {I_xe:.2f} it needs "
          f"{p_match/100:.2f} atm of krypton ({p_match/100 + 0.2:.2f} atm absolute with 20% oxygen). "
          f"At 1 atm of krypton the block is only {I_kr_1atm:.2f}.", flush=True)

    # --- 2. predicted isotope differences
    res['effects'] = []
    print('\nradical      placement  r (/s)    129Xe-132Xe        83Kr-84Kr (matched P)   83Kr-84Kr (1 atm)')
    for rad, place, a129, a83 in ROWS:
        for r in RS:
            y0 = singlet_yield(0, 0, B_MT, K_RP, A_BG, r)
            y129 = singlet_yield(0.5, a129, B_MT, K_RP, A_BG, r)
            y83 = singlet_yield(4.5, a83, B_MT, K_RP, A_BG, r)
            rel129, rel83 = (y129 - y0) / y0, (y83 - y0) / y0
            row = dict(radical=rad, placement=place, r=r, a129_mT=a129, a83_mT=a83,
                       rel129=rel129, rel83=rel83,
                       pts_xe=points(rel129, I_xe), pts_kr_matched=points(rel83, I_xe),
                       pts_kr_1atm=points(rel83, I_kr_1atm))
            res['effects'].append(row)
            print(f"{rad:12s} {place:9s} {r:7.0e}   {row['pts_xe']:5.2f} pts          "
                  f"{row['pts_kr_matched']:5.2f} pts              {row['pts_kr_1atm']:5.2f} pts", flush=True)

    # --- 3. cells needed, at the relaxation rate that is realistic for a protein-bound radical
    res['n_cells'] = {}
    for row in [x for x in res['effects'] if x['r'] in (0.0, 1e8)]:
        key = f"{row['radical']}/{row['placement']}/r={row['r']:.0e}"
        res['n_cells'][key] = {f'sigma={s}': dict(xe=n_cells(row['pts_xe'], s),
                                                  kr_matched=n_cells(row['pts_kr_matched'], s),
                                                  kr_1atm=n_cells(row['pts_kr_1atm'], s)) for s in (4, 6, 8)}

    # --- 4. what has to be bought
    res['supply'] = dict(abundance_pct=ABUND,
                         enrichment_needed=dict(spin_arm='83Kr, 11.5% natural: enrichment required',
                                                control_arm='84Kr, 57.0% natural: the majority isotope'),
                         xenon_for_comparison=dict(spin_arm='129Xe, 26.4%', control_arm='132Xe, 26.9%'))

    # --- 5. the two confounds
    dm_kr = (MASS['84Kr'] - MASS['83Kr']) / MASS['83Kr']
    dm_xe = (MASS['132Xe'] - MASS['129Xe']) / MASS['129Xe']
    res['confounds'] = dict(
        mass_difference=dict(kr_rel=dm_kr, xe_rel=dm_xe, ratio=dm_xe / dm_kr,
                             note='any mass-dependent artefact (diffusion ~ 1/sqrt(m), zero-point energy ~ 1/sqrt(m)) '
                                  'is smaller for the krypton pair by this factor'),
        diffusion_ratio=dict(kr=np.sqrt(MASS['84Kr'] / MASS['83Kr']), xe=np.sqrt(MASS['132Xe'] / MASS['129Xe'])),
        pressure_reversal=dict(
            note='a few atmospheres antagonise anaesthesia (pressure reversal), so a krypton experiment at 2.5 atm '
                 'cannot be compared with a xenon experiment at 1 atm. It can be compared with 84Kr at the same '
                 'pressure: both arms sit at identical total pressure, so the reversal cancels in the isotope '
                 'difference. The cost is that the absolute block is smaller than pressure-free potency predicts, '
                 'which is why the block has to be measured in the chamber rather than assumed.'),
        nuclear_relaxation=dict(
            note='83Kr has I = 9/2 and a large quadrupole moment (0.259 b), so its nuclear relaxation in a cavity is '
                 'fast by NMR standards - T1 of order 1-10 ms, i.e. 1e2-1e3 /s. That is 1e4-1e5 times slower than '
                 'the radical pair recombines (5.5e7 /s), so the nuclear spin is static over the pair lifetime and '
                 'the quadrupole cannot wash the effect out. The relaxation that matters is the electron\'s.'))
    print(f"\nmass difference: 84Kr/83Kr {100*dm_kr:.2f}%, 132Xe/129Xe {100*dm_xe:.2f}% "
          f"-> any mass artefact is {dm_xe/dm_kr:.1f}x smaller in the krypton pair", flush=True)
    json.dump(res, open(os.path.join(OUT, 'krypton_design.json'), 'w'), indent=1, default=float)
    print('\nwrote', os.path.join(OUT, 'krypton_design.json'), flush=True)
