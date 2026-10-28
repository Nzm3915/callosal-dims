"""docs/03 section 6: order-of-magnitude bounds for every route by which a xenon (or lithium) isotope could change a
biological response, at body temperature (310 K). Prints a table and writes results/quantum/energy_scales.json.

1. Nuclear Zeeman energy in the geomagnetic field, relative to kT (thermodynamic spin effect).
2. Hyperfine energy with a contacting radical (a = 1 mT, the largest value from xe_hyperfine.py), relative to kT.
3. Equilibrium mass isotope effect on binding/solubility of a monatomic solute (Bigeleisen-Mayer, first quantum
   correction):  ln(K_1/K_2) = hbar^2 / (24 (kT)^2) * (1/m_1 - 1/m_2) * sum_axes (k_site - k_ref)
   with an upper-end force constant of the binding cage k = 10 N/m (van der Waals cages are ~0.1-2 N/m).
4. Kinetic mass effect on diffusion/equilibration, D ~ m^-1/2 (changes the time to equilibrium, not the potency).
5. Lithium: measured transport fractionation by Na+/H+ exchangers (Poet et al. 2023): up to ~2% faster 6Li uptake.
"""
import json, os, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
hbar, kB, u, h = 1.054571817e-34, 1.380649e-23, 1.66053906660e-27, 6.62607015e-34
kT = kB * 310.0
gam = {'129Xe': -74.52e6, '131Xe': 22.09e6, '6Li': 39.37e6, '7Li': 103.98e6}      # rad s^-1 T^-1
mass = {'129Xe': 128.9048, '132Xe': 131.9042, '136Xe': 135.9072, '6Li': 6.0151, '7Li': 7.0160}
res = {}
res['zeeman_129Xe_50uT_over_kT'] = abs(hbar * gam['129Xe'] * 50e-6) / kT
res['hyperfine_129Xe_1mT_over_kT'] = h * 28.025e6 / kT          # a = 1 mT  <->  A = 28 MHz
for (a, b) in (('129Xe', '132Xe'), ('132Xe', '136Xe'), ('6Li', '7Li')):
    dinv = 1 / (mass[a] * u) - 1 / (mass[b] * u)
    res[f'equilibrium_mass_effect_{a}_{b}_k10'] = hbar ** 2 / (24 * kT ** 2) * dinv * 3 * 10.0
    res[f'diffusion_ratio_{a}_{b}'] = np.sqrt(mass[b] / mass[a]) - 1
res['Li_transport_fractionation_NHE_max'] = 0.0213
res['Li2018_reported_potency_ratio_spin_vs_spin0'] = 1.45
for k, v in res.items():
    print(f'{k:48s} {v:.2e}')
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, 'energy_scales.json'), 'w'), indent=1)
