"""Merge the screening runs of docs/03 section 7.1 into one table, fill in what the first pass had to skip, and
re-test the explanation of why one nucleus beats another.

Three jobs:

  1. merge. quantum/screen.py was run twice: once for the small radicals (results/quantum/screen.json) and again
     for the ring radicals after the in-plane geometry was fixed (screen_bio.json). Where both contain a row, the
     later run wins, because the earlier phenoxyl 'node' rows were computed with the guest inside a hydrogen.
  2. fill. The first pass skipped the relaxation numbers for 83Kr (I = 9/2) because the Liouvillian is 6400 x 6400.
     radical_pair.py now applies the superoperator matrix-free, so those rows can be completed - and nothing has to
     be recomputed by DFT, since the hyperfine coupling is already stored. The spin dynamics is redone for every
     row, which also re-checks the rows that were not skipped.
  3. re-test. docs/03 section 7.1 claims that what decides a nucleus's effect is not its hyperfine coupling but the
     effective field it puts on the electron, a sqrt(I(I+1)). That was checked on a table that still contained the
     broken phenoxyl rows. It is checked again here on the merged, corrected table.

usage: python screen_summary.py OUTDIR
"""
import sys, os, json, numpy as np
from scipy.stats import spearmanr
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
_rp = open(os.path.join(here, 'radical_pair.py')).read()
exec(_rp[:_rp.index("\nif __name__ == '__main__':")].split('"""', 2)[2])

I132, MARGIN, A_BG, K_RP, B_MT = 0.48, 2.0, 0.347, 5.5e7, 0.05
RS = np.r_[0.0, np.geomspace(1e6, 1e10, 9)]


def effective_field(a, I):
    """the field the nucleus puts on the electron, which is what the effect tracks"""
    return abs(a) * np.sqrt(I * (I + 1))


if __name__ == '__main__':
    rows = {}
    for fn in ('screen.json', 'screen_bio.json'):            # later file wins on a shared key
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print('missing', p); continue
        for r in json.load(open(p)):
            rows[(r['radical'], r['placement'], r['guest'])] = dict(r, source=fn)
    print(f'{len(rows)} rows merged from the screening runs', flush=True)

    # --- fill the spin dynamics for every row, 83Kr included
    for key, r in rows.items():
        I, a = r['I'], r['a_mT']
        y0 = singlet_yield(0, 0, B_MT, K_RP, A_BG, 0.0)
        r['rel_norelax'] = 100 * (singlet_yield(I, a, B_MT, K_RP, A_BG, 0.0) - y0) / y0
        y0r = singlet_yield(0, 0, B_MT, K_RP, A_BG, 1e8)
        r['rel_r1e8'] = 100 * (singlet_yield(I, a, B_MT, K_RP, A_BG, 1e8) - y0r) / y0r
        rmax = 0.0
        for rr in RS:
            z0 = singlet_yield(0, 0, B_MT, K_RP, A_BG, rr); z = singlet_yield(I, a, B_MT, K_RP, A_BG, rr)
            if I132 * abs(100 * (z - z0) / z0) > MARGIN:
                rmax = float(rr)
        r['r_max_detectable'] = rmax
        r['relaxation_skipped'] = False
        r['eff_field_mT'] = effective_field(a, I)

    print(f"\n{'radical':11s} {'place':5s} {'guest':6s} {'contact':>7s} {'a (mT)':>8s} {'a sqrt(I(I+1))':>15s}"
          f" {'r=0':>8s} {'r=1e8':>8s} {'r_max':>8s}")
    for key in sorted(rows):
        r = rows[key]
        print(f"{r['radical']:11s} {r['placement']:5s} {r['guest']:6s} {r.get('min_contact_A', float('nan')):7.2f}"
              f" {r['a_mT']:+8.3f} {r['eff_field_mT']:15.3f} {r['rel_norelax']:+7.2f}% {r['rel_r1e8']:+7.2f}%"
              f" {r['r_max_detectable']:8.0e}", flush=True)

    # --- the explanation: effect ratio against effective-field ratio, and against bare-coupling ratio
    ratios = []
    for (rad, place, g), r in rows.items():
        if g == '129Xe':
            continue
        ref = rows.get((rad, place, '129Xe'))
        if ref is None or abs(ref['rel_norelax']) < 1e-3:
            continue
        ratios.append(dict(radical=rad, placement=place, guest=g,
                           effect_ratio=r['rel_norelax'] / ref['rel_norelax'],
                           field_ratio=r['eff_field_mT'] / (ref['eff_field_mT'] + 1e-15),
                           coupling_ratio=abs(r['a_mT']) / (abs(ref['a_mT']) + 1e-15)))
    e = np.array([x['effect_ratio'] for x in ratios])
    f = np.array([x['field_ratio'] for x in ratios]); c = np.array([x['coupling_ratio'] for x in ratios])
    rf, pf = spearmanr(e, f); rc, pc = spearmanr(e, c)
    summary = dict(n_rows=len(rows), n_ratios=len(ratios),
                   field=dict(spearman=float(rf), p=float(pf), median_abs_dev=float(np.median(np.abs(e - f)))),
                   coupling=dict(spearman=float(rc), p=float(pc), median_abs_dev=float(np.median(np.abs(e - c)))))
    print(f"\n{len(ratios)} (radical x placement x nucleus) ratios against 129Xe:")
    print(f"  effect ratio vs effective-field ratio: rho = {rf:+.2f} (p = {pf:.2g}), "
          f"median |deviation| {summary['field']['median_abs_dev']:.2f}")
    print(f"  effect ratio vs bare-coupling ratio:   rho = {rc:+.2f} (p = {pc:.2g}), "
          f"median |deviation| {summary['coupling']['median_abs_dev']:.2f}")
    # the same test restricted to the rows where the guest actually touches spin density (the pi lobe)
    m = np.array([x['placement'] == 'lobe' or x['radical'] == 'superoxide' for x in ratios])
    if m.sum() >= 6:
        rf2, pf2 = spearmanr(e[m], f[m]); rc2, pc2 = spearmanr(e[m], c[m])
        summary['lobe_only'] = dict(n=int(m.sum()), field_spearman=float(rf2), field_p=float(pf2),
                                    coupling_spearman=float(rc2), coupling_p=float(pc2))
        print(f"  restricted to the {m.sum()} rows with the guest on the orbital: field rho = {rf2:+.2f} "
              f"(p = {pf2:.2g}), coupling rho = {rc2:+.2f} (p = {pc2:.2g})")
    json.dump(dict(summary=summary, ratios=ratios, rows=[rows[k] for k in sorted(rows)]),
              open(os.path.join(OUT, 'screen_summary.json'), 'w'), indent=1, default=float)
    print('\nwrote', os.path.join(OUT, 'screen_summary.json'))
