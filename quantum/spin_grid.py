"""Precompute the xenon isotope effect over the whole parameter space, for the interactive explorer
(docs/03 section 6) and for the feasibility screen.

Axes: a (129Xe hyperfine, mT), k (recombination rate, 1/s), r (electron-spin relaxation rate, 1/s), at the
geomagnetic field; plus a field scan at a few (a, r) for the magnetic-field-effect panel. For every point the exact
singlet yields of spin-0, 129Xe and 131Xe are computed with radical_pair.py (Liouville solution including
relaxation), and stored as the relative change in percent, rounded to 2 decimals.

usage: python spin_grid.py OUTDIR [--fast]
"""
import sys, os, json, time, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else 'results/quantum'
FAST = '--fast' in sys.argv
exec(open(os.path.join(here, 'radical_pair.py')).read().split("if __name__")[0])
A_BG = 0.347

A = np.round(np.geomspace(0.02, 3.0, 9 if FAST else 17), 4)
K = np.geomspace(1e6, 1e10, 9 if FAST else 13)
Rr = np.r_[0.0, np.geomspace(1e5, 1e10, 7 if FAST else 11)]
B = np.round(np.r_[0.0, np.geomspace(0.01, 5.0, 11 if FAST else 19)], 4)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    y0 = np.zeros((len(K), len(Rr)))
    for i, k in enumerate(K):
        for j, r in enumerate(Rr):
            y0[i, j] = singlet_yield(0, 0, 0.05, k, A_BG, r)
    print('spin-0 grid done', round(time.time() - t0), 's', flush=True)
    rel = np.zeros((2, len(A), len(K), len(Rr)))
    for ia, a in enumerate(A):
        for i, k in enumerate(K):
            for j, r in enumerate(Rr):
                y9 = singlet_yield(0.5, a, 0.05, k, A_BG, r)
                y1 = singlet_yield(1.5, R131 * a, 0.05, k, A_BG, r)
                rel[0, ia, i, j] = 100 * (y9 - y0[i, j]) / y0[i, j]
                rel[1, ia, i, j] = 100 * (y1 - y0[i, j]) / y0[i, j]
        print(f'a = {a} mT done, {round(time.time() - t0)} s', flush=True)
    # field scan: how the isotope effect changes with a static field (the signature of a radical pair)
    AB = [0.36, 0.76, 1.9]; RB = [1e6, 1e8]
    field = np.zeros((len(AB), len(RB), len(B), 3))
    for ia, a in enumerate(AB):
        for ir, r in enumerate(RB):
            for ib, b in enumerate(B):
                z0 = singlet_yield(0, 0, b, 5.5e7, A_BG, r)
                field[ia, ir, ib] = [z0, 100 * (singlet_yield(0.5, a, b, 5.5e7, A_BG, r) - z0) / z0,
                                     100 * (singlet_yield(1.5, R131 * a, b, 5.5e7, A_BG, r) - z0) / z0]
        print('field scan a =', a, round(time.time() - t0), 's', flush=True)
    out = dict(a_mT=A.tolist(), k=K.tolist(), r=Rr.tolist(), B_mT=B.tolist(), a_bg_mT=A_BG,
               phi0=np.round(y0, 4).tolist(), rel129=np.round(rel[0], 2).tolist(), rel131=np.round(rel[1], 2).tolist(),
               field_a=AB, field_r=RB, field=np.round(field, 3).tolist(),
               dft=dict(free_average=0.36, locked_min=0.76, locked_close=1.29, smith_fit=1.896))
    json.dump(out, open(os.path.join(OUT, 'spin_grid.json'), 'w'), separators=(',', ':'))
    print('wrote', os.path.getsize(os.path.join(OUT, 'spin_grid.json')) // 1024, 'kB in', round(time.time() - t0), 's')
