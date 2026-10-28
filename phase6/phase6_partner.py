"""Phase 6-2b: is the state dependence (c1 < 0) simply the loss of a correlated partner?

If A's input to B carries A's own choice state, silencing A removes an amount proportional to A's *true* state.
A's true state correlates with B's (the two hemispheres agree), but the regression only sees a noisy estimate cA.
The part of A's state that cA misses is then credited to cB, and c1 comes out negative even with a purely additive,
fixed link (an errors-in-variables effect, no gate, no gain).

Test: measure A's pre-laser state better (longer windows: 0.4, 0.8, 1.2 s; and all of A's units vs half) and watch
c1. Partner loss predicts |c1| shrinks as cA becomes more reliable; a receiver-side mechanism (gain, attractor) does
not care how well A is measured. As a control, the same manipulation is applied to cB's window.
Reliability of each estimate is measured on control trials by split-halves of units (Spearman-Brown corrected).

    y = b0 + b1 cB + b2 cA + stim (c0 + c1 cB + c2 cA)       y: B's confidence during the laser (-1.7..-0.9)

usage: python phase6_partner.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from phase6_ramp import load, Axes, zs, wls, rate

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(63)
WINS = {'0.4': (-2.1, -1.7), '0.8': (-2.5, -1.7), '1.2': (-2.9, -1.7)}
LAS = (-1.7, -0.9)


def reliab(AxA, AxB, trials):
    """split-half (units) reliability of a confidence estimate on these trials"""
    a, b = AxA.conf(trials[0], *trials[1]), AxB.conf(trials[0], *trials[1])
    r = np.corrcoef(a, b)[0, 1]
    return 2 * r / (1 + r) if r > -0.99 else np.nan


def pair_rows(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ub, ua = d['idx'][rec], d['idx'][sil]
    if min(len(ua), len(ub)) < 8 or len(stim) < 20:
        return []
    rows = []
    fold = rng.permutation(len(ctrl_all)) % 2
    perm_a = rng.permutation(len(ua)); a1, a2 = ua[perm_a[:len(ua) // 2]], ua[perm_a[len(ua) // 2:]]
    store = {}
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        AB, AA, A1, A2 = Axes(d, ub, fit), Axes(d, ua, fit), Axes(d, a1, fit), Axes(d, a2, fit)
        for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
            s = store.setdefault(st, {})
            s.setdefault('y', []).append(AB.conf(trials, *LAS))
            for wn, wv in WINS.items():
                s.setdefault(f'cB{wn}', []).append(AB.conf(trials, *wv))
                s.setdefault(f'cA{wn}', []).append(AA.conf(trials, *wv))
                s.setdefault(f'cAh{wn}', []).append(A1.conf(trials, *wv))
                if st == 0:
                    s.setdefault(f'relA{wn}', []).append(reliab(A1, A2, (trials, wv)))
    X = {st: {k: (np.concatenate(v) if k[:3] != 'rel' else np.nanmean(v)) for k, v in s.items()} for st, s in store.items()}
    y = np.r_[X[0.0]['y'], X[1.0]['y']]
    stv = np.r_[np.zeros(len(X[0.0]['y'])), np.ones(len(X[1.0]['y']))]
    w = np.where(stv == 0, 1.0, 0.5); one = np.ones_like(y); m0 = stv == 0
    for bw in WINS:
        for aw in list(WINS) + ['none']:
            for akind in ('all', 'half'):
                if aw == 'none' and akind == 'half':
                    continue
                cB = np.r_[X[0.0][f'cB{bw}'], X[1.0][f'cB{bw}']]; cB = zs(cB, cB[m0])
                if aw == 'none':
                    D = np.c_[one, cB, stv, stv * cB]
                    b = wls(D, y, w); c1, c2 = b[3], np.nan; rel = np.nan
                else:
                    key = 'cA' if akind == 'all' else 'cAh'
                    cA = np.r_[X[0.0][f'{key}{aw}'], X[1.0][f'{key}{aw}']]; cA = zs(cA, cA[m0])
                    D = np.c_[one, cB, cA, stv, stv * cB, stv * cA]
                    b = wls(D, y, w); c1, c2 = b[4], b[5]
                    rel = X[0.0][f'relA{aw}'] if akind == 'all' else (lambda r: r / (2 - r))(X[0.0][f'relA{aw}'])
                rows.append(dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, winB=bw,
                                 winA=aw, Aunits=akind, b1=b[1], c1=c1, c2=c2, relA=rel,
                                 r_AB=np.corrcoef(cB[m0], cA[m0])[0, 1] if aw != 'none' else np.nan))
    return rows


if __name__ == '__main__':
    lab = pd.read_csv(LABELS)
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            rows += pair_rows(d, sil, rec, code)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase6_partner.csv'), index=False)
    print(df.groupby(['winB', 'winA', 'Aunits'])[['c1', 'c2', 'b1', 'relA', 'r_AB']].median().round(3).to_string())
