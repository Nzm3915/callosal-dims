"""Phase 6-6: after A is silenced, is A's plan rewritten from B? (Li et al. 2016; Inagaki et al. 2019)

Inagaki et al. found that after a perturbation ALM activity snaps back to one of two discrete end points, and Li et al.
that recovery after unilateral silencing relies on the other hemisphere. If B restores A, A's state at the end of the
delay should, on silence-A trials, depend MORE on B's state and LESS on A's own state before the laser:

    zA_end = a0 + a1 cA_pre + a2 cB_las + stim (d0 + d1 cA_pre + d2 cB_las)

cA_pre: A's confidence before the laser (-2.1..-1.7 s); cB_las: B's confidence during the laser (-1.7..-0.9 s, B is
not silenced); zA_end: A's confidence at the end of the delay (-0.4..0). "Rewrite" predicts d1 < 0 (A's own memory is
erased) and d2 > 0 (B's state is copied in). On error trials, B and A can disagree with the instruction; the agreement
of A's end-point sign with B's laser-window sign is compared between stim and control trials, separately on trials
where B "goes wrong" (B's laser-window sign opposes the instruction).

usage: python phase6_rewrite.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from phase6_ramp import load, Axes, zs, wls

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(68)


def pair_row(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ua, ub = d['idx'][sil], d['idx'][rec]
    if min(len(ua), len(ub)) < 6 or len(stim) < 20:
        return None
    fold = rng.permutation(len(ctrl_all)) % 2
    S = {k: [] for k in ('st', 'w', 'cA', 'cB', 'yA', 'sB', 'sA')}
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        AA, AB = Axes(d, ua, fit), Axes(d, ub, fit)
        for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
            S['st'].append(np.full(len(trials), st)); S['w'].append(np.full(len(trials), 1.0 if st == 0 else 0.5))
            S['cA'].append(AA.conf(trials, -2.1, -1.7)); S['cB'].append(AB.conf(trials, -1.7, -0.9))
            S['yA'].append(AA.conf(trials, -0.4, 0.0))
            S['sB'].append(AB.conf(trials, -1.7, -0.9) > 0); S['sA'].append(AA.conf(trials, -0.4, 0.0) > 0)
    X = {k: np.concatenate(v) for k, v in S.items()}
    st, w = X['st'], X['w']; m0 = st == 0; one = np.ones_like(st)
    cA, cB = zs(X['cA'], X['cA'][m0]), zs(X['cB'], X['cB'][m0])
    b = wls(np.c_[one, cA, cB, st, st * cA, st * cB], X['yA'], w)
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, n_stim=len(stim),
               a1=b[1], a2=b[2], d0=b[3], d1=b[4], d2=b[5])
    # sign agreement (signed confidence > 0 means "on the instructed side")
    agree = X['sA'] == X['sB']
    for nm, msk in (('ctrl', m0), ('stim', ~m0)):
        row[f'agree_{nm}'] = agree[msk].mean()
        bad = msk & ~X['sB']                                   # B on the wrong side during the laser
        row[f'agree_Bwrong_{nm}'] = agree[bad].mean() if bad.sum() >= 3 else np.nan
        row[f'n_Bwrong_{nm}'] = int(bad.sum())
    return row


if __name__ == '__main__':
    lab = pd.read_csv(LABELS)
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            r = pair_row(d, sil, rec, code)
            if r: rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase6_rewrite.csv'), index=False)
    from scipy.stats import wilcoxon
    print(len(df), 'pairs')
    for c in ('a1', 'a2', 'd0', 'd1', 'd2', 'agree_ctrl', 'agree_stim', 'agree_Bwrong_ctrl', 'agree_Bwrong_stim'):
        print(f'{c:18s} median {df[c].median():+.3f}  p0={wilcoxon(df[c].dropna()).pvalue:.1e}')
    print('agree stim - ctrl', (df.agree_stim - df.agree_ctrl).median().round(3), f'p={wilcoxon(df.agree_stim - df.agree_ctrl).pvalue:.1e}')
    v = (df.agree_Bwrong_stim - df.agree_Bwrong_ctrl).dropna()
    print('B-wrong agreement stim - ctrl', v.median().round(3), f'p={wilcoxon(v).pvalue:.1e}', len(v))
