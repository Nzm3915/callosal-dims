"""Phase 6-3 (B2'): is the receiver's inhibitory (FS) activity the carrier of the state dependence?

Callosal input mainly drives B's fast-spiking (FS) cells (Phase 5). If FS activity sets the gain of B's choice
representation, the slope of B's later confidence on its earlier confidence should depend on B's FS rate F:

    y = b0 + b1 cB + e F + f F cB                                  (fitted on CONTROL trials only)

Silencing A lowers F by dF (measured). With f and e from control trials, the gain account predicts, with no free
parameter fitted to stimulation trials,

    c1_pred = f dF ,     c0_pred = e dF

which is compared with the observed c0, c1 of  y = b0 + b1 cB + stim (c0 + c1 cB). Mediation check: add
e F + f F cB to the full model and see whether c1 shrinks toward 0. Same analysis with B's mean pyramidal rate P
as the carrier (comparison). F and P are mean z-scored rates during the laser window (-1.7..-0.9 s); cB is B's
signed confidence before the laser (-2.1..-1.7), y during the laser. FS units of B: putative FS units activated by
B's own laser (Phase 5 rule), disjoint from the pyramidal units used for cB and y.

usage: python phase6_fsgain.py CACHE_DIR RESULTS_DIR OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from phase6_ramp import Axes, zs, wls, rate

CACHE, R, OUT = sys.argv[1:4]
rng = np.random.default_rng(64)
PRE, LAS = (-2.1, -1.7), (-1.7, -0.9)

fr = pd.read_csv(os.path.join(R, 'units_fr_all.csv'))
fs = fr[(fr.celltype == 'FS') & (fr.fr0 >= 2)].copy(); fs['r1'] = fs.fr1 / fs.fr0; fs['r2'] = fs.fr2 / fs.fr0
fs = fs[np.maximum(fs.r1, fs.r2) >= 2]; fs['hemi'] = np.where(fs.r1 >= fs.r2, 'code1', 'code2')
lab = pd.read_csv(os.path.join(R, 'units_labeled.csv'))


def load(path):
    z = np.load(path)
    sess = os.path.basename(path)[:-4].replace('__', '/')
    uid = list(z['unit_ids'])
    l = lab[lab.session == sess]; hemi = dict(zip(l.unit, l.hemi))
    idx = {h: np.array([i for i, u in enumerate(uid) if hemi.get(u) == h]) for h in ('code1', 'code2')}
    f = fs[fs.session == sess]
    fsi = {h: np.array([uid.index(u) for u in f[f.hemi == h].unit]) for h in ('code1', 'code2')}
    ok = (z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) &
          (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05) &
          ((z['HitR'] + z['HitL'] + z['ErrR'] + z['ErrL']) > 0))
    return dict(sess=sess, mouse=sess.split('/')[0], c=z['counts'].astype(np.float32), t=z['edges'][:-1], idx=idx,
                fs=fsi, ok=ok, instR=(z['HitR'] + z['ErrR']) > 0, correct=(z['HitR'] + z['HitL']) > 0,
                pt=z['PhotostimulationType'])


def meanz(d, units, trials, ref, a, b):
    x = rate(d, units, trials, a, b); r0 = rate(d, units, ref, a, b)
    return ((x - r0.mean(0)) / (r0.std(0) + 1.0)).mean(1)


def pair_row(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ub, fb = d['idx'][rec], d['fs'][rec]
    if len(ub) < 6 or len(fb) < 2 or len(stim) < 20:
        return None
    fold = rng.permutation(len(ctrl_all)) % 2
    S = {k: [] for k in ('st', 'w', 'cB', 'y', 'F', 'P', 'Fpre')}
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        AB = Axes(d, ub, fit)
        for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
            S['st'].append(np.full(len(trials), st)); S['w'].append(np.full(len(trials), 1.0 if st == 0 else 0.5))
            S['cB'].append(AB.conf(trials, *PRE)); S['y'].append(AB.conf(trials, *LAS))
            S['F'].append(meanz(d, fb, trials, fit, *LAS)); S['Fpre'].append(meanz(d, fb, trials, fit, *PRE))
            S['P'].append(meanz(d, ub, trials, fit, *LAS))
    X = {k: np.concatenate(v) for k, v in S.items()}
    st, w, y = X['st'], X['w'], X['y']; m0 = st == 0; one = np.ones_like(y)
    cB = zs(X['cB'], X['cB'][m0])
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, nFS=len(fb), n_stim=len(stim))
    b = wls(np.c_[one, cB, st, st * cB], y, w)
    row.update(obs_b1=b[1], obs_c0=b[2], obs_c1=b[3])
    for nm in ('F', 'P', 'Fpre'):
        g = X[nm]; s0 = g[m0].std() + 1e-9; gz = (g - g[m0].mean()) / s0
        bc = np.linalg.lstsq(np.c_[one[m0], cB[m0], gz[m0], gz[m0] * cB[m0]], y[m0], rcond=None)[0]
        dG = gz[~m0].mean()
        row.update({f'{nm}_e': bc[2], f'{nm}_f': bc[3], f'{nm}_dz': dG, f'{nm}_ratio': g[~m0].mean() / (g[m0].mean() + 1e-9),
                    f'{nm}_c1pred': bc[3] * dG, f'{nm}_c0pred': bc[2] * dG})
        bm = wls(np.c_[one, cB, gz, gz * cB, st, st * cB], y, w)
        row.update({f'{nm}_med_c1': bm[5], f'{nm}_med_c0': bm[4], f'{nm}_med_f': bm[3]})
        # gain inside stimulation trials: does the slope depend on the carrier among stim trials?
        bs = np.linalg.lstsq(np.c_[one[~m0], cB[~m0], gz[~m0], gz[~m0] * cB[~m0]], y[~m0], rcond=None)[0]
        row[f'{nm}_f_stim'] = bs[3]
    # raw FS rate change during the laser (ratio of mean rates)
    row['FS_rate_ratio'] = (rate(d, fb, stim, *LAS).mean() / (rate(d, fb, ctrl_all, *LAS).mean() + 1e-9))
    return row


if __name__ == '__main__':
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            r = pair_row(d, sil, rec, code)
            if r: rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase6_fsgain.csv'), index=False)
    from scipy.stats import wilcoxon
    print(len(df), 'pairs,', df.mouse.nunique(), 'mice')
    for c in df.columns[6:]:
        v = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f'{c:16s} median {v.median():+.3f}  p={wilcoxon(v).pvalue:.1e}')
