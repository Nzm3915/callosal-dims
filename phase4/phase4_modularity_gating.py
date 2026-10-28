"""Phase 4a: individual differences (modularity, per mouse) and a model-free test of state-dependent gating.

1. Modularity (Chen et al. 2021): fraction of trial-type selectivity a hemisphere keeps during the first 0.8 s of
   the delay while the other hemisphere is silenced; recovery: fraction of selectivity in the last 200 ms before
   the go cue after silencing, averaged over both hemispheres. Delay-selective labeled pyramidal units only
   (rank-sum p < 0.05 on delay spike counts, correct control trials). Chen's exclusion criteria applied.

2. Model-free gating test. For every trial, the state of each hemisphere just before the laser comes on
   (400-ms window ending at delay onset) is read out as a signed confidence along its choice decoder:
       conf = s * (CD.r - boundary) / sd,   s = +1 for "lick right" trials, -1 for "lick left"
   (CD, boundary and sd from correct control trials of the other cross-validation fold).
   Outcome: B's signed confidence during the laser epoch (first 0.8 s of the delay) and at the end of the delay.
   Per hemisphere pair, ordinary least squares over control and "silence A" trials:
       y = b0 + b1 cB + b2 cA + stim * (c0 + c1 cB + c2 cA)
   c0: average effect of silencing A on B. Precision/state gating (H4) predicts
       c1 > 0  (B listens less to A when B is already sure -> the loss of A hurts B less)
       c2 < 0  (losing A hurts B more when A was sure)
   Control trials give the baseline relation (persistence, regression to the mean), so only the interaction
   with stim is attributed to A's input.

usage: python phase4_modularity_gating.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import ranksums

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(4)


def load(path, lab):
    z = np.load(path)
    sess = os.path.basename(path)[:-4].replace('__', '/')
    l = lab[lab.session == sess]
    hemi = dict(zip(l.unit, l.hemi))
    uid = z['unit_ids']
    idx = {h: np.array([i for i, u in enumerate(uid) if hemi.get(u) == h]) for h in ('code1', 'code2')}
    ok = (z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) &
          (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05) &
          ((z['HitR'] + z['HitL'] + z['ErrR'] + z['ErrL']) > 0))
    c = z['counts'].astype(np.float32)
    t0 = z['edges'][:-1]
    return dict(sess=sess, mouse=sess.split('/')[0], c=c, t0=t0, idx=idx, ok=ok,
                instR=(z['HitR'] + z['ErrR']) > 0, correct=(z['HitR'] + z['HitL']) > 0,
                pt=z['PhotostimulationType'])


def rate(d, units, trials, a, b):
    w = (d['t0'] >= a - 1e-6) & (d['t0'] < b - 1e-6)
    return d['c'][np.ix_(trials, units)][:, :, w].sum(-1) / (w.sum() * 0.05)       # trials x units, Hz


def selective_units(d, h):
    u = d['idx'][h]
    cc = np.where(d['ok'] & (d['pt'] == 0) & d['correct'])[0]
    R, L = cc[d['instR'][cc]], cc[~d['instR'][cc]]
    rR, rL = rate(d, u, R, -1.7, 0), rate(d, u, L, -1.7, 0)
    keep, pref = [], []
    for i in range(len(u)):
        if ranksums(rR[:, i], rL[:, i]).pvalue < 0.05:
            keep.append(i); pref.append(rR[:, i].mean() > rL[:, i].mean())
    return u[keep], np.array(pref, bool)


def selectivity(d, units, pref, trials, a, b):
    """mean over units of (rate on preferred-type trials - rate on non-preferred-type trials)"""
    R, L = trials[d['instR'][trials]], trials[~d['instR'][trials]]
    if min(len(R), len(L)) < 3: return np.nan
    dR, dL = rate(d, units, R, a, b).mean(0), rate(d, units, L, a, b).mean(0)
    return np.mean(np.where(pref, dR - dL, dL - dR))


def modularity_rows(d):
    sel = {h: selective_units(d, h) for h in ('code1', 'code2')}
    ctrl = np.where(d['ok'] & (d['pt'] == 0))[0]
    out = []
    for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
        stim = np.where(d['ok'] & (d['pt'] == code))[0]
        u, pref = sel[rec]
        row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, n_sel_rec=len(u),
                   n_sel_sil=len(sel[sil][0]))
        if len(u) >= 1:
            se, sl = selectivity(d, u, pref, ctrl, -1.7, -0.9), selectivity(d, u, pref, ctrl, -0.9, 0)
            row.update(ctrl_sel_early=se, ctrl_sel_late=sl,
                       modularity=selectivity(d, u, pref, stim, -1.7, -0.9) / se)
            # recovery in both hemispheres (last 200 ms), averaged as in Chen et al.
            recs = []
            for h in ('code1', 'code2'):
                uh, ph = sel[h]
                if len(uh) == 0: continue
                recs.append(selectivity(d, uh, ph, stim, -0.2, 0) / selectivity(d, uh, ph, ctrl, -0.2, 0))
            row['recovery'] = np.nanmean(recs) if recs else np.nan
        out.append(row)
    return out


def cd_conf(d, h, fit_tr, trials, a, b):
    """signed confidence along h's CD, CD/boundary/sd from correct control trials fit_tr"""
    u = d['idx'][h]
    R, L = fit_tr[d['instR'][fit_tr]], fit_tr[~d['instR'][fit_tr]]
    xR, xL = rate(d, u, R, a, b), rate(d, u, L, a, b)
    # CD from the delay epoch (as in Chen), applied to window (a, b)
    cd = rate(d, u, R, -1.7, 0).mean(0) - rate(d, u, L, -1.7, 0).mean(0)
    cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = xR @ cd, xL @ cd
    bnd = (pR.mean() + pL.mean()) / 2
    sd = np.sqrt((pR.var() + pL.var()) / 2) + 1e-6
    p = rate(d, u, trials, a, b) @ cd
    s = np.where(d['instR'][trials], 1.0, -1.0)
    return s * (p - bnd) / sd


def gating_rows(d):
    out = []
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
        if min(len(d['idx'][sil]), len(d['idx'][rec])) < 5: continue
        stim = np.where(d['ok'] & (d['pt'] == code))[0]
        if len(stim) < 20: continue
        fold = rng.permutation(len(ctrl_all)) % 2
        Xs, Ys1, Ys2 = [], [], []
        for f in (0, 1):
            fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
            ev_ctrl = ctrl_all[fold == f]
            for trials, st in ((ev_ctrl, 0.0), (stim, 0.5)):          # stim trials split their weight over folds
                cB = cd_conf(d, rec, fit, trials, -2.1, -1.7)
                cA = cd_conf(d, sil, fit, trials, -2.1, -1.7)
                y1 = cd_conf(d, rec, fit, trials, -1.7, -0.9)
                y2 = cd_conf(d, rec, fit, trials, -0.4, 0.0)
                w = np.full(len(trials), 1.0 if st == 0 else 0.5)
                Xs.append(np.c_[np.full(len(trials), st > 0), cB, cA, w]); Ys1.append(y1); Ys2.append(y2)
        X = np.concatenate(Xs); y1 = np.concatenate(Ys1); y2 = np.concatenate(Ys2)
        stim_ind, cB, cA, w = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # standardise predictors using control trials
        m = stim_ind == 0
        cB = (cB - cB[m].mean()) / (cB[m].std() + 1e-9); cA = (cA - cA[m].mean()) / (cA[m].std() + 1e-9)
        D = np.c_[np.ones_like(cB), cB, cA, stim_ind, stim_ind * cB, stim_ind * cA]
        row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, n_stim=len(stim),
                   n_ctrl=len(ctrl_all), r_cA_cB=np.corrcoef(cA[m], cB[m])[0, 1])
        for name, y in (('laser', y1), ('end', y2)):
            sw = np.sqrt(w)
            beta = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)[0]
            for k, nm in enumerate(['b0', 'b1_cB', 'b2_cA', 'c0_stim', 'c1_stim_x_cB', 'c2_stim_x_cA']):
                row[f'{name}_{nm}'] = beta[k]
        out.append(row)
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    mods, gates = [], []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        mods += modularity_rows(d)
        gates += gating_rows(d)
    md = pd.DataFrame(mods); gd = pd.DataFrame(gates)
    md.to_csv(os.path.join(OUT, 'phase4_modularity.csv'), index=False)
    gd.to_csv(os.path.join(OUT, 'phase4_gating_test.csv'), index=False)
    print('modularity rows', len(md), 'gating pairs', len(gd))
