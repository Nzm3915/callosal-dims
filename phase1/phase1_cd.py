"""Phase 1, step 2: reproduce Chen et al. 2021 Fig. 1 with the Phase 0 hemisphere labels,
and test whether the two hemispheres agree on the (wrong) choice in error trials.

Per session:
  1. control trials (no photostimulation, GoodTrials, no early lick, a normal 1.7-s delay, a lick)
  2. spike rates in 400-ms windows sliding in 50-ms steps, aligned to the go cue
  3. choice decoder CD per hemisphere = mean over delay windows of (mean rate "lick right" - "lick left"),
     from correct trials; 2-fold cross-validation so every correct trial is projected with a CD that
     did not see it. Error trials are projected with the CD from all correct trials.
  4. endpoint = projection in the last 400 ms before the go cue
  5. metrics: Spearman correlation of endpoints between hemispheres within each trial type (Fig. 1G),
     correlation over time, fraction of trials with the same predicted choice (Fig. 1H),
     prediction of the actual lick, and the same quantities on error trials.

usage: python phase1_cd.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr

CACHE, LABELS, OUT = sys.argv[1:4]
WIN = 8                      # 8 x 50 ms = 400 ms
MIN_UNITS, MIN_TRIALS = 5, 10
rng = np.random.default_rng(0)


def load(path, lab):
    z = np.load(path)
    sess = os.path.basename(path)[:-4].replace('__', '/')
    l = lab[lab.session == sess]
    uid = z['unit_ids']
    hemi = {u: h for u, h in zip(l.unit, l.hemi)}
    ia = np.array([i for i, u in enumerate(uid) if hemi.get(u) == 'code1'])
    ib = np.array([i for i, u in enumerate(uid) if hemi.get(u) == 'code2'])
    pt, good, early = z['PhotostimulationType'], z['GoodTrials'].astype(bool), z['LickEarly'].astype(bool)
    delay = z['cue'] - z['PoleOutTime']
    hitR, hitL, errR, errL = (z[k].astype(bool) for k in ['HitR', 'HitL', 'ErrR', 'ErrL'])
    ctrl = (pt == 0) & good & ~early & (np.abs(delay - 1.7) < 0.05) & (hitR | hitL | errR | errL)
    edges = z['edges']
    c = z['counts'].astype(np.float32)
    # rolling 400-ms window: window k covers bins k..k+7; its end time is edges[k+8]
    cs = np.cumsum(np.concatenate([np.zeros_like(c[..., :1]), c], axis=-1), axis=-1)
    rate = (cs[..., WIN:] - cs[..., :-WIN]) / (WIN * 0.05)          # trials x units x windows, Hz
    t_end = edges[WIN:]
    return dict(sess=sess, rate=rate, t_end=t_end, ia=ia, ib=ib, ctrl=ctrl,
                instR=hitR | errR, lickR=hitR | errL, correct=hitR | hitL)


def cd_vector(r, isR, delay_w):
    """r: trials x units x windows. CD = mean over delay windows of (mean R - mean L), unit norm."""
    d = r[isR][:, :, delay_w].mean(0) - r[~isR][:, :, delay_w].mean(0)
    v = d.mean(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def detrend(x, half=10):
    """Remove slow drift across the session: subtract the mean of the neighbouring trials
    (up to `half` on each side, the trial itself excluded). x: trials (in session order) x windows."""
    out = np.empty_like(x)
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        nb = np.r_[lo:i, i + 1:hi]
        out[i] = x[i] - x[nb].mean(0)
    return out


def boundary(pR, pL):
    vR, vL = pR.var() + 1e-9, pL.var() + 1e-9
    return (pR.mean() / vR + pL.mean() / vL) / (1 / vR + 1 / vL)


def analyse(d):
    ctrl = d['ctrl']
    corr_tr = np.where(ctrl & d['correct'])[0]
    err_tr = np.where(ctrl & ~d['correct'])[0]
    isR_all = d['instR']
    nR, nL = (isR_all[corr_tr]).sum(), (~isR_all[corr_tr]).sum()
    if min(len(d['ia']), len(d['ib'])) < MIN_UNITS or min(nR, nL) < MIN_TRIALS:
        return None, None
    t_end = d['t_end']
    delay_w = (t_end - 0.4 >= -1.7 - 1e-6) & (t_end <= 1e-6)          # windows fully inside the delay
    end_w = int(np.argmin(np.abs(t_end - 0.0)))                        # window ending at the go cue
    ntr = d['rate'].shape[0]
    proj = {h: np.full((ntr, len(t_end)), np.nan) for h in 'AB'}
    bnd = {h: {} for h in 'AB'}
    # 2-fold split stratified by trial type
    fold = np.zeros(ntr, int)
    for side in (True, False):
        idx = corr_tr[isR_all[corr_tr] == side]
        idx = rng.permutation(idx)
        fold[idx[len(idx) // 2:]] = 1
    for h, iu in (('A', d['ia']), ('B', d['ib'])):
        r = d['rate'][:, iu, :]
        for k in (0, 1):
            tr_tr = corr_tr[fold[corr_tr] != k]; te_tr = corr_tr[fold[corr_tr] == k]
            v = cd_vector(r[tr_tr], isR_all[tr_tr], delay_w)
            p_tr = np.einsum('tuw,u->tw', r[tr_tr], v)
            bnd[h][k] = boundary(p_tr[isR_all[tr_tr], end_w], p_tr[~isR_all[tr_tr], end_w])
            proj[h][te_tr] = np.einsum('tuw,u->tw', r[te_tr], v)
        v = cd_vector(r[corr_tr], isR_all[corr_tr], delay_w)
        p_all = np.einsum('tuw,u->tw', r[corr_tr], v)
        bnd[h]['all'] = boundary(p_all[isR_all[corr_tr], end_w], p_all[~isR_all[corr_tr], end_w])
        if len(err_tr):
            proj[h][err_tr] = np.einsum('tuw,u->tw', r[err_tr], v)

    def pred(h, trials):                                  # predicted "lick right"
        out = np.zeros(len(trials), bool)
        for j, t in enumerate(trials):
            b = bnd[h][fold[t]] if t in set(corr_tr) else bnd[h]['all']
            out[j] = proj[h][t, end_w] > b
        return out

    res = dict(session=d['sess'], nA=len(d['ia']), nB=len(d['ib']), n_corrR=int(nR), n_corrL=int(nL),
               n_err=len(err_tr))
    eA, eB = proj['A'][:, end_w], proj['B'][:, end_w]
    for name, side in (('R', True), ('L', False)):
        t = corr_tr[isR_all[corr_tr] == side]
        res[f'rho_{name}'] = spearmanr(eA[t], eB[t])[0]
        res[f'rho_{name}_shuf'] = spearmanr(eA[t], rng.permutation(eB[t]))[0]
        ta = np.concatenate([t, err_tr[isR_all[err_tr] == side]])   # Chen Fig. 1F used correct + error trials
        res[f'rho_{name}_withErr'] = spearmanr(eA[ta], eB[ta])[0]
    res['rho_both'] = spearmanr(eA[corr_tr], eB[corr_tr])[0]
    res['rho_both_shuf'] = spearmanr(eA[corr_tr], rng.permutation(eB[corr_tr]))[0]
    # correlation over time, within trial type (average of R and L); raw and after removing slow drift
    rt, rt_dt = [], []
    dt = {}
    for s in (True, False):
        t = corr_tr[isR_all[corr_tr] == s]                     # already in session order
        dt[s] = (detrend(proj['A'][t]), detrend(proj['B'][t]))
    for w in range(len(t_end)):
        vals = [spearmanr(proj['A'][corr_tr[isR_all[corr_tr] == s], w],
                          proj['B'][corr_tr[isR_all[corr_tr] == s], w])[0] for s in (True, False)]
        rt.append(np.nanmean(vals))
        rt_dt.append(np.nanmean([spearmanr(dt[s][0][:, w], dt[s][1][:, w])[0] for s in (True, False)]))
    res['rho_R_dt'] = spearmanr(dt[True][0][:, end_w], dt[True][1][:, end_w])[0]
    res['rho_L_dt'] = spearmanr(dt[False][0][:, end_w], dt[False][1][:, end_w])[0]
    pre_w = int(np.argmin(np.abs(t_end - (-3.0))))             # window ending at sample onset (pre-sample)
    res['rho_pre'] = rt[pre_w]; res['rho_pre_dt'] = rt_dt[pre_w]
    # agreement and choice prediction
    pa, pb = pred('A', corr_tr), pred('B', corr_tr)
    lick = d['lickR'][corr_tr]
    res['agree_corr'] = np.mean(pa == pb)
    res['predA_corr'] = np.mean(pa == lick); res['predB_corr'] = np.mean(pb == lick)
    if len(err_tr):
        ea, eb = pred('A', err_tr), pred('B', err_tr)
        le = d['lickR'][err_tr]
        res['agree_err'] = np.mean(ea == eb)
        res['predA_err'] = np.mean(ea == le); res['predB_err'] = np.mean(eb == le)
        # expected agreement if the two hemispheres were independent given the trial type
        exp = 0.0
        for s in (True, False):
            m = d['instR'][err_tr] == s
            if m.sum() == 0: continue
            qa, qb = ea[m].mean(), eb[m].mean()
            exp += m.sum() * (qa * qb + (1 - qa) * (1 - qb))
        res['agree_err_indep'] = exp / len(err_tr)
        res['n_err_agree'] = int((ea == eb).sum())
        res['n_err_both_actual'] = int(((ea == le) & (eb == le)).sum())
        res['n_err_both_instructed'] = int(((ea != le) & (eb != le)).sum())
    ex = dict(t_end=t_end, rt=np.array(rt), rt_dt=np.array(rt_dt), projA=proj['A'], projB=proj['B'], corr_tr=corr_tr, err_tr=err_tr,
              instR=isR_all, end_w=end_w)
    return res, ex


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    rows, ts, ts_dt, examples = [], [], [], {}
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        res, ex = analyse(load(p, lab))
        if res is None: continue
        rows.append(res); ts.append(ex['rt']); ts_dt.append(ex['rt_dt']); examples[res['session']] = ex
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase1_sessions.csv'), index=False)
    np.save(os.path.join(OUT, 'phase1_rho_time.npy'), np.array(ts))
    np.save(os.path.join(OUT, 'phase1_rho_time_detrended.npy'), np.array(ts_dt))
    np.save(os.path.join(OUT, 'phase1_t_end.npy'), ex['t_end'])
    # keep the example with the most units for plotting
    best = df.assign(n=df.nA + df.nB).sort_values('n').session.iloc[-1]
    np.savez(os.path.join(OUT, 'phase1_example.npz'), session=best, **examples[best])
    print(len(df), 'sessions')
    print(df[['rho_R', 'rho_L', 'rho_R_dt', 'rho_L_dt', 'rho_pre', 'rho_pre_dt', 'rho_both', 'rho_R_shuf', 'rho_L_shuf', 'rho_both_shuf',
              'agree_corr', 'predA_corr', 'predB_corr', 'agree_err', 'agree_err_indep',
              'predA_err', 'predB_err']].describe().round(3).T[['count', 'mean', '50%', 'std']])
