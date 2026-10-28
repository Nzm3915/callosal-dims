"""Phase 7-1: is the rest of the state dependence (c1) the loss of an interhemispheric round trip?

Round-trip hypothesis: B's trial-specific state is partly carried forward through A (B -> A -> B). On control trials,
A's state during the laser window then depends on B's earlier state cB (beyond A's own earlier state cA). Silencing A
removes A's input w * xA(t) to B; the removed amount correlates with cB, so c1 < 0 without any gain change.

Part 1 (model-light ratio estimator). Regress A's confidence during the laser window xA on [1, cB, cA] separately on
control and on silence-A trials (beta). If B's loss is w times the loss of A's choice signal,
    c0 = w (beta0_stim - beta0_ctrl),   c2 = w (betaA_stim - betaA_ctrl),   c1_loop = w (betaB_stim - betaB_ctrl)
w is calibrated from c0 or from c2 (two independent calibrations; their agreement is a check), and c1_loop is the
prediction. Observational w (the coupling seen in control trials) is reported for comparison.

Part 2 (coupled dynamical model). Both hemispheres' choice axes (400-ms windows, 100-ms steps) follow
    xA(t+1) = xA + aA(t) + kAA xA + kAB xB          xB(t+1) = xB + aB(t) + kBB xB + kBA xA
On silence-A trials, A's observed trajectory is used (A is clamped by the laser). The couplings are fitted on control
trials plus silencing trials (one-step regression; the silencing trials make the A->B coupling identifiable because A
is driven far from its normal range). The model is run forward from the pre-laser state on held-out trials and scored
with the Phase 4 regression. Two counterfactuals isolate the round trip: (i) kAB = 0 in A's equation on control
trials (A no longer receives B, so A's trajectory carries no B history), (ii) A's control trajectory replaced by its
trial-type mean (A carries no trial-specific information at all).

usage: python phase7_loop.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase6'))
from phase6_ramp import load, Axes, zs, wls

CACHE, LABELS, OUT = sys.argv[1:4]
rng = np.random.default_rng(71)
PRE, LAS = (-2.1, -1.7), (-1.7, -0.9)


def regress(D, y, w=None):
    w = np.ones(len(y)) if w is None else w
    return wls(D, y, w)


def part1(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ua, ub = d['idx'][sil], d['idx'][rec]
    if min(len(ua), len(ub)) < 6 or len(stim) < 20:
        return None
    fold = rng.permutation(len(ctrl_all)) % 2
    S = {k: [] for k in ('st', 'w', 'cA', 'cB', 'y', 'xA')}
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        AA, AB = Axes(d, ua, fit), Axes(d, ub, fit)
        for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
            S['st'].append(np.full(len(trials), st)); S['w'].append(np.full(len(trials), 1.0 if st == 0 else 0.5))
            S['cA'].append(AA.conf(trials, *PRE)); S['cB'].append(AB.conf(trials, *PRE))
            S['y'].append(AB.conf(trials, *LAS)); S['xA'].append(AA.conf(trials, *LAS))
    X = {k: np.concatenate(v) for k, v in S.items()}
    st, w = X['st'], X['w']; m0 = st == 0; one = np.ones_like(st)
    cA, cB = zs(X['cA'], X['cA'][m0]), zs(X['cB'], X['cB'][m0])
    b = regress(np.c_[one, cB, cA, st, st * cB, st * cA], X['y'], w)
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, n_stim=len(stim),
               b1=b[1], b2=b[2], c0=b[3], c1=b[4], c2=b[5])
    bc = regress(np.c_[one[m0], cB[m0], cA[m0]], X['xA'][m0]); bs = regress(np.c_[one[~m0], cB[~m0], cA[~m0]], X['xA'][~m0])
    d0, dB, dA = bs - bc
    row.update(betaB_ctrl=bc[1], betaA_ctrl=bc[2], beta0_ctrl=bc[0], beta0_stim=bs[0], betaB_stim=bs[1], betaA_stim=bs[2],
               dbeta0=d0, dbetaB=dB, dbetaA=dA)
    w_c0 = row['c0'] / d0 if abs(d0) > 1e-6 else np.nan
    w_c2 = row['c2'] / dA if abs(dA) > 0.05 else np.nan
    # observational coupling: does xA (laser window) predict y beyond cB, cA on control trials?
    bo = regress(np.c_[one[m0], cB[m0], cA[m0], zs(X['xA'][m0], X['xA'][m0])], X['y'][m0])
    row.update(w_c0=w_c0, w_c2=w_c2, w_obs=bo[3] / (X['xA'][m0].std() + 1e-9),
               c1_loop_c0=w_c0 * dB, c1_loop_c2=w_c2 * dB if np.isfinite(w_c2) else np.nan,
               c2_pred_from_c0=w_c0 * dA)
    return row


# ------------------------------------------------------------------ part 2: coupled dynamical model
def windows(d):
    c = d['c']
    cs = np.cumsum(np.concatenate([np.zeros_like(c[..., :1]), c], -1), -1)
    r = ((cs[..., 8:] - cs[..., :-8]) / 0.4)[..., ::2]
    tw = d['t'][8:][::2] + 0.05                                      # window end times
    return r, tw


def proj_axis(r, d, units, fit, D):
    rr = r[:, units, :]
    R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
    cd = rr[R][:, :, D].mean((0, 2)) - rr[L][:, :, D].mean((0, 2)); cd /= np.linalg.norm(cd) + 1e-12
    pR, pL = np.einsum('tub,u->tb', rr[R], cd), np.einsum('tub,u->tb', rr[L], cd)
    bnd = (pR.mean(0) + pL.mean(0)) / 2; sd = np.sqrt((pR.var(0) + pL.var(0)) / 2).mean() + 1e-6
    s = np.where(d['instR'], 1.0, -1.0)[:, None]
    return s * (np.einsum('tub,u->tb', rr, cd) - bnd) / sd             # all trials x windows


def part2(d, sil, rec, code, r, tw):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ua, ub = d['idx'][sil], d['idx'][rec]
    if min(len(ua), len(ub)) < 6 or len(stim) < 20:
        return None
    D = np.where((tw > -1.3 - 1e-6) & (tw <= 1e-6))[0]
    pre = int(np.argmin(np.abs(tw - (-1.7)))); steps = np.arange(pre, pre + 8)       # 8 x 100 ms = laser epoch
    out = dict(session=d['sess'], silenced=sil, receiving=rec)
    acc = {}
    for f in (0, 1):
        # split control and stim trials in halves (balanced by trial type)
        sets = {}
        for nm, pool in (('c', ctrl_all), ('s', stim)):
            a_, b_ = [], []
            for side in (True, False):
                p = np.random.default_rng(72).permutation(pool[d['instR'][pool] == side]); h = len(p) // 2
                a_.append(p[:h]); b_.append(p[h:])
            A_, B_ = np.concatenate(a_), np.concatenate(b_)
            sets[nm] = (A_, B_) if f == 0 else (B_, A_)
        fit = sets['c'][0][d['correct'][sets['c'][0]]]
        xA = proj_axis(r, d, ua, fit, D); xB = proj_axis(r, d, ub, fit, D)
        tr_c, tr_s = sets['c'][0], sets['s'][0]; te_c, te_s = sets['c'][1], sets['s'][1]
        # one-step regression with time-specific offsets; A's equation only from control trials (A clamped on stim)
        def design(trials, own, other, with_other=True):
            Xs, Ys = [], []
            for j, tt in enumerate(steps):
                T = np.zeros((len(trials), len(steps))); T[:, j] = 1
                cols = [T, own[trials, tt:tt + 1]] + ([other[trials, tt:tt + 1]] if with_other else [])
                Xs.append(np.hstack(cols)); Ys.append(own[trials, tt + 1] - own[trials, tt])
            return np.vstack(Xs), np.concatenate(Ys)
        XB, YB = design(np.r_[tr_c, tr_s], xB, xA)
        wB = np.linalg.lstsq(XB, YB, rcond=None)[0]                   # [aB(t)..., kBB, kBA]
        XA, YA = design(tr_c, xA, xB)
        wA = np.linalg.lstsq(XA, YA, rcond=None)[0]                   # [aA(t)..., kAA, kAB]
        # B-alone model with a stim offset (null: no coupling)
        Xn, Yn = design(np.r_[tr_c, tr_s], xB, xA, with_other=False)
        stim_col = np.concatenate([np.r_[np.zeros(len(tr_c)), np.ones(len(tr_s))]] * len(steps))[:, None]
        wN = np.linalg.lstsq(np.c_[Xn, stim_col], Yn, rcond=None)[0]
        te = np.r_[te_c, te_s]; is_s = np.r_[np.zeros(len(te_c)), np.ones(len(te_s))]
        x0B = xB[te, pre]
        meanA = {s: xA[te_c[d['instR'][te_c] == s]].mean(0) for s in (True, False)}

        def run(mode):
            b = xB[te, pre].copy(); a = xA[te, pre].copy(); traj = []
            for j, tt in enumerate(steps):
                T = np.zeros((len(te), len(steps))); T[:, j] = 1
                # A: observed on stim trials (clamped); simulated on control trials
                a_obs = xA[te, tt]
                if mode == 'null':
                    b = b + np.c_[T, b[:, None], is_s[:, None]] @ wN
                else:
                    a_in = np.where(is_s == 1, a_obs, a)
                    if mode == 'meanA':                            # A carries no trial-specific information
                        mA = np.array([meanA[s][tt] for s in d['instR'][te]])
                        a_in = np.where(is_s == 1, a_obs, mA)
                    b_new = b + np.c_[T, b[:, None], a_in[:, None]] @ wB
                    kAB = 0.0 if mode == 'noBtoA' else wA[-1]
                    a = a + np.c_[T, a[:, None], np.zeros((len(te), 1))] @ np.r_[wA[:-1], 0.0] + kAB * b
                    b = b_new
                traj.append(b)
            y = np.stack(traj, 1).mean(1)
            m = is_s == 0; xs = (x0B - x0B[m].mean()) / (x0B[m].std() + 1e-9)
            bb = np.linalg.lstsq(np.c_[np.ones_like(xs), xs, is_s, is_s * xs], y, rcond=None)[0]
            return bb
        yobs = xB[te][:, steps + 1].mean(1)
        m = is_s == 0; xs = (x0B - x0B[m].mean()) / (x0B[m].std() + 1e-9)
        bo = np.linalg.lstsq(np.c_[np.ones_like(xs), xs, is_s, is_s * xs], yobs, rcond=None)[0]
        for nm, bb in (('obs', bo), ('full', run('full')), ('noBtoA', run('noBtoA')), ('meanA', run('meanA')), ('null', run('null'))):
            acc.setdefault(f'{nm}_c0', []).append(bb[2]); acc.setdefault(f'{nm}_c1', []).append(bb[3]); acc.setdefault(f'{nm}_b1', []).append(bb[1])
        acc.setdefault('kBA', []).append(wB[-1]); acc.setdefault('kAB', []).append(wA[-1])
        acc.setdefault('kBB', []).append(wB[-2]); acc.setdefault('kAA', []).append(wA[-2])
    out.update({k: float(np.mean(v)) for k, v in acc.items()})
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    r1, r2 = [], []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        r, tw = windows(d)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            a = part1(d, sil, rec, code)
            if a: r1.append(a)
            b = part2(d, sil, rec, code, r, tw)
            if b: r2.append(b)
    d1, d2 = pd.DataFrame(r1), pd.DataFrame(r2)
    d1.to_csv(os.path.join(OUT, 'phase7_loop_ratio.csv'), index=False)
    d2.to_csv(os.path.join(OUT, 'phase7_loop_model.csv'), index=False)
    from scipy.stats import wilcoxon
    print('PART 1', len(d1), 'pairs')
    for c in ('c0', 'c1', 'c2', 'dbeta0', 'dbetaB', 'dbetaA', 'betaB_ctrl', 'w_c0', 'w_c2', 'w_obs', 'c1_loop_c0', 'c1_loop_c2', 'c2_pred_from_c0'):
        v = d1[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f'{c:16s} median {v.median():+.3f}  p0={wilcoxon(v).pvalue:.1e}  n={len(v)}')
    print('PART 2', len(d2), 'pairs')
    for c in [x for x in d2.columns if x.endswith(('_c0', '_c1', '_b1'))] + ['kBA', 'kAB', 'kBB', 'kAA']:
        print(f'{c:12s} median {d2[c].median():+.3f}')
