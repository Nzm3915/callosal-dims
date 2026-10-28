"""Phase 6-2: is the non-selective ramping mode the carrier of the state dependence?

Finkelstein et al. (2021, Fig. 5f): in single trials, a stronger non-selective ramping mode before a distractor made
the plan more robust (fewer switches). In their model the ramp moves the two choice attractors apart. Translated to
this data set (silence A at delay onset, look at B):

 (a) Robustness of B's choice axis. Regression over control + silence-A trials, held-out decoders (2 folds):
        y = b0 + b1 cB + b2 cA + b3 rB + b4 rA + stim * (c0 + c1 cB + c2 cA + c3 rB + c4 rA)
     cB, cA: signed confidence along each hemisphere's CD just before the laser (-2.1..-1.7 s);
     rB, rA: ramping-mode amplitude in the same window (RM = late delay - pre-sample, both trial types pooled,
     orthogonal to CD; Finkelstein's definition). y: B's confidence during the laser (-1.7..-0.9).
     Ramp hypothesis: c3 > 0 (strong ramp protects), and c1 moves toward 0 once rB is in the model.
 (b) Behaviour (Finkelstein's procedure): stim trials split into ramp tertiles of rB (and of rA); in each tertile the
     error rate on stim trials minus the error rate on control trials of the same tertile.
 (c) Trial-wise gain during the laser: among stim trials, y depends on cB more steeply when B kept more of its ramp?
        y = a + b cB + e m + f m cB     (stim trials; same model on control trials for comparison)
     m = B's ramping-mode amplitude during the laser, computed from a random half of B's units; y and cB from the
     other half (no shared spike-count noise). Gain hypothesis: f_stim > 0 and f_stim > f_ctrl.

usage: python phase6_ramp.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.optimize import minimize

CACHE, LABELS, OUT = (sys.argv[1:4] + [None] * 3)[:3]
rng = np.random.default_rng(62)
PRE, LAS = (-2.1, -1.7), (-1.7, -0.9)


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
    return dict(sess=sess, mouse=sess.split('/')[0], c=z['counts'].astype(np.float32), t=z['edges'][:-1], idx=idx,
                ok=ok, instR=(z['HitR'] + z['ErrR']) > 0, correct=(z['HitR'] + z['HitL']) > 0,
                pt=z['PhotostimulationType'])


def rate(d, units, trials, a, b):
    w = (d['t'] >= a - 1e-6) & (d['t'] < b - 1e-6)
    return d['c'][np.ix_(trials, units)][:, :, w].sum(-1) / (w.sum() * 0.05)


class Axes:
    """CD and ramping mode of a unit set, fitted on correct control trials"""
    def __init__(self, d, units, fit):
        R, L = fit[d['instR'][fit]], fit[~d['instR'][fit]]
        cd = rate(d, units, R, -1.7, 0).mean(0) - rate(d, units, L, -1.7, 0).mean(0)
        self.cd = cd / (np.linalg.norm(cd) + 1e-12)
        late = 0.5 * (rate(d, units, R, -0.5, 0).mean(0) + rate(d, units, L, -0.5, 0).mean(0))
        pre = 0.5 * (rate(d, units, R, -3.6, -3.0).mean(0) + rate(d, units, L, -3.6, -3.0).mean(0))
        rm = late - pre; rm = rm - (rm @ self.cd) * self.cd
        self.rm = rm / (np.linalg.norm(rm) + 1e-12)
        self.units, self.d, self.fit = units, d, fit
        self.cal = {}

    def conf(self, trials, a, b):
        d = self.d
        if (a, b) not in self.cal:
            R, L = self.fit[d['instR'][self.fit]], self.fit[~d['instR'][self.fit]]
            pR, pL = rate(d, self.units, R, a, b) @ self.cd, rate(d, self.units, L, a, b) @ self.cd
            self.cal[(a, b)] = ((pR.mean() + pL.mean()) / 2, np.sqrt((pR.var() + pL.var()) / 2) + 1e-6)
        bnd, sd = self.cal[(a, b)]
        s = np.where(d['instR'][trials], 1.0, -1.0)
        return s * (rate(d, self.units, trials, a, b) @ self.cd - bnd) / sd

    def ramp(self, trials, a, b):
        return rate(self.d, self.units, trials, a, b) @ self.rm


def zs(x, ref):
    return (x - ref.mean()) / (ref.std() + 1e-9)


def wls(D, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)[0]


def pair_rows(d, sil, rec, code):
    ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]
    stim = np.where(d['ok'] & (d['pt'] == code))[0]
    ub, ua = d['idx'][rec], d['idx'][sil]
    if min(len(ua), len(ub)) < 6 or len(stim) < 20:
        return None
    fold = rng.permutation(len(ctrl_all)) % 2
    cols = {k: [] for k in ('st', 'w', 'cB', 'cA', 'rB', 'rA', 'y', 'err', 'yh', 'cBh', 'mh', 'mh_c')}
    hsplit = rng.permutation(len(ub)); h1, h2 = ub[hsplit[: len(ub) // 2]], ub[hsplit[len(ub) // 2:]]
    for f in (0, 1):
        fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
        AB, AA = Axes(d, ub, fit), Axes(d, ua, fit)
        H1, H2 = Axes(d, h1, fit), Axes(d, h2, fit)             # neuron halves: H1 for choice, H2 for the ramp
        for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
            n = len(trials)
            cols['st'].append(np.full(n, st)); cols['w'].append(np.full(n, 1.0 if st == 0 else 0.5))
            cols['cB'].append(AB.conf(trials, *PRE)); cols['cA'].append(AA.conf(trials, *PRE))
            cols['rB'].append(AB.ramp(trials, *PRE)); cols['rA'].append(AA.ramp(trials, *PRE))
            cols['y'].append(AB.conf(trials, *LAS)); cols['err'].append(~d['correct'][trials])
            cols['yh'].append(H1.conf(trials, *LAS)); cols['cBh'].append(H1.conf(trials, *PRE))
            cols['mh'].append(H2.ramp(trials, *LAS)); cols['mh_c'].append(H2.ramp(trials, *PRE))
    X = {k: np.concatenate(v) for k, v in cols.items()}
    st, w = X['st'], X['w']; m0 = st == 0
    for k in ('cB', 'cA', 'rB', 'rA', 'cBh'):
        X[k] = zs(X[k], X[k][m0])
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, n_stim=len(stim), n_ctrl=len(ctrl_all),
               r_cB_rB=np.corrcoef(X['cB'][m0], X['rB'][m0])[0, 1], r_rA_rB=np.corrcoef(X['rA'][m0], X['rB'][m0])[0, 1])
    one = np.ones_like(st)
    # (a) Phase-4 model, then with ramp terms
    D4 = np.c_[one, X['cB'], X['cA'], st, st * X['cB'], st * X['cA']]
    b = wls(D4, X['y'], w)
    row.update(p4_b1=b[1], p4_c0=b[3], p4_c1=b[4], p4_c2=b[5])
    D6 = np.c_[one, X['cB'], X['cA'], X['rB'], X['rA'], st, st * X['cB'], st * X['cA'], st * X['rB'], st * X['rA']]
    b = wls(D6, X['y'], w)
    for k, nm in enumerate(['b0', 'b1', 'b2', 'b3', 'b4', 'c0', 'c1', 'c2', 'c3', 'c4']):
        row[f'r_{nm}'] = b[k]
    # (b) behaviour by ramp tertile (tertiles from all trials of the session, as in Finkelstein)
    for nm in ('rB', 'rA', 'cB'):
        q = np.quantile(X[nm], [1 / 3, 2 / 3]); bins = np.digitize(X[nm], q)
        for j in range(3):
            s_ = (bins == j) & ~m0; c_ = (bins == j) & m0
            row[f'excess_err_{nm}{j}'] = (X['err'][s_].mean() - X['err'][c_].mean()) if s_.sum() >= 4 and c_.sum() >= 4 else np.nan
    # logistic: error ~ cB + rB + stim*(cB + rB)
    E = X['err'].astype(float)
    Dl = np.c_[one, X['cB'], X['rB'], st, st * X['cB'], st * X['rB']]
    def nll(bb):
        p = 1 / (1 + np.exp(-np.clip(Dl @ bb, -30, 30))); p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(w * (E * np.log(p) + (1 - E) * np.log(1 - p))) + 1e-2 * np.sum(bb[1:] ** 2)
    bl = minimize(nll, np.zeros(Dl.shape[1]), method='L-BFGS-B').x
    row.update(lg_cB=bl[1], lg_rB=bl[2], lg_stim=bl[3], lg_stim_cB=bl[4], lg_stim_rB=bl[5])
    # (c) trial-wise gain with a neuron split
    for nm, msk in (('stim', ~m0), ('ctrl', m0)):
        mm = X['mh'][msk]; mm = (mm - X['mh'][m0].mean()) / (X['mh'][m0].std() + 1e-9)
        cb = X['cBh'][msk]
        Dc = np.c_[np.ones(msk.sum()), cb, mm, mm * cb]
        bc = np.linalg.lstsq(Dc, X['yh'][msk], rcond=None)[0]
        row.update({f'g_{nm}_b': bc[1], f'g_{nm}_e': bc[2], f'g_{nm}_f': bc[3]})
    # ramp loss on stim trials (during the laser, half-2 units), in control s.d.
    row['ramp_drop_sd'] = (X['mh'][~m0].mean() - X['mh'][m0].mean()) / (X['mh'][m0].std() + 1e-9)
    row['ramp_own_sd'] = (X['mh'][m0].mean() - X['mh_c'][m0].mean()) / (X['mh'][m0].std() + 1e-9)
    return row


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
            r = pair_rows(d, sil, rec, code)
            if r: rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase6_ramp.csv'), index=False)
    from scipy.stats import wilcoxon
    print(len(df), 'pairs')
    for c in df.columns[6:]:
        v = df[c].dropna()
        if len(v) > 10:
            print(f'{c:22s} median {v.median():+.3f}  p={wilcoxon(v).pvalue:.1e}')
