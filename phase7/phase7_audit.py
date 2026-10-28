"""Phase 7-3: red-team audit of the main claims.

A1  Pseudo-replication. Most tests so far treated hemisphere pairs as independent, but pairs come from the same
    session (2 per session) and sessions from the same mouse (median 3). Every headline effect is re-tested with
    one value per mouse (median over its pairs), Wilcoxon over mice; pooled ratios get a bootstrap over mice.
A2  Rewrite test without the attenuation caveat. Phase 6-6 used B's state DURING the laser, which the laser itself
    degrades. Here both predictors are pre-laser: zA_end = a0 + a1 cA_pre + a2 cB_pre + stim (d0 + d1 cA_pre + d2 cB_pre).
    Rewrite-from-B predicts d2 > 0 and d1 < 0.
A3  Selection of units. Units silenced by both or neither unilateral condition were left unlabelled; how many could
    be strongly callosally driven cells that the analyses miss?

usage: python phase7_audit.py CACHE_DIR RESULTS_DIR
"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase6'))
from phase6_ramp import load, Axes, zs, wls

CACHE, R = sys.argv[1:3]
out = {}


def mouse_of(s):
    return s.split('/')[0]


def mouse_test(df, col, ref=None):
    v = df.assign(mouse=df.session.map(mouse_of))
    x = v[col] - (v[ref] if ref else 0)
    m = x.groupby(v.mouse).median().dropna()
    return dict(n_mice=len(m), median=float(m.median()), p=float(wilcoxon(m).pvalue) if len(m) > 5 else np.nan,
                n_pairs=int(x.notna().sum()), p_pairs=float(wilcoxon(x.dropna()).pvalue))


# ---------------- A1
p2 = pd.read_csv(os.path.join(R, 'phase2', 'phase2_pairs.csv')); p2d = p2[p2.epoch == 'during']
p6r = pd.read_csv(os.path.join(R, 'phase6', 'phase6_ramp.csv'))
p6m = pd.read_csv(os.path.join(R, 'phase6', 'phase6_modes.csv'))
ov = pd.read_csv(os.path.join(R, 'phase6', 'phase6_dlag_overlap.csv')); ov = ov[(ov.rel_den > 0) & (ov.pa > 0)]
rw = pd.read_csv(os.path.join(R, 'phase6', 'phase6_rewrite.csv'))
fg = pd.read_csv(os.path.join(R, 'phase6', 'phase6_fsgain.csv'))
lp = pd.read_csv(os.path.join(R, 'phase7', 'phase7_loop_ratio.csv'))
tests = {
    'phase2: receiving(3) - correlated(3) subspace': (p2d, 'f_rec3', 'f_corr3'),
    'phase4: c0 (silencing lowers B confidence)': (p6r, 'p4_c0', None),
    'phase4: c1 (state dependence)': (p6r, 'p4_c1', None),
    'phase6-1: top receiving direction - ramp mode': (p6m, 'during_f_rec1', 'during_f_RM_early'),
    'phase6-1: choice mode - random 1-D': (p6m, 'during_f_CD', 'f_rand1'),
    'phase6-2: stim x ramp (c3)': (p6r, 'r_c3', None),
    'phase6-3: c1 predicted from FS gain - observed': (fg, 'F_c1pred', 'obs_c1'),
    'phase6-5: DLAG across - receiving (same k)': (ov, 'f_dlag_across', 'f_rec_pa'),
    'phase6-5: DLAG across - random (same k)': (ov, 'f_dlag_across', 'f_rand_pa'),
    'phase6-6: d1 (own memory, stim x cA_pre)': (rw, 'd1', None),
    'phase7-1: c1 from round trip (c2-calibrated)': (lp, 'c1_loop_c2', None),
}
out['A1'] = {k: mouse_test(df, a, b) for k, (df, a, b) in tests.items()}
# pooled single-neuron gain, bootstrap over mice
u = pd.read_csv(os.path.join(R, 'phase6', 'phase6_neurongain_units.csv'))
rows = []
for (s, sil), d in u.groupby(['session', 'silenced']):
    uc, us, G = d.u_ctrl.values, d.u_stim.values, d.g.values
    W = np.abs(uc); tot = np.sum(W * G); S = (tot - W * G) / (W.sum() - W + 1e-12)
    rows.append(dict(mouse=mouse_of(s), obs=np.sum(us - uc), own=np.sum(uc * (G - 1)), loop1=np.sum(uc * (G * S - 1))))
P = pd.DataFrame(rows); M = P.groupby('mouse').sum()
rng = np.random.default_rng(0)
for c in ('own', 'loop1'):
    bs = []
    for i in range(2000):
        idx = rng.choice(len(M), len(M)); q = M.iloc[idx]
        bs.append(q[c].sum() / q.obs.sum())
    out['A1'][f'phase7-2: fraction of c1 by {c} (bootstrap over mice)'] = dict(
        fraction=float(M[c].sum() / M.obs.sum()), ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])], n_mice=len(M))

# ---------------- A2
lab = pd.read_csv(os.path.join(R, 'units_labeled.csv'))
rng2 = np.random.default_rng(73); rows = []
for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
    d = load(p, lab)
    for sil, rec, code in (('code1', 'code2', 1), ('code2', 'code1', 2)):
        ctrl_all = np.where(d['ok'] & (d['pt'] == 0))[0]; stim = np.where(d['ok'] & (d['pt'] == code))[0]
        ua, ub = d['idx'][sil], d['idx'][rec]
        if min(len(ua), len(ub)) < 6 or len(stim) < 20: continue
        fold = rng2.permutation(len(ctrl_all)) % 2
        S_ = {k: [] for k in ('st', 'w', 'cA', 'cB', 'yA')}
        for f in (0, 1):
            fit = ctrl_all[(fold != f) & d['correct'][ctrl_all]]
            AA, AB = Axes(d, ua, fit), Axes(d, ub, fit)
            for trials, st in ((ctrl_all[fold == f], 0.0), (stim, 1.0)):
                S_['st'].append(np.full(len(trials), st)); S_['w'].append(np.full(len(trials), 1.0 if st == 0 else 0.5))
                S_['cA'].append(AA.conf(trials, -2.1, -1.7)); S_['cB'].append(AB.conf(trials, -2.1, -1.7))
                S_['yA'].append(AA.conf(trials, -0.4, 0.0))
        X = {k: np.concatenate(v) for k, v in S_.items()}
        st, w = X['st'], X['w']; m0 = st == 0; one = np.ones_like(st)
        cA, cB = zs(X['cA'], X['cA'][m0]), zs(X['cB'], X['cB'][m0])
        b = wls(np.c_[one, cA, cB, st, st * cA, st * cB], X['yA'], w)
        rows.append(dict(session=d['sess'], silenced=sil, a1=b[1], a2=b[2], d0=b[3], d1=b[4], d2=b[5]))
A2 = pd.DataFrame(rows); A2.to_csv(os.path.join(R, 'phase7', 'phase7_rewrite_pre.csv'), index=False)
out['A2'] = {c: mouse_test(A2, c) for c in ('a1', 'a2', 'd1', 'd2')}

# ---------------- A3
amb = lab[lab.hemi == 'ambiguous']
out['A3'] = dict(n_labeled=int((lab.hemi != 'ambiguous').sum()), n_ambiguous=len(amb),
                 silenced_by_both=int(((amb.S1 < 0.2) & (amb.S2 < 0.2)).sum()),
                 silenced_by_neither=int(((amb.S1 >= 0.2) & (amb.S2 >= 0.2)).sum()))
json.dump(out, open(os.path.join(R, 'phase7', 'phase7_audit.json'), 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
