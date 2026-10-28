"""Phase 6-5b: does DLAG, fitted to normal trials only, find the callosal channel?

For every session fitted by phase6_dlag.py and both silencing directions (silence A, look at B):
  1. Overlap with the receiving response. Delta_B = mean sqrt-count response of B on silence-A trials minus control
     (laser window -1.7..-0.9 s, 50-ms bins, both trial types). Fraction of the reliable response inside a subspace S
     (split halves, as in Phase 2):  f(S) = <P Delta1, P Delta2> / <Delta1, Delta2>.
     Subspaces of B from DLAG: all across-hemisphere loadings, the A-leads ones (signal flows A -> B), the B-leads
     ones, the within-B loadings. References: cross-validated top-k receiving directions (ceiling, k matched) and k/n.
  2. Direction. DLAG flow index of the session (variance-weighted sign of the delays, + = code1 leads) against the
     causal asymmetry measured by silencing (how much more code2 depends on code1 than the reverse, Phase 4 c0).

usage: python phase6_dlag_analysis.py CACHE_DIR RESULTS_DIR
"""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

CACHE, R = sys.argv[1:3]
rng = np.random.default_rng(67)
NSPLIT = 30


def orth(M, k=None):
    if M.shape[1] == 0:
        return M
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    k = M.shape[1] if k is None else k
    return u[:, :k]


rows, sess_rows = [], []
for f in sorted(glob.glob(os.path.join(R, 'phase6', 'dlag', '*.npz'))):
    m = np.load(f)
    sess = os.path.basename(f)[:-4].replace('__', '/')
    z = np.load(os.path.join(CACHE, os.path.basename(f)))
    ok = (z['GoodTrials'].astype(bool) & ~z['LickEarly'].astype(bool) &
          (np.abs(z['cue'] - z['PoleOutTime'] - 1.7) < 0.05) & ((z['HitR'] + z['HitL'] + z['ErrR'] + z['ErrL']) > 0))
    instR = (z['HitR'] + z['ErrR']) > 0; pt = z['PhotostimulationType']
    t = z['edges'][:-1]; W = (t >= -1.7 - 1e-6) & (t < -0.9 - 1e-6)
    delays = m['delays']; pa = len(delays)
    va = m['var_across1'] + m['var_across2'] if pa else np.zeros(0)
    flow = float(np.sum(np.sign(delays) * va) / (va.sum() + 1e-12)) if pa else np.nan
    sess_rows.append(dict(session=sess, pa=pa, flow=flow, n_dir=int(m['directional'].sum()) if pa else 0))
    for sil, rec, code, units_key, Ca_key, Cw_key, lead_sign in (
            ('code1', 'code2', 1, 'units2', 'Ca2', 'Cw2', +1), ('code2', 'code1', 2, 'units1', 'Ca1', 'Cw1', -1)):
        ub = m[units_key]; Ca, Cw = m[Ca_key], m[Cw_key]
        ctrl = {s: np.where(ok & (pt == 0) & (instR == s))[0] for s in (True, False)}
        stim = {s: np.where(ok & (pt == code) & (instR == s))[0] for s in (True, False)}
        if min(len(v) for v in stim.values()) < 6:
            continue
        Y = np.sqrt(z['counts'][:, ub][:, :, W].astype(float))
        subs = {}
        if pa:
            subs['dlag_across'] = orth(Ca)
            AtoB = (np.sign(delays) == lead_sign)
            subs['dlag_AtoB'] = orth(Ca[:, AtoB]); subs['dlag_BtoA'] = orth(Ca[:, ~AtoB])
        subs['dlag_within'] = orth(Cw)
        subs['dlag_shared'] = orth(np.c_[Ca, Cw]) if (pa + Cw.shape[1]) else np.zeros((len(ub), 0))
        acc = {k: 0.0 for k in list(subs) + ['den', 'rec_pa', 'rec_shared']}
        k_pa, k_sh = pa, subs['dlag_shared'].shape[1]
        for _ in range(NSPLIT):
            D = []
            for _h in range(2):
                pass
            halves = {s: (rng.permutation(stim[s]), rng.permutation(ctrl[s])) for s in (True, False)}
            D1 = np.concatenate([Y[halves[s][0][:len(halves[s][0]) // 2]].mean(0) - Y[halves[s][1][:len(halves[s][1]) // 2]].mean(0)
                                 for s in (True, False)], 1)
            D2 = np.concatenate([Y[halves[s][0][len(halves[s][0]) // 2:]].mean(0) - Y[halves[s][1][len(halves[s][1]) // 2:]].mean(0)
                                 for s in (True, False)], 1)
            acc['den'] += np.sum(D1 * D2)
            for k, V in subs.items():
                if V.shape[1]:
                    acc[k] += np.sum((V.T @ D1) * (V.T @ D2))
            # conservative ceiling: direction from one third of stim (vs one third of controls), scored on the other thirds
            th = {s_: (rng.permutation(stim[s_]), rng.permutation(ctrl[s_])) for s_ in (True, False)}
            def part(i, s_):
                st_, ct_ = th[s_]; a, b = len(st_) // 3, len(ct_) // 3
                return Y[st_[i * a:(i + 1) * a]].mean(0) - Y[ct_[i * b:(i + 1) * b]].mean(0)
            G3 = np.concatenate([part(2, s_) for s_ in (True, False)], 1)
            G1 = np.concatenate([part(0, s_) for s_ in (True, False)], 1)
            G2 = np.concatenate([part(1, s_) for s_ in (True, False)], 1)
            acc['den3'] = acc.get('den3', 0.0) + np.sum(G1 * G2)
            if k_pa:
                U3 = orth(G3, k_pa)
                acc['g3_rec_pa'] = acc.get('g3_rec_pa', 0.0) + np.sum((U3.T @ G1) * (U3.T @ G2))
                acc['g3_dlag_across'] = acc.get('g3_dlag_across', 0.0) + np.sum((subs['dlag_across'].T @ G1) * (subs['dlag_across'].T @ G2))
            for nm, kk in (('rec_pa', k_pa), ('rec_shared', k_sh)):
                if kk:
                    U1, U2 = orth(D1, kk), orth(D2, kk)
                    acc[nm] += 0.5 * (np.sum((U1.T @ D1) * (U1.T @ D2)) + np.sum((U2.T @ D2) * (U2.T @ D1)))
        row = dict(session=sess, silenced=sil, receiving=rec, nB=len(ub), pa=pa, k_shared=k_sh,
                   n_AtoB=int(subs['dlag_AtoB'].shape[1]) if pa else 0, rel_den=acc['den'] / NSPLIT)
        for k in list(subs) + ['rec_pa', 'rec_shared']:
            row[f'f_{k}'] = acc[k] / acc['den'] if acc['den'] > 0 else np.nan
        row['f_rand_pa'] = pa / len(ub); row['f_rand_shared'] = k_sh / len(ub)
        if pa and acc.get('den3', 0) > 0:
            row['g3_rec_pa'] = acc['g3_rec_pa'] / acc['den3']; row['g3_dlag_across'] = acc['g3_dlag_across'] / acc['den3']
        rows.append(row)

df = pd.DataFrame(rows); ss = pd.DataFrame(sess_rows)
df.to_csv(os.path.join(R, 'phase6', 'phase6_dlag_overlap.csv'), index=False)
ss.to_csv(os.path.join(R, 'phase6', 'phase6_dlag_sessions.csv'), index=False)
good = df[df.rel_den > 0]
print(len(ss), 'sessions;', len(good), 'pairs with a reliable response')
print('pa distribution', ss.pa.value_counts().sort_index().to_dict())
print(good[[c for c in good.columns if c.startswith('f_')]].median().round(3).to_string())
g = good[good.pa > 0]
for a, b in (('f_dlag_across', 'f_rand_pa'), ('f_dlag_across', 'f_rec_pa'), ('f_dlag_shared', 'f_rec_shared'),
             ('f_dlag_shared', 'f_rand_shared')):
    print(a, 'vs', b, 'p =', f'{wilcoxon(g[a] - g[b]).pvalue:.1e}')
# direction: flow index vs causal asymmetry
r6 = pd.read_csv(os.path.join(R, 'phase6', 'phase6_ramp.csv'))
c0 = r6.pivot_table(index='session', columns='silenced', values='p4_c0')
c0['causal'] = c0['code2'] - c0['code1']      # > 0: silencing code1 hurts code2 more than the reverse -> code1 drives
mm = ss.merge(c0[['causal']], left_on='session', right_index=True).dropna()
rho, p = spearmanr(mm.flow, mm.causal)
print(f'flow index vs causal asymmetry: rho = {rho:+.2f}, p = {p:.2f}, n = {len(mm)}')

# summary of the fitted models (variable-width CSVs)
sys.path.insert(0, os.path.dirname(__file__))
from dlag_summary import read_summary
d = read_summary(R)
dl = np.concatenate([np.array(x) for x in d.delays if len(x)]); dr = np.concatenate([np.array(x) for x in d.directional if len(x)]).astype(bool)
taus = np.concatenate([np.array(x) for x in d.taus if len(x)])
summ = dict(n_sessions=len(d), n_mice=d.mouse.nunique(), frac_pa_pos=(d.pa > 0).mean(), pa_median=d.pa.median(),
            n_latents=len(dl), frac_directional=dr.mean(), med_abs_delay_dir=np.median(np.abs(dl[dr])),
            frac_code1_leads=np.mean(dl[dr] > 0), across_strength=np.median(np.r_[d.across_strength1[d.pa > 0], d.across_strength2[d.pa > 0]]),
            tau_median_ms=np.median(taus), flow_rho=rho, flow_p=p, flow_n=len(mm), n_pairs=len(g))
for k in ('g3_rec_pa', 'g3_dlag_across', 'f_dlag_across', 'f_dlag_AtoB', 'f_dlag_BtoA', 'f_dlag_within', 'f_dlag_shared', 'f_rec_pa', 'f_rec_shared', 'f_rand_pa', 'f_rand_shared'):
    summ[k] = g[k].median()
summ['p_across_vs_rand'] = wilcoxon(g.f_dlag_across - g.f_rand_pa).pvalue
summ['p_across_vs_rec'] = wilcoxon(g.f_dlag_across - g.f_rec_pa).pvalue
summ['p_across_vs_rec3'] = wilcoxon((g.g3_dlag_across - g.g3_rec_pa).dropna()).pvalue
summ['p_AtoB_vs_BtoA'] = wilcoxon((g.f_dlag_AtoB - g.f_dlag_BtoA).dropna()).pvalue
import json
json.dump(summ, open(os.path.join(R, 'phase6', 'phase6_dlag_summary.json'), 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
