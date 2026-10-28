"""Phase 6-1: what does the callosal input look like in the receiver's own coordinates?

Finkelstein et al. 2021 and Inagaki et al. 2019 describe ALM activity with two main modes: the choice mode (CD) and a
non-selective "ramping mode" (RM, the same for lick-left and lick-right trials), which they interpret as an external
input that moves the two choice attractors apart and makes the plan robust. Economo et al. 2018 split the choice
information into an early (sample), a late (end of delay) and a movement (go) coding direction.

For every pair (silence A, look at B) we ask how much of B's reliable silencing response Delta_B lies along each
mode (noise-corrected fraction, as in Phase 2):

    f(m) = <m.Delta1, m.Delta2> / <Delta1, Delta2>,     Delta1, Delta2 from disjoint halves of the trials

Modes are estimated from a separate third of the control trials (z-scored rates, 100-ms bins):
  CD        choice mode: lick-right minus lick-left, whole delay
  RM_early  non-selective mode of the laser window: mean(-1.7..-0.9) - mean(pre-sample), both trial types pooled,
            Gram-Schmidt orthogonalised to CD
  RM_late   Finkelstein's ramping mode: mean(last 0.5 s of delay) - mean(pre-sample), orthogonalised to CD
  SM        sample-onset non-selective mode: mean(-3.0..-2.6) - pre-sample, orthogonalised to CD
  NS3       the span of RM_early, RM_late, SM (3-D non-selective subspace)
  CDe, CDl, CDgo  Economo's coding directions (sample -3.0..-2.6; delay end -0.4..0; after go 0..0.4, CDgo
            orthogonalised to CDl)
  uniform   all units together

Also: how much of B's own non-selective delay modulation is removed by silencing A,
    loss_RM = - <RM_early, mean Delta_B> / <RM_early, mean(control laser window) - mean(control pre-sample)>
and how much choice selectivity survives along CDe / CDl / CDgo (stim / control, split halves).

usage: python phase6_modes.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, numpy as np, pandas as pd

CACHE, LABELS, OUT = sys.argv[1:4]
NSPLIT = 30
MIN_UNITS = 5
rng = np.random.default_rng(61)


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
    nb = c.shape[-1] // 2
    r = (c[..., :2 * nb:2] + c[..., 1:2 * nb:2]) / 0.1
    t0 = z['edges'][:-1][:2 * nb:2]
    return dict(sess=sess, mouse=sess.split('/')[0], r=r, t0=t0, idx=idx, ok=ok,
                instR=(z['HitR'] + z['ErrR']) > 0, lickR=(z['HitR'] + z['ErrL']) > 0,
                correct=(z['HitR'] + z['HitL']) > 0, pt=z['PhotostimulationType'])


def W(t0, a, b):
    return (t0 >= a - 1e-6) & (t0 < b - 1e-6)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def gs(v, basis):
    for b in basis:
        v = v - (v @ b) * b
    return unit(v)


def analyse_pair(d, sil, rec):
    ia, ib = d['idx'][sil], d['idx'][rec]
    if min(len(ia), len(ib)) < MIN_UNITS:
        return None
    code = 1 if sil == 'code1' else 2
    t0 = d['t0']
    wpre, wsamp, wlas, wafter = W(t0, -3.6, -3.0), W(t0, -3.0, -2.6), W(t0, -1.7, -0.9), W(t0, -0.9, 0.0)
    wdel, wlate, wend, wgo = W(t0, -1.7, 0.0), W(t0, -0.5, 0.0), W(t0, -0.4, 0.0), W(t0, 0.0, 0.4)
    ctrl = {s: np.where(d['ok'] & (d['pt'] == 0) & (d['instR'] == s))[0] for s in (True, False)}
    stim = {s: np.where(d['ok'] & (d['pt'] == code) & (d['instR'] == s))[0] for s in (True, False)}
    if min(len(v) for v in stim.values()) < 6 or min(len(v) for v in ctrl.values()) < 30:
        return None
    rB = d['r'][:, ib, :]
    acc, n = {}, 0

    def add(k, v):
        acc[k] = acc.get(k, 0.0) + v

    for _ in range(NSPLIT):
        est, c1, c2, s1, s2 = {}, {}, {}, {}, {}
        for s in (True, False):
            p = rng.permutation(ctrl[s]); n3 = len(p) // 3
            est[s], c1[s], c2[s] = p[:n3], p[n3:2 * n3], p[2 * n3:3 * n3]
            q = rng.permutation(stim[s]); h = len(q) // 2
            s1[s], s2[s] = q[:h], q[h:2 * h]
        E = np.r_[est[True], est[False]]
        mu = rB[E][:, :, wdel].mean((0, 2)); sd = rB[E][:, :, wdel].std((0, 2)) + 1.0
        z = (rB - mu[None, :, None]) / sd[None, :, None]
        m = {s: z[est[s]].mean(0) for s in (True, False)}                  # n x T
        mall = 0.5 * (m[True] + m[False])
        cd = unit((m[True] - m[False])[:, wdel].mean(1))
        pre = mall[:, wpre].mean(1)
        rm_e = gs(mall[:, wlas].mean(1) - pre, [cd])
        rm_l = gs(mall[:, wlate].mean(1) - pre, [cd])
        sm = gs(mall[:, wsamp].mean(1) - pre, [cd])
        ns = np.linalg.qr(np.c_[rm_e, rm_l, sm])[0]
        cde = unit((m[True] - m[False])[:, wsamp].mean(1))
        cdl = unit((m[True] - m[False])[:, wend].mean(1))
        cdgo = gs((m[True] - m[False])[:, wgo].mean(1), [cdl])
        uni = np.ones(len(ib)) / np.sqrt(len(ib))
        modes = dict(CD=cd, RM_early=rm_e, RM_late=rm_l, SM=sm, CDe=cde, CDl=cdl, CDgo=cdgo, uniform=uni)
        for e, w in (('during', wlas), ('after', wafter)):
            D1 = np.concatenate([z[s1[s]][:, :, w].mean(0) - z[c1[s]][:, :, w].mean(0) for s in (True, False)], 1)
            D2 = np.concatenate([z[s2[s]][:, :, w].mean(0) - z[c2[s]][:, :, w].mean(0) for s in (True, False)], 1)
            add(f'{e}_den', np.sum(D1 * D2))
            for k, v in modes.items():
                add(f'{e}_f_{k}', np.sum((v @ D1) * (v @ D2)))
            add(f'{e}_f_NS3', np.sum((ns.T @ D1) * (ns.T @ D2)))
            # cross-validated top receiving direction (for reference) and its cosine with RM_early
            U1 = np.linalg.svd(D1, full_matrices=False)[0][:, 0]
            add(f'{e}_f_rec1', 0.5 * (np.sum((U1 @ D1) * (U1 @ D2)) + np.sum((np.linalg.svd(D2, full_matrices=False)[0][:, 0] @ D2) *
                                                                         (np.linalg.svd(D2, full_matrices=False)[0][:, 0] @ D1))))
            add(f'{e}_cos_rec1_RMe', abs(U1 @ rm_e))
            add(f'{e}_cos_rec1_CD', abs(U1 @ cd))
            # does the input land in a direction B itself uses? variance of B's own activity along U1
            # (estimated on held-out control trials c2), relative to an average direction (alignment index).
            C2 = np.r_[c2[True], c2[False]]
            Xt = z[C2][:, :, wdel].transpose(0, 2, 1).reshape(-1, len(ib))          # total (signal + noise)
            Xn = np.concatenate([z[c2[s]][:, :, wdel] - z[c2[s]][:, :, wdel].mean(0) for s in (True, False)])
            Xn = Xn.transpose(0, 2, 1).reshape(-1, len(ib))                            # trial-to-trial residuals
            Xs = np.concatenate([z[c2[s]][:, :, wdel].mean(0) for s in (True, False)], 1).T
            for nm, X in (('tot', Xt), ('noise', Xn), ('sig', Xs)):
                X = X - X.mean(0)
                v_u = np.mean((X @ U1) ** 2); v_avg = np.mean(np.sum(X ** 2, 1)) / len(ib)
                v_cd = np.mean((X @ cd) ** 2)
                add(f'{e}_align_{nm}', v_u / (v_avg + 1e-12)); add(f'{e}_alignCD_{nm}', v_cd / (v_avg + 1e-12))
            # mean (over trial types and time) response along RM_early, relative to B's own modulation
            Dm = 0.5 * (D1 + D2).mean(1)
            own = (0.5 * (z[c1[True]].mean(0) + z[c1[False]].mean(0)))[:, w].mean(1) - \
                  (0.5 * (z[c1[True]].mean(0) + z[c1[False]].mean(0)))[:, wpre].mean(1)
            add(f'{e}_loss_RMe', -(rm_e @ Dm) / (rm_e @ own + 1e-9))
            add(f'{e}_own_RMe', rm_e @ own)
            add(f'{e}_proj_RMe', rm_e @ Dm)
            # selectivity retention along Economo's directions (split halves to avoid shared noise)
            for k in ('CD', 'CDe', 'CDl', 'CDgo'):
                v = modes[k]
                sel_s = v @ (z[s1[True]][:, :, w].mean((0, 2)) - z[s1[False]][:, :, w].mean((0, 2)))
                sel_c = v @ (z[c2[True]][:, :, w].mean((0, 2)) - z[c2[False]][:, :, w].mean((0, 2)))
                add(f'{e}_selS_{k}', sel_s); add(f'{e}_selC_{k}', sel_c)
        n += 1
    row = dict(session=d['sess'], mouse=d['mouse'], silenced=sil, receiving=rec, nA=len(ia), nB=len(ib),
               n_stim=len(stim[True]) + len(stim[False]))
    for e in ('during', 'after'):
        den = acc[f'{e}_den']
        row[f'{e}_den'] = den / n
        for k in list(acc):
            if k.startswith(f'{e}_f_'):
                row[k] = acc[k] / den
        for k in ('cos_rec1_RMe', 'cos_rec1_CD', 'loss_RMe', 'own_RMe', 'proj_RMe', 'align_tot', 'align_noise',
                  'align_sig', 'alignCD_tot', 'alignCD_noise', 'alignCD_sig'):
            row[f'{e}_{k}'] = acc[f'{e}_{k}'] / n
        for k in ('CD', 'CDe', 'CDl', 'CDgo'):
            row[f'{e}_ret_{k}'] = acc[f'{e}_selS_{k}'] / (acc[f'{e}_selC_{k}'] + 1e-9)
            row[f'{e}_selC_{k}'] = acc[f'{e}_selC_{k}'] / n
    row['f_rand1'] = 1 / len(ib)
    return row


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        d = load(p, lab)
        for sil, rec in (('code1', 'code2'), ('code2', 'code1')):
            r = analyse_pair(d, sil, rec)
            if r: rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'phase6_modes.csv'), index=False)
    cols = [c for c in df.columns if c.startswith('during_f_')] + ['f_rand1']
    print(len(df), 'pairs')
    print(df[cols].median().round(3).to_string())
    print(df[[c for c in df.columns if c.startswith('after_f_')]].median().round(3).to_string())
    print(df[[c for c in df.columns if 'ret_' in c or 'loss' in c or 'cos' in c or 'align' in c]].median().round(3).to_string())
