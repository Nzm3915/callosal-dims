"""Referee point: the hemisphere labels of Phase 0 come from the silencing responses themselves, and the NWB files
carry no usable probe/hemisphere metadata (electrode coordinates are placeholders; channel numbers 1-64 repeat on the
two probes). Is there a check of the labels that does not use the photostimulation trials at all?

Physical prediction: strongly co-fluctuating unit pairs are neighbours on the same probe. If the labels are right,
(i) strong pairs are far more common among same-label pairs recorded on nearby channel numbers than among
different-label pairs on the same channel numbers (those sit on different probes), and (ii) only same-label pairs
show the fall-off with channel distance. If the labels were wrong, channel distance would matter regardless of label.

Fast co-fluctuations: control trials only (no laser), 50-ms counts over the whole trial, minus the PSTH of the
trial type and minus each trial's own mean; Pearson r over all trial x bin samples. A pair is 'strong' if r > 0.1.
Per-mouse test on pairs with channel distance <= 2.

usage: python phase7_label_validation.py CACHE_DIR LABELS_CSV OUTDIR
"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
CACHE, LABELS, OUT = sys.argv[1:4]
TH = 0.1

if __name__ == '__main__':
    lab = pd.read_csv(LABELS); lab['key'] = lab.session.str.replace('/', '__') + '.npz'
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, '*.npz'))):
        z = np.load(p, allow_pickle=True); name = os.path.basename(p); L = lab[lab.key == name]
        uid = list(z['unit_ids']); ok = z['GoodTrials'].astype(bool) & (z['PhotostimulationType'] == 0)
        tt = np.where(z['HitR'][ok].astype(bool) | z['ErrL'][ok].astype(bool), 1, 0)
        C = z['counts'][ok].astype(float)
        for c in (0, 1):
            C[tt == c] -= C[tt == c].mean(0, keepdims=True)
        C -= C.mean(2, keepdims=True); X = C.transpose(0, 2, 1).reshape(-1, C.shape[1])
        L = L[L.unit.isin(uid) & (L.hemi != 'ambiguous')]
        idx = np.array([uid.index(u) for u in L.unit]); good = X[:, idx].std(0) > 0
        L, idx = L[good], idx[good]; R = np.corrcoef(X[:, idx].T)
        h, e = L.hemi.values, L.electrode.values.astype(int)
        iu, ju = np.triu_indices(len(idx), 1)
        rows.append(pd.DataFrame(dict(session=name, mouse=name.split('__')[0], same=h[iu] == h[ju],
                                      de=np.abs(e[iu] - e[ju]), r=R[iu, ju])))
    df = pd.concat(rows)
    df['dist'] = pd.cut(df.de, [-1, 0, 2, 8, 64], labels=['0', '1-2', '3-8', '>8']).astype(str)
    tab = df.groupby(['dist', 'same']).r.agg(n='size', frac_strong=lambda r: float((r > TH).mean()), mean='mean').reset_index()
    print(tab.round(4).to_string(index=False))
    near = df[df.de <= 2]
    pm = near.groupby(['mouse', 'same']).r.apply(lambda r: (r > TH).mean()).unstack().dropna()
    out = dict(table=tab.to_dict('records'),
               near_same_frac=float((near[near.same].r > TH).mean()), near_diff_frac=float((near[~near.same].r > TH).mean()),
               mice=len(pm), mice_same_gt_diff=float((pm[True] > pm[False]).mean()),
               p_mouse=float(wilcoxon(pm[True], pm[False]).pvalue))
    print({k: v for k, v in out.items() if k != 'table'})
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, 'phase7_label_validation.json'), 'w'), indent=1)
