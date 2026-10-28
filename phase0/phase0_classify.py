"""Phase 0 (part 2): classify each unit's hemisphere from photoinhibition and make summary figures."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1]); outdir = sys.argv[2]
df['S1'], df['S2'], df['S3'] = df.fr1/df.fr0, df.fr2/df.fr0, df.fr3/df.fr0
pyr = df[(df.celltype == 'pyramidal') & (df.fr0 >= 1.0)].copy()

def label(r):
    # silenced (<0.2 of control) by one code, and at least 4x more strongly than by the other code
    m, M = min(r.S1, r.S2), max(r.S1, r.S2)
    if not (m < 0.2 and (M == 0 or m / M < 0.25)): return 'ambiguous'
    return 'code1' if r.S1 < r.S2 else 'code2'
pyr['hemi'] = pyr.apply(label, axis=1)

print('pyramidal units with fr0>=1 Hz:', len(pyr), 'sessions:', pyr.session.nunique())
print(pyr.hemi.value_counts().to_string())
print('code3 silences (S3<0.3) fraction:', (pyr.S3 < 0.3).mean().round(3))
print('median S by group:'); print(pyr.groupby('hemi')[['S1','S2','S3']].median().round(3).to_string())

# electrode block vs inferred hemisphere
pyr['eblock'] = np.where(pyr.electrode < 64, 'e<64', 'e>=64')
print('electrode block x hemisphere:'); print(pd.crosstab(pyr.eblock, pyr.hemi).to_string())
# per session: is there a clean electrode split?
def split(g):
    g = g[g.hemi != 'ambiguous']
    if g.hemi.nunique() < 2: return pd.Series(dict(n1=(g.hemi=='code1').sum(), n2=(g.hemi=='code2').sum(), e1max=np.nan, e2min=np.nan, clean=np.nan))
    e1, e2 = g[g.hemi=='code1'].electrode, g[g.hemi=='code2'].electrode
    return pd.Series(dict(n1=len(e1), n2=len(e2), e1min=e1.min(), e1max=e1.max(), e2min=e2.min(), e2max=e2.max(),
                          clean=float(e1.max() < e2.min() or e2.max() < e1.min())))
ss = pyr.groupby('session').apply(split)
print('sessions with both hemispheres labeled:', ss.clean.notna().sum(), ' clean electrode separation:', int(ss.clean.sum()))
ss.to_csv(f'{outdir}/session_split.csv')
pyr.to_csv(f'{outdir}/units_labeled.csv', index=False)

# FS units: VGAT-ChR2 drives inhibitory cells in the illuminated hemisphere -> FS rate should rise there
fs = df[(df.celltype == 'FS') & (df.fr0 >= 1.0)].copy()
fs['up1'] = (fs.S1 > 1.5) & (fs.S2 < 1.2); fs['up2'] = (fs.S2 > 1.5) & (fs.S1 < 1.2)
print('FS units:', len(fs), ' median S3 (bilateral):', round(fs.S3.median(), 2),
      ' clearly driven up by only code1:', int(fs.up1.sum()), ' only code2:', int(fs.up2.sum()), ' neither/both:', int((~fs.up1 & ~fs.up2).sum()))
# sanity: in the same session, FS cells driven by code k should sit with pyramidal cells silenced by code k (same hemisphere)


# figure 1: S1 vs S2 scatter
fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
eps = 0.02
c = {'code1': '#2a6fdb', 'code2': '#e0702a', 'ambiguous': '#9a9a9a'}
for h, g in pyr.groupby('hemi'):
    ax[0].scatter(np.clip(g.S1, eps, 5), np.clip(g.S2, eps, 5), s=6, alpha=.5, color=c[h], label=f'{h} (n={len(g)})')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('rate ratio during PhotostimulationType 1'); ax[0].set_ylabel('rate ratio during PhotostimulationType 2')
ax[0].axvline(1, color='k', lw=.5); ax[0].axhline(1, color='k', lw=.5); ax[0].legend(fontsize=8, loc='lower left')
ax[0].set_title('Each pyramidal unit is silenced by type 1 or type 2, not both')
ax[1].hist(np.clip(pyr.S3, 0, 2), bins=40, color='#555')
ax[1].set_xlabel('rate ratio during PhotostimulationType 3'); ax[1].set_ylabel('units')
ax[1].set_title('Type 3 silences nearly all units (bilateral)')
plt.tight_layout(); plt.savefig(f'{outdir}/fig_phase0_labels.png', dpi=150)
