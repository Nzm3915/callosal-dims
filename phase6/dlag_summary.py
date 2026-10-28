"""Read the per-session DLAG summary CSVs (rows have a variable number of cvll_pa* columns)."""
import csv, glob, os, numpy as np, pandas as pd


def read_summary(R):
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'phase6', 'phase6_dlag_*_*.csv')) + glob.glob(os.path.join(R, 'phase6_dlag_*_*.csv'))):
        with open(f) as fh:
            rd = csv.reader(fh); next(rd)
            for r in rd:
                p1, p2 = int(r[5]), int(r[6]); n = min(p1, p2, 4) + 1
                cv = [float(x) for x in r[8:8 + n]]
                rest = r[8 + n:]
                rows.append(dict(session=r[0], mouse=r[1], q1=int(r[2]), q2=int(r[3]), n_trials=int(r[4]), p1=p1, p2=p2,
                                 pa=int(r[7]), cvll=cv, cv_gain=max(cv) - cv[0], across_strength1=float(rest[0]),
                                 across_strength2=float(rest[1]),
                                 delays=[float(x) for x in rest[2].split(';')] if rest[2] else [],
                                 directional=[int(x) for x in rest[3].split(';')] if rest[3] else [],
                                 taus=[float(x) for x in rest[4].split(';')] if rest[4] else []))
    return pd.DataFrame(rows)
