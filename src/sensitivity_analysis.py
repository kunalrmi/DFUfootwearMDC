"""Sensitivity strategies (run_sensitivity.py) vs their matched S1: EM, Sub-EM, accuracy and corrected-t 95% intervals.
Usage: PAKDD_DATA=<csv> python3 -I sensitivity_analysis.py <main_out> <sensitivity_dir>"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from common import m, Y as Y0
MAIN, TB = sys.argv[1], sys.argv[2]
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{MAIN}/rep[0-9]*.pkl'))]
for r in reps:
    for p in glob.glob(f"{MAIN}/extra_*_rep{r['rep']}.pkl") + [f"{TB}/extra_sensitivity_rep{r['rep']}.pkl"] + glob.glob(f"{TB}/extra_sensitivity_tfm_rep{r['rep']}.pkl"):
        r['store'].update(pickle.load(open(p, 'rb'))['store'])
def fv(r, k, thr): e = (r['store'][k]['Y'] != Y0).sum(1) <= thr; return np.array([e[r['fold'] == f].mean() for f in range(5)])
def lvl(k, thr): return 100 * np.mean([((r['store'][k]['Y'] != Y0).sum(1) <= thr).mean() for r in reps])
rows = []
for k, base in [(('LR', 'S3-rev'), ('LR', 'S1')), (('GBT', 'S3-rev'), ('GBT', 'S1')), (('RF', 'S3-rev'), ('RF', 'S1')),
                (('LR', 'S3-rev'), ('LR', 'S3')), (('GBT', 'S3-rev'), ('GBT', 'S3')), (('RF', 'S3-rev'), ('RF', 'S3')),
                (('LR', 'S1-merged'), ('LR', 'S1')), (('GBT', 'S1-merged'), ('GBT', 'S1')), (('RF', 'S1-merged'), ('RF', 'S1')),
                (('RF-leaf5', 'S1'), ('RF', 'S1')), (('RF-leaf5', 'S2'), ('RF-leaf5', 'S1')), (('RF-leaf5', 'S6'), ('RF-leaf5', 'S1')),
                (('GBT-reg', 'S1'), ('GBT', 'S1')), (('GBT-reg', 'S2'), ('GBT-reg', 'S1')),
                (('TFM', 'S3-rev'), ('TFM', 'S1')), (('TFM', 'S3-rev'), ('TFM', 'S3')), (('TFM', 'S1-merged'), ('TFM', 'S1'))]:
    row = dict(variant='/'.join(k), baseline='/'.join(base))
    for nm, thr in [('EM', 0), ('Sub', 1)]:
        d = np.concatenate([fv(r, k, thr) - fv(r, base, thr) for r in reps]); kk = len(d); se = np.sqrt((1 / kk + 0.25) * d.var(ddof=1)); q = tdist.ppf(0.975, kk - 1)
        row.update({nm: lvl(k, thr), f'{nm}_base': lvl(base, thr), f'd_{nm}': 100 * d.mean(), f'{nm}_lo': 100 * (d.mean() - q * se), f'{nm}_hi': 100 * (d.mean() + q * se),
                    f'{nm}_p': 2 * tdist.sf(abs(d.mean() / se), kk - 1)})
    rows.append(row)
t = pd.DataFrame(rows); t.to_csv(f'{MAIN}/table_sensitivity.csv', index=False); pd.set_option('display.width', 250); print(t.round(2).to_string())
