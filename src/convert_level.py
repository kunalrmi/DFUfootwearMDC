"""Convert Kaggle run 6 CSVs (predictions_level_tfm.csv / predictions_level_mlp.csv) into extra_kaglevel_<l>_rep<r>.pkl.
Usage: PAKDD_DATA=<csv> python3 -I convert_level.py <kaggle_csv_dir> <out_dir>"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import K, m, n
KAG, OUT = sys.argv[1], sys.argv[2]
for path in sorted(glob.glob(f'{KAG}/predictions_level_*.csv')):
    lname = os.path.basename(path)[len('predictions_level_'):-4]; d = pd.read_csv(path); per = {}
    for (rep, L, S), g in d.groupby(['rep', 'learner', 'strategy']):
        s = {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)}
        for j, gj in g.groupby('decision'):
            rows = gj.row.values; s['P'][j][rows] = gj[[f'p{c}' for c in range(K[j])]].values
            s['Y'][rows, j] = gj.y_dec.values; s['Yraw'][rows, j] = gj.y_raw.values
        assert g.row.nunique() == n, f'{path} rep {rep} {L} {S} covers {g.row.nunique()}/{n} records'
        per.setdefault(int(rep), {})[(L, S)] = s
    for rep, store in per.items():
        pickle.dump(dict(rep=rep, store=store), open(f'{OUT}/extra_kaglevel_{lname}_rep{rep}.pkl', 'wb'))
    print(lname, {r: sorted(v) for r, v in per.items()})
