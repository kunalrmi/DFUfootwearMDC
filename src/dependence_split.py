"""Per-decision teacher-forced dependence gain on the outer test folds, split into the part recovered by renormalising
BR onto coherent classes (implied by the constraints) and the residual. LR/GBT from rep*.pkl; TFM from teacher_tfm.csv.
Usage: PAKDD_DATA=<csv> python3 -I dependence_split.py <outdir> [teacher_tfm.csv]"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import TGT, K, m, n, Y as Y0, NONE, ROCK, FFD, FFM
OUT = sys.argv[1]; TT = sys.argv[2] if len(sys.argv) > 2 else None
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{OUT}/rep[0-9]*.pkl'))]
ex = {int(p[-5]): pickle.load(open(p, 'rb')) for p in glob.glob(f'{OUT}/extra_tfm_rep*.pkl')}
def ll(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1)).mean()
def renorm(j, pb):
    if not (j in ROCK[1:] or j == FFM): return pb
    par = ROCK[0] if j in ROCK[1:] else FFD; pn = Y0[:, par] == NONE[par]
    mask = np.ones_like(pb, bool); mask[pn] = False; mask[pn, NONE[j]] = True; mask[~pn, NONE[j]] = False
    q = np.where(mask, pb, 0); return q / np.clip(q.sum(1, keepdims=True), 1e-15, None)
rows = []
def add(L, Pb_by_rep, T_by_rep):
    for j in range(m):
        br = np.mean([ll(Pb_by_rep[r][j], Y0[:, j]) for r in Pb_by_rep]); brc = np.mean([ll(renorm(j, Pb_by_rep[r][j]), Y0[:, j]) for r in Pb_by_rep])
        te = np.mean([ll(T_by_rep[r][j], Y0[:, j]) for r in T_by_rep])
        rows.append(dict(learner=L, decision=TGT[j], gain=br - te, implied=br - brc, residual=brc - te))
for L in ['LR', 'GBT']:
    add(L, {r['rep']: r['store'][(L, 'S1')]['P'] for r in reps}, {r['rep']: r['chain_aux'][L]['teacher'] for r in reps})
if TT and ex:
    t = pd.read_csv(TT); T = {}
    for rep, g in t.groupby('rep'):
        T[rep] = [g[g.decision == j].sort_values('row')[[f'p{c}' for c in range(K[j])]].values for j in range(m)]
    add('TFM', {r: ex[r]['store'][('TFM', 'S1')]['P'] for r in T}, T)
d = pd.DataFrame(rows); d.to_csv(f'{OUT}/table_dependence_split.csv', index=False)
s = d.groupby('learner')[['gain', 'implied', 'residual']].sum(); s['share_implied'] = s.implied / s.gain
print(s.round(3)); print(d[d.gain.abs() > 0.03].round(3).to_string())
