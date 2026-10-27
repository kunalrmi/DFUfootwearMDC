"""MLP strategies (S1: one single-head net per decision; S6: one multi-head net) on the same folds as rep<r>.pkl.
Usage: PAKDD_DATA=<csv> python3 -I run_mlp.py <repeat_id> <outdir>"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
from mlp_numpy import MultiHeadMLP
rep, OUT = int(sys.argv[1]), sys.argv[2]
base = pickle.load(open(f'{OUT}/rep{rep}.pkl', 'rb')); fold = base['fold']; seed = base['seed']
store = {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in [('MLP', 'S1'), ('MLP', 'S6')]}
t0 = time.time()
CK = f'{OUT}/ckpt_mlp{rep}.pkl'; f0 = 0
if os.path.exists(CK): ck = pickle.load(open(CK, 'rb')); f0 = ck['f'] + 1; store = ck['store']; print(f'resumed at fold {f0}', flush=True)
for f in range(f0, 5):
    tr, te = fold != f, fold == f
    tri = np.where(tr)[0]
    val = iter_strat(Y[tr], 10, seed=seed + 300 + f) == 0          # same stratified 10% validation split for S1 and S6
    enc = Enc('onehot').fit(X.iloc[tri[~val]])
    A, V, T = enc.transform(X.iloc[tri[~val]]), enc.transform(X.iloc[tri[val]]), enc.transform(X[te])
    Ya, Yv = Y[tr][~val], Y[tr][val]
    net = MultiHeadMLP(A.shape[1], K, seed=seed + f).fit(A, Ya, V, Yv)
    P6 = net.predict_proba(T)
    P1 = []
    for j in range(m):
        nj = MultiHeadMLP(A.shape[1], [K[j]], seed=seed + 1000 * (j + 1) + f).fit(A, Ya[:, [j]], V, Yv[:, [j]])
        P1.append(nj.predict_proba(T)[0])
    for key, P in [(('MLP', 'S1'), P1), (('MLP', 'S6'), P6)]:
        for j in range(m): store[key]['P'][j][te] = P[j]
        store[key]['Y'][te] = decode(P); store[key]['Yraw'][te] = np.column_stack([p.argmax(1) for p in P])
    print(f'mlp rep {rep} fold {f} done {time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(f=f, store=store), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)
pickle.dump(dict(rep=rep, store=store), open(f'{OUT}/extra_mlp_rep{rep}.pkl', 'wb'))
print('saved', flush=True)
