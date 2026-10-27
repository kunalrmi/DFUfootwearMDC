"""Levelled design: strategy-learner combinations missing from the first run, on the SAME folds as rep<r>.pkl.
  GBT: S4 (ensemble of ten random-order chains, same orders as the LR ensemble)
  RF : S2 (stacked BR), S3 (chain: beam 5, greedy, Monte Carlo marginals), S4, S5 (domain powerset)
Usage: PAKDD_DATA=<csv> python3 -I run_level.py <repeat_id> <outdir>
Writes <outdir>/extra_level_rep<id>.pkl (merged by stats.py)."""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common as C
from common import *

rep, OUT = int(sys.argv[1]), sys.argv[2]
base = pickle.load(open(f'{OUT}/rep{rep}.pkl', 'rb')); fold = base['fold']; seed = base['seed']
order = list(range(m))
# ensemble orders: replay the generator of the LR ensemble (run_repeat.py), so every learner uses the same ten orders per fold
r_ord = np.random.default_rng(seed + 7); ORDERS = [[list(r_ord.permutation(m)) for _ in range(10)] for f in range(5)]
# stacking: replay the shared generator of run_repeat.py and reuse the LR out-of-fold split of each fold
r_s2 = np.random.default_rng(seed); S2SPLIT = []
for f in range(5):
    ntr = int((fold != f).sum())
    a = np.arange(ntr) % 3; r_s2.shuffle(a); S2SPLIT.append(a)          # LR
    b = np.arange(ntr) % 3; r_s2.shuffle(b)                               # GBT (consumed, not used)
class Fixed:   # stands in for common.rng inside s2(): returns the LR split
    def __init__(self, a): self.a = a
    def shuffle(self, x): x[:] = self.a

KEYS = [('GBT', 'S4'), ('RF', 'S2'), ('RF', 'S3'), ('RF', 'S3-greedy'), ('RF', 'S4'), ('RF', 'S5')]
store = {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in KEYS}
def put(key, te, P, Yp, Yraw):
    for j in range(m): store[key]['P'][j][te] = P[j]
    store[key]['Y'][te] = Yp; store[key]['Yraw'][te] = Yraw
def raw(P): return np.column_stack([p.argmax(1) for p in P])

def s4(L, Xtr, Ytr, Xte, ncat, f):
    Pe = [np.zeros((len(Xte), K[j])) for j in range(m)]
    for o in ORDERS[f]:
        pg, _ = chain_greedy(fit_chain(L, Xtr, Ytr, o, ncat), Xte, o)
        for j in range(m): Pe[j] += pg[j] / 10
    return Pe

t0 = time.time(); CK = f'{OUT}/ckpt_level{rep}.pkl'; f0 = 0
if os.path.exists(CK): ck = pickle.load(open(CK, 'rb')); f0 = ck['f'] + 1; store = ck['store']; print(f'resumed at fold {f0}', flush=True)
for f in range(f0, 5):
    tr, te = fold != f, fold == f; Ytr = Y[tr]
    # ---- GBT: S4 ----
    enc = Enc('ordinal').fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te]); ncat = len(CAT)
    P = s4('GBT', Xtr, Ytr, Xte, ncat, f); put(('GBT', 'S4'), te, P, decode(P), raw(P))
    print(f'rep {rep} fold {f} GBT-S4 {time.time()-t0:.0f}s', flush=True)
    # ---- RF: S2, S3, S4, S5 ----
    enc = Enc('onehot').fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te]); ncat = 0
    C.rng = Fixed(S2SPLIT[f]); P = s2('RF', Xtr, Ytr, Xte, ncat); put(('RF', 'S2'), te, P, decode(P), raw(P))
    models = fit_chain('RF', Xtr, Ytr, order, ncat)
    Pg, Yg = chain_greedy(models, Xte, order); Pmc = chain_mc(models, Xte, order, S=100, seed=seed * 10 + f)
    Yb = chain_beam(models, Xte, order, B=5)
    put(('RF', 'S3'), te, Pmc, Yb, Yg); put(('RF', 'S3-greedy'), te, Pg, Yg, Yg)
    P = s4('RF', Xtr, Ytr, Xte, ncat, f); put(('RF', 'S4'), te, P, decode(P), raw(P))
    P5, Y5, _ = s5_predict('RF', Xtr, Ytr, Xte, ncat); put(('RF', 'S5'), te, P5, Y5, Y5)
    print(f'rep {rep} fold {f} done {time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(f=f, store=store), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)
pickle.dump(dict(rep=rep, store=store), open(f'{OUT}/extra_level_rep{rep}.pkl', 'wb'))
print('saved', flush=True)
