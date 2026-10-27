"""Sensitivity strategies on the same folds as rep<r>.pkl.
  S3-rev     : classifier chain in reverse domain order (beam 5 with coherence pruning in any order), LR/GBT/RF
  S1-merged  : binary relevance after re-encoding each constraint group (rocker: 4 decisions; forefoot wedge: 2) as one
               decision over its observed combinations (rare ones mapped as in S5), LR/GBT/RF
  Regularised variants (keys 'RF-leaf5', 'GBT-reg'): RF with min_samples_leaf=5 (S1, S2, S6); GBT with min_samples_leaf=20 and
               l2_regularization=1 (S1, S2)
Usage: PAKDD_DATA=<csv> python3 -I run_sensitivity.py <repeat_id> <outdir>  -> <outdir>/extra_sensitivity_rep<id>.pkl"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common as C
from common import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

rep, OUT = int(sys.argv[1]), sys.argv[2]
base = pickle.load(open(f'{OUT}/rep{rep}.pkl', 'rb')); fold = base['fold']; seed = base['seed']
REV = [j for d in reversed(list(DOMIDX)) for j in DOMIDX[d]]       # reverse domain order
_orig = C.make
def make_patched(learner, ncat=0, multi=False):
    if learner == 'RF-leaf5': return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=2, random_state=0)
    if learner == 'GBT-reg':
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=20, l2_regularization=1.0,
                                              categorical_features=list(range(ncat)) if ncat else None, random_state=0)
    return _orig(learner, ncat, multi)
C.make = make_patched
C.ENCKIND['RF-leaf5'] = 'onehot'; C.ENCKIND['GBT-reg'] = 'ordinal'
r_s2 = np.random.default_rng(seed); SPL = []
for f in range(5):
    ntr = int((fold != f).sum()); a = np.arange(ntr) % 3; r_s2.shuffle(a); SPL.append(a); b = np.arange(ntr) % 3; r_s2.shuffle(b)
class Fixed:
    def __init__(self, a): self.a = a
    def shuffle(self, x): x[:] = self.a

GRP = [DOMIDX['Rocker'], [FFD, FFM]]
def merged_br(L, Xtr, Ytr, Xte, ncat):
    P = [None] * m; Yp = np.zeros((len(Xte), m), int); ing = set(sum(GRP, []))
    for j in range(m):
        if j in ing: continue
        P[j] = fullproba(fitc(L, ncat, Xtr, Ytr[:, j]), Xte, K[j]); Yp[:, j] = P[j].argmax(1)
    for idx in GRP:
        combos, inv, cnt = np.unique(Ytr[:, idx], axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel(); freq = np.where(cnt >= 3)[0]; mapped = inv.copy()
        for c in np.where(cnt < 3)[0]:
            dist = (combos[freq] != combos[c]).sum(1); best = freq[np.lexsort((-cnt[freq], dist))[0]]; mapped[inv == c] = best
        clf = fitc(L, ncat, Xtr, mapped); pc = np.zeros((len(Xte), len(combos))); pc[:, clf.classes_] = clf.predict_proba(Xte)
        Yp[:, idx] = combos[pc.argmax(1)]
        for i, j in enumerate(idx): P[j] = np.stack([pc[:, combos[:, i] == c].sum(1) for c in range(K[j])], 1)
    return P, Yp

KEYS = [(L, 'S3-rev') for L in ['LR', 'GBT', 'RF']] + [(L, 'S1-merged') for L in ['LR', 'GBT', 'RF']] + \
       [('RF-leaf5', 'S1'), ('RF-leaf5', 'S2'), ('RF-leaf5', 'S6'), ('GBT-reg', 'S1'), ('GBT-reg', 'S2')]
store = {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in KEYS}
def put(key, te, P, Yp, Yraw):
    for j in range(m): store[key]['P'][j][te] = P[j]
    store[key]['Y'][te] = Yp; store[key]['Yraw'][te] = Yraw
def raw(P): return np.column_stack([p.argmax(1) for p in P])
t0 = time.time(); CK = f'{OUT}/ckpt_sensitivity{rep}.pkl'; f0 = 0
if os.path.exists(CK): ck = pickle.load(open(CK, 'rb')); f0 = ck['f'] + 1; store = ck['store']; print(f'resumed at fold {f0}', flush=True)
for f in range(f0, 5):
    tr, te = fold != f, fold == f; Ytr = Y[tr]
    for L in ['LR', 'GBT', 'RF', 'RF-leaf5', 'GBT-reg']:
        enc = Enc(ENCKIND[L]).fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te]); ncat = len(CAT) if L.startswith('GBT') else 0
        if L in ('LR', 'GBT', 'RF'):
            models = fit_chain(L, Xtr, Ytr, REV, ncat); Pg, Yg = chain_greedy(models, Xte, REV); Yb = chain_beam_any(models, Xte, REV, B=5)
            put((L, 'S3-rev'), te, Pg, Yb, Yg)
            P, Yp = merged_br(L, Xtr, Ytr, Xte, ncat); put((L, 'S1-merged'), te, P, Yp, Yp)
        else:
            P1 = s1(L, Xtr, Ytr, Xte, ncat); put((L, 'S1'), te, P1, decode(P1), raw(P1))
            C.rng = Fixed(SPL[f]); P2 = s2(L, Xtr, Ytr, Xte, ncat); put((L, 'S2'), te, P2, decode(P2), raw(P2))
            if L == 'RF-leaf5':
                rf = make_patched('RF-leaf5').fit(Xtr, Ytr); pr = rf.predict_proba(Xte); P6 = []
                for j in range(m):
                    p = np.zeros((te.sum(), K[j])); p[:, rf.classes_[j].astype(int)] = pr[j]; P6.append(p)
                put((L, 'S6'), te, P6, decode(P6), raw(P6))
        print(f'rep {rep} fold {f} {L} {time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(f=f, store=store), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)
pickle.dump(dict(rep=rep, store=store), open(f'{OUT}/extra_sensitivity_rep{rep}.pkl', 'wb')); print('saved', flush=True)
