"""Public MDC benchmarks (no hard constraints): S1-S5 with the paper's fixed learner settings, plus the dependence diagnostic.
S1 binary relevance; S2 stacked BR (3-fold out-of-fold); S3 classifier chain (index order, true-label training, beam 5);
S4 ensemble of ten random-order chains (averaged greedy marginals); S5 label powerset over all dimensions (combinations
seen < 3 times mapped to the nearest frequent one). Diagnostic: teacher-forced chain log-loss vs BR log-loss on the test fold.
3 repeats x 5-fold CV (seeds 101/202/303). Usage:
    python3 -I bench2.py <out_dir> <learners: LR,GBT,RF[,TFM]> <file.mat> [...]
Writes <out_dir>/bench2_<dataset>_<learner>.csv (one row per seed x fold x strategy); skips existing files (resumable)."""
import sys, os, time
import numpy as np, pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def make(L):
    if L == 'LR': return LogisticRegression(C=1.0, max_iter=3000)
    if L == 'GBT': return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=5, random_state=0)
    if L == 'RF': return RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0)
    if L == 'TFM':
        from tfm_ecoc import ECOC_TFM
        return ECOC_TFM()
    raise ValueError(L)
class Const:
    def __init__(self, c): self.classes_ = np.array([c])
    def predict_proba(self, X): return np.ones((len(X), 1))
def fit(L, X, y):
    u = np.unique(y)
    return Const(u[0]) if len(u) < 2 else make(L).fit(X, y)
def proba(clf, X, k):
    p = np.zeros((len(X), k)); p[:, np.asarray(clf.classes_).astype(int)] = clf.predict_proba(X); return p
def oh(Y, K, cols): return np.hstack([np.eye(K[j])[Y[:, j]] for j in cols]) if len(cols) else np.zeros((len(Y), 0))
def ll(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))

def fit_chain(L, X, Y, K, order):
    return [fit(L, np.hstack([X, oh(Y, K, order[:p])]), Y[:, order[p]]) for p in range(len(order))]
def chain_greedy(models, X, K, order):
    m = len(K); Yp = np.zeros((len(X), m), int); P = [None] * m
    for p, j in enumerate(order):
        P[j] = proba(models[p], np.hstack([X, oh(Yp, K, order[:p])]), K[j]); Yp[:, j] = P[j].argmax(1)
    return P, Yp
def chain_beam(models, X, K, order, B=5):
    m = len(K); N = len(X); beams = np.zeros((N, 1, m), int); score = np.zeros((N, 1))
    for p, j in enumerate(order):
        nb = beams.shape[1]; flat = beams.reshape(N * nb, m)
        pr = proba(models[p], np.hstack([np.repeat(X, nb, 0), oh(flat, K, order[:p])]), K[j])
        cand = (np.log(np.clip(pr, 1e-15, 1)) + score.reshape(-1, 1)).reshape(N, nb * K[j]); keep = min(B, nb * K[j])
        top = np.argsort(-cand, 1)[:, :keep]; bi, ci = top // K[j], top % K[j]
        nbm = np.take_along_axis(beams, bi[:, :, None], 1).copy(); nbm[:, :, j] = ci; beams, score = nbm, np.take_along_axis(cand, top, 1)
    return beams[:, 0, :]
def chain_teacher(models, X, Yt, K, order):
    return {j: proba(models[p], np.hstack([X, oh(Yt, K, order[:p])]), K[j]) for p, j in enumerate(order)}

def run_fold(L, Xtr, Ytr, Xte, Yte, K, seed, f):
    m = len(K); out = {}; order = list(range(m))
    P1 = [proba(fit(L, Xtr, Ytr[:, j]), Xte, K[j]) for j in range(m)]
    out['S1'] = np.column_stack([p.argmax(1) for p in P1])
    fo = np.arange(len(Xtr)) % 3; np.random.default_rng(seed * 10 + f).shuffle(fo); oof = [np.zeros((len(Xtr), K[j])) for j in range(m)]
    for k in range(3):
        a, v = fo != k, fo == k
        for j in range(m): oof[j][v] = proba(fit(L, Xtr[a], Ytr[a, j]), Xtr[v], K[j])
    out['S2'] = np.column_stack([proba(fit(L, np.hstack([Xtr] + [oof[i] for i in range(m) if i != j]), Ytr[:, j]),
                                       np.hstack([Xte] + [P1[i] for i in range(m) if i != j]), K[j]).argmax(1) for j in range(m)])
    models = fit_chain(L, Xtr, Ytr, K, order)
    out['S3'] = chain_beam(models, Xte, K, order)
    T = chain_teacher(models, Xte, Yte, K, order)
    llbr = np.array([ll(P1[j], Yte[:, j]).sum() for j in range(m)]); lltf = np.array([ll(T[j], Yte[:, j]).sum() for j in range(m)])
    r = np.random.default_rng(seed + 7 + 1000 * f); Pe = [np.zeros((len(Xte), K[j])) for j in range(m)]
    for e in range(10):
        o = list(r.permutation(m)); pg, _ = chain_greedy(fit_chain(L, Xtr, Ytr, K, o), Xte, K, o)
        for j in range(m): Pe[j] += pg[j] / 10
    out['S4'] = np.column_stack([p.argmax(1) for p in Pe])
    keys = [tuple(x) for x in Ytr]; vc = pd.Series(keys).value_counts(); freq = [c for c in vc.index if vc[c] >= 3] or list(vc.index[:1]); F = np.array(freq)
    idx = {c: i for i, c in enumerate(freq)}
    def near(x):
        d = (F != np.array(x)).sum(1); best = np.flatnonzero(d == d.min()); return best[np.argmax([vc[freq[b]] for b in best])]
    yc = np.array([idx[k] if k in idx else near(k) for k in keys])
    out['S5'] = F[proba(fit(L, Xtr, yc), Xte, len(freq)).argmax(1)]
    return out, llbr, lltf

if __name__ == '__main__':
    OUT, LEARN = sys.argv[1], sys.argv[2].split(','); os.makedirs(OUT, exist_ok=True)
    for path in sys.argv[3:]:
        d = loadmat(path); name = os.path.basename(path)[:-4]
        X = np.asarray(d['data']['norm'][0][0], float); Yraw = np.asarray(d['target'])
        Y = np.column_stack([np.unique(Yraw[:, j], return_inverse=True)[1] for j in range(Yraw.shape[1])]); K = [int(Y[:, j].max()) + 1 for j in range(Y.shape[1])]
        n, m = Y.shape
        for L in LEARN:
            fn = f'{OUT}/bench2_{name}_{L}.csv'
            if os.path.exists(fn): continue
            rows = []; t0 = time.time(); part = fn + '.part.csv'
            if os.path.exists(part): rows = pd.read_csv(part).to_dict('records')
            donef = {(r['seed'], r['fold']) for r in rows}
            for seed in (101, 202, 303):
                for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
                    if (seed, f) in donef: continue
                    sc = StandardScaler().fit(X[tr]); o, llbr, lltf = run_fold(L, sc.transform(X[tr]), Y[tr], sc.transform(X[te]), Y[te], K, seed, f)
                    for S, Yp in o.items():
                        err = (Yp != Y[te]).sum(1)
                        rows.append(dict(dataset=name, n=n, dims=m, classes='/'.join(map(str, K)), learner=L, strategy=S, seed=seed, fold=f, n_test=int(len(te)),
                                         n_em=int((err == 0).sum()), n_sub=int((err <= 1).sum()), n_correct=int((Yp == Y[te]).sum()),
                                         ll_br=float(llbr.sum()) if S == 'S1' else np.nan, ll_tf=float(lltf.sum()) if S == 'S3' else np.nan,
                                         ll_br_first=float(llbr[0]) if S == 'S1' else np.nan))
                    pd.DataFrame(rows).to_csv(part + '.tmp', index=False); os.replace(part + '.tmp', part)   # per-fold checkpoint
            pd.DataFrame(rows).to_csv(fn + '.tmp', index=False); os.replace(fn + '.tmp', fn)
            if os.path.exists(part): os.remove(part)
            print(f'{name} n={n} K={K} {L}: {time.time()-t0:.0f}s', flush=True)
