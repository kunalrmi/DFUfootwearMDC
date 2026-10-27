"""Replication on public multi-dimensional classification benchmarks (no hard constraints).
Strategies: S1 (binary relevance), S2 (stacked BR, 3-fold OOF), S3 (classifier chain, true-label training, beam 5),
S5 (label powerset over all dimensions, combinations seen < 3 times mapped to the nearest frequent one).
3 repeats x 5-fold CV, same fixed learner settings as the main study (LR C=1; GBT 100 trees, 31 leaves, lr 0.1, 5/leaf).
Usage: python3 -I benchmarks.py <out_csv> <file.mat> [<file.mat> ...]"""
import sys, os, time
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.stats import t as tdist
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def make(L):
    if L == 'LR': return LogisticRegression(C=1.0, max_iter=3000)
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=5, random_state=0)
def proba(clf, X, k):
    p = np.zeros((len(X), k)); p[:, clf.classes_] = clf.predict_proba(X); return p
def fit(L, X, y):
    u = np.unique(y)
    if len(u) < 2:
        class C1:
            classes_ = u
            def predict_proba(self, Z): return np.ones((len(Z), 1))
        return C1()
    return make(L).fit(X, y)
def oh(Y, K, cols): return np.hstack([np.eye(K[j])[Y[:, j]] for j in cols]) if cols else np.zeros((len(Y), 0))

def run(L, Xtr, Ytr, Xte, K):
    m = len(K); out = {}
    P1 = [proba(fit(L, Xtr, Ytr[:, j]), Xte, K[j]) for j in range(m)]
    out['S1'] = np.column_stack([p.argmax(1) for p in P1])
    # S2
    fo = np.arange(len(Xtr)) % 3; oof = [np.zeros((len(Xtr), K[j])) for j in range(m)]
    for k in range(3):
        a, v = fo != k, fo == k
        for j in range(m): oof[j][v] = proba(fit(L, Xtr[a], Ytr[a, j]), Xtr[v], K[j])
    out['S2'] = np.column_stack([proba(fit(L, np.hstack([Xtr] + [oof[i] for i in range(m) if i != j]), Ytr[:, j]),
                                       np.hstack([Xte] + [P1[i] for i in range(m) if i != j]), K[j]).argmax(1) for j in range(m)])
    # S3: chain in index order, beam 5
    models = [fit(L, np.hstack([Xtr, oh(Ytr, K, list(range(p)))]), Ytr[:, p]) for p in range(m)]
    N = len(Xte); beams = np.zeros((N, 1, m), int); score = np.zeros((N, 1))
    for p in range(m):
        nb = beams.shape[1]; flat = beams.reshape(N * nb, m)
        pr = proba(models[p], np.hstack([np.repeat(Xte, nb, 0), oh(flat, K, list(range(p)))]), K[p])
        cand = (np.log(np.clip(pr, 1e-15, 1)) + score.reshape(-1, 1)).reshape(N, nb * K[p]); keep = min(5, nb * K[p])
        top = np.argsort(-cand, 1)[:, :keep]; bi, ci = top // K[p], top % K[p]
        nbm = np.take_along_axis(beams, bi[:, :, None], 1).copy(); nbm[:, :, p] = ci; beams, score = nbm, np.take_along_axis(cand, top, 1)
    out['S3'] = beams[:, 0, :]
    # S5: label powerset with pruning
    keys = [tuple(r) for r in Ytr]; vc = pd.Series(keys).value_counts(); freq = [c for c in vc.index if vc[c] >= 3] or list(vc.index[:1])
    F = np.array(freq)
    def near(r):
        d = (F != np.array(r)).sum(1); best = np.flatnonzero(d == d.min()); return best[np.argmax([vc[freq[b]] for b in best])]
    yc = np.array([freq.index(k) if k in freq else near(k) for k in keys])
    clf = fit(L, Xtr, yc); pp = proba(clf, Xte, len(freq))
    out['S5'] = F[pp.argmax(1)]
    return out

if __name__ == '__main__':
    rows = []
    for path in sys.argv[2:]:
        d = loadmat(path); name = os.path.basename(path)[:-4]
        X = np.asarray(d['data']['norm'][0][0], float); Yraw = np.asarray(d['target'])
        Y = np.column_stack([np.unique(Yraw[:, j], return_inverse=True)[1] for j in range(Yraw.shape[1])]); K = [int(Y[:, j].max()) + 1 for j in range(Y.shape[1])]
        n, m = Y.shape; t0 = time.time()
        for L in ['LR', 'GBT']:
            per = {S: {'em': [], 'acc': [], 'fold_em': []} for S in ['S1', 'S2', 'S3', 'S5']}
            for seed in (101, 202, 303):
                pred = {S: np.zeros_like(Y) for S in per}; fid = np.zeros(n, int)
                for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
                    sc = StandardScaler().fit(X[tr]); o = run(L, sc.transform(X[tr]), Y[tr], sc.transform(X[te]), K); fid[te] = f
                    for S in per: pred[S][te] = o[S]
                for S in per:
                    ok = (pred[S] == Y).all(1); per[S]['em'].append(ok.mean()); per[S]['acc'].append((pred[S] == Y).mean())
                    per[S]['fold_em'].append([ok[fid == f].mean() for f in range(5)])
            base = np.concatenate(per['S1']['fold_em'])
            for S in per:
                dd = np.concatenate(per[S]['fold_em']) - base; k = len(dd); se = np.sqrt((1 / k + 0.25) * dd.var(ddof=1)) if dd.var() > 0 else np.nan
                p = 2 * tdist.sf(abs(dd.mean() / se), k - 1) if se and se == se and se > 0 else 1.0
                rows.append(dict(dataset=name, n=n, dims=m, classes='/'.join(map(str, K)), learner=L, strategy=S, acc=100 * np.mean(per[S]['acc']),
                                 em=100 * np.mean(per[S]['em']), d_em_vs_S1=100 * dd.mean(), p_corrected_t=p))
        print(name, n, K, f'{time.time()-t0:.0f}s', flush=True)
    t = pd.DataFrame(rows); t.to_csv(sys.argv[1], index=False); print(t.round(3).to_string())
