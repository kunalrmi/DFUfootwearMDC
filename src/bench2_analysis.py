"""Aggregate bench2 results: per dataset x learner x strategy exact match and difference from S1 (corrected t over 15 folds),
cross-dataset tests per learner and strategy (Wilcoxon signed-rank; Bayesian signed-rank with ROPE of +-1 point), and H3 across
datasets (relative teacher-forced gain vs realised exact-match gain of S3 and S4).
Usage: python3 -I bench2_analysis.py <dir with bench2_*.csv> <out_prefix>"""
import sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import t as tdist, wilcoxon, spearmanr
D, OUTP = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D}/bench2_*.csv')) if not f.endswith('.part.csv')])
from sklearn.model_selection import KFold
def fold_size(row):   # recompute (early runs stored a wrong n_test)
    return len(list(KFold(5, shuffle=True, random_state=int(row.seed)).split(np.zeros(int(row.n))))[int(row.fold)][1])
sz = {k: fold_size(pd.Series(dict(seed=k[0], fold=k[1], n=k[2]))) for k in set(zip(d.seed, d.fold, d.n))}
d['n_test'] = [sz[k] for k in zip(d.seed, d.fold, d.n)]
d['em'] = d.n_em / d.n_test; d['sub'] = d.n_sub / d.n_test
rows = []
for (ds, L), g in d.groupby(['dataset', 'learner']):
    b = g[g.strategy == 'S1'].set_index(['seed', 'fold'])
    G = (b.ll_br - g[g.strategy == 'S3'].set_index(['seed', 'fold']).ll_tf).sum() / b.n_test.sum()
    rel = G / (b.ll_br.sum() / b.n_test.sum())
    for S, h in g.groupby('strategy'):
        h = h.set_index(['seed', 'fold']).loc[b.index]; dd = (h.em - b.em).values; k = len(dd)
        se = np.sqrt((1 / k + 0.25) * dd.var(ddof=1)) if dd.var() > 0 else np.nan
        p = 2 * tdist.sf(abs(dd.mean() / se), k - 1) if se == se and se > 0 else 1.0
        rows.append(dict(dataset=ds, n=int(g.n.iloc[0]), dims=int(g.dims.iloc[0]), learner=L, strategy=S, em=100 * h.n_em.sum() / h.n_test.sum(),
                         sub=100 * h.n_sub.sum() / h.n_test.sum(), d_em=100 * dd.mean(), p=p, G=G, G_rel=rel))
t = pd.DataFrame(rows); t.to_csv(f'{OUTP}_table.csv', index=False)

def bayes_signrank(x, rope=1.0, ns=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al. 2017): P(left), P(rope), P(right) for differences x (in points)."""
    r = np.random.default_rng(seed); x = np.concatenate([[0.0], np.asarray(x, float)]); n = len(x)
    W = (x[:, None] + x[None, :]) / 2; L = (W < -rope); R = (W > rope); M = ~(L | R)
    a = np.ones(n); a[0] = 0.5; out = np.zeros(3)
    for _ in range(ns // 100):
        w = r.dirichlet(a, size=100)
        pl = np.einsum('si,ij,sj->s', w, L, w); pm = np.einsum('si,ij,sj->s', w, M, w); pr = np.einsum('si,ij,sj->s', w, R, w)
        win = np.argmax(np.stack([pl, pm, pr], 1), 1); out += np.bincount(win, minlength=3)
    return out / out.sum()
cross = []
for (L, S), g in t[t.strategy != 'S1'].groupby(['learner', 'strategy']):
    x = g.d_em.values
    pw = wilcoxon(x).pvalue if np.any(x != 0) and len(x) >= 5 else np.nan
    bl, bm, br = bayes_signrank(x)
    cross.append(dict(learner=L, strategy=S, n_datasets=len(x), median_d=np.median(x), mean_d=x.mean(), wins=int((x > 0).sum()), losses=int((x < 0).sum()),
                      n_sig_pos=int(((g.d_em > 0) & (g.p < 0.05)).sum()), n_sig_neg=int(((g.d_em < 0) & (g.p < 0.05)).sum()),
                      wilcoxon_p=pw, p_left=bl, p_rope=bm, p_right=br))
cr = pd.DataFrame(cross); cr.to_csv(f'{OUTP}_cross.csv', index=False)
h3 = {}
for S in ['S3', 'S4']:
    q = t[t.strategy == S]; rho = spearmanr(q.G_rel, q.d_em); h3[S] = dict(rho=rho.statistic, p=rho.pvalue, n=len(q))
json.dump(dict(h3=h3, datasets=sorted(t.dataset.unique()), learners=sorted(t.learner.unique())), open(f'{OUTP}_summary.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 220)
print(t.pivot_table(index=['dataset', 'learner'], columns='strategy', values='d_em').round(1).to_string())
print(t[t.strategy == 'S1'][['dataset', 'learner', 'em', 'G_rel']].round(3).to_string())
print(cr.round(3).to_string()); print(h3)
