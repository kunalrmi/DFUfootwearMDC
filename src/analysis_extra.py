"""Additional analyses on the out-of-fold predictions.
Usage: PAKDD_DATA=<csv> python3 -I analysis_extra.py <outdir>
Writes <outdir>/extra_analyses.json and <outdir>/table_baselines.csv, table_domain_tests.csv, table_ci_corrected.csv."""
import sys, os, glob, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist, binomtest
from sklearn.neighbors import KNeighborsClassifier
from common import *

OUT = sys.argv[1]
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{OUT}/rep[0-9]*.pkl'))]
for r in reps:
    for p in glob.glob(f"{OUT}/extra_*_rep{r['rep']}.pkl"):
        ex = pickle.load(open(p, 'rb')); r['store'].update(ex['store']); r.setdefault('lc_extra', []).extend(ex.get('lc', []))
Y0 = reps[0]['Y']; R = len(reps); res = {}
keys = [k for k in reps[0]['store'] if all(k in r['store'] for r in reps)]

# ---------- 1. corrected repeated k-fold t intervals for EM / Sub-EM differences ----------
def fold_vec(r, Yd, thr, cols=None):
    c = list(range(m)) if cols is None else cols
    e = ((Yd[:, c] != Y0[:, c]).sum(1) <= thr)
    return np.array([e[r['fold'] == f].mean() for f in range(5)])
rows = []
for k in keys:
    L, S = k
    if S == 'S1' or (L, 'S1') not in reps[0]['store']: continue
    for name, thr in [('EM', 0), ('SubEM', 1)]:
        d = np.concatenate([fold_vec(r, r['store'][k]['Y'], thr) - fold_vec(r, r['store'][(L, 'S1')]['Y'], thr) for r in reps])
        kk = len(d); se = np.sqrt((1 / kk + 0.25) * d.var(ddof=1)); q = tdist.ppf(0.975, kk - 1)
        rows.append(dict(learner=L, strategy=S, measure=name, diff_pts=100 * d.mean(), lo=100 * (d.mean() - q * se), hi=100 * (d.mean() + q * se)))
ci = pd.DataFrame(rows); ci.to_csv(f'{OUT}/table_ci_corrected.csv', index=False)
fam = ci[~ci.strategy.str.contains('greedy')]
res['max_upper_corrected'] = {m_: float(fam[fam.measure == m_].hi.max()) for m_ in ['EM', 'SubEM']}

# ---------- 2. domain exact match: S3/S2/S6 vs S1, bootstrap CI + McNemar on majority outcome ----------
brng = np.random.default_rng(2027); drows = []
for k in keys:
    L, S = k
    if S in ('S1',) or 'greedy' in S or (L, 'S1') not in reps[0]['store']: continue
    for dname, idx in DOMIDX.items():
        if len(idx) < 2: continue
        v1 = np.array([(r['store'][k]['Y'][:, idx] == Y0[:, idx]).all(1) for r in reps]).mean(0)
        v0 = np.array([(r['store'][(L, 'S1')]['Y'][:, idx] == Y0[:, idx]).all(1) for r in reps]).mean(0)
        dd = v1 - v0; bs = [dd[brng.integers(0, n, n)].mean() for _ in range(2000)]
        a, b = v0 > 0.5, v1 > 0.5; n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        fd = np.concatenate([np.array([((r['store'][k]['Y'][:, idx] == Y0[:, idx]).all(1)[r['fold'] == f].mean() - (r['store'][(L, 'S1')]['Y'][:, idx] == Y0[:, idx]).all(1)[r['fold'] == f].mean()) for f in range(5)]) for r in reps])
        kk = len(fd); se = np.sqrt((1 / kk + 0.25) * fd.var(ddof=1)); pt = 2 * tdist.sf(abs(fd.mean() / se), kk - 1) if se > 0 else 1.0
        drows.append(dict(learner=L, strategy=S, domain=dname, s1=100 * v0.mean(), strat=100 * v1.mean(), diff=100 * dd.mean(),
                          lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5), p_mcnemar=p, p=pt))
dt = pd.DataFrame(drows)
dt['holm'] = np.nan
for L, g in dt.groupby('learner'):          # Holm within learner over its strategy x domain tests
    p = g.p.values; o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, ix in enumerate(o): run = max(run, min(1, (len(p) - i) * p[ix])); adj[ix] = run
    dt.loc[g.index, 'holm'] = adj
dt.to_csv(f'{OUT}/table_domain_tests.csv', index=False)
res['domain_sig_after_holm'] = dt[dt.holm < 0.05][['learner', 'strategy', 'domain', 'diff']].round(2).to_dict('records')

# ---------- 3. decoded (delivered) Hamming accuracy ----------
res['decoded_hamming_acc'] = {f'{L}-{S}': float(np.mean([(r['store'][(L, S)]['Y'] == Y0).mean() for r in reps])) for (L, S) in keys}

# ---------- 4. trivial baselines: majority class and 1-NN (copies the neighbour's complete prescription) ----------
brow = []
for name in ['Majority', '1-NN', '5-NN']:
    em, sub, acc = [], [], []
    for r in reps:
        Yd = np.zeros_like(Y0)
        for f in range(5):
            tr, te = r['fold'] != f, r['fold'] == f
            if name == 'Majority':
                Yd[te] = [np.bincount(Y0[tr, j], minlength=K[j]).argmax() for j in range(m)]
            else:
                enc = Enc('onehot').fit(X[tr]); A, B = enc.transform(X[tr]), enc.transform(X[te])
                kn = KNeighborsClassifier(n_neighbors=int(name[0])).fit(A, np.arange(tr.sum()))
                if name == '1-NN':
                    Yd[te] = Y0[tr][kn.predict(B)]
                else:                                   # per-decision vote over 5 neighbours, then coherent decoding
                    nb = kn.kneighbors(B, return_distance=False); Ytr = Y0[tr]
                    P = [np.stack([np.bincount(Ytr[nb[i], j], minlength=K[j]) / 5 for i in range(len(nb))]) for j in range(m)]
                    Yd[te] = decode(P)
        err = (Yd != Y0).sum(1); em.append((err == 0).mean()); sub.append((err <= 1).mean()); acc.append((Yd == Y0).mean())
    brow.append(dict(baseline=name, acc=100 * np.mean(acc), em=100 * np.mean(em), subem=100 * np.mean(sub), viol=100 * (1 - coherent_rows(Yd).mean())))
bt = pd.DataFrame(brow); bt.to_csv(f'{OUT}/table_baselines.csv', index=False)

# ---------- 5. learning-curve corrected t-tests for every learner ----------
lc = pd.DataFrame(sum([r['lc'] for r in reps], []) + sum([r.get('lc_extra', []) for r in reps], []))
lct = {}
for L in sorted(set(lc.learner)):
    for fr in sorted(set(lc.frac)):
        a = lc[(lc.learner == L) & (lc.frac == fr)]
        p = a.pivot_table(index=['rep', 'fold'], columns='strategy', values=['n_em', 'n'])
        d = ((p[('n_em', 'S3')] - p[('n_em', 'S1')]) / p[('n', 'S1')]).values; k = len(d); ratio = 97 / (390 * fr)
        tt = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
        lct[f'{L}@{fr}'] = dict(diff_pts=float(100 * d.mean()), p=float(2 * tdist.sf(abs(tt), k - 1)))
res['learning_curve_ttests'] = lct

# ---------- 6. oracle gain: share carried by constraint-linked decisions (inner CV) ----------
o = pd.read_csv(f'{OUT}/table_oracle.csv'); C = [TGT[j] for j in ROCK[1:]] + [TGT[FFM]]
res['inner_cv_share_constraint_linked'] = {L: float(o[(o.learner == L) & o.decision.isin(C)].g_pre.sum() / o[o.learner == L].g_pre.sum()) for L in ['LR', 'GBT']}
res['negative_gj'] = {L: int((o[(o.learner == L) & (o.pos > 0)].g_pre < 0).sum()) for L in ['LR', 'GBT']}
json.dump(res, open(f'{OUT}/extra_analyses.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 200)
print(ci.round(2).to_string()); print(bt.round(2).to_string()); print(dt.round(3).to_string())
print(json.dumps({k: v for k, v in res.items() if k != 'domain_sig_after_holm'}, indent=1, default=lambda x: round(float(x), 4)))
print('domain sig:', res['domain_sig_after_holm'])
