"""Dependence decomposition and equivalence analyses on the out-of-fold predictions (no refitting).
A. Dependence gain on the outer test folds for LR, GBT and TFM (one estimation set), split into
   (i) oracle-parent renormalisation (BR marginals renormalised onto classes coherent with the TRUE parent) and
   (ii) deployed group renormalisation (BR's product distribution renormalised over each constraint group's coherent
        assignments, i.e. the distribution whose mode the coherent decoder returns), with record-bootstrap 95% CIs.
B. Equivalence-style reporting: 90% corrected repeated k-fold t intervals (= TOST at alpha 0.05) for EM and Sub-EM
   differences, and the minimum detectable effect (80% power, two-sided 5%) from the corrected standard error.
C. H1 with the corrected t interval (90%) for mean per-decision accuracy.
Usage: PAKDD_DATA=<csv> python3 -I dependence_equivalence.py <outdir> <teacher_tfm.csv>"""
import sys, os, glob, json, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import t as tdist, norm
from common import TGT, K, m, n, Y as Y0, NONE, ROCK, FFD, FFM, DOMIDX, GROUPS, GA
OUT, TT = sys.argv[1], sys.argv[2]
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{OUT}/rep[0-9]*.pkl'))]
for r in reps:
    for p in glob.glob(f"{OUT}/extra_*_rep{r['rep']}.pkl"):
        r['store'].update(pickle.load(open(p, 'rb'))['store'])
res = {}; brng = np.random.default_rng(2027)
def llv(P, y): return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))

# ---------------- A ----------------
t = pd.read_csv(TT); TF = {}
for rep, g in t.groupby('rep'):
    TF[rep] = [g[g.decision == j].sort_values('row')[[f'p{c}' for c in range(K[j])]].values for j in range(m)]
def oracle_renorm(j, pb):
    if not (j in ROCK[1:] or j == FFM): return pb
    par = ROCK[0] if j in ROCK[1:] else FFD; pn = Y0[:, par] == NONE[par]
    mask = np.ones_like(pb, bool); mask[pn] = False; mask[pn, NONE[j]] = True; mask[~pn, NONE[j]] = False
    q = np.where(mask, pb, 0); return q / np.clip(q.sum(1, keepdims=True), 1e-15, None)
def logZ(P):   # per record: log of the BR probability mass on coherent assignments, summed over the constraint groups
    out = np.zeros(n)
    for g, A in zip(GROUPS, GA):
        s = np.zeros(n)
        for a in A: s += np.prod([P[j][:, a[i]] for i, j in enumerate(g)], axis=0)
        out += np.log(np.clip(s, 1e-300, None))
    return out
A = {}
for L in ['LR', 'GBT', 'TFM']:
    per = []   # per-record arrays averaged over repeats
    br, orc, dep, tf = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n); R = 0
    for r in reps:
        P = r['store'][(L, 'S1')]['P']
        T = TF[r['rep']] if L == 'TFM' else [r['chain_aux'][L]['teacher'][j] for j in range(m)]
        br += sum(llv(P[j], Y0[:, j]) for j in range(m)); tf += sum(llv(T[j], Y0[:, j]) for j in range(m))
        orc += sum(llv(oracle_renorm(j, P[j]), Y0[:, j]) for j in range(m))
        dep += sum(llv(P[j], Y0[:, j]) for j in range(m)) + logZ(P); R += 1
    br, orc, dep, tf = br / R, orc / R, dep / R, tf / R
    G = br - tf; Go = br - orc; Gd = br - dep
    def stat(idx): return [G[idx].mean(), Go[idx].mean() / G[idx].mean(), Gd[idx].mean() / G[idx].mean(), G[idx].mean() / br[idx].mean()]
    est = stat(np.arange(n)); bs = np.array([stat(brng.integers(0, n, n)) for _ in range(2000)])
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    A[L] = dict(G=est[0], G_ci=[lo[0], hi[0]], share_oracle=est[1], share_oracle_ci=[lo[1], hi[1]], share_deployed=est[2], share_deployed_ci=[lo[2], hi[2]],
                rel_to_br=est[3], rel_ci=[lo[3], hi[3]], joint_ll_br=br.mean(), joint_ll_chain_tf=tf.mean(), joint_ll_br_deployed=dep.mean())
res['dependence_test_folds'] = A

# ---------------- B, C ----------------
def fold_vec(r, Yd, thr):
    e = (Yd != Y0).sum(1) <= thr; return np.array([e[r['fold'] == f].mean() for f in range(5)])
def fold_acc(r, P):
    a = np.mean([(P[j].argmax(1) == Y0[:, j]) for j in range(m)], 0); return np.array([a[r['fold'] == f].mean() for f in range(5)])
def ct(d, level):
    k = len(d); se = np.sqrt((1 / k + 0.25) * d.var(ddof=1)); q = tdist.ppf(0.5 + level / 2, k - 1); return d.mean(), d.mean() - q * se, d.mean() + q * se, se
cmp = pd.read_csv(f'{OUT}/table_comparisons.csv'); rows = []
keys = [k for k in reps[0]['store'] if k[1] in ('S2', 'S3', 'S4', 'S5', 'S6') and (k[0], 'S1') in reps[0]['store'] and k[0] in ('LR', 'GBT', 'TFM', 'RF', 'MLP')]
for (L, S) in keys:
    row = dict(learner=L, strategy=S)
    for name, thr in [('EM', 0), ('SubEM', 1)]:
        d = np.concatenate([fold_vec(r, r['store'][(L, S)]['Y'], thr) - fold_vec(r, r['store'][(L, 'S1')]['Y'], thr) for r in reps])
        mu, lo, hi, se = ct(d, 0.90); row.update({f'{name}_diff': 100 * mu, f'{name}_ci90_lo': 100 * lo, f'{name}_ci90_hi': 100 * hi,
                                                  f'{name}_mde80': 100 * (tdist.ppf(0.975, len(d) - 1) + norm.ppf(0.8)) * se})
    d = np.concatenate([fold_acc(r, r['store'][(L, S)]['P']) - fold_acc(r, r['store'][(L, 'S1')]['P']) for r in reps])
    mu, lo, hi, se = ct(d, 0.90); row.update(acc_diff=100 * mu, acc_ci90_lo=100 * lo, acc_ci90_hi=100 * hi)
    c = cmp[(cmp.learner == L) & (cmp.strategy == S)]
    if len(c): row.update(EM_disc=c.EM_disc.values[0], EM_holm=c.EM_holm_p.values[0], SubEM_holm=c.SubEM_holm_p.values[0])
    rows.append(row)
eq = pd.DataFrame(rows); eq.to_csv(f'{OUT}/table_equivalence.csv', index=False)
main = eq[eq.strategy.isin(['S2', 'S3', 'S4'])]
res['equivalence'] = dict(max_EM_hi90_S2S4=float(main.EM_ci90_hi.max()), max_EM_hi90_all=float(eq.EM_ci90_hi.max()),
                          min_EM_lo90_S2S4=float(main.EM_ci90_lo.min()), median_EM_mde80=float(eq.EM_mde80.median()),
                          range_EM_mde80=[float(eq.EM_mde80.min()), float(eq.EM_mde80.max())],
                          H1_within_1pt=eq[(eq.acc_ci90_lo > -1) & (eq.acc_ci90_hi < 1)][['learner', 'strategy']].values.tolist(),
                          H1_fail=eq[~((eq.acc_ci90_lo > -1) & (eq.acc_ci90_hi < 1))][['learner', 'strategy', 'acc_diff', 'acc_ci90_lo', 'acc_ci90_hi']].round(2).values.tolist())
json.dump(res, open(f'{OUT}/dependence_equivalence.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(json.dumps(res['dependence_test_folds'], indent=1, default=lambda x: round(float(x), 3)))
print(eq.round(2).to_string()); print(json.dumps(res['equivalence'], indent=1, default=str))
