"""Aggregate repeats and run the pre-specified analyses.
Usage: PAKDD_DATA=<csv> python3 -I stats.py <outdir>"""
import sys, os, glob, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import binomtest, t as tdist, spearmanr
from common import TGT, DOM, DOMIDX, K, m, n, coherent_rows
from sklearn.metrics import balanced_accuracy_score
OUT = sys.argv[1]
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{OUT}/rep[0-9]*.pkl'))]
for r in reps:   # merge extra learners (MLP here; TFM from Kaggle) run on the same folds
    for p in glob.glob(f"{OUT}/extra_*_rep{r['rep']}.pkl"):
        ex = pickle.load(open(p, 'rb')); r['store'].update(ex['store']); r['lc'] = r['lc'] + ex.get('lc', [])
common_keys = set.intersection(*[set(r['store']) for r in reps])
for r in reps: r['store'] = {k: v for k, v in r['store'].items() if k in common_keys}
R = len(reps); Y = reps[0]['Y']
NT = [TGT.index(c) for c in json.load(open(f'{OUT}/nontrivial.json'))['0.95']]
keys = list(reps[0]['store'].keys())
boot_rng = np.random.default_rng(2027); B = 2000
def ll(Pj, yj): return -np.log(np.clip(Pj[np.arange(len(yj)), yj], 1e-15, 1))

# ---------- main table ----------
rows = []
for k in keys:
    acc, ba, lls, em, sub, viol, emraw, accnt, emnt, subnt = ([] for _ in range(10)); dom = {d: [] for d, v in DOMIDX.items() if len(v) > 1}
    for r in reps:
        s = r['store'][k]; err = (s['Y'] != Y).sum(1); errnt = (s['Y'][:, NT] != Y[:, NT]).sum(1)
        acc.append(np.mean([(s['P'][j].argmax(1) == Y[:, j]).mean() for j in range(m)]))
        ba.append(np.mean([balanced_accuracy_score(Y[:, j], s['P'][j].argmax(1)) for j in range(m)]))
        lls.append(np.mean([ll(s['P'][j], Y[:, j]).mean() for j in range(m)]))
        em.append((err == 0).mean()); sub.append((err <= 1).mean())
        viol.append(1 - coherent_rows(s['Yraw']).mean()); emraw.append(((s['Yraw'] != Y).sum(1) == 0).mean())
        accnt.append(np.mean([(s['P'][j].argmax(1) == Y[:, j]).mean() for j in NT])); emnt.append((errnt == 0).mean()); subnt.append((errnt <= 1).mean())
        for d in dom: dom[d].append((s['Y'][:, DOMIDX[d]] == Y[:, DOMIDX[d]]).all(1).mean())
    rows.append(dict(learner=k[0], strategy=k[1], acc=np.mean(acc), ba=np.mean(ba), logloss=np.mean(lls), em=np.mean(em), em_sd=np.std(em),
                     subem=np.mean(sub), viol_raw=np.mean(viol), em_raw=np.mean(emraw), acc_nt=np.mean(accnt), em_nt=np.mean(emnt), subem_nt=np.mean(subnt),
                     **{f'dom_{d}': np.mean(v) for d, v in dom.items()}))
main = pd.DataFrame(rows); main.to_csv(f'{OUT}/table_main.csv', index=False)

# ---------- comparisons vs matched S1 ----------
def per_patient(k):
    acc = np.mean([(r['store'][k]['Y'] == Y).mean(1) for r in reps], 0)          # decoded per-patient accuracy, repeat-averaged
    accP = np.mean([np.mean([(r['store'][k]['P'][j].argmax(1) == Y[:, j]) for j in range(m)], 0) for r in reps], 0)
    emv = np.array([((r['store'][k]['Y'] != Y).sum(1) == 0) for r in reps]); subv = np.array([((r['store'][k]['Y'] != Y).sum(1) <= 1) for r in reps])
    return accP, emv, subv
def fold_em(k, thr):
    out = []
    for r in reps:
        e = (r['store'][k]['Y'] != Y).sum(1) <= thr
        out += [e[r['fold'] == f].mean() for f in range(5)]
    return np.array(out)
cmp = []
for k in keys:
    L, S = k
    if S in ('S1',) or (L, 'S1') not in reps[0]['store']: continue
    base = (L, 'S1'); aP0, em0, sub0 = per_patient(base); aP1, em1, sub1 = per_patient(k)
    d_acc = aP1 - aP0; bs = [d_acc[boot_rng.integers(0, n, n)].mean() for _ in range(B)]
    row = dict(learner=L, strategy=S, d_acc_pts=100 * d_acc.mean(), d_acc_ci90_lo=100 * np.percentile(bs, 5), d_acc_ci90_hi=100 * np.percentile(bs, 95))
    for name, v0, v1, thr in [('EM', em0, em1, 0), ('SubEM', sub0, sub1, 1)]:
        a = v0.mean(0) > 0.5; b = v1.mean(0) > 0.5          # majority outcome over repeats
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        dd = v1.mean(0).astype(float) - v0.mean(0).astype(float); bsd = [dd[boot_rng.integers(0, n, n)].mean() for _ in range(B)]
        df = fold_em(k, thr) - fold_em(base, thr); kr = len(df)
        if df.var(ddof=1) > 0:
            tt = df.mean() / np.sqrt((1 / kr + 0.25) * df.var(ddof=1)); pt = 2 * tdist.sf(abs(tt), kr - 1)
        else: pt = 1.0
        row.update({f'd_{name}_pts': 100 * dd.mean(), f'{name}_ci95_lo': 100 * np.percentile(bsd, 2.5), f'{name}_ci95_hi': 100 * np.percentile(bsd, 97.5),
                    f'{name}_mcnemar_p': p, f'{name}_ttest_p': pt, f'{name}_disc': f'{n01}/{n10}'})
    cmp.append(row)
cmp = pd.DataFrame(cmp)
fam = ~cmp.strategy.str.contains('greedy')   # pre-specified family: strategies vs matched S1 (greedy chain is a sensitivity variant)
for name in ['EM', 'SubEM']:   # Holm per measure over the family
    p = cmp.loc[fam, f'{name}_mcnemar_p'].values; o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, idx in enumerate(o): run = max(run, min(1, (len(p) - i) * p[idx])); adj[idx] = run
    cmp[f'{name}_holm_p'] = np.nan; cmp.loc[fam, f'{name}_holm_p'] = adj
    # primary test: Holm over the corrected repeated k-fold t-test p-values, the same procedure as the intervals
    p = cmp.loc[fam, f'{name}_ttest_p'].values; o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, idx in enumerate(o): run = max(run, min(1, (len(p) - i) * p[idx])); adj[idx] = run
    cmp[f'{name}_t_holm'] = np.nan; cmp.loc[fam, f'{name}_t_holm'] = adj
cmp.to_csv(f'{OUT}/table_comparisons.csv', index=False)

# ---------- oracle gains, propagation cost ----------
orc = []
for L in ['LR', 'GBT']:
    gpre = np.mean([r['oracle'][L]['g_pre'].mean(0) for r in reps], 0); gall = np.mean([r['oracle'][L]['g_all'].mean(0) for r in reps], 0)
    llbr = np.mean([[ll(r['store'][(L, 'S1')]['P'][j], Y[:, j]).mean() for j in range(m)] for r in reps], 0)
    llmc = np.mean([[ll(r['store'][(L, 'S3')]['P'][j], Y[:, j]).mean() for j in range(m)] for r in reps], 0)
    llgr = np.mean([[ll(r['chain_aux'][L]['greedyP'][j], Y[:, j]).mean() for j in range(m)] for r in reps], 0)
    for j in range(m):
        orc.append(dict(learner=L, decision=TGT[j], pos=j, domain=[d for d, v in DOM.items() if TGT[j] in v][0], g_pre=gpre[j], g_all=gall[j],
                        realised_gain=llbr[j] - llmc[j], prop_cost=llgr[j] - llmc[j]))
orc = pd.DataFrame(orc); orc.to_csv(f'{OUT}/table_oracle.csv', index=False)
summary = {}
for L in ['LR', 'GBT']:
    q = orc[orc.learner == L]; q2 = q[q.pos > 0]
    rho = spearmanr(q2.g_pre, q2.realised_gain)
    summary[L] = dict(G=q.g_pre.sum(), G_all=q.g_all.sum(), G_D={d: q[q.domain == d].g_pre.sum() for d in DOM}, mean_prop_cost=q.prop_cost.mean(),
                      spearman_g_vs_realised=(rho.statistic, rho.pvalue))
dg = []
for L in ['LR', 'GBT']:
    for S in ['S3', 'S5']:
        for d, v in DOMIDX.items():
            if len(v) < 2: continue
            gain = main[(main.learner == L) & (main.strategy == S)][f'dom_{d}'].values[0] - main[(main.learner == L) & (main.strategy == 'S1')][f'dom_{d}'].values[0]
            dg.append(dict(learner=L, strategy=S, domain=d, G_D=summary[L]['G_D'][d], dom_em_gain_pts=100 * gain))
dg = pd.DataFrame(dg); dg.to_csv(f'{OUT}/table_domain_gain.csv', index=False)

# ---------- ECC members vs clinical order ----------
ecc = []
for r in reps:
    for e, Yb in enumerate(r['ecc_Ybeam']): ecc.append(((Yb != Y).sum(1) == 0).mean())
s3_em = main[(main.learner == 'LR') & (main.strategy == 'S3')].em.values[0]
summary['ecc_member_em'] = dict(mean=float(np.mean(ecc)), min=float(np.min(ecc)), max=float(np.max(ecc)), clinical_S3_em=float(s3_em),
                                share_members_below_clinical=float(np.mean(np.array(ecc) < s3_em)))
# ---------- learning curve ----------
lc = pd.DataFrame(sum([r['lc'] for r in reps], []))
lcs = lc.groupby(['learner', 'strategy', 'frac']).sum(numeric_only=True).reset_index()
lcs['em'] = lcs.n_em / lcs.n; lcs['subem'] = lcs.n_sub / lcs.n; lcs['acc'] = lcs.n_correct / (lcs.n * m)
for L in sorted(set(lcs.learner)):
    for S in ['S1', 'S3']:
        rr = main[(main.learner == L) & (main.strategy == S)].iloc[0]
        lcs = pd.concat([lcs, pd.DataFrame([dict(learner=L, strategy=S, frac=1.0, em=rr.em, subem=rr.subem, acc=rr.acc)])])
lcs = lcs.sort_values(['learner', 'strategy', 'frac']); lcs[['learner', 'strategy', 'frac', 'em', 'subem', 'acc']].to_csv(f'{OUT}/table_learning_curve.csv', index=False)
# ---------- S5 ceiling ----------
s5c = {L: {d: np.mean([r['s5_unreach'][L][d] for r in reps]) for d in reps[0]['s5_unreach'][L]} for L in ['LR', 'GBT']}
summary['s5_unreachable_test_combos_per_repeat'] = s5c; summary['repeats'] = R
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(f'=== repeats: {R} ===')
print(main.round(3).to_string(index=False))
print(cmp.round(4).to_string(index=False))
print(json.dumps(summary, indent=1, default=lambda x: round(float(x), 4)))
print(dg.round(3).to_string(index=False))
print(lcs[['learner', 'strategy', 'frac', 'em', 'subem', 'acc']].round(3).to_string(index=False))
