"""One repeat of the full experiment (5-fold iterative-stratified CV) for LR, GBT (sklearn HGB) and RF.
Usage: PAKDD_DATA=<csv> python3 -I run_repeat.py <repeat_id> <seed> <outdir>
Writes <outdir>/rep<id>.pkl with out-of-fold predictions, per-fold measures, oracle gains,
ECC member results, learning-curve results and S5 ceilings."""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common as C
from common import *

rep, seed, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
C.rng = np.random.default_rng(seed)
rng_local = np.random.default_rng(seed + 7)
fold = iter_strat(Y, 5, seed=seed)
order = list(range(m))
LEARNERS = ['LR', 'GBT', 'RF']
store = {}
def put(key, te, P, Yp, Yraw):
    s = store.setdefault(key, {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)})
    for j in range(m): s['P'][j][te] = P[j]
    s['Y'][te] = Yp; s['Yraw'][te] = Yraw

def ll_vec(Pj, yj): return -np.log(np.clip(Pj[np.arange(len(yj)), yj], 1e-15, 1))
oracle = {L: {'g_pre': np.zeros((5, m)), 'g_all': np.zeros((5, m))} for L in ['LR', 'GBT']}
chain_aux = {L: {'teacher': [np.zeros((n, K[j])) for j in range(m)], 'greedyP': [np.zeros((n, K[j])) for j in range(m)]} for L in ['LR', 'GBT']}
ecc_members = []   # (fold, member, order, Ybeam over test idx)
ecc_Ybeam = [np.zeros((n, m), int) for _ in range(10)]
s5_unreach = {L: {} for L in ['LR', 'GBT']}
lc = []            # learning curve rows
t0 = time.time()
CK = f'{OUT}/ckpt_rep{rep}.pkl'; f0 = 0
if os.path.exists(CK):   # resume after an interrupted run (container restarts)
    ck = pickle.load(open(CK, 'rb')); f0 = ck['f'] + 1
    store, oracle, chain_aux, ecc_members, ecc_Ybeam, s5_unreach, lc = (ck[k] for k in ['store', 'oracle', 'chain_aux', 'ecc_members', 'ecc_Ybeam', 's5_unreach', 'lc'])
    C.rng, rng_local = ck['rng'], ck['rng_local']; print(f'resumed at fold {f0}', flush=True)
for f in range(f0, 5):
    tr, te = fold != f, fold == f
    for L in LEARNERS:
        enc = Enc(ENCKIND[L]).fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te])
        ncat = len(CAT) if L == 'GBT' else 0
        Ytr = Y[tr]
        P1 = s1(L, Xtr, Ytr, Xte, ncat)
        put((L, 'S1'), te, P1, decode(P1), np.column_stack([p.argmax(1) for p in P1]))
        if L == 'RF':
            rf = make('RF').fit(Xtr, Ytr); pr = rf.predict_proba(Xte); P6 = []
            for j in range(m):
                p = np.zeros((te.sum(), K[j])); p[:, rf.classes_[j].astype(int)] = pr[j]; P6.append(p)
            put((L, 'S6'), te, P6, decode(P6), np.column_stack([p.argmax(1) for p in P6]))
            continue
        P2 = s2(L, Xtr, Ytr, Xte, ncat)
        put((L, 'S2'), te, P2, decode(P2), np.column_stack([p.argmax(1) for p in P2]))
        models = fit_chain(L, Xtr, Ytr, order, ncat)
        Pg, Yg = chain_greedy(models, Xte, order)
        Pmc = chain_mc(models, Xte, order, S=100, seed=seed * 10 + f)
        Yb = chain_beam(models, Xte, order, B=5)
        put((L, 'S3'), te, Pmc, Yb, Yg)
        put((L, 'S3-greedy'), te, Pg, Yg, Yg)
        T = chain_teacher(models, Xte, Y[te], order)
        for j in range(m):
            chain_aux[L]['teacher'][j][te] = T[j]; chain_aux[L]['greedyP'][j][te] = Pg[j]
        P5, Y5, unr = s5_predict(L, Xtr, Ytr, Xte, ncat)
        put((L, 'S5'), te, P5, Y5, Y5)
        for dname, idx in DOMIDX.items():
            if len(idx) > 1:
                ok = (Y[te][:, idx][:, None, :] == unr[dname][None]).all(2).any(1)
                s5_unreach[L][dname] = s5_unreach[L].get(dname, 0) + int((~ok).sum())
        # ---- oracle gain by inner 5-fold CV within the training fold ----
        inner = iter_strat(Ytr, 5, seed=seed + 100 + f)
        llbr = np.zeros(m); llpre = np.zeros(m); llall = np.zeros(m); ntot = 0
        Xtr_df = X[tr].reset_index(drop=True)
        for g in range(5):
            itr, iva = inner != g, inner == g
            e2 = Enc(ENCKIND[L]).fit(Xtr_df[itr]); A, B = e2.transform(Xtr_df[itr]), e2.transform(Xtr_df[iva])
            Ya, Yv = Ytr[itr], Ytr[iva]
            pb = s1(L, A, Ya, B, ncat)
            mods = fit_chain(L, A, Ya, order, ncat)
            tp = chain_teacher(mods, B, Yv, order)
            for j in range(m):
                llbr[j] += ll_vec(pb[j], Yv[:, j]).sum(); llpre[j] += ll_vec(tp[j], Yv[:, j]).sum()
                others = [i for i in range(m) if i != j]
                ca = fitc(L, ncat, np.hstack([A, onehot_y(Ya[:, others], others)]), Ya[:, j])
                llall[j] += ll_vec(fullproba(ca, np.hstack([B, onehot_y(Yv[:, others], others)]), K[j]), Yv[:, j]).sum()
            ntot += iva.sum()
        oracle[L]['g_pre'][f] = (llbr - llpre) / ntot; oracle[L]['g_all'][f] = (llbr - llall) / ntot
        # ---- ECC (LR only): average of 10 random-order chains; members also decoded by beam search ----
        if L == 'LR':
            Pe = [np.zeros((te.sum(), K[j])) for j in range(m)]
            for e in range(10):
                o = list(rng_local.permutation(m)); me = fit_chain(L, Xtr, Ytr, o, ncat)
                pg, _ = chain_greedy(me, Xte, o)
                for j in range(m): Pe[j] += pg[j] / 10
                # beam with coherence pruning only valid when constraint parents precede children
                ecc_Ybeam[e][te] = chain_beam_any(me, Xte, o)
            put((L, 'S4'), te, Pe, decode(Pe), np.column_stack([p.argmax(1) for p in Pe]))
        # ---- learning curve: S1 vs S3 at 25/50/75% of the training fold (100% = main run) ----
        tri = np.where(tr)[0]
        for frac in [0.25, 0.5, 0.75]:
            sub = iter_strat(Ytr, 4, seed=seed + 200 + f)  # quarters, stratified
            keep = sub < int(round(frac * 4))
            Xs_df = X.iloc[tri[keep]]
            e3 = Enc(ENCKIND[L]).fit(Xs_df); A, B = e3.transform(Xs_df), e3.transform(X[te])
            pa = s1(L, A, Ytr[keep], B, ncat); ya = decode(pa)
            ms = fit_chain(L, A, Ytr[keep], order, ncat); yb = chain_beam(ms, B, order, B=5)
            for S, yp in [('S1', ya), ('S3', yb)]:
                err = (yp != Y[te]).sum(1)
                lc.append(dict(rep=rep, fold=f, learner=L, frac=frac, strategy=S, n_em=int((err == 0).sum()), n_sub=int((err <= 1).sum()),
                               n_correct=int((yp == Y[te]).sum()), n=int(te.sum())))
    print(f'rep {rep} fold {f} done {time.time()-t0:.0f}s', flush=True)
    pickle.dump(dict(f=f, store=store, oracle=oracle, chain_aux=chain_aux, ecc_members=ecc_members, ecc_Ybeam=ecc_Ybeam, s5_unreach=s5_unreach, lc=lc, rng=C.rng, rng_local=rng_local), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)

pickle.dump(dict(rep=rep, seed=seed, fold=fold, Y=Y, store={k: {'P': v['P'], 'Y': v['Y'], 'Yraw': v['Yraw']} for k, v in store.items()},
                 oracle=oracle, chain_aux=chain_aux, ecc_Ybeam=ecc_Ybeam, s5_unreach=s5_unreach, lc=lc), open(f'{OUT}/rep{rep}.pkl', 'wb'))
print(f'rep {rep} saved, total {time.time()-t0:.0f}s', flush=True)
