"""Inference checks (same folds as the main run).
(a) beam-width sweep for the clinical-order chain (B = 1, 5, 20, 100) with LR and GBT: exact match and how often a wider
    beam finds a prescription with higher chain probability than the narrower one;
(b) GBT with within-fold sigmoid calibration (3-fold CalibratedClassifierCV) for S1, S2 and S3 (beam 5);
(c) joint-mode voting for the ensemble of chains (from the stored member predictions).
Usage: PAKDD_DATA=<csv> python3 -I inference_checks.py <main_outdir> <outdir>   (checkpoints after every fold)"""
import sys, os, glob, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import common as C
from common import *

MAIN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
reps = [pickle.load(open(p, 'rb')) for p in sorted(glob.glob(f'{MAIN}/rep[0-9]*.pkl'))]
order = list(range(m)); BW = [1, 5, 20, 100]

def chain_logp(models, Xte, Yp):
    lp = np.zeros(len(Xte))
    for p in range(m):
        j = order[p]; pr = fullproba(models[p], np.hstack([Xte, onehot_y(Yp[:, order[:p]], order[:p])]), K[j])
        lp += np.log(np.clip(pr[np.arange(len(Xte)), Yp[:, j]], 1e-15, 1))
    return lp

_orig = C.make
def make_cal(learner, ncat=0, multi=False):
    if learner == 'GBTcal':
        return CalibratedClassifierCV(_orig('GBT', ncat), method='sigmoid', cv=3)
    return _orig(learner, ncat, multi)
C.make = make_cal
def fitc_safe(L, ncat, Xa, y):           # calibration needs >= 3 records per class; fall back to the uncalibrated model
    u, cnt = np.unique(y, return_counts=True)
    if L == 'GBTcal' and (len(u) < 2 or cnt.min() < 3): return C.fitc('GBT', ncat, Xa, y)
    return C.fitc(L, ncat, Xa, y)

for r in reps:
    rep = r['rep']; ck = f'{OUT}/inf_rep{rep}.pkl'
    st = pickle.load(open(ck, 'rb')) if os.path.exists(ck) else {'done': [], 'Y': {}, 'lp': {}}
    C.rng = np.random.default_rng(r['seed'])
    for f in range(5):
        if f in st['done']: continue
        t0 = time.time(); tr, te = r['fold'] != f, r['fold'] == f; ti = np.where(te)[0]
        for L in ['LR', 'GBT']:
            enc = Enc(ENCKIND[L]).fit(X[tr]); A, B = enc.transform(X[tr]), enc.transform(X[te]); ncat = len(CAT) if L == 'GBT' else 0
            models = fit_chain(L, A, Y[tr], order, ncat)
            for bw in BW:
                Yb = chain_beam(models, B, order, B=bw)
                st['Y'].setdefault((L, f'beam{bw}'), np.zeros((n, m), int))[ti] = Yb
                st['lp'].setdefault((L, f'beam{bw}'), np.zeros(n))[ti] = chain_logp(models, B, Yb)
        # calibrated GBT: S1, S2, S3 (beam 5)
        enc = Enc('ordinal').fit(X[tr]); A, B = enc.transform(X[tr]), enc.transform(X[te]); ncat = len(CAT)
        P1 = [fullproba(fitc_safe('GBTcal', ncat, A, Y[tr][:, j]), B, K[j]) for j in range(m)]
        st['Y'].setdefault(('GBTcal', 'S1'), np.zeros((n, m), int))[ti] = decode(P1)
        st.setdefault('P', {}).setdefault(('GBTcal', 'S1'), [np.zeros((n, K[j])) for j in range(m)])
        for j in range(m): st['P'][('GBTcal', 'S1')][j][ti] = P1[j]
        # S2 with calibrated first and second stage (3-fold OOF first stage, as in the main run)
        rr = C.rng; idx = rr.permutation(tr.sum()); fo = np.arange(tr.sum()) % 3; fo = fo[np.argsort(idx)]
        oof = [np.zeros((tr.sum(), K[j])) for j in range(m)]
        for k in range(3):
            a, v = fo != k, fo == k
            for j in range(m): oof[j][v] = fullproba(fitc_safe('GBTcal', ncat, A[a], Y[tr][a, j]), A[v], K[j])
        P2 = []
        for j in range(m):
            others = [jj for jj in range(m) if jj != j]
            A2 = np.hstack([A] + [oof[jj] for jj in others]); B2 = np.hstack([B] + [P1[jj] for jj in others])
            P2.append(fullproba(fitc_safe('GBTcal', ncat, A2, Y[tr][:, j]), B2, K[j]))
        st['Y'].setdefault(('GBTcal', 'S2'), np.zeros((n, m), int))[ti] = decode(P2)
        models = [fitc_safe('GBTcal', ncat, np.hstack([A, onehot_y(Y[tr][:, order[:p]], order[:p])]), Y[tr][:, order[p]]) for p in range(m)]
        st['Y'].setdefault(('GBTcal', 'S3'), np.zeros((n, m), int))[ti] = chain_beam(models, B, order, B=5)
        st['done'].append(f); pickle.dump(st, open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck)
        print(f'rep {rep} fold {f} done in {time.time()-t0:.0f}s', flush=True)

# ---------------- summary ----------------
sts = {r['rep']: pickle.load(open(f'{OUT}/inf_rep{r["rep"]}.pkl', 'rb')) for r in reps}
rows = []
def em(Yd): return ((Yd != Y).sum(1) == 0).mean()
def sub(Yd): return ((Yd != Y).sum(1) <= 1).mean()
for L in ['LR', 'GBT']:
    for bw in BW:
        k = (L, f'beam{bw}')
        better = np.mean([(sts[r['rep']]['lp'][k] > sts[r['rep']]['lp'][(L, 'beam1')] + 1e-9).mean() for r in reps])
        better5 = np.mean([(sts[r['rep']]['lp'][k] > sts[r['rep']]['lp'][(L, 'beam5')] + 1e-9).mean() for r in reps])
        rows.append(dict(learner=L, variant=f'beam {bw}', em=100 * np.mean([em(sts[r['rep']]['Y'][k]) for r in reps]),
                         subem=100 * np.mean([sub(sts[r['rep']]['Y'][k]) for r in reps]),
                         higher_prob_than_B1=100 * better, higher_prob_than_B5=100 * better5,
                         mean_logp=np.mean([sts[r['rep']]['lp'][k].mean() for r in reps])))
for S in ['S1', 'S2', 'S3']:
    k = ('GBTcal', S)
    row = dict(learner='GBT (calibrated)', variant=S, em=100 * np.mean([em(sts[r['rep']]['Y'][k]) for r in reps]),
               subem=100 * np.mean([sub(sts[r['rep']]['Y'][k]) for r in reps]))
    if S == 'S1':
        P = [np.mean([sts[r['rep']]['P'][k][j] for r in reps], 0) for j in range(m)]
        row['logloss'] = np.mean([np.mean([-np.log(np.clip(sts[r['rep']]['P'][k][j][np.arange(n), Y[:, j]], 1e-15, 1)).mean() for j in range(m)]) for r in reps])
    rows.append(row)
# ensemble of chains with joint-mode voting over the members' beam-decoded prescriptions
ev_em, ev_sub = [], []
for r in reps:
    Ym = np.stack(r['ecc_Ybeam'], 1)   # n x 10 x m
    S4 = r['store'][('LR', 'S4')]['Y']; Yv = S4.copy()
    for i in range(n):
        vals, cnt = np.unique(Ym[i], axis=0, return_counts=True)
        if cnt.max() > 1: Yv[i] = vals[cnt.argmax()]
    ev_em.append(em(Yv)); ev_sub.append(sub(Yv))
rows.append(dict(learner='LR', variant='ensemble, joint-mode vote', em=100 * np.mean(ev_em), subem=100 * np.mean(ev_sub)))
t = pd.DataFrame(rows); t.to_csv(f'{OUT}/table_inference_checks.csv', index=False)
print(t.round(3).to_string())
