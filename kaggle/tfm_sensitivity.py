"""TabPFN v2 sensitivity runs and benchmarks (Kaggle, GPU).
  --part clinical : TabPFN S3-rev (chain in reverse domain order) and S1-merged (BR with each constraint group re-encoded as one
                    decision) on the clinical data, same folds as the paper. Output: results_tfm_sensitivity/predictions_sensitivity_tfm.csv
  --part bench    : TabPFN S1-S5 and the dependence diagnostic on every public MDC benchmark (*.mat) in the working folder,
                    via bench2.py. Output: results_tfm_sensitivity/bench2_<dataset>_TFM.csv
Usage:
    !PAKDD_DATA=structured_coded.csv python tfm_sensitivity.py --part clinical
    !python tfm_sensitivity.py --part bench
Both parts resume after an interruption (per-fold checkpoints / per-dataset files). STANDIN=1 tests without a GPU."""
import sys, os, glob, time, argparse, pickle, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--part', default='clinical'); args = ap.parse_args()
OUT = 'results_tfm_sensitivity'; os.makedirs(OUT, exist_ok=True); ONE = os.environ.get('ONE_FOLD') == '1'
from tfm_ecoc import ECOC_TFM, STANDIN
t0 = time.time()
if args.part == 'bench':
    import bench2
    files = sorted(f for f in glob.glob('*.mat'))
    print('benchmarks:', files, flush=True)
    LRN = os.environ.get('BENCH_LEARNERS', 'TFM,LR,GBT,RF')   # TFM first; the CPU learners repeat the paper's runs as a backup
    for f in files:
        os.system(f'{sys.executable} bench2.py {OUT} {LRN} "{f}"')
else:
    import common as C
    from common import *
    _orig = C.make
    C.make = lambda learner, ncat=0, multi=False: ECOC_TFM(list(range(ncat)) if ncat else None) if learner == 'TFM' else _orig(learner, ncat, multi)
    C.ENCKIND['TFM'] = 'ordinal'
    folds = pd.read_csv('folds.csv'); assert len(folds) == n
    REV = [j for d in reversed(list(DOMIDX)) for j in DOMIDX[d]]; GRP = [DOMIDX['Rocker'], [FFD, FFM]]
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
    keys = [('TFM', 'S3-rev'), ('TFM', 'S1-merged')]; allrows = []
    for col in [c for c in folds.columns if c.startswith('rep')]:
        rep = int(col[3:]); fold = folds[col].values; CK = f'{OUT}/ckpt_sensitivity_rep{rep}.pkl'
        if os.path.exists(CK): ck = pickle.load(open(CK, 'rb')); store, done, f0 = ck['store'], ck['done'], ck['f'] + 1
        else: store = {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in keys}; done = np.zeros(n, bool); f0 = 0
        for f in range(f0, 5):
            tr, te = fold != f, fold == f; ncat = len(CAT)
            enc = Enc('ordinal').fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te]); Ytr = Y[tr]
            models = fit_chain('TFM', Xtr, Ytr, REV, ncat); Pg, Yg = chain_greedy(models, Xte, REV); Yb = chain_beam_any(models, Xte, REV, B=5)
            P, Yp = merged_br('TFM', Xtr, Ytr, Xte, ncat)
            for key, PP, YY, YR in [(keys[0], Pg, Yb, Yg), (keys[1], P, Yp, Yp)]:
                for j in range(m): store[key]['P'][j][te] = PP[j]
                store[key]['Y'][te] = YY; store[key]['Yraw'][te] = YR
            done |= te; print(f'clinical rep {rep} fold {f} done {time.time()-t0:.0f}s', flush=True)
            if not ONE: pickle.dump(dict(store=store, done=done, f=f), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)
            if ONE: break
        idx = np.where(done)[0]
        for (L, S), s in store.items():
            for j in range(m):
                d = pd.DataFrame({'rep': rep, 'row': idx, 'learner': L, 'strategy': S, 'decision': j, 'y_dec': s['Y'][idx, j], 'y_raw': s['Yraw'][idx, j]})
                for c in range(4): d[f'p{c}'] = s['P'][j][idx, c] if c < K[j] else np.nan
                allrows.append(d)
        if ONE: break
    pd.concat(allrows).to_csv(f'{OUT}/predictions_sensitivity_tfm.csv', index=False)
info = {'part': args.part, 'tfm_version': os.environ.get('TFM_VERSION', 'v2'), 'standin': STANDIN, 'seconds': round(time.time() - t0), 'python': platform.python_version()}
for mod in ['sklearn', 'torch', 'tabpfn']:
    try: info[mod] = __import__(mod).__version__
    except Exception: info[mod] = 'not used'
pd.DataFrame([info]).to_csv(f'{OUT}/run_info_{args.part}.csv', index=False); print('done', info)
