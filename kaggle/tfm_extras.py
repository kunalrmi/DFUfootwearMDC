"""Kaggle: additional TabPFN v2 analyses.
(1) Teacher-forced chain probabilities on the outer test folds (for the TabPFN dependence gain), same folds as the paper.
(2) TabPFN on the public MDC benchmarks (Edm, Jura, Song): S1, S2, S3 and S5 (output codes for more than ten combinations).
Usage (after setting TABPFN_TOKEN as in README.md):
    !PAKDD_DATA=structured_coded.csv python tfm_extras.py --part oracle      # ~30-60 min on GPU
    !python tfm_extras.py --part bench                                       # ~30-60 min on GPU
Outputs in ./results_tfm_extras/: teacher_tfm.csv, bench_tfm.csv, run_info.csv. Zip that folder and send it back.
Test switch: STANDIN=1 replaces TabPFN with scikit-learn boosting (no GPU needed); ONE_FOLD=1 runs one fold."""
import sys, os, argparse, time, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--part', default='oracle'); args = ap.parse_args()
OUT = 'results_tfm_extras'; os.makedirs(OUT, exist_ok=True)
STANDIN = os.environ.get('STANDIN') == '1'; ONE = os.environ.get('ONE_FOLD') == '1'
TFM_VERSION = os.environ.get('TFM_VERSION', 'v2')

def make_tfm(cat_idx=None):
    if STANDIN:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=100, random_state=0)
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')
    from tabpfn import TabPFNClassifier
    kw = dict(device='auto', random_state=0)
    if cat_idx: kw['categorical_features_indices'] = cat_idx
    if hasattr(TabPFNClassifier, 'create_default_for_version'):
        ver = TFM_VERSION
        try:
            from tabpfn.constants import ModelVersion
            ver = getattr(ModelVersion, TFM_VERSION.upper().replace('.', '_'), TFM_VERSION)
        except Exception: pass
        try: return TabPFNClassifier.create_default_for_version(ver, **kw)
        except TypeError:
            clf = TabPFNClassifier.create_default_for_version(ver)
            for k, v in kw.items():
                try: clf.set_params(**{k: v})
                except Exception: pass
            return clf
    try: return TabPFNClassifier(**kw)
    except TypeError: return TabPFNClassifier(device='auto', random_state=0)

t0 = time.time()
if args.part == 'oracle':
    import common as C
    from common import *
    folds = pd.read_csv('folds.csv'); order = list(range(m)); rows = []
    def fit1(Xa, y, ncat):
        u = np.unique(y)
        if len(u) < 2: return C.Const(u[0])
        return make_tfm(list(range(ncat))).fit(Xa, y)
    for rep in (1, 2, 3):
        fold = folds[f'rep{rep}'].values
        for f in range(5):
            tr, te = fold != f, fold == f; ti = np.where(te)[0]
            enc = Enc('ordinal').fit(X[tr]); A, B = enc.transform(X[tr]), enc.transform(X[te]); ncat = len(CAT)
            for p in range(m):
                j = order[p]
                mdl = fit1(np.hstack([A, onehot_y(Y[tr][:, order[:p]], order[:p])]), Y[tr][:, j], ncat)
                P = fullproba(mdl, np.hstack([B, onehot_y(Y[te][:, order[:p]], order[:p])]), K[j])
                for i, r in enumerate(ti):
                    rows.append([rep, r, j] + list(P[i]) + [np.nan] * (4 - K[j]))
            print(f'oracle rep {rep} fold {f} done {time.time()-t0:.0f}s', flush=True)
            if ONE: break
        if ONE: break
    pd.DataFrame(rows, columns=['rep', 'row', 'decision', 'p0', 'p1', 'p2', 'p3']).to_csv(f'{OUT}/teacher_tfm.csv', index=False)
else:
    import benchmarks as BM
    from tfm_ecoc import ECOC_TFM
    _make = BM.make
    BM.make = lambda L: ECOC_TFM() if L == 'TFM' else _make(L)   # output codes when a powerset has more than ten classes
    from scipy.io import loadmat
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    rows = []
    for name in ['Edm', 'Jura', 'Song']:
        d = loadmat(f'{name}.mat'); X = np.asarray(d['data']['norm'][0][0], float); Yr = np.asarray(d['target'])
        Y = np.column_stack([np.unique(Yr[:, j], return_inverse=True)[1] for j in range(Yr.shape[1])]); K = [int(Y[:, j].max()) + 1 for j in range(Y.shape[1])]
        n = len(Y); allow_s5 = True
        for seed in (101, 202, 303):
            preds = {}; fid = np.zeros(n, int)
            for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
                sc = StandardScaler().fit(X[tr]); o = BM.run('TFM', sc.transform(X[tr]), Y[tr], sc.transform(X[te]), K); fid[te] = f
                for S, v in o.items():
                    if S == 'S5' and not allow_s5: continue
                    preds.setdefault(S, np.zeros_like(Y))[te] = v
                if ONE: break
            for S, v in preds.items():
                pad = [np.nan] * (3 - len(K))
                for i in range(n): rows.append([name, seed, i, fid[i], S] + list(v[i]) + pad + list(Y[i]) + pad)
            print(f'bench {name} seed {seed} done {time.time()-t0:.0f}s', flush=True)
            if ONE: break
    kk = 3
    pd.DataFrame(rows, columns=['dataset', 'seed', 'row', 'fold', 'strategy'] + [f'pred{j}' for j in range(kk)] + [f'true{j}' for j in range(kk)]).to_csv(f'{OUT}/bench_tfm.csv', index=False)
info = {'part': args.part, 'tfm_version': TFM_VERSION, 'standin': STANDIN, 'seconds': round(time.time() - t0), 'python': platform.python_version()}
for mod in ['sklearn', 'tabpfn', 'torch']:
    try: info[mod] = __import__(mod).__version__
    except Exception: info[mod] = 'not used'
pd.DataFrame([info]).to_csv(f'{OUT}/run_info_{args.part}.csv', index=False); print('done', info)
