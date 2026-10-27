"""Kaggle runner: LightGBM (GBT), PyTorch MLP and, optionally, TabPFN v2 on the SAME folds as the main run.
Files needed in the working directory: common.py, structured_coded.csv, folds.csv, run_kaggle.py

Usage (Kaggle notebook cell):
    !PAKDD_DATA=structured_coded.csv python run_kaggle.py --learners lgbm,mlp      # CPU is fine
    !PAKDD_DATA=structured_coded.csv python run_kaggle.py --learners tfm            # needs GPU + internet
Outputs in ./results_csv/ : predictions_<learner>.csv, oracle_<learner>.csv, learning_curve_<learner>.csv,
s5_unreachable_<learner>.csv and run_info.csv. Zip that folder and send it back.
Test switches (no LightGBM/torch/TabPFN needed): STANDIN=1 uses scikit-learn/NumPy stand-ins; ONE_FOLD=1 runs one fold."""
import sys, os, time, argparse, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--learners', default='lgbm,mlp'); ap.add_argument('--out', default='results_csv')
args = ap.parse_args(); LEARN = [x.strip() for x in args.learners.split(',')]; OUT = args.out; os.makedirs(OUT, exist_ok=True)
STANDIN = os.environ.get('STANDIN') == '1'; ONE = os.environ.get('ONE_FOLD') == '1'
folds = pd.read_csv('folds.csv'); assert len(folds) == n, 'folds.csv must have 487 rows'
order = list(range(m))

# ---------------- learners ----------------
def make_lgbm(ncat):
    if STANDIN: return C.make('GBT', ncat)
    import lightgbm as lgb
    class LGBM(lgb.LGBMClassifier):
        def fit(self, X, y):
            return super().fit(X, y, categorical_feature=list(range(ncat)) if ncat else 'auto')
    # LightGBM's categorical defaults (min_data_per_group=100, cat_smooth=10) are meant for large data; with ~390 records they
    # block almost every split on the 55 categorical inputs. Match the paper's small-data setting (at least 5 records per group/leaf).
    return LGBM(n_estimators=100, num_leaves=31, learning_rate=0.1, min_child_samples=5, min_data_per_group=5, cat_smooth=1.0,
                cat_l2=1.0, random_state=0, verbose=-1, n_jobs=4)
TFM_VERSION = os.environ.get('TFM_VERSION', 'v2')   # the paper uses TabPFN v2; newer tabpfn releases default to v3.5
def make_tfm(ncat):
    if STANDIN: return C.make('GBT', ncat)
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')     # never wait for an interactive login (set TABPFN_TOKEN instead)
    from tabpfn import TabPFNClassifier
    kw = dict(device='auto', random_state=0, categorical_features_indices=list(range(ncat)))
    if hasattr(TabPFNClassifier, 'create_default_for_version'):   # tabpfn >= 6: choose the model version explicitly
        ver = TFM_VERSION
        try:
            from tabpfn.constants import ModelVersion
            ver = getattr(ModelVersion, TFM_VERSION.upper().replace('.', '_'), TFM_VERSION)
        except Exception:
            pass
        try:
            return TabPFNClassifier.create_default_for_version(ver, **kw)
        except TypeError:
            clf = TabPFNClassifier.create_default_for_version(ver)
            for k, v in kw.items():
                try: clf.set_params(**{k: v})
                except Exception: pass
            return clf
    try: return TabPFNClassifier(**kw)                    # older tabpfn 2.x: v2 is the default model
    except TypeError: return TabPFNClassifier(device='auto', random_state=0)
_orig = C.make
def make_patched(learner, ncat=0, multi=False):
    if learner == 'LGBM': return make_lgbm(ncat)
    if learner == 'TFM': return make_tfm(ncat)
    return _orig(learner, ncat, multi)
C.make = make_patched
C.ENCKIND['LGBM'] = 'ordinal'; C.ENCKIND['TFM'] = 'ordinal'

# ---------------- PyTorch multi-head MLP (same architecture as the paper) ----------------
def train_mlp(A, Ya, V, Yv, T, head_sizes, seed):
    if STANDIN:
        from mlp_numpy import MultiHeadMLP
        return MultiHeadMLP(A.shape[1], head_sizes, seed=seed).fit(A, Ya, V, Yv).predict_proba(T)
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.trunk = nn.Sequential(nn.Linear(A.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2))
            s.heads = nn.ModuleList([nn.Linear(128, k) for k in head_sizes])
        def forward(s, x):
            h = s.trunk(x); return [hd(h) for hd in s.heads]
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss()
    tA, tY = torch.tensor(A, dtype=torch.float32, device=dev), torch.tensor(Ya, dtype=torch.long, device=dev)
    tV, tYv = torch.tensor(V, dtype=torch.float32, device=dev), torch.tensor(Yv, dtype=torch.long, device=dev)
    best, best_state, wait = np.inf, None, 0; g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(200):
        net.train(); perm = torch.randperm(len(tA), generator=g)
        for s0 in range(0, len(tA), 64):
            b = perm[s0:s0 + 64].to(dev); out = net(tA[b])
            loss = sum(ce(o, tY[b, k]) for k, o in enumerate(out)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(ce(o, tYv[:, k]).item() for k, o in enumerate(net(tV)))
        if vl < best - 1e-6: best, best_state, wait = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 20: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return [torch.softmax(o, 1).cpu().numpy() for o in net(torch.tensor(T, dtype=torch.float32, device=dev))]

# ---------------- helpers ----------------
def ll_vec(Pj, yj): return -np.log(np.clip(Pj[np.arange(len(yj)), yj], 1e-15, 1))
def new_store(keys): return {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in keys}
def put(store, key, te, P, Yp, Yraw):
    for j in range(m): store[key]['P'][j][te] = P[j]
    store[key]['Y'][te] = Yp; store[key]['Yraw'][te] = Yraw
def store_to_rows(store, rep, mask):
    rows = []
    idx = np.where(mask)[0]
    for (L, S), s in store.items():
        for j in range(m):
            P = s['P'][j][idx]
            d = pd.DataFrame({'rep': rep, 'row': idx, 'learner': L, 'strategy': S, 'decision': j,
                              'y_dec': s['Y'][idx, j], 'y_raw': s['Yraw'][idx, j]})
            for c in range(4): d[f'p{c}'] = P[:, c] if c < K[j] else np.nan
            rows.append(d)
    return pd.concat(rows)

def run_tree_like(L, rep, fold, seed):
    """S1, S2, S3 (beam, greedy, Monte Carlo), S5, oracle (inner CV) and learning curve for LGBM or TFM."""
    ncat = len(CAT); keys = [(L, 'S1'), (L, 'S2'), (L, 'S3'), (L, 'S3-greedy')] + ([(L, 'S5')] if L != 'TFM' else [])
    store = new_store(keys); orc, lc, unr_rows = [], [], []; done = np.zeros(n, bool)
    for f in range(5):
        tr, te = fold != f, fold == f; done |= te
        enc = Enc('ordinal').fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te]); Ytr = Y[tr]
        P1 = s1(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S1'), te, P1, decode(P1), np.column_stack([p.argmax(1) for p in P1]))
        P2 = s2(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S2'), te, P2, decode(P2), np.column_stack([p.argmax(1) for p in P2]))
        models = fit_chain(L, Xtr, Ytr, order, ncat)
        Pg, Yg = chain_greedy(models, Xte, order); Pmc = chain_mc(models, Xte, order, S=100, seed=seed * 10 + f)
        Yb = chain_beam(models, Xte, order, B=5)
        put(store, (L, 'S3'), te, Pmc, Yb, Yg); put(store, (L, 'S3-greedy'), te, Pg, Yg, Yg)
        if L != 'TFM':
            P5, Y5, unr = s5_predict(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S5'), te, P5, Y5, Y5)
            for dname, idx in DOMIDX.items():
                if len(idx) > 1:
                    ok = (Y[te][:, idx][:, None, :] == unr[dname][None]).all(2).any(1)
                    unr_rows.append(dict(rep=rep, fold=f, learner=L, domain=dname, n=int((~ok).sum())))
            # oracle gain by inner 5-fold CV within the training fold
            inner = iter_strat(Ytr, 5, seed=seed + 100 + f); Xtr_df = X[tr].reset_index(drop=True)
            llbr, llpre, llall, ntot = np.zeros(m), np.zeros(m), np.zeros(m), 0
            for g in range(5):
                itr, iva = inner != g, inner == g
                e2 = Enc('ordinal').fit(Xtr_df[itr]); A, B = e2.transform(Xtr_df[itr]), e2.transform(Xtr_df[iva]); Ya, Yv = Ytr[itr], Ytr[iva]
                pb = s1(L, A, Ya, B, ncat); tp = chain_teacher(fit_chain(L, A, Ya, order, ncat), B, Yv, order)
                for j in range(m):
                    llbr[j] += ll_vec(pb[j], Yv[:, j]).sum(); llpre[j] += ll_vec(tp[j], Yv[:, j]).sum()
                    oth = [i for i in range(m) if i != j]
                    ca = fitc(L, ncat, np.hstack([A, onehot_y(Ya[:, oth], oth)]), Ya[:, j])
                    llall[j] += ll_vec(fullproba(ca, np.hstack([B, onehot_y(Yv[:, oth], oth)]), K[j]), Yv[:, j]).sum()
                ntot += iva.sum()
            for j in range(m): orc.append(dict(rep=rep, fold=f, learner=L, decision=j, g_pre=(llbr[j] - llpre[j]) / ntot, g_all=(llbr[j] - llall[j]) / ntot))
        # learning curve (S1 vs S3 at 25/50/75%)
        tri = np.where(tr)[0]; sub = iter_strat(Ytr, 4, seed=seed + 200 + f)
        for frac in [0.25, 0.5, 0.75]:
            keep = sub < int(round(frac * 4)); Xs = X.iloc[tri[keep]]
            e3 = Enc('ordinal').fit(Xs); A, B = e3.transform(Xs), e3.transform(X[te])
            ya = decode(s1(L, A, Ytr[keep], B, ncat)); yb = chain_beam(fit_chain(L, A, Ytr[keep], order, ncat), B, order, B=5)
            for S, yp in [('S1', ya), ('S3', yb)]:
                err = (yp != Y[te]).sum(1)
                lc.append(dict(rep=rep, fold=f, learner=L, frac=frac, strategy=S, n_em=int((err == 0).sum()), n_sub=int((err <= 1).sum()), n_correct=int((yp == Y[te]).sum()), n=int(te.sum())))
        print(f'{L} rep {rep} fold {f} done', flush=True)
        if ONE: break
    return store_to_rows(store, rep, done), orc, lc, unr_rows

def run_mlp(rep, fold, seed):
    store = new_store([('MLP', 'S1'), ('MLP', 'S6')]); done = np.zeros(n, bool)
    for f in range(5):
        tr, te = fold != f, fold == f; done |= te; tri = np.where(tr)[0]
        val = iter_strat(Y[tr], 10, seed=seed + 300 + f) == 0
        enc = Enc('onehot').fit(X.iloc[tri[~val]])
        A, V, T = enc.transform(X.iloc[tri[~val]]), enc.transform(X.iloc[tri[val]]), enc.transform(X[te]); Ya, Yv = Y[tr][~val], Y[tr][val]
        P6 = train_mlp(A, Ya, V, Yv, T, K, seed + f)
        P1 = [train_mlp(A, Ya[:, [j]], V, Yv[:, [j]], T, [K[j]], seed + 1000 * (j + 1) + f)[0] for j in range(m)]
        for key, P in [(('MLP', 'S1'), P1), (('MLP', 'S6'), P6)]:
            put(store, key, te, P, decode(P), np.column_stack([p.argmax(1) for p in P]))
        print(f'MLP rep {rep} fold {f} done', flush=True)
        if ONE: break
    return store_to_rows(store, rep, done)

# ---------------- main ----------------
t0 = time.time()
for lname in LEARN:
    L = {'lgbm': 'LGBM', 'tfm': 'TFM', 'mlp': 'MLP'}[lname]
    preds, orcs, lcs, unrs = [], [], [], []
    for col in [c for c in folds.columns if c.startswith('rep')]:
        rep = int(col[3:]); fold = folds[col].values; seed = 100 * rep + rep
        C.rng = np.random.default_rng(seed)
        if L == 'MLP': preds.append(run_mlp(rep, fold, seed))
        else:
            p, o, l, u = run_tree_like(L, rep, fold, seed); preds.append(p); orcs += o; lcs += l; unrs += u
        if ONE: break
    pd.concat(preds).to_csv(f'{OUT}/predictions_{lname}.csv', index=False)
    if orcs: pd.DataFrame(orcs).to_csv(f'{OUT}/oracle_{lname}.csv', index=False)
    if lcs: pd.DataFrame(lcs).to_csv(f'{OUT}/learning_curve_{lname}.csv', index=False)
    if unrs: pd.DataFrame(unrs).to_csv(f'{OUT}/s5_unreachable_{lname}.csv', index=False)
    print(f'{lname} finished after {time.time()-t0:.0f}s', flush=True)
info = {'tfm_version': TFM_VERSION if 'tfm' in LEARN else '', 'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'standin': STANDIN, 'one_fold': ONE, 'seconds': round(time.time() - t0)}
for mod in ['sklearn', 'lightgbm', 'torch', 'tabpfn']:
    try: info[mod] = __import__(mod).__version__
    except Exception: info[mod] = 'not used'
pd.DataFrame([info]).to_csv(f'{OUT}/run_info.csv', index=False)
print('all done', info)
