"""Kaggle: strategy-learner combinations missing from the first run, on the SAME folds and preprocessing as the paper.
  --learners tfm : TabPFN v2 with S4 (ensemble of ten random-order chains) and S5 (domain powerset; output codes for >10 classes)
  --learners mlp : PyTorch MLP (same architecture and early stopping as the paper's MLP) with S2, S3 (beam 5, greedy, Monte Carlo), S4, S5
Usage:
    !PAKDD_DATA=structured_coded.csv python level_kaggle.py --learners tfm      # GPU
    !PAKDD_DATA=structured_coded.csv python level_kaggle.py --learners mlp      # GPU or CPU
Outputs in ./results_level/: predictions_level_<learner>.csv and run_info_<learner>.csv.
Per-fold checkpoints in ./results_level/ckpt_*.pkl: if the session stops, rerun the same command and it resumes.
Test switches: STANDIN=1 (scikit-learn / NumPy stand-ins, no GPU, TabPFN or torch), ONE_FOLD=1 (one fold of repeat 1)."""
import sys, os, time, argparse, platform, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import common as C
from common import *
from tfm_ecoc import ECOC_TFM, STANDIN

ap = argparse.ArgumentParser(); ap.add_argument('--learners', default='tfm,mlp'); args = ap.parse_args()
LEARN = [x.strip() for x in args.learners.split(',')]; OUT = 'results_level'; os.makedirs(OUT, exist_ok=True)
ONE = os.environ.get('ONE_FOLD') == '1'
folds = pd.read_csv('folds.csv'); assert len(folds) == n, 'folds.csv must have 487 rows'
order = list(range(m))

# ---------------- MLP as a single-output classifier (one softmax head), same settings as the paper ----------------
VAL = {'mask': None}      # validation rows of the current training fold (identical to the paper's MLP S1 split)
SEEDBOX = {'s': 0}
class MLPClf:
    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True); SEEDBOX['s'] += 1; seed = SEEDBOX['s']
        if VAL['mask'] is not None and len(X) == len(VAL['mask']): va = VAL['mask']
        else:   # first-stage stacking models fit on a subset: carve a 10% validation split of that subset
            va = np.zeros(len(X), bool); va[np.random.default_rng(seed).permutation(len(X))[:max(1, len(X) // 10)]] = True
        A, V, Ya, Yv = X[~va], X[va], yi[~va], yi[va]; k = len(self.classes_)
        if STANDIN:
            from mlp_numpy import MultiHeadMLP
            self.net = MultiHeadMLP(A.shape[1], [k], seed=seed).fit(A, Ya[:, None], V, Yv[:, None]); return self
        import torch, torch.nn as nn
        torch.manual_seed(seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.dev = dev
        net = nn.Sequential(nn.Linear(A.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, k)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss()
        tA, tY = torch.tensor(A, dtype=torch.float32, device=dev), torch.tensor(Ya, dtype=torch.long, device=dev)
        tV, tYv = torch.tensor(V, dtype=torch.float32, device=dev), torch.tensor(Yv, dtype=torch.long, device=dev)
        best, best_state, wait = np.inf, None, 0; g = torch.Generator(device='cpu').manual_seed(seed)
        for ep in range(200):
            net.train(); perm = torch.randperm(len(tA), generator=g)
            for s0 in range(0, len(tA), 64):
                b = perm[s0:s0 + 64].to(dev); loss = ce(net(tA[b]), tY[b]); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): vl = ce(net(tV), tYv).item()
            if vl < best - 1e-6: best, best_state, wait = vl, {kk: v.detach().clone() for kk, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 20: break
        net.load_state_dict(best_state); net.eval(); self.net = net; return self
    def predict_proba(self, X):
        if STANDIN: return self.net.predict_proba(X)[0]
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor(X, dtype=torch.float32, device=self.dev)), 1).cpu().numpy()

_orig = C.make
def make_patched(learner, ncat=0, multi=False):
    if learner == 'TFM': return ECOC_TFM(list(range(ncat)) if ncat else None)
    if learner == 'MLP': return MLPClf()
    return _orig(learner, ncat, multi)
C.make = make_patched

# ---------------- helpers ----------------
def new_store(keys): return {k: {'P': [np.zeros((n, K[j])) for j in range(m)], 'Y': np.zeros((n, m), int), 'Yraw': np.zeros((n, m), int)} for k in keys}
def put(store, key, te, P, Yp, Yraw):
    for j in range(m): store[key]['P'][j][te] = P[j]
    store[key]['Y'][te] = Yp; store[key]['Yraw'][te] = Yraw
def raw(P): return np.column_stack([p.argmax(1) for p in P])
def store_to_rows(store, rep, mask):
    rows = []; idx = np.where(mask)[0]
    for (L, S), s in store.items():
        for j in range(m):
            P = s['P'][j][idx]
            d = pd.DataFrame({'rep': rep, 'row': idx, 'learner': L, 'strategy': S, 'decision': j, 'y_dec': s['Y'][idx, j], 'y_raw': s['Yraw'][idx, j]})
            for c in range(4): d[f'p{c}'] = P[:, c] if c < K[j] else np.nan
            rows.append(d)
    return pd.concat(rows)
class Fixed:
    def __init__(self, a): self.a = a
    def shuffle(self, x): x[:] = self.a

def s4(L, Xtr, Ytr, Xte, ncat, orders):
    Pe = [np.zeros((len(Xte), K[j])) for j in range(m)]
    for o in orders:
        pg, _ = chain_greedy(fit_chain(L, Xtr, Ytr, o, ncat), Xte, o)
        for j in range(m): Pe[j] += pg[j] / 10
    return Pe

def run(L, rep, fold, seed):
    keys = [('TFM', 'S4'), ('TFM', 'S5')] if L == 'TFM' else [('MLP', 'S2'), ('MLP', 'S3'), ('MLP', 'S3-greedy'), ('MLP', 'S4'), ('MLP', 'S5')]
    CK = f'{OUT}/ckpt_{L}_rep{rep}.pkl'
    if os.path.exists(CK): ck = pickle.load(open(CK, 'rb')); store, done, f0 = ck['store'], ck['done'], ck['f'] + 1; print(f'{L} rep {rep}: resumed at fold {f0}', flush=True)
    else: store, done, f0 = new_store(keys), np.zeros(n, bool), 0
    # same ten chain orders per fold as the other learners' ensembles; same stacking split as LR
    r_ord = np.random.default_rng(seed + 7); ORD = [[list(r_ord.permutation(m)) for _ in range(10)] for f in range(5)]
    r_s2 = np.random.default_rng(seed); SPL = []
    for f in range(5):
        ntr = int((fold != f).sum()); a = np.arange(ntr) % 3; r_s2.shuffle(a); SPL.append(a); b = np.arange(ntr) % 3; r_s2.shuffle(b)
    for f in range(f0, 5):
        tr, te = fold != f, fold == f; Ytr = Y[tr]; t0 = time.time()
        if L == 'TFM':
            ncat = len(CAT); enc = Enc('ordinal').fit(X[tr]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te])
            P = s4(L, Xtr, Ytr, Xte, ncat, ORD[f]); put(store, (L, 'S4'), te, P, decode(P), raw(P))
            P5, Y5, _ = s5_predict(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S5'), te, P5, Y5, Y5)
        else:
            ncat = 0; tri = np.where(tr)[0]
            val = iter_strat(Ytr, 10, seed=seed + 300 + f) == 0          # the paper's MLP validation split
            enc = Enc('onehot').fit(X.iloc[tri[~val]]); Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te])
            VAL['mask'] = val; SEEDBOX['s'] = seed * 100000 + f * 10000
            C.rng = Fixed(SPL[f]); P = s2(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S2'), te, P, decode(P), raw(P))
            models = fit_chain(L, Xtr, Ytr, order, ncat)
            Pg, Yg = chain_greedy(models, Xte, order); Pmc = chain_mc(models, Xte, order, S=100, seed=seed * 10 + f)
            Yb = chain_beam(models, Xte, order, B=5)
            put(store, (L, 'S3'), te, Pmc, Yb, Yg); put(store, (L, 'S3-greedy'), te, Pg, Yg, Yg)
            P = s4(L, Xtr, Ytr, Xte, ncat, ORD[f]); put(store, (L, 'S4'), te, P, decode(P), raw(P))
            P5, Y5, _ = s5_predict(L, Xtr, Ytr, Xte, ncat); put(store, (L, 'S5'), te, P5, Y5, Y5)
            VAL['mask'] = None
        done |= te
        print(f'{L} rep {rep} fold {f} done ({time.time()-t0:.0f}s)', flush=True)
        if not ONE: pickle.dump(dict(store=store, done=done, f=f), open(CK + '.tmp', 'wb')); os.replace(CK + '.tmp', CK)
        if ONE: break
    return store_to_rows(store, rep, done)

t0 = time.time()
for lname in LEARN:
    L = {'tfm': 'TFM', 'mlp': 'MLP'}[lname]; preds = []
    for col in [c for c in folds.columns if c.startswith('rep')]:
        rep = int(col[3:]); seed = 100 * rep + rep
        preds.append(run(L, rep, folds[col].values, seed))
        if ONE: break
    pd.concat(preds).to_csv(f'{OUT}/predictions_level_{lname}.csv', index=False)
    info = {'learner': lname, 'tfm_version': os.environ.get('TFM_VERSION', 'v2'), 'standin': STANDIN, 'one_fold': ONE, 'seconds': round(time.time() - t0), 'python': platform.python_version()}
    for mod in ['sklearn', 'torch', 'tabpfn']:
        try: info[mod] = __import__(mod).__version__
        except Exception: info[mod] = 'not used'
    pd.DataFrame([info]).to_csv(f'{OUT}/run_info_{lname}.csv', index=False)
    print(f'{lname} finished after {time.time()-t0:.0f}s', info, flush=True)
