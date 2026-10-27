"""Shared data loading and strategy code for the full experiment (structured, coded fields only)."""
import sys, json, time, itertools, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)
import os
DATA = os.environ["PAKDD_DATA"]

# ---------------- data ----------------
d = pd.read_csv(DATA)
d = d[d.Fastener_Type != 'BOA'].reset_index(drop=True)
d['Wedge_Forefoot_Magnitude'] = d['Wedge_Forefoot_Magnitude'].replace({'Medium': 'Medium/High', 'High': 'Medium/High'})
DOM = {'Fastener': ['Fastener_Type'],
       'Rocker': ['Rocker_Profile', 'Rocker_Angle', 'Rocker_Apex_Position', 'Rocker_Apex_Angle_Direction'],
       'Outsole': ['Outsole_Heavy_Duty_Work'],
       'Relasting': ['Relast_Forefoot', 'Relast_Midfoot', 'Relast_Hindfoot'],
       'Wedge': ['Wedge_Forefoot_Direction', 'Wedge_Forefoot_Magnitude', 'Wedge_Midfoot_Direction', 'Wedge_Hindfoot_Direction'],
       'Insole additions': ['Insole_Metatarsal_Bar', 'Insole_Metatarsal_Dome', 'Insole_Metatarsal_Pad', 'Insole_Localised_Deflection', 'Insole_First_Ray_Drop'],
       'Insole targets': ['Insole_Target_Forefoot', 'Insole_Target_Hallux', 'Insole_Target_Lesser_Digits', 'Insole_Target_1st_MTH', 'Insole_Target_4th_5th_MTH', 'Insole_Target_IPJ_2_3', 'Insole_Target_Hindfoot', 'Insole_Target_Heel']}
TGT = sum(DOM.values(), [])
m = len(TGT)
DOMIDX = {k: [TGT.index(c) for c in v] for k, v in DOM.items()}
INP = list(d.columns[1:66])
# ---- input preprocessing (label-agnostic, fixed before the final run; PAKDD_PREPROC=0 reproduces the earlier run) ----
PREPROC = os.environ.get('PAKDD_PREPROC', '1') == '1'
DROPPED = {}
if PREPROC:
    # 1. constant in all records (carry no information)
    DROPPED.update({'Amputation_Midfoot': 'constant', 'Amputation_Other': 'constant'})
    # 2. exact duplicates of another input in all records (the first-named input is kept)
    DROPPED.update({'Neuropathy_Sensory': 'identical to Neuropathy_Present',
                    'Foot_Structure_Shape': 'identical to Deformity_Present',
                    'Activity_Shopping': 'identical to Activity_Walking_Stairs',
                    'Needs_Specialised_Transport': 'identical to Access_To_Clinic',
                    'Previous_Footwear_Used': 'encoded by Previous_Footwear_Outcome = Not applicable'})
    INP = [c for c in INP if c not in DROPPED]
    # 3. structural missingness made explicit (severity is recorded only when a deformity is present)
    d['Deformity_Severity'] = d['Deformity_Severity'].fillna('Not applicable')
    # 4. rare levels (< 10 records) merged into the clinically nearest level
    d['Primary_Target_Lesion_Type'] = d['Primary_Target_Lesion_Type'].replace(
        {'Ulcer': 'Ulcer only', 'Ulcer + callus': 'Ulcer + hyperkeratosis', 'Ulcer + callus + corn': 'Ulcer + hyperkeratosis',
         'Ulcer + corn': 'Ulcer + hyperkeratosis', 'Callus': 'Hyperkeratosis only', 'Callus + corn': 'Hyperkeratosis only', 'Corn': 'Hyperkeratosis only'})
    d['Primary_Target_Specific_Site'] = d['Primary_Target_Specific_Site'].replace({'IPJ hallux': 'Hallux'})
    d['Employment_Status'] = d['Employment_Status'].replace({'Retired': 'Not employed', 'Unemployed': 'Not employed'})
NUM =['Age_Years', 'BMI_kg_m2', 'Peak_Plantar_Pressure_kPa']
for c in NUM:
    d[c] = pd.to_numeric(d[c], errors='coerce')
CAT = [c for c in INP if c not in NUM]
d[CAT] = d[CAT].astype(str)
classes = [sorted(d[c].unique()) for c in TGT]
K = [len(c) for c in classes]
Y = np.column_stack([pd.Categorical(d[c], categories=classes[j]).codes for j, c in enumerate(TGT)])
n = len(d)
X = d[INP]
NONE = [next((i for i, v in enumerate(classes[j]) if v.startswith('No ')), None) for j in range(m)]
ROCK = DOMIDX['Rocker']; FFD, FFM = TGT.index('Wedge_Forefoot_Direction'), TGT.index('Wedge_Forefoot_Magnitude')


# ---------------- iterative stratification (Sechidis et al. 2011) ----------------
def iter_strat(Y, k, seed):
    r = np.random.default_rng(seed)
    L = np.column_stack([(Y[:, j] == c).astype(int) for j in range(m) for c in range(K[j])])
    desired_lab = np.outer(np.full(k, 1 / k), L.sum(0))
    desired_n = np.full(k, len(Y) / k)
    fold = -np.ones(len(Y), int); remaining = np.ones(len(Y), bool)
    while remaining.any():
        counts = L[remaining].sum(0).astype(float); counts[counts == 0] = np.inf
        l = int(np.argmin(counts))
        idx = np.where(remaining & (L[:, l] == 1))[0]; r.shuffle(idx)
        for i in idx:
            cand = np.flatnonzero(desired_lab[:, l] == desired_lab[:, l].max())
            if len(cand) > 1:
                cand = cand[desired_n[cand] == desired_n[cand].max()]
            f = r.choice(cand); fold[i] = f; remaining[i] = False
            desired_lab[f] -= L[i]; desired_n[f] -= 1
    return fold

# ---------------- encoders ----------------
class Enc:
    def __init__(self, kind): self.kind = kind
    def fit(self, X):
        if self.kind == 'onehot':
            self.oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X[CAT])
            self.med = X[NUM].median(); self.sc = StandardScaler().fit(X[NUM].fillna(self.med))
        else:
            self.oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X[CAT])
        return self
    def transform(self, X):
        if self.kind == 'onehot':
            return np.hstack([self.oh.transform(X[CAT]), self.sc.transform(X[NUM].fillna(self.med))])
        return np.hstack([self.oe.transform(X[CAT]), X[NUM].values])

def make(learner, ncat=0, multi=False):
    if learner == 'LR': return LogisticRegression(C=1.0, max_iter=3000)
    if learner == 'GBT':
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=5,
                                              categorical_features=list(range(ncat)) if ncat else None, random_state=0)
    if learner == 'RF': return RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0)

class Const:
    def __init__(self, c): self.classes_ = np.array([c])
    def predict_proba(self, X): return np.ones((len(X), 1))
def fitc(L, ncat, X, y):
    u = np.unique(y)
    return Const(u[0]) if len(u) < 2 else make(L, ncat).fit(X, y)

ENCKIND = {'LR': 'onehot', 'GBT': 'ordinal', 'RF': 'onehot'}

def fullproba(clf, X, Kj):
    p = np.zeros((len(X), Kj)); p[:, clf.classes_.astype(int)] = clf.predict_proba(X); return p

def onehot_y(Yp, cols):
    return np.hstack([np.eye(K[j])[Yp[:, i]] for i, j in enumerate(cols)]) if cols else np.zeros((len(Yp), 0))

# ---------------- coherence ----------------
def coherent_rows(Yp):
    ok = np.ones(len(Yp), bool)
    prof_none = Yp[:, ROCK[0]] == NONE[ROCK[0]]
    for j in ROCK[1:]: ok &= prof_none == (Yp[:, j] == NONE[j])
    ok &= (Yp[:, FFD] == NONE[FFD]) == (Yp[:, FFM] == NONE[FFM])
    return ok

GROUPS = [ROCK, [FFD, FFM]]
def group_assignments(g):
    out = []
    for combo in itertools.product(*[range(K[j]) for j in g]):
        y = np.zeros(m, int); y[g] = combo
        Yt = np.zeros((1, m), int); Yt[0, g] = combo
        # coherence restricted to the group
        if g == ROCK:
            pn = combo[0] == NONE[ROCK[0]]
            if all(pn == (combo[i] == NONE[j]) for i, j in enumerate(g) if i > 0): out.append(combo)
        else:
            if (combo[0] == NONE[FFD]) == (combo[1] == NONE[FFM]): out.append(combo)
    return np.array(out)
GA = [group_assignments(g) for g in GROUPS]


def decode(P):
    Yp = np.column_stack([p.argmax(1) for p in P])
    for g, A in zip(GROUPS, GA):
        score = np.zeros((len(Yp), len(A)))
        for i, j in enumerate(g): score += np.log(np.clip(P[j][:, A[:, i]], 1e-15, 1))
        Yp[:, g] = A[score.argmax(1)]
    return Yp

# ---------------- strategies ----------------
def s1(L, Xtr, Ytr, Xte, ncat):
    return [fullproba(fitc(L, ncat, Xtr, Ytr[:, j]), Xte, K[j]) for j in range(m)]

def s2(L, Xtr, Ytr, Xte, ncat):
    oof = [np.zeros((len(Xtr), K[j])) for j in range(m)]
    f3 = np.arange(len(Xtr)) % 3; rng.shuffle(f3)
    for f in range(3):
        tr, va = f3 != f, f3 == f
        for j in range(m): oof[j][va] = fullproba(fitc(L, ncat, Xtr[tr], Ytr[tr, j]), Xtr[va], K[j])
    te1 = s1(L, Xtr, Ytr, Xte, ncat)
    out = []
    for j in range(m):
        A = np.hstack([Xtr] + [oof[i] for i in range(m) if i != j]); B = np.hstack([Xte] + [te1[i] for i in range(m) if i != j])
        out.append(fullproba(fitc(L, ncat, A, Ytr[:, j]), B, K[j]))
    return out

def fit_chain(L, Xtr, Ytr, order, ncat):
    return [fitc(L, ncat, np.hstack([Xtr, onehot_y(Ytr[:, order[:p]], order[:p])]), Ytr[:, order[p]]) for p in range(m)]

def chain_teacher(models, Xte, Yte, order):
    return {order[p]: fullproba(models[p], np.hstack([Xte, onehot_y(Yte[:, order[:p]], order[:p])]), K[order[p]]) for p in range(m)}

def chain_greedy(models, Xte, order):
    Yp = np.zeros((len(Xte), m), int); P = [None] * m
    for p in range(m):
        j = order[p]; pr = fullproba(models[p], np.hstack([Xte, onehot_y(Yp[:, order[:p]], order[:p])]), K[j])
        P[j] = pr; Yp[:, j] = pr.argmax(1)
    return P, Yp

def chain_mc(models, Xte, order, S=100, seed=0):
    # Rao-Blackwellised marginals: average the conditional distributions over ancestral samples
    r = np.random.default_rng(seed); N = len(Xte)
    Xr = np.repeat(Xte, S, axis=0); Ys = np.zeros((N * S, m), int); out = [None] * m
    for p in range(m):
        j = order[p]; pr = fullproba(models[p], np.hstack([Xr, onehot_y(Ys[:, order[:p]], order[:p])]), K[j])
        out[j] = pr.reshape(N, S, K[j]).mean(1)
        u = r.random(len(pr))[:, None]; Ys[:, j] = (pr.cumsum(1) < u).sum(1).clip(max=K[j] - 1)
    return out

def allowed(j, partial):
    """Mask of coherent classes for decision j given partial assignment (clinical order)."""
    mask = np.ones((len(partial), K[j]), bool)
    if j in ROCK[1:]:
        pn = partial[:, ROCK[0]] == NONE[ROCK[0]]
        mask[pn] = False; mask[pn, NONE[j]] = True; mask[~pn, NONE[j]] = False
    if j == FFM:
        dn = partial[:, FFD] == NONE[FFD]
        mask[dn] = False; mask[dn, NONE[j]] = True; mask[~dn, NONE[j]] = False
    return mask

def chain_beam(models, Xte, order, B=5):
    N = len(Xte); beams = np.zeros((N, 1, m), int); score = np.zeros((N, 1))
    for p in range(m):
        j = order[p]; nb = beams.shape[1]
        flat = beams.reshape(N * nb, m)
        pr = fullproba(models[p], np.hstack([np.repeat(Xte, nb, axis=0), onehot_y(flat[:, order[:p]], order[:p])]), K[j])
        cand = np.log(np.clip(pr, 1e-15, 1)) + score.reshape(-1, 1)
        cand[~allowed(j, flat)] = -np.inf
        cand = cand.reshape(N, nb * K[j]); keep = min(B, nb * K[j])
        top = np.argsort(-cand, 1)[:, :keep]
        bi, ci = top // K[j], top % K[j]
        newb = np.take_along_axis(beams, bi[:, :, None], 1).copy(); newb[:, :, j] = ci
        beams, score = newb, np.take_along_axis(cand, top, 1)
    return beams[:, 0, :]

def s5(L, Xtr, Ytr, Xte, ncat):
    P = [None] * m
    for dname, idx in DOMIDX.items():
        if len(idx) == 1:
            j = idx[0]; P[j] = fullproba(fitc(L, ncat, Xtr, Ytr[:, j]), Xte, K[j]); continue
        combos, inv, cnt = np.unique(Ytr[:, idx], axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel(); freq = np.where(cnt >= 3)[0]
        mapped = inv.copy()
        for c in np.where(cnt < 3)[0]:
            dist = (combos[freq] != combos[c]).sum(1); best = freq[np.lexsort((-cnt[freq], dist))[0]]
            mapped[inv == c] = best
        clf = fitc(L, ncat, Xtr, mapped)
        pc = np.zeros((len(Xte), len(combos))); pc[:, clf.classes_] = clf.predict_proba(Xte)
        for i, j in enumerate(idx):
            P[j] = np.stack([pc[:, combos[:, i] == c].sum(1) for c in range(K[j])], 1)
    # joint prediction within domains = argmax combination
    return P

def s5_predict(L, Xtr, Ytr, Xte, ncat):
    Yp = np.zeros((len(Xte), m), int); P = [None] * m; unreachable = {}
    for dname, idx in DOMIDX.items():
        if len(idx) == 1:
            j = idx[0]; P[j] = fullproba(fitc(L, ncat, Xtr, Ytr[:, j]), Xte, K[j]); Yp[:, j] = P[j].argmax(1); continue
        combos, inv, cnt = np.unique(Ytr[:, idx], axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel(); freq = np.where(cnt >= 3)[0]; mapped = inv.copy()
        for c in np.where(cnt < 3)[0]:
            dist = (combos[freq] != combos[c]).sum(1); best = freq[np.lexsort((-cnt[freq], dist))[0]]; mapped[inv == c] = best
        clf = fitc(L, ncat, Xtr, mapped)
        pc = np.zeros((len(Xte), len(combos))); pc[:, clf.classes_] = clf.predict_proba(Xte)
        Yp[:, idx] = combos[pc.argmax(1)]
        for i, j in enumerate(idx): P[j] = np.stack([pc[:, combos[:, i] == c].sum(1) for c in range(K[j])], 1)
        unreachable[dname] = combos[freq]
    return P, Yp, unreachable



def allowed_any(j, partial, assigned):
    """Coherent classes for decision j given already-assigned decisions (any order):
    members of a constraint group must agree on whether they are 'none'."""
    mask = np.ones((len(partial), K[j]), bool)
    for g in GROUPS:
        if j not in g: continue
        prev = [i for i in g if i in assigned]
        if not prev: continue
        status = partial[:, prev[0]] == NONE[prev[0]]
        mask[status] = False; mask[status, NONE[j]] = True; mask[~status, NONE[j]] = False
    return mask

def chain_beam_any(models, Xte, order, B=5):
    N = len(Xte); beams = np.zeros((N, 1, m), int); score = np.zeros((N, 1)); assigned = set()
    for p in range(m):
        j = order[p]; nb = beams.shape[1]; flat = beams.reshape(N * nb, m)
        pr = fullproba(models[p], np.hstack([np.repeat(Xte, nb, axis=0), onehot_y(flat[:, order[:p]], order[:p])]), K[j])
        cand = np.log(np.clip(pr, 1e-15, 1)) + score.reshape(-1, 1)
        cand[~allowed_any(j, flat, assigned)] = -np.inf
        cand = cand.reshape(N, nb * K[j]); keep = min(B, nb * K[j])
        top = np.argsort(-cand, 1)[:, :keep]; bi, ci = top // K[j], top % K[j]
        newb = np.take_along_axis(beams, bi[:, :, None], 1).copy(); newb[:, :, j] = ci
        beams, score = newb, np.take_along_axis(cand, top, 1); assigned.add(j)
    return beams[:, 0, :]
