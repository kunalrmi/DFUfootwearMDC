"""TabPFN v2 constructor and an output-code wrapper for more than ten classes.
TabPFN v2 handles at most ten classes. For a powerset class space with C > 10 combinations, ECOC_TFM fits R = 6 TabPFN models,
each on a random balanced recoding of the C classes into 10 symbols (all codewords distinct), and scores class c by
sum_r log p_r(symbol_r(c)), renormalised with a softmax. With C <= 10 it is plain TabPFN.
STANDIN=1 replaces TabPFN with scikit-learn boosting (testing without a GPU)."""
import os
import numpy as np

STANDIN = os.environ.get('STANDIN') == '1'
TFM_VERSION = os.environ.get('TFM_VERSION', 'v2')

def make_tfm(cat_idx=None):
    if STANDIN:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=5,
                                              categorical_features=cat_idx or None, random_state=0)
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')
    from tabpfn import TabPFNClassifier
    kw = dict(device='auto', random_state=0)
    if cat_idx: kw['categorical_features_indices'] = list(cat_idx)
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

class ECOC_TFM:
    def __init__(self, cat_idx=None, alphabet=10, R=6, seed=0):
        self.cat_idx, self.alphabet, self.R, self.seed = cat_idx, alphabet, R, seed
    def fit(self, X, y):
        self.classes_, yi = np.unique(y, return_inverse=True); C = len(self.classes_)
        if C <= self.alphabet:
            self.models = [make_tfm(self.cat_idx).fit(X, yi)]; self.codes = None; return self
        r = np.random.default_rng(self.seed + C)
        while True:   # balanced random codebooks with distinct codewords
            codes = np.stack([r.permutation(C) % self.alphabet for _ in range(self.R)], 1)
            if len({tuple(c) for c in codes}) == C: break
        self.codes = codes; self.models = []
        for k in range(self.R):
            yk = codes[yi, k]; self.models.append((make_tfm(self.cat_idx).fit(X, yk), np.unique(yk)))
        return self
    def predict_proba(self, X):
        if self.codes is None:
            m = self.models[0]; p = np.zeros((len(X), len(self.classes_))); p[:, m.classes_.astype(int)] = m.predict_proba(X); return p
        score = np.zeros((len(X), len(self.classes_)))
        for k, (mdl, sym) in enumerate(self.models):
            ps = np.full((len(X), self.alphabet), 1e-15); ps[:, mdl.classes_.astype(int)] = mdl.predict_proba(X)
            score += np.log(np.clip(ps[:, self.codes[:, k]], 1e-15, 1))
        score -= score.max(1, keepdims=True); e = np.exp(score); return e / e.sum(1, keepdims=True)
