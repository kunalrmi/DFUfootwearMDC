"""Define non-trivial decisions once: S1-LR balanced accuracy < threshold in a separate 5-fold run (seed 999)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
f = iter_strat(Y, 5, seed=999); P = [np.zeros((n, K[j])) for j in range(m)]
for k in range(5):
    tr, te = f != k, f == k
    e = Enc('onehot').fit(X[tr]); p = s1('LR', e.transform(X[tr]), Y[tr], e.transform(X[te]), 0)
    for j in range(m): P[j][te] = p[j]
ba = {TGT[j]: balanced_accuracy_score(Y[:, j], P[j].argmax(1)) for j in range(m)}
out = {str(t): [c for c, v in ba.items() if v < t] for t in [0.90, 0.95, 0.98]}
out['ba'] = ba
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print({k: len(v) for k, v in out.items() if k != 'ba'}); print(out['0.95'])
