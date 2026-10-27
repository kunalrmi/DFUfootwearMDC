"""Small NumPy MLP with a shared trunk and one softmax head per decision.
Trunk: input -> 256 -> 128 (ReLU, dropout 0.2). Loss: sum of per-head cross-entropies.
Adam (lr 1e-3), batch 64, max 200 epochs, early stopping (patience 20) on a validation split."""
import numpy as np

class MultiHeadMLP:
    def __init__(self, n_in, head_sizes, h1=256, h2=128, drop=0.2, lr=1e-3, seed=0):
        r = np.random.default_rng(seed); self.r = r; self.drop = drop; self.lr = lr; self.K = list(head_sizes)
        he = lambda a, b: r.normal(0, np.sqrt(2 / a), (a, b))
        self.p = {'W1': he(n_in, h1), 'b1': np.zeros(h1), 'W2': he(h1, h2), 'b2': np.zeros(h2)}
        for k, Kk in enumerate(self.K):
            self.p[f'V{k}'] = r.normal(0, np.sqrt(1 / h2), (h2, Kk)); self.p[f'c{k}'] = np.zeros(Kk)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0

    def _forward(self, X, train):
        p = self.p
        z1 = X @ p['W1'] + p['b1']; a1 = np.maximum(z1, 0)
        d1 = (self.r.random(a1.shape) > self.drop) / (1 - self.drop) if train else 1.0
        a1d = a1 * d1
        z2 = a1d @ p['W2'] + p['b2']; a2 = np.maximum(z2, 0)
        d2 = (self.r.random(a2.shape) > self.drop) / (1 - self.drop) if train else 1.0
        a2d = a2 * d2
        probs = []
        for k in range(len(self.K)):
            s = a2d @ p[f'V{k}'] + p[f'c{k}']; s -= s.max(1, keepdims=True); e = np.exp(s); probs.append(e / e.sum(1, keepdims=True))
        return probs, (X, z1, a1d, d1, z2, a2d, d2)

    def loss_grads(self, X, Ys):
        probs, (X, z1, a1d, d1, z2, a2d, d2) = self._forward(X, True); n = len(X); p = self.p; g = {}
        loss = 0.0; da2 = np.zeros_like(a2d)
        for k, Kk in enumerate(self.K):
            y = Ys[:, k]; loss -= np.log(np.clip(probs[k][np.arange(n), y], 1e-15, 1)).mean()
            ds = probs[k].copy(); ds[np.arange(n), y] -= 1; ds /= n
            g[f'V{k}'] = a2d.T @ ds; g[f'c{k}'] = ds.sum(0); da2 += ds @ p[f'V{k}'].T
        dz2 = da2 * d2 * (z2 > 0)
        g['W2'] = a1d.T @ dz2; g['b2'] = dz2.sum(0)
        da1 = dz2 @ p['W2'].T; dz1 = da1 * d1 * (z1 > 0)
        g['W1'] = X.T @ dz1; g['b1'] = dz1.sum(0)
        return loss, g

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]; self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def predict_proba(self, X):
        return self._forward(X, False)[0]

    def val_loss(self, X, Ys):
        pr = self.predict_proba(X)
        return -sum(np.log(np.clip(pr[k][np.arange(len(X)), Ys[:, k]], 1e-15, 1)).mean() for k in range(len(self.K)))

    def fit(self, X, Ys, Xv, Yv, epochs=200, batch=64, patience=20):
        best, best_p, wait = np.inf, None, 0
        for ep in range(epochs):
            idx = self.r.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]; _, g = self.loss_grads(X[b], Ys[b]); self._adam(g)
            vl = self.val_loss(Xv, Yv)
            if vl < best - 1e-6: best, best_p, wait = vl, {k: v.copy() for k, v in self.p.items()}, 0
            else:
                wait += 1
                if wait >= patience: break
        self.p = best_p
        return self

def grad_check(seed=0):
    r = np.random.default_rng(seed); X = r.normal(size=(7, 5)); Ys = np.column_stack([r.integers(0, 3, 7), r.integers(0, 2, 7)])
    net = MultiHeadMLP(5, [3, 2], h1=6, h2=4, drop=0.0, seed=1)
    _, g = net.loss_grads(X, Ys); err = 0
    for k in ['W1', 'b1', 'W2', 'V0', 'c1']:
        P = net.p[k]
        for _ in range(4):
            i = tuple(r.integers(0, s) for s in P.shape); old = P[i]
            P[i] = old + 1e-5; lp, _ = net.loss_grads(X, Ys); P[i] = old - 1e-5; lm, _ = net.loss_grads(X, Ys); P[i] = old
            err = max(err, abs((lp - lm) / 2e-5 - g[k][i]))
    return err
