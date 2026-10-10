import numpy as np
class Node:
    pass
def gini(y, w, k):
    c = np.bincount(y, weights=w, minlength=k); t = c.sum()
    return 0.0 if t <= 0 else 1 - ((c / t) ** 2).sum()
def fit(X, y, w, names, k, depth=0, max_depth=5, min_leaf=8):
    n = Node(); n.counts = np.bincount(y, weights=w, minlength=k); n.pred = int(n.counts.argmax())
    n.leaf = True
    if depth >= max_depth or len(y) < 2 * min_leaf or gini(y, w, k) < 1e-6:
        return n
    best = None; g0 = gini(y, w, k); W = w.sum()
    for j in range(X.shape[1]):
        xj = X[:, j]; ok = ~np.isnan(xj)
        if ok.sum() < 2 * min_leaf: continue
        vals = np.unique(xj[ok])
        if len(vals) < 2: continue
        cuts = (vals[:-1] + vals[1:]) / 2
        if len(cuts) > 60: cuts = cuts[np.linspace(0, len(cuts) - 1, 60).astype(int)]
        for c in cuts:
            L = ok & (xj <= c); R = ok & (xj > c)
            if L.sum() < min_leaf or R.sum() < min_leaf: continue
            g = (w[L].sum() * gini(y[L], w[L], k) + w[R].sum() * gini(y[R], w[R], k)) / (w[L].sum() + w[R].sum())
            if best is None or g < best[0]: best = (g, j, c)
    if best is None or g0 - best[0] < 1e-4: return n
    g, j, c = best; xj = X[:, j]
    nan = np.isnan(xj); L = (xj <= c) | nan   # NaN go left
    n.leaf = False; n.j = j; n.c = c; n.name = names[j]
    n.left = fit(X[L], y[L], w[L], names, k, depth + 1, max_depth, min_leaf)
    n.right = fit(X[~L], y[~L], w[~L], names, k, depth + 1, max_depth, min_leaf)
    return n
def show(n, classes, ind=0):
    pad = '  ' * ind
    if n.leaf:
        t = n.counts.sum(); top = np.argsort(-n.counts)[:3]
        print(pad + 'leaf n=%.0f: ' % t + ', '.join('%s %.0f' % (classes[i], n.counts[i]) for i in top if n.counts[i] > 0))
    else:
        print(pad + '%s <= %.4g ?' % (n.name, n.c)); show(n.left, classes, ind + 1); print(pad + 'else'); show(n.right, classes, ind + 1)
