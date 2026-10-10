import json, sys
import numpy as np
import flat, cart
path = sys.argv[1]; min_snr = float(sys.argv[2]); depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5
recs = []
for l in open(path):
    d = json.loads(l)
    if 'error' in d: continue
    if d['feat'].get('stage') != 'ok': continue
    if d['snr'] < min_snr: continue
    recs.append(d)
fl = [flat.flatten(d['feat']) for d in recs]
names = sorted({k for f in fl for k in f if k not in ('stage_ok',)})
X = np.array([[f.get(n, np.nan) for n in names] for f in fl], float)
fam = [d['truth']['family'] for d in recs]
# finer classes for 'other'
lab = []
for d in recs:
    v = d['variant']
    if d['truth']['family'] == 'other': lab.append(v)
    elif d['truth']['family'] == 'multi': lab.append('multi')
    else: lab.append(d['truth']['family'])
classes = sorted(set(lab)); y = np.array([classes.index(l) for l in lab])
# balance classes
cnt = np.bincount(y); w = np.array([1.0 / cnt[i] for i in y]) * len(y) / len(classes)
print('n', len(y), dict(zip(classes, cnt)))
tree = cart.fit(X, y, w, names, len(classes), max_depth=depth, min_leaf=6)
cart.show(tree, classes)
