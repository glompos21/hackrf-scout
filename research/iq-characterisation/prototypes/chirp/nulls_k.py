"""Null of the k-consecutive-blocks statistic: min over k consecutive hop-L/2 blocks of R (same direction), white z.
usage: python3 nulls_k.py L NBLOCKS_MILLIONS -> results/null_lagk_L.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from chirp import css

L = int(sys.argv[1]); nm = float(sys.argv[2])
hop = L // 2
rng = np.random.default_rng(L + 17)
batch = 100000
res = {1: [], 2: [], 3: []}
tot = 0
t0 = time.time()
while tot < nm * 1e6:
    n = (batch - 1) * hop + L
    z = (rng.standard_normal(n, dtype=np.float32) + 1j * rng.standard_normal(n, dtype=np.float32)).astype(np.complex64)
    win = np.lib.stride_tricks.sliding_window_view(z, L)[::hop][:batch]
    up, dn = css.block_stats(win, L)
    for R in (up, dn):
        res[1].append(R)
        res[2].append(np.minimum(R[:-1], R[1:]))
        res[3].append(np.minimum(np.minimum(R[:-2], R[1:-1]), R[2:]))
    tot += batch
out = {}
for k in (1, 2, 3):
    r = np.concatenate(res[k])
    out[str(k)] = {'%g' % p: float(np.quantile(r, 1 - p)) for p in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)}
    out[str(k)]['n'] = int(len(r))
    print('L=%d k=%d n=%d' % (L, k, len(r)), {a: round(b, 2) for a, b in out[str(k)].items() if a != 'n'}, '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open('results/null_lagk_%d.json' % L, 'w'), indent=1)
