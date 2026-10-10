"""Monte-Carlo null of the lag-line block statistic (white Gaussian decimated z, i.i.d. complex normal).
usage: python3 nulls.py [nblocks_millions]   -> writes results/null_lag.json (quantile table per L)"""
import sys, json, time, os
import numpy as np

def mc(L, nblk, seed=0, q_div=8, batch=200000):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chirp import css
    rng = np.random.default_rng(seed)
    out = np.empty((nblk,), np.float32)
    outd = np.empty((nblk,), np.float32)
    for s in range(0, nblk, batch):
        e = min(nblk, s + batch)
        z = (rng.standard_normal((e - s, L), dtype=np.float32) + 1j * rng.standard_normal((e - s, L), dtype=np.float32)).astype(np.complex64)
        out[s:e], outd[s:e] = css.block_stats(z, L, q_div)
    return np.concatenate([out, outd])

if __name__ == '__main__':
    nm = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    res = {}
    for L in (64, 128):
        t = time.time()
        r = mc(L, int(nm * 1e6 * (1.0 if L == 64 else 0.5)), seed=L)
        qs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        res[str(L)] = {('%g' % p): float(np.quantile(r, 1 - p)) for p in qs}
        res[str(L)]['n'] = int(len(r))
        print('L=%d  n=%d  %.1fs' % (L, len(r), time.time() - t), {k: round(v, 2) for k, v in res[str(L)].items() if k != 'n'})
    os.makedirs('results', exist_ok=True)
    json.dump(res, open('results/null_lag.json', 'w'), indent=1)
