"""Runtime / memory of the whole pipeline on a real-size capture (5 s), streaming from a ci8 file."""
import sys, os, time, resource, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np, gen, ampfeat as A, bench

CASES = [('nfm 5s@2M (12.5 kHz in 2 Msps)', 'nfm', 2e6, 5.0, 15, {}), ('ook 5s@2M gated', 'ook', 2e6, 5.0, 15, dict(on_fraction=0.3)),
         ('gfsk 5s@4M', 'gfsk', 4e6, 5.0, 15, {}), ('ofdm 5s@10M', 'ofdm', 10e6, 5.0, 15, {}), ('pulsed 5s@2M', 'pulsed', 2e6, 5.0, 15, {})]
if __name__ == '__main__':
    which = sys.argv[1:] 
    d = tempfile.mkdtemp(dir=os.path.dirname(HERE) + '/amplitude/runs')
    for label, kind, rate, sec, snr, kw in CASES:
        if which and kind not in which: continue
        r = gen.make(kind, rate, sec, snr, seed=7, dc=6, cfo_hz=2500, return_ci8=True, **kw)
        path = os.path.join(d, kind + '.ci8'); gen.write_ci8(r, path); T = r['truth']; del r
        lo, hi = bench.hull(T['bands_hz']); B = hi - lo
        prior = (0.5 * (lo + hi) + 20e3, max(1, np.ceil(B / 100e3)) * 100e3)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        t = time.perf_counter(); R = A.analyze(path, rate, prior=prior); dt = time.perf_counter() - t
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
        tm = R['timing']
        print('%-34s file %.0f MB | total %.1fs  pass1 %.1f band %.2f pass2 %.1f feats(HMM etc) %.1f | ticks %d g=%d tick %.1f us | peak RSS %.0f MB (before %.0f) | duty %.3f (truth %.3f) runs %d' % (
            label, os.path.getsize(path) / 1e6, dt, tm['pass1'], tm['band'], tm['pass2'], tm['feats'], R['channel']['T'], R['channel']['g'], R['channel']['tick_s'] * 1e6, rss, rss0, R['duty'], T['on_fraction'], R['runs']['n_runs']), flush=True)
        os.remove(path)
