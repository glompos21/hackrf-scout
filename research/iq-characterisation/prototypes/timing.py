import sys, time, resource, numpy as np
sys.path.insert(0, '.')
import gen
secs = float(sys.argv[1]); rate = float(sys.argv[2]); kinds = sys.argv[3:]
for kind in kinds:
    kw = {}
    if kind == 'pulsed': kw = dict(pw_s=2e-6)
    t = time.time()
    r = gen.make(kind, rate, secs, 12.0, seed=3, **kw)
    dt = time.time() - t
    print('%-7s %.1fs @%.0fM  gen %.2fs  ci8=%d MB  rss peak %.0f MB  clip=%.5f' % (kind, secs, rate/1e6, dt, len(r['ci8'])/1e6, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3, r['truth']['clip_fraction']))
    del r
