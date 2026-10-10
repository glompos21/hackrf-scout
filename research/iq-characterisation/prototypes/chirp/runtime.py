"""Streaming runtime / memory: write a 5 s ci8 file (separate process) and analyse it with Ci8FileSource.
usage: python3 runtime.py gen KIND RATE SNR [kw=json] FILE   |   python3 runtime.py run FILE RATE [est_bw]"""
import json, os, resource, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

if sys.argv[1] == 'gen':
    import gen
    kind, rate, snr = sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 6 else {}
    path = sys.argv[-1]
    r = gen.make(kind, rate, 5.0, snr, seed=11, on_fraction=0.2, offset_hz=30e3, cfo_hz=2e3, dc=5 + 4j, **kw)
    open(path, 'wb').write(r['ci8'])
    print('wrote', path, len(r['ci8']) / 1e6, 'MB')
else:
    from chirp import analyze as an, frontend as fe
    path, rate = sys.argv[2], float(sys.argv[3])
    meta = dict(est_bw_hz=float(sys.argv[4])) if len(sys.argv) > 4 else None
    src = fe.Ci8FileSource(path, rate)
    t0 = time.time()
    res = an.characterise(src, meta=meta)
    dt = time.time() - t0
    print('RUN %s: %.2fs wall for %.1f s of IQ  peak RSS %.0f MB  timing %s' % (os.path.basename(path), dt, src.n / rate, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, {k: round(v, 2) for k, v in res['timing'].items()}))
    c = res['chirp']
    print('  chirp', c['flag'], (c.get('best') or {}).get('sf'), (c.get('best') or {}).get('bw'), 'pulsed', res['pulsed']['flag'], res['pulsed']['n_events'], 'label', res['label'])
