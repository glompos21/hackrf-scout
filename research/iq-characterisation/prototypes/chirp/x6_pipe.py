import sys, time
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe

META = dict(est_bw_hz=200e3)
def one(kind, kw, snr, seed=100, rate=2e6, secs=5.0, onfrac=0.2, **mk):
    r = gen.make(kind, rate, secs, snr, seed=seed, on_fraction=onfrac, dc=6, cfo_hz=3000, offset_hz=40e3, **dict(kw, **mk))
    src = fe.ArraySource(r['iq'], rate)
    res = an.characterise(src, meta=META)
    c, p = res['chirp'], res['pulsed']
    b = c.get('best')
    print('%-7s %-24s snr %3g | chirp=%s %s | pulsed=%s n_ev=%d | label=%s | %.2fs' % (kind, kw, snr, c['flag'],
          ('bw=%g sf=%d dir=%s R=%.0f score=%.2f hits=%d/%d' % (b['bw'], b['sf'], b['dir'], b['R'], b['score'], b['n_hits'], b['n_blocks'])) if b else 'max_score=%.2f' % c.get('max_score', -1),
          p['flag'], p['n_events'], res['label'], res['timing']['total']))
    return res

if __name__ == '__main__':
    one('noise', {}, 0)
    for s in (20, 10, 5, 0, -5):
        one('lora', {}, s)
    for s in (10, 0):
        one('lora', dict(sf=9), s)
        one('lora', dict(sf=12), s)
    for s in (20, 10):
        one('pulsed', {}, s)
        one('fsk2', {}, s)
