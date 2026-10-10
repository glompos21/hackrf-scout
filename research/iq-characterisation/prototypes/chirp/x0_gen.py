import sys, time
sys.path.insert(0, '..')
import numpy as np, gen
for kind, kw in [('lora', {}), ('lora', dict(sf=9)), ('lora', dict(sf=12)), ('pulsed', {}), ('pulsed', dict(mod='lfm'))]:
    t = time.time()
    r = gen.make(kind, 2e6, 5.0, 10.0, seed=1, on_fraction=0.2, **kw)
    tr = r['truth']
    print(kind, kw, 'gen %.2fs' % (time.time() - t), 'n=%d' % tr['n'], 'nb=%d' % len(tr['bursts']), 'onfrac=%.3f' % tr['on_fraction'],
          'bw=%s' % tr['occupied_bw_hz'], 'sr=%s' % tr['symbol_rate'], 'clip=%.5f' % tr['clip_fraction'], 'noise_rms=%.2f' % tr['noise_rms_lsb'])
    if tr['bursts']:
        print('   first bursts', [(round(a, 4), round(b, 4)) for a, b in tr['bursts'][:3]])
