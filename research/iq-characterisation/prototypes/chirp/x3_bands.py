import sys, time
sys.path.insert(0, '..')
import numpy as np, gen, frontend as fe
cases = [('noise', {}, 0), ('lora', {}, 5), ('lora', {}, 0), ('lora', dict(sf=9), 0), ('lora', dict(bw_hz=250e3, sf=8), 5), ('lora', dict(bw_hz=500e3, sf=7), 5),
         ('pulsed', {}, 10), ('pulsed', {}, 0), ('pulsed', dict(mod='lfm'), 10), ('fsk2', {}, 10), ('ofdm', {}, 10), ('hopper', {}, 10), ('cw', {}, 10), ('nfm', {}, 10), ('wfm', {}, 10), ('ook', {}, 10)]
for kind, kw, snr in cases:
    r = gen.make(kind, 2e6, 5.0, snr, seed=100, on_fraction=0.2 if kind in ('lora', 'pulsed') else 1.0, dc=6, cfo_hz=3000, offset_hz=40e3, **kw)
    tr = r['truth']
    t = time.time()
    src = fe.ArraySource(r['iq'], 2e6)
    p1 = fe.pass1(src)
    fl, bands, info = fe.find_bands(p1)
    dt = time.time() - t
    print('%-7s %-22s snr %3d  truth c=%7.0f bw=%s | p1 %.2fs floor %.4g (n0 truth %.4g) K=%.0f' % (kind, kw, snr, tr['center_hz'], tr['occupied_bw_hz'], dt, fl, tr['noise_psd_lsb2_per_hz'], info['K']))
    for b in bands[:3]:
        print('      region lo %8.0f hi %8.0f centre %8.0f bw3 %8.0f full %8.0f plateau %4.1f dB' % (b['lo_hz'], b['hi_hz'], b['centre_hz'], b['bw3_hz'], b['bw_full_hz'], b['plateau_db']))
