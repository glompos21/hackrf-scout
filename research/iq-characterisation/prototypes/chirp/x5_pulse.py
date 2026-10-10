import sys, time
sys.path.insert(0, '..')
import numpy as np, gen, frontend as fe, pulse as pl

def run(kind, kw, snr, rate=2e6, seed=100, onfrac=0.2, secs=5.0, verbose=True):
    r = gen.make(kind, rate, secs, snr, seed=seed, on_fraction=onfrac, dc=6, cfo_hz=3000, offset_hz=40e3, **kw)
    tr = r['truth']
    x = r['iq']
    src = fe.ArraySource(x, rate)
    p1 = fe.pass1(src)
    fl, bands, info = fe.find_bands(p1)
    mean = p1['mean']
    noise = fl * rate
    t0 = time.time()
    def chunks():
        for c in src.chunks(1 << 20):
            cc = c - np.complex64(mean)
            yield (cc.real ** 2 + cc.imag ** 2).astype(np.float32)
    ms = pl.detect_pulses(chunks(), noise, rate)
    t_det = time.time() - t0
    n_true = len(tr['on_intervals'])
    w = np.array([m['width'] for m in ms]) / rate
    st = np.array([m['t0'] for m in ms]) / rate
    pri = pl.estimate_pri(st, rate)
    if verbose:
        print('%s %s snr %g: truth pulses %d  events %d (%.2fs)  width med %.2f us (truth pw %.1f us)  PRI %s jitter %s frac_mult %.2f' % (
            kind, kw, snr, n_true, len(ms), t_det, np.median(w) * 1e6 if len(w) else float('nan'), tr['parameters']['pw_s'] * 1e6 if 'pw_s' in tr['parameters'] else float('nan'),
            None if pri['pri'] is None else '%.1f us' % (pri['pri'] * 1e6), pri.get('jitter'), pri['frac_multiple']))
    return ms, tr

if __name__ == '__main__':
    for snr in (30, 20, 15, 10, 5, 0):
        run('pulsed', {}, snr)

    print('--- stacked width')
    for snr in (30, 20, 15, 10, 7):
        rate = 2e6
        ms, tr = run('pulsed', {}, snr, verbose=False)
        r = gen.make('pulsed', rate, 5.0, snr, seed=100, on_fraction=0.2, dc=6, cfo_hz=3000, offset_hz=40e3)
        x = r['iq']; src = fe.ArraySource(x, rate); p1 = fe.pass1(src); fl, _, _ = fe.find_bands(p1)
        xc = x - np.complex64(p1['mean']); p = (xc.real ** 2 + xc.imag ** 2).astype(np.float32)
        t = time.time()
        st = pl.stack_profile(p, ms, fl * rate)
        print('snr %g: events %d  stacked width %.2f us (50%%-amp truth 10 us, power-FWHM expected ~9.4)  snr_stack %.1f dB  %.2fs' % (snr, len(ms), st['width_samples'] / rate * 1e6 if st else float('nan'), 10 * np.log10(st['snr']) if st else float('nan'), time.time() - t))
