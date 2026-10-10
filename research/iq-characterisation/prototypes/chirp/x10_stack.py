import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-hackrf-scout/f99c7e11-f52c-5034-af0a-858cb1074c59/scratchpad/iqchar')
import numpy as np, gen
from chirp import analyze as an, frontend as fe, cases, pulse as pl

def stack(src, mean, events, noise, max_events=600, iters=2, maxshift=None):
    m = int(np.median([e['scale'] for e in events]))
    h = int(max(24, 5 * m))
    sel = events[:max_events]
    cen = np.array([0.5 * (e['t0'] + e['t1']) for e in sel])
    segs = []
    for c in np.round(cen).astype(np.int64):
        x = src.read(int(c) - 2 * h, 4 * h + 1)
        if len(x) < 4 * h + 1: continue
        x = x - np.complex64(mean)
        segs.append((x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2) - noise)
    W = np.array(segs); k = len(W)
    T = W.mean(0)
    if maxshift is None: maxshift = h
    ms = int(maxshift)
    for _ in range(iters if ms > 0 else 0):
        Tc = T[h:-h]
        sc = np.empty((k, 2 * ms + 1))
        for j, sft in enumerate(range(-ms, ms + 1)):
            sc[:, j] = (W[:, h + sft:W.shape[1] - h + sft] * Tc[None, :]).sum(1)
        sh = np.argmax(sc, axis=1) - ms
        T = np.zeros(W.shape[1])
        for i in range(k): T += np.roll(W[i], -sh[i])
        T /= k
    T = np.convolve(T, np.ones(3) / 3.0, mode='same')
    kk = int(np.argmax(T[h:-h])) + h
    pk = float(T[kk]); half = 0.5 * pk
    lo = kk
    while lo > 0 and T[lo] > half: lo -= 1
    hi = kk
    while hi < len(T) - 1 and T[hi] > half: hi += 1
    tl = lo + (half - T[lo]) / (T[lo + 1] - T[lo]); th = hi - 1 + (T[hi - 1] - half) / (T[hi - 1] - T[hi])
    return th - tl

cfg = an.load_cfg()
print('name snr truth3dB | A(full align) B(no align) C(+-m/2)')
for name in ('pulsed_10us_1ms', 'pulsed_5us_2ms', 'pulsed_20us_jit', 'pulsed_lfm_10us'):
    for snr in (5, 8, 10, 15, 20):
        spec = cases.SPECS[name]
        gk, imp, rng = cases.case_params(name, snr, 100)
        r = gen.make(spec['kind'], spec['rate'], 5.0, float(snr), seed=100, return_ci8=False, **gk)
        src = fe.ArraySource(r['iq'], spec['rate'])
        p1 = fe.pass1(src); fl, _, _ = fe.find_bands(p1); noise = fl * spec['rate']
        mean = np.complex64(p1['mean'])
        def ch():
            for c in src.chunks(1 << 20):
                cc = c - mean; yield (cc.real ** 2 + cc.imag ** 2).astype(np.float32)
        ev = pl.detect_pulses(ch(), noise, spec['rate'])
        tl = cases.truth_labels(name, r['truth'])
        if len(ev) < 10:
            print('%-18s %2d  events %d' % (name, snr, len(ev))); continue
        m = int(np.median([e['scale'] for e in ev]))
        res = []
        for kw in (dict(), dict(maxshift=0), dict(maxshift=max(2, m // 2))):
            try: res.append(stack(src, mean, ev, noise, **kw) / spec['rate'] * 1e6)
            except Exception as e: res.append(float('nan'))
        print('%-18s %2d  events %5d truth %.2f | A %.2f  B %.2f  C %.2f us' % (name, snr, len(ev), tl['pw3db_s'] * 1e6, *res))
