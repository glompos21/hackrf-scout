p = open('pulse.py').read()
a = p.index('def stack_profile(')
b = p.index('# ----------------------------------------------------------------------------------------------\n# PRI')
new = '''def stack_profile(src, mean, events, noise, max_events=600, iters=3):
    """Align the envelope windows around the events on a running template (integer-sample cross-correlation)
    and average them.  The windows are read back from the capture (random access).  Returns dict(width_samples,
    peak, snr, n, profile) with the -3 dB (half power) full width of the stacked profile by linear
    interpolation; None if it cannot be measured.  Robust to noisy individual detections (low SNR) and to PRI
    jitter (no PRI grid is needed)."""
    if len(events) < 3:
        return None
    m = int(np.median([e['scale'] for e in events]))
    h = int(max(24, 5 * m))
    sel = events[:max_events]
    cen = np.array([0.5 * (e['t0'] + e['t1']) for e in sel])
    segs = []
    for c in np.round(cen).astype(np.int64):
        x = src.read(int(c) - 2 * h, 4 * h + 1)
        if len(x) < 4 * h + 1:
            continue
        x = x - np.complex64(mean)
        segs.append((x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2) - noise)
    if len(segs) < 3:
        return None
    W = np.array(segs)
    k = len(W)
    shifts = np.zeros(k, np.int64)
    T = W.mean(0)
    for _ in range(iters):
        Tc = T[h:-h]
        # cross-correlate every window with the template core for shifts -h..h (vectorised)
        sc = np.empty((k, 2 * h + 1))
        for j, sft in enumerate(range(-h, h + 1)):
            sc[:, j] = (W[:, h + sft:W.shape[1] - h + sft] * Tc[None, :]).sum(1)
        shifts = np.argmax(sc, axis=1) - h
        T = np.zeros(W.shape[1])
        for i in range(k):
            T += np.roll(W[i], -shifts[i])
        T /= k
    T = np.convolve(T, np.ones(3) / 3.0, mode='same')
    kk = int(np.argmax(T[h:-h])) + h
    pk = float(T[kk])
    if pk <= 0:
        return None
    half = 0.5 * pk
    lo = kk
    while lo > 0 and T[lo] > half:
        lo -= 1
    hi = kk
    while hi < len(T) - 1 and T[hi] > half:
        hi += 1
    if lo == 0 or hi == len(T) - 1:
        return None
    tl = lo + (half - T[lo]) / (T[lo + 1] - T[lo])
    th = hi - 1 + (T[hi - 1] - half) / (T[hi - 1] - T[hi])
    return dict(width_samples=float(th - tl), peak=pk, snr=pk / noise, n=k, profile=T, k=kk, h=h)


'''
p = p[:a] + new + p[b:]
open('pulse.py', 'w').write(p)

a = open('analyze.py').read()
a = a.replace('''        k = min(len(events), 600)
        lo = max(0, int(min(e['t0'] for e in events[:k])) - 1024)
        hi = int(max(e['t1'] for e in events[:k])) + 1024
        out['events_head'] = k
        out['stack_pending'] = True
        out['chirp_pulse'] = pl.intrapulse_slope(src, mean, events, rate)''', '''        st = pl.stack_profile(src, mean, events, noise)
        if st is not None:
            out['width_s'] = st['width_samples'] / rate
            out['stack_snr_db'] = 10 * math.log10(max(st['snr'], 1e-9))
            out['stack_n'] = st['n']
        out['chirp_pulse'] = pl.intrapulse_slope(src, mean, events, rate)''')
open('analyze.py', 'w').write(a)
