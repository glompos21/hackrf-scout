s = open('frontend.py').read()
s = s.replace('''    def chunks(self, size=CHUNK):
        for a in range(0, self.n, size):
            yield self.iq[a:a + size]
''', '''    def chunks(self, size=CHUNK):
        for a in range(0, self.n, size):
            yield self.iq[a:a + size]

    def read(self, a, n):
        return self.iq[max(a, 0):max(a, 0) + n]
''')
s = s.replace('''    def chunks(self, size=CHUNK):
        with open(self.path, 'rb') as fh:''', '''    def read(self, a, n):
        a = max(int(a), 0)
        with open(self.path, 'rb') as fh:
            fh.seek(2 * a)
            buf = fh.read(2 * int(n))
        buf = buf[:len(buf) - (len(buf) & 1)]
        return np.frombuffer(buf, np.int8).astype(np.float32).view(np.complex64)

    def chunks(self, size=CHUNK):
        with open(self.path, 'rb') as fh:''')
open('frontend.py', 'w').write(s)

# pulse.py: intrapulse chirp from raw segments
p = open('pulse.py').read()
a = p.index('def intrapulse_slope(')
p = p[:a] + '''def intrapulse_slope(src, mean, events, rate, lead=0.15, min_samples=8, max_events=300, band_hz=None, fc_hz=0.0):
    """Frequency ramp inside the pulses (linear FM).  Each pulse is read back from the capture (random access),
    optionally band-limited to [fc - band/2, fc + band/2] by an FFT mask, and the lag-1 phase increments over its
    core (trim `lead` of the -3 dB width at both ends) are fitted with a line.  Pooled over the pulses:
    slope (Hz/s), t = mean / (std / sqrt(n)), fraction of pulses with the same sign, swept bandwidth."""
    sl, wl = [], []
    for ev in events[:max_events]:
        w = ev['t1'] - ev['t0']
        a = int(math.floor(ev['t0'] - 0.5 * w - 8))
        b = int(math.ceil(ev['t1'] + 0.5 * w + 8))
        x = src.read(a, b - a)
        if len(x) < b - a:
            continue
        x = x - np.complex64(mean)
        if band_hz is not None and band_hz < 0.9 * rate:
            X = np.fft.fft(x)
            f = np.fft.fftfreq(len(x), 1.0 / rate)
            X *= (np.abs(f - fc_hz) <= band_hz / 2).astype(np.float32)
            x = np.fft.ifft(X).astype(np.complex64)
        c0 = int(math.ceil(ev['t0'] - a + lead * w))
        c1 = int(math.floor(ev['t1'] - a - lead * w))
        if c1 - c0 < min_samples:
            continue
        z = x[c0 + 1:c1 + 1] * np.conj(x[c0:c1])
        ph = np.angle(z).astype(np.float64)
        xx = np.arange(len(ph)) - 0.5 * (len(ph) - 1)
        sl.append(float((xx * ph).sum() / (xx * xx).sum()))
        wl.append(len(ph))
    if len(sl) < 5:
        return None
    sl = np.asarray(sl)
    mean_s = float(sl.mean())
    sd = float(sl.std(ddof=1)) + 1e-12
    mu = mean_s * rate * rate / (2 * math.pi)
    return dict(slope_hz_per_s=mu, t=float(mean_s / (sd / math.sqrt(len(sl)))), n=len(sl),
                frac_same_sign=float((np.sign(sl) == np.sign(mean_s)).mean()),
                swept_hz=float(abs(mu) * np.median(wl) / rate))
'''
open('pulse.py', 'w').write(p)

a = open('analyze.py').read()
a = a.replace('''def pulse_stage(events, rate, cfg):
    out = dict(n_events=len(events))''', '''def pulse_stage(events, rate, cfg, src=None, mean=0j, noise=None, env_peek=None):
    out = dict(n_events=len(events))''')
a = a.replace("res['pulsed'] = pulse_stage(pdet.finish(), rate, cfg)", "res['pulsed'] = pulse_stage(pdet.finish(), rate, cfg, src=src, mean=p1['mean'], noise=noise_full)")
a = a.replace('''    flag, info = pl.decide_pulsed(w, t0s, pri, cfg['pulse'])
    out.update(flag=bool(flag), info=info)
    return out''', '''    flag, info = pl.decide_pulsed(w, t0s, pri, cfg['pulse'])
    out.update(flag=bool(flag), info=info)
    if flag and src is not None:
        # width of the stacked profile (robust at low SNR) and intra-pulse frequency ramp
        k = min(len(events), 600)
        lo = max(0, int(min(e['t0'] for e in events[:k])) - 1024)
        hi = int(max(e['t1'] for e in events[:k])) + 1024
        out['events_head'] = k
        out['stack_pending'] = True
        out['chirp_pulse'] = pl.intrapulse_slope(src, mean, events, rate)
    return out''')
open('analyze.py', 'w').write(a)
