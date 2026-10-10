"""Amplitude / envelope statistics for captured HackRF IQ (family: amplitude statistics).

numpy only, Python >= 3.9, streaming: the capture is read in chunks (an ndarray or a ci8 file) and
never held in memory as a whole.  Two passes over the data:

  pass 1  chunked Welch PSD (Hann, 50 % overlap), DC estimate, raw wideband amplitude moments
          (the "RF-Sentinel" kurtosis of |iq| on the raw wideband capture), clip counter.
  band    blind noise-floor + occupied-band estimate from the PSD (no truth, no prior needed).
  pass 2  FFT channeliser (overlap-save, spectral cropping = mix + brick-ish filter + decimate to the
          channel), then per-"tick" additive moment sums (n, sum r .. r^4, I^4, Q^4, max r^2).  All
          later statistics (whole capture, on-state only, off-state only, jackknife) are sums of ticks,
          which is what makes the whole thing streamable.
  detect  noise-referenced multi-scale energy detector on the tick power trace -> on/off mask.
  feats   envelope kurtosis (several definitions), noise-compensated (M2M4) kurtosis, PAPR, CV,
          I/Q kurtosis, Otsu split / bimodality, duty, burst statistics, chip-run quantisation,
          spectral carrier-line test.

Nothing here reads the generator's truth.  `analyze()` takes (source, rate) and optional priors.
"""
import math
import time

import numpy as np

# ----------------------------------------------------------------------------------------------
# configuration (physics-based defaults; see README_amplitude notes in the report for provenance)
# ----------------------------------------------------------------------------------------------
CFG = dict(
    chunk=1 << 20,            # samples per read
    alpha_det=1e-3,           # target false alarm probability per capture of the burst detector
    alpha_band=1e-3,          # same for the PSD band finder
    smooth_bins=5,            # PSD smoothing for band finding
    floor_q=0.30,             # quantile used for the first floor estimate
    eps_sys=0.02,             # systematic noise-level margin on detector thresholds (2 % = 0.09 dB)
    tick_s=2e-6,              # target tick duration
    max_ticks=400_000,
    chan_margin=1.10,         # channel bandwidth = cluster extent * margin + 2 smoothing blurs
    rho=0.25,                 # channel filter roll-off (fraction of bw)
    min_chan_bins=8,          # minimum channel width in PSD bins
    blk=1 << 16,              # channeliser FFT size
    ov=0.25,                  # overlap fraction
    ring_trans=3.0,           # drop 3/transition seconds at the ends
    subtract_dc=True,         # remove the capture mean (LO leakage / DC offset) before everything else
    lam_fine=3.0,             # switch cost (nats) of the fine HMM pass inside active regions
)


# ----------------------------------------------------------------------------------------------
# small numerics: regularised upper incomplete gamma and the Gamma(k, 1/k) upper quantile
# ----------------------------------------------------------------------------------------------
def gammaincc(a, x):
    """Q(a, x) = P(Gamma(a,1) > x)  (Numerical-Recipes series / continued fraction)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(100000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_upper(k, p):
    """x with P(X > x) = p for X ~ Gamma(shape k, scale 1/k) (mean 1).  k may be fractional."""
    k = max(float(k), 0.05)
    lo, hi = 0.0, 1.0
    while gammaincc(k, k * hi) > p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gammaincc(k, k * mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gamma_lower(k, p):
    """x with P(X < x) = p (mean-1 Gamma)."""
    return gamma_upper(k, 1.0 - p)


# ----------------------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------------------
def iter_chunks(src, chunk=1 << 20):
    """Yield complex64 chunks from an ndarray (complex) or from a ci8 file path."""
    if isinstance(src, str):
        with open(src, 'rb') as f:
            while True:
                raw = np.fromfile(f, dtype=np.int8, count=2 * chunk)
                m = raw.size // 2
                if m == 0:
                    break
                z = np.empty(m, np.complex64)
                z.real = raw[0:2 * m:2]
                z.imag = raw[1:2 * m:2]
                yield z
    else:
        for a in range(0, len(src), chunk):
            yield np.asarray(src[a:a + chunk], dtype=np.complex64)


def pick_nfft(rate):
    t = rate / 500.0
    nf = 1 << int(round(math.log2(max(t, 1024))))
    return int(min(max(nf, 1024), 32768))


# ----------------------------------------------------------------------------------------------
# pass 1
# ----------------------------------------------------------------------------------------------
def _welch_acc(x, w, nfft, hop, acc, batch):
    nseg = (len(x) - nfft) // hop + 1
    if nseg <= 0:
        return 0
    st = x.strides[0]
    segs = np.lib.stride_tricks.as_strided(x, shape=(nseg, nfft), strides=(hop * st, st), writeable=False)
    for s in range(0, nseg, batch):
        e = min(nseg, s + batch)
        F = np.fft.fft(segs[s:e] * w, axis=1)
        acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
    return nseg


def pass1(src, rate, nfft=None, chunk=None, subtract_dc=True):
    chunk = chunk or CFG['chunk']
    nfft = nfft or pick_nfft(rate)
    hop = nfft // 2
    w = np.hanning(nfft).astype(np.float32)
    acc = np.zeros(nfft)
    nseg = 0
    mean0 = None
    n = 0
    sx = 0j
    S = np.zeros(5)           # raw amplitude moments of |x|: n, r, r2, r3, r4 (float64 sums)
    iq = np.zeros(4)          # raw I^2, I^4, Q^2, Q^4
    clip = 0
    tail = np.zeros(0, np.complex64)
    batch = max(1, (1 << 22) // nfft)
    for z in iter_chunks(src, chunk):
        if mean0 is None:
            mean0 = complex(z.mean()) if subtract_dc else 0j
        n += len(z)
        sx += complex(z.sum(dtype=np.complex128))
        re, im = z.real, z.imag
        r2 = re * re + im * im
        r = np.sqrt(r2)
        S[0] += len(z)
        S[1] += r.sum(dtype=np.float64)
        S[2] += r2.sum(dtype=np.float64)
        S[3] += (r2 * r).sum(dtype=np.float64)
        S[4] += (r2 * r2).sum(dtype=np.float64)
        i2, q2 = re * re, im * im
        iq += [i2.sum(dtype=np.float64), (i2 * i2).sum(dtype=np.float64),
               q2.sum(dtype=np.float64), (q2 * q2).sum(dtype=np.float64)]
        clip += int(np.count_nonzero((re >= 127) | (re <= -128) | (im >= 127) | (im <= -128)))
        x = np.concatenate([tail, z - np.complex64(mean0)]) if len(tail) else z - np.complex64(mean0)
        ns = _welch_acc(x, w, nfft, hop, acc, batch)
        nseg += ns
        used = (ns - 1) * hop + nfft if ns > 0 else 0
        tail = x[ns * hop:] if ns > 0 else x
        if len(tail) > nfft:        # never keep more than needed
            tail = tail[-(nfft - hop):]
    mean = (sx / max(n, 1)) if subtract_dc else 0j
    psd = acc / (max(nseg, 1) * rate * float((w.astype(np.float64) ** 2).sum()))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
    psd = np.fft.fftshift(psd)
    m1, m2, m3, m4 = S[1] / n, S[2] / n, S[3] / n, S[4] / n
    c2 = m2 - m1 ** 2
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    raw = dict(
        rfs_kurt=float(c4 / c2 ** 2),                       # RF-Sentinel: central kurtosis of |iq| (noise: 3.245)
        m4=float(m4 / m2 ** 2),                             # E|x|^4 / (E|x|^2)^2 (noise: 2)
        cv=float(math.sqrt(c2) / m1),                       # coefficient of variation of |x| (noise: 0.5227)
        iq_kurt=float(0.5 * (n * iq[1] / iq[0] ** 2 + n * iq[3] / iq[2] ** 2)),   # raw I/Q kurtosis (noise 3)
        rms=float(math.sqrt(m2 / 2.0)),
        clip_frac=clip / max(n, 1),
    )
    return dict(psd=psd, f=f, rate=rate, nfft=nfft, nseg=nseg, df=rate / nfft, mean=mean, n=n, raw=raw)


# ----------------------------------------------------------------------------------------------
# band finder (blind)
# ----------------------------------------------------------------------------------------------
def _movavg(a, s):
    if s <= 1:
        return a.copy()
    c = np.cumsum(np.concatenate([[0.0], a]))
    out = np.empty(len(a))
    h = s // 2
    ap = np.pad(a, (h, s - 1 - h), mode='edge')
    c = np.cumsum(np.concatenate([[0.0], ap]))
    return (c[s:] - c[:-s]) / s


def k_eff_psd(nseg):
    """Effective number of independent averages of a Hann 50 %-overlap Welch bin."""
    return max(nseg / 1.056, 1.0)


def _dof(K, s):
    """Degrees of freedom of a bin-smoothed Hann/50 %-overlap Welch PSD (calibrated on white noise)."""
    return K * s / (1.0 if s <= 1 else (1.8 if s < 16 else 2.0))


def find_band(p1, cfg=None):
    cfg = cfg or CFG
    psd, f, df, nseg = p1['psd'], p1['f'], p1['df'], p1['nseg']
    nb = len(psd)
    s = cfg['smooth_bins']
    sf = 16
    K = k_eff_psd(nseg)
    dof_s, dof_f = _dof(K, s), _dof(K, sf)
    dc = np.abs(f) < 3 * df             # bins damaged by DC removal / spike
    valid = ~dc
    ps = _movavg(psd, s)
    pf = _movavg(psd, sf)
    # first floor: low quantile of the 16-bin smoothed PSD with Gamma correction
    q = cfg['floor_q']
    f0 = float(np.quantile(pf[valid], q)) / gamma_lower(dof_f, q)
    thr = f0 * gamma_upper(dof_s, cfg['alpha_band'] / nb)
    flag = ps > thr
    # grow flagged set by the smoothing blur, re-estimate the floor from the median of the rest
    guard = sf + 2
    gm = np.convolve(flag.astype(np.int8), np.ones(2 * guard + 1, np.int8), mode='same') > 0
    clean = valid & ~gm
    if clean.sum() < max(0.15 * nb, 64):
        floor, floor_ok = f0, False        # almost everything is signal: no trustworthy floor
    else:
        floor, floor_ok = float(np.median(pf[clean])) / gamma_lower(dof_f, 0.5), True
        thr = floor * gamma_upper(dof_s, cfg['alpha_band'] / nb)
        flag = ps > thr
    clusters = []
    idx = np.flatnonzero(flag)
    if len(idx):
        gap = max(3, s + 1)
        brk = np.flatnonzero(np.diff(idx) > gap)
        st = np.concatenate([[0], brk + 1])
        en = np.concatenate([brk, [len(idx) - 1]])
        for a, b in zip(st, en):
            lo, hi = int(idx[a]), int(idx[b])
            lo2, hi2 = max(0, lo - s), min(nb - 1, hi + s)
            seg = slice(lo2, hi2 + 1)
            exc = float((psd[seg] - floor).clip(0).sum() * df)
            clusters.append(dict(lo_bin=lo2, hi_bin=hi2, f_lo=float(f[lo2] - df / 2), f_hi=float(f[hi2] + df / 2),
                                 excess=exc, nbins=hi2 - lo2 + 1))
    clusters.sort(key=lambda c: -c['excess'])
    tot = float(psd.sum() * df)
    out = dict(floor=floor, floor_ok=floor_ok, floor0=f0, clusters=clusters, n_clusters=len(clusters),
               dof_s=dof_s, K=K, total_power=tot, df=df, smooth=ps, thr=thr)
    if clusters:
        c = clusters[0]
        bw = c['f_hi'] - c['f_lo']
        out.update(f_lo=c['f_lo'], f_hi=c['f_hi'], fc=0.5 * (c['f_lo'] + c['f_hi']), bw=bw,
                   band_snr_db=10 * math.log10(max(c['excess'], 1e-30) / (floor * bw)),
                   band_excess_frac=c['excess'] / tot)
    return out


def line_test(p1, band, cfg=None):
    """Narrow spectral-line ("carrier present") test on the Welch PSD.

    line_over_thr_db : 3-bin-smoothed peak / floor, in dB above the analytic false-alarm threshold (Gamma upper
                       quantile for alpha/(nbins/2), times the 2 % systematic floor margin).  > 0 means a line.
    line_frac        : share of the in-band excess power inside +-2 bins of the peak (1 = pure tone)."""
    cfg = cfg or CFG
    psd, df = p1['psd'], p1['df']
    f = p1['f']
    nb = len(psd)
    K = k_eff_psd(p1['nseg'])
    floor = band['floor']
    dof3 = _dof(K, 3)
    thr = (1 + cfg['eps_sys']) * gamma_upper(dof3, cfg['alpha_band'] / (nb / 2.0))
    ps3 = _movavg(psd, 3)
    valid = np.abs(f) >= 3 * df
    out = dict(line_frac=float('nan'))
    if band['clusters']:
        c = band['clusters'][0]
        seg = slice(c['lo_bin'], c['hi_bin'] + 1)
        ex = psd[seg] - floor
        p5 = np.convolve(ex, np.ones(5), mode='same')
        tot = float(ex.clip(0).sum())
        out['line_frac'] = float(max(p5.max(), 0) / max(tot, 1e-30))
    i = int(np.argmax(np.where(valid, ps3, 0)))
    ratio = ps3[i] / floor
    out.update(line_z=float((ratio - 1.0) * math.sqrt(dof3)), line_f=float(f[i]),
               line_ratio_db=10 * math.log10(max(ratio, 1e-9)),
               line_over_thr_db=10 * math.log10(max(ratio, 1e-9) / thr))
    return out


# ----------------------------------------------------------------------------------------------
# channeliser + tick accumulation (pass 2)
# ----------------------------------------------------------------------------------------------
class Channel:
    def __init__(self, rate, fc, bw, rho=None, N=None, ov=None):
        rho = CFG['rho'] if rho is None else rho
        N = N or CFG['blk']
        ov = CFG['ov'] if ov is None else ov
        self.rate, self.fc, self.bw, self.rho = rate, fc, bw, rho
        self.N = N
        self.O = int(N * ov) // 8 * 8
        self.hop = N - self.O
        span = bw * (1 + rho)
        M = int(math.ceil(N * span / rate / 8.0) * 8)
        M = min(max(M, 64), N)
        self.M = M
        self.rate_d = rate * M / N
        self.kc = int(round(fc * N / rate))
        j = np.arange(-M // 2, M // 2)
        self.idx = (self.kc + j) % N
        frel = (self.kc + j) * rate / N - fc
        a = np.abs(frel)
        f1, f2 = bw / 2.0, bw / 2.0 * (1 + rho)
        H = np.where(a <= f1, 1.0, np.where(a >= f2, 0.0, 0.5 * (1 + np.cos(math.pi * (a - f1) / max(f2 - f1, 1e-9)))))
        self.H = H.astype(np.float32)
        self.noise_gain = float((H ** 2).sum() / N)        # N_ch = sigma^2 * noise_gain, sigma^2 = floor_psd * rate
        self.enbw = float((H ** 2).sum() * rate / N)       # equivalent noise bandwidth (Hz)
        self.keep0 = (self.O // 2) * M // N
        # autocorrelation of the filtered noise at lag l (output samples): rho_l = IDFT(H^2)/R0
        R = np.fft.ifft(H ** 2)
        rho2 = (np.abs(R[1:M // 2]) / abs(R[0])) ** 2
        lags = np.arange(1, M // 2)
        self._c1 = np.concatenate([[0.0], np.cumsum(rho2)])                  # sum_{l<n} rho_l^2 (index n-1)
        self._c2 = np.concatenate([[0.0], np.cumsum(lags * rho2)])
        self.trans_hz = max((f2 - f1), 1.0)

    def dof(self, n):
        """Gamma shape of the mean of |y|^2 over n consecutive output samples of the filtered white noise:
        k = n / (1 + 2 sum_{l<n} (1 - l/n) rho_l^2)."""
        n = int(max(1, n))
        j = min(n - 1, len(self._c1) - 1)
        s1, s2 = self._c1[j], self._c2[j]
        return n / (1.0 + 2.0 * (s1 - s2 / n))

    def block(self, xb):
        X = np.fft.fft(xb)
        Z = np.fft.ifftshift(X[self.idx] * self.H)          # bins were gathered in ascending order -> natural order
        y = np.fft.ifft(Z) * np.float32(self.M / self.N)
        return y.astype(np.complex64)


class TickAcc:
    FIELDS = ('s1', 's2', 's3', 's4', 'i2', 'i4', 'q2', 'q4', 'mx')

    def __init__(self, g):
        self.g = g
        self.carry = np.zeros(0, np.complex64)
        self.parts = {k: [] for k in self.FIELDS}
        self.nsamp = 0

    def push(self, y):
        g = self.g
        if len(self.carry):
            y = np.concatenate([self.carry, y])
        m = (len(y) // g) * g
        self.carry = y[m:]
        if m == 0:
            return
        y = y[:m]
        re, im = y.real, y.imag
        r2 = re * re + im * im
        r = np.sqrt(r2)
        i2, q2 = re * re, im * im
        sh = (-1, g)
        P = self.parts
        P['s1'].append(r.reshape(sh).sum(1, dtype=np.float64))
        P['s2'].append(r2.reshape(sh).sum(1, dtype=np.float64))
        P['s3'].append((r2 * r).reshape(sh).sum(1, dtype=np.float64))
        P['s4'].append((r2 * r2).reshape(sh).sum(1, dtype=np.float64))
        P['i2'].append(i2.reshape(sh).sum(1, dtype=np.float64))
        P['i4'].append((i2 * i2).reshape(sh).sum(1, dtype=np.float64))
        P['q2'].append(q2.reshape(sh).sum(1, dtype=np.float64))
        P['q4'].append((q2 * q2).reshape(sh).sum(1, dtype=np.float64))
        P['mx'].append(r2.reshape(sh).max(1).astype(np.float64))
        self.nsamp += m

    def finish(self):
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.parts.items()}


class FinePSD:
    """Streaming Hann Welch PSD (50 % overlap) of the channelised signal, ~5 Hz resolution when the channel allows."""

    def __init__(self, rate_d, n_expected=None, target_df=5.0, max_nfft=1 << 15):
        n = 1 << int(round(math.log2(max(rate_d / target_df, 256))))
        self.nfft = int(min(max(n, 256), max_nfft))
        if n_expected is not None:                    # keep >= ~16 half-overlapped segments (K ~ 8 independent averages)
            self.nfft = int(max(64, min(self.nfft, 1 << int(math.floor(math.log2(max(n_expected / 16.0, 64)))))))
        self.rate_d = rate_d
        self.hop = self.nfft // 2
        self.w = np.hanning(self.nfft).astype(np.float32)
        self.acc = np.zeros(self.nfft)
        self.nseg = 0
        self.carry = np.zeros(0, np.complex64)

    def push(self, y):
        x = np.concatenate([self.carry, y]) if len(self.carry) else y
        ns = (len(x) - self.nfft) // self.hop + 1
        if ns <= 0:
            self.carry = x
            return
        st = x.strides[0]
        segs = np.lib.stride_tricks.as_strided(x, shape=(ns, self.nfft), strides=(self.hop * st, st), writeable=False)
        batch = max(1, (1 << 21) // self.nfft)
        for a in range(0, ns, batch):
            F = np.fft.fft(segs[a:a + batch] * self.w, axis=1)
            self.acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        self.nseg += ns
        self.carry = x[ns * self.hop:]

    def result(self):
        psd = self.acc / (max(self.nseg, 1) * self.rate_d * float((self.w.astype(np.float64) ** 2).sum()))
        f = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / self.rate_d))
        return f, np.fft.fftshift(psd), self.nseg


def channelize_ticks(src, rate, ch, mean, g, chunk=None, fine=None):
    """Stream `src` through the channeliser, accumulate tick sums.  Returns (ticks dict, n_out_samples)."""
    chunk = chunk or CFG['chunk']
    N, O, hop, M = ch.N, ch.O, ch.hop, ch.M
    k0 = ch.keep0
    acc = TickAcc(g)
    buf = np.zeros(O // 2, np.complex64)
    mean = np.complex64(mean)
    nin = 0
    for z in iter_chunks(src, chunk):
        nin += len(z)
        buf = np.concatenate([buf, z - mean])
        while len(buf) >= N:
            y = ch.block(buf[:N])
            acc.push(y[k0:M - k0])
            if fine is not None:
                fine.push(y[k0:M - k0])
            buf = buf[hop:]
    # tail: buf holds (up to) the last O samples already seen + the unseen remainder
    rem = len(buf)
    if rem > O // 2:
        pad = np.zeros(N, np.complex64)
        pad[:rem] = buf
        y = ch.block(pad)
        nkeep = int((rem - O // 2) * M / N)
        acc.push(y[k0:k0 + nkeep])
        if fine is not None:
            fine.push(y[k0:k0 + nkeep])
    return acc.finish(), acc.nsamp


# ----------------------------------------------------------------------------------------------
# burst detection on the tick power trace
# ----------------------------------------------------------------------------------------------
def runs_of(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _csum(a):
    return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])


def presence_scan(p, noise, kfun, cfg=None, alpha=None):
    """Noise-referenced multi-scale energy detector (presence only).  Windows of 1,2,4,.. ticks; threshold at
    the Gamma(k*m) upper quantile for a false alarm probability alpha/(n_scales * n_windows) times the
    systematic noise margin.  Returns the best z-score and scale; `any` = something is above the floor."""
    cfg = cfg or CFG
    alpha = cfg['alpha_det'] if alpha is None else alpha
    T = len(p)
    out = dict(any=False, zbest=0.0, mbest=0, n_scales=0)
    if T < 16:
        return out
    scales = []
    m = 1
    while m <= T // 4:
        scales.append(m)
        m *= 2
    ns = len(scales)
    cs = _csum(p)
    zbest, mbest = -1e9, 0
    anyflag = False
    for m in scales:
        q = (cs[m:] - cs[:-m]) / m
        k = kfun(m)
        am = alpha / (ns * max(T / m, 1.0))
        thr = noise * (1 + cfg['eps_sys']) * gamma_upper(k, am)
        qm = float(q.max())
        if qm > thr:
            anyflag = True
        z = (qm - noise) / (noise / math.sqrt(k))
        # margin-aware z: how far above the *threshold* in sigma units
        if z > zbest:
            zbest, mbest = z, m
    out.update(any=anyflag, zbest=float(zbest), mbest=mbest, n_scales=ns)
    return out


def fit_two_level(p, noise, k, iters=40):
    """EM for a two-component Gamma mixture with the noise component FIXED at mean `noise`:
    p_t ~ (1-d) Gamma(k, mean noise) + d Gamma(k, mean L).  Returns (d, L, loglik_gain_nats)."""
    p = np.maximum(p, 1e-12 * noise)
    lp = math.lgamma(k)
    lnp = np.log(p)

    def lf(mu):
        return (k - 1) * lnp - k * p / mu + k * math.log(k / mu) - lp
    ln_noise = lf(noise)
    # init: on level from the top decile
    top = np.sort(p)[int(0.9 * len(p)):]
    L = max(float(top.mean()), 1.05 * noise)
    d = 0.2
    for _ in range(iters):
        a = math.log(d) + lf(L)
        b = math.log(1 - d) + ln_noise
        m = np.maximum(a, b)
        r = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))
        d_new = float(np.clip(r.mean(), 1e-4, 1 - 1e-4))
        L_new = float((r * p).sum() / max(r.sum(), 1e-9))
        L_new = max(L_new, 1.0001 * noise)
        if abs(d_new - d) < 1e-5 and abs(L_new - L) < 1e-5 * L:
            d, L = d_new, L_new
            break
        d, L = d_new, L_new
    a = math.log(d) + lf(L)
    b = math.log(1 - d) + ln_noise
    mix = np.logaddexp(a, b)
    return d, L, float((mix - ln_noise).sum())


def viterbi2(cn, cl, lam):
    """Two-state Viterbi (pure python loop).  cn/cl: per-tick cost (neg log-lik) of off/on.  lam: switch cost."""
    T = len(cn)
    cn, cl = cn.tolist(), cl.tolist()
    b0, b1 = bytearray(T), bytearray(T)
    a0, a1 = cn[0], cl[0]
    for t in range(1, T):
        y0 = a1 + lam
        if y0 < a0:
            n0 = y0 + cn[t]
            b0[t] = 1
        else:
            n0 = a0 + cn[t]
        y1 = a0 + lam
        if y1 < a1:
            n1 = y1 + cl[t]
            b1[t] = 1
        else:
            n1 = a1 + cl[t]
        a0, a1 = n0, n1
    st = 0 if a0 <= a1 else 1
    out = bytearray(T)
    for t in range(T - 1, -1, -1):
        out[t] = st
        if st == 0:
            if b0[t]:
                st = 1
        else:
            if b1[t]:
                st = 0
    return np.frombuffer(bytes(out), dtype=np.uint8).astype(bool), min(a0, a1)


def segment_hmm(p, noise, k_tick, cfg=None):
    """On/off segmentation by a two-state HMM with Gamma emissions; the off level is the known noise power,
    the on level is estimated (EM, then Viterbi training).

    Two passes: (1) switch cost 0.5 ln T + 4 nats (the martingale bound P(false run) <= exp(-2 lambda) per tick
    gives ~1e-3 false runs per capture); (2) inside the dilated active region only, a cheaper switch cost
    (3 nats) to resolve short gaps (OOK chips).  Outside the active region pass 2 stays off."""
    cfg = cfg or CFG
    T = len(p)
    d, L, gain = fit_two_level(p, noise, k_tick)
    res = dict(em_d=d, em_L=L, em_gain=gain, on=np.zeros(T, bool), L=L, llr_nats=0.0)
    if L < 1.02 * noise:
        return res
    lam = 0.5 * math.log(max(T, 2)) + 4.0
    pp = np.maximum(p, 1e-12 * noise)
    cn = k_tick * pp / noise + k_tick * math.log(noise / k_tick)
    on = np.zeros(T, bool)
    for it in range(3):
        cl = k_tick * pp / L + k_tick * math.log(L / k_tick)
        on, cost = viterbi2(cn, cl, lam)
        if not on.any():
            break
        Ln = float(pp[on].mean())
        if abs(Ln - L) < 0.01 * L:
            L = Ln
            break
        L = max(Ln, 1.02 * noise)
    if on.any() and cfg.get('lam_fine', 3.0) < lam:
        act = _dilate(on, 20)
        cl = k_tick * pp / L + k_tick * math.log(L / k_tick)
        cl = np.where(act, cl, 1e6)
        on2, _ = viterbi2(cn, cl, cfg.get('lam_fine', 3.0))
        if on2.any():
            on = on2
            L = max(float(pp[on].mean()), 1.02 * noise)
    cl = k_tick * pp / L + k_tick * math.log(L / k_tick)
    res.update(on=on, L=L)
    res['llr_nats'] = float(cn.sum() - np.where(on, cl, cn).sum()) if on.any() else 0.0
    return res


# ----------------------------------------------------------------------------------------------
# statistics from tick sums
# ----------------------------------------------------------------------------------------------
def _moments(tk, sel, g):
    """Return additive sums over the ticks selected by boolean `sel` (or index array)."""
    n = float(np.count_nonzero(sel) if getattr(sel, 'dtype', None) == bool else len(sel)) * g
    return dict(n=n, **{k: float(tk[k][sel].sum()) for k in ('s1', 's2', 's3', 's4', 'i2', 'i4', 'q2', 'q4')},
                mx=float(tk['mx'][sel].max()) if n > 0 else 0.0)


def stats_from_sums(sm, noise=None):
    """Envelope statistics from additive moment sums.  `noise` (power of the additive complex Gaussian noise
    per sample inside the channel) enables the noise-compensated estimators."""
    n = sm['n']
    if n < 8:
        return None
    m1, m2, m3, m4 = sm['s1'] / n, sm['s2'] / n, sm['s3'] / n, sm['s4'] / n
    # s2 = sum r^2 ; careful: the field names above are sums of r^1..r^4 (s1..s4), r^2 power = s2
    pw = m2                                 # E|y|^2
    e4 = m4                                 # E|y|^4
    c2 = m2 - m1 ** 2
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    st = dict(power=pw, m4=e4 / pw ** 2, amp_kurt=c4 / c2 ** 2, cv=math.sqrt(max(c2, 0)) / m1,
              papr_db=10 * math.log10(max(sm['mx'], 1e-30) / pw),
              iq_kurt=0.5 * (n * sm['i4'] / sm['i2'] ** 2 + n * sm['q4'] / sm['q2'] ** 2))
    if noise is not None:
        S = pw - noise
        st['snr_ch_db'] = 10 * math.log10(max(S, 1e-9 * noise) / noise)
        if S > 0.05 * noise:
            st['m4_sig'] = (e4 - 4 * S * noise - 2 * noise ** 2) / S ** 2
        else:
            st['m4_sig'] = float('nan')
    return st


def m4_expected_ce(snr_lin):
    """E|y|^4/(E|y|^2)^2 of a constant-envelope signal (power S) plus complex Gaussian noise (power N)."""
    s = snr_lin
    return (s * s + 4 * s + 2) / (1 + s) ** 2


def jackknife_se(tk, sel_idx, g, noise, nblk=16):
    """Delete-one-block jackknife standard errors of m4, m4_sig, amp_kurt and snr_ch_db over the selected ticks."""
    if len(sel_idx) < nblk * 2:
        return {}
    parts = np.array_split(sel_idx, nblk)
    vals = []
    for i in range(nblk):
        idx = np.concatenate([parts[j] for j in range(nblk) if j != i])
        st = stats_from_sums(_moments(tk, idx, g), noise)
        if st is None:
            return {}
        vals.append(st)
    out = {}
    for k in ('m4', 'm4_sig', 'amp_kurt', 'snr_ch_db', 'cv'):
        v = np.array([s.get(k, float('nan')) for s in vals], float)
        if np.isnan(v).any():
            out['se_' + k] = float('nan')
        else:
            out['se_' + k] = float(math.sqrt((nblk - 1) / nblk * ((v - v.mean()) ** 2).sum()))
    return out


# ----------------------------------------------------------------------------------------------
# Otsu / bimodality on the (log) power trace
# ----------------------------------------------------------------------------------------------
def otsu(v, nb=128):
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo, 0.0, 0.0
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    p = h / h.sum()
    w = np.cumsum(p)
    c = 0.5 * (e[:-1] + e[1:])
    mu = np.cumsum(p * c)
    mt = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (mt * w - mu) ** 2 / (w * (1 - w))
    sb[~np.isfinite(sb)] = 0
    k = int(np.argmax(sb))
    var = float((p * (c - mt) ** 2).sum())
    return float(e[k + 1]), float(sb[k] / var) if var > 0 else 0.0, float(w[k])


def bimodality_coef(v):
    n = len(v)
    if n < 10:
        return float('nan')
    m = v.mean()
    d = v - m
    s2 = (d ** 2).mean()
    if s2 <= 0:
        return float('nan')
    sk = (d ** 3).mean() / s2 ** 1.5
    ku = (d ** 4).mean() / s2 ** 2          # non-excess
    return float((sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


# ----------------------------------------------------------------------------------------------
# run statistics (burst count / lengths / chip quantisation)
# ----------------------------------------------------------------------------------------------
def run_stats(on, tick_s):
    s, e = runs_of(on)
    d = dict(n_runs=int(len(s)), duty=float(on.mean()) if len(on) else 0.0)
    if len(s) == 0:
        return d
    ln = (e - s) * tick_s
    d.update(run_med_s=float(np.median(ln)), run_min_s=float(ln.min()), run_max_s=float(ln.max()),
             run_cv=float(ln.std() / ln.mean()))
    if len(s) > 1:
        gp = (s[1:] - e[:-1]) * tick_s
        pr = np.diff(s) * tick_s
        d.update(gap_med_s=float(np.median(gp)), gap_min_s=float(gp.min()), gap_max_s=float(gp.max()),
                 pri_med_s=float(np.median(pr)), pri_cv=float(pr.std() / pr.mean()))
    return d


def chip_estimate(on, tick_s):
    """Symbol/chip-rate hint from on/off run lengths: smallest 'quantum' q such that run lengths are
    near-integer multiples of q.  Returns (rate_hz, quality in [0,1]) or (nan, 0)."""
    s, e = runs_of(on)
    if len(s) < 6:
        return float('nan'), 0.0
    on_len = (e - s).astype(float)
    off_len = (s[1:] - e[:-1]).astype(float)
    L = np.concatenate([on_len, off_len])
    # drop frame gaps (very long compared with the typical run): keep runs below 6x the 25th percentile
    q25 = np.percentile(L, 25)
    Lk = L[L <= 6 * q25]
    if len(Lk) < 6:
        return float('nan'), 0.0
    best = (float('nan'), 0.0)
    cands = np.linspace(0.6 * np.percentile(Lk, 10), 1.6 * np.percentile(Lk, 10), 120)
    scores = []
    for q in cands:
        if q < 1.5:
            scores.append(0.0)
            continue
        r = Lk / q
        # raised-cosine comb score: 1 when every length is an integer multiple of q
        sc = float(np.mean(np.cos(2 * math.pi * r)))
        scores.append(sc)
    i = int(np.argmax(scores))
    return float(1.0 / (cands[i] * tick_s)), float(scores[i])


# ----------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------
def _erode_runs(on, e_ticks):
    s, e = runs_of(on)
    d = np.zeros(len(on) + 1, np.int32)
    for a, b in zip(s, e):
        k = min(e_ticks, (b - a) // 4)
        if b - k > a + k:
            d[a + k] += 1
            d[b - k] -= 1
    return np.cumsum(d[:-1]) > 0


def _dilate(mask, w):
    if w <= 0:
        return mask
    c = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    n = len(mask)
    lo = np.clip(np.arange(n) - w, 0, n)
    hi = np.clip(np.arange(n) + w + 1, 0, n)
    return (c[hi] - c[lo]) > 0


def carrier_test(fine, ch, noise):
    """Carrier-present test on the fine (~5 Hz) PSD of the channelised signal.

    carrier_frac : share of the in-passband excess power (PSD minus the known noise level) inside +-2 fine bins of the
                   strongest bin.  A tone is ~1, AM carrier 0.5-0.97, FM/FSK/PSK/OOK/OFDM <= 0.5.
    carrier_hz   : frequency of the peak relative to the channel centre.
    carrier_over_thr_db : 3-bin peak / noise level in dB above the analytic false-alarm threshold (> 0: a narrow peak exists)."""
    f, P, nseg = fine.result()
    df = f[1] - f[0]
    floor_f = noise / ch.enbw                               # noise PSD (power/Hz) inside the channel
    inb = np.abs(f) <= 0.5 * ch.bw
    out = dict(carrier_frac=float('nan'), carrier_z=float('nan'), carrier_over_thr_db=float('nan'), carrier_hz=float('nan'),
               carrier_df=float(df), carrier_nseg=int(nseg))
    if nseg < 4 or inb.sum() < 8:
        return out
    ex = np.where(inb, P - floor_f, 0.0)
    tot = float(ex.sum())                                    # unclipped: the noise term averages out
    p3 = np.convolve(P, np.ones(3) / 3.0, mode='same')
    i3 = int(np.argmax(np.where(inb, p3, 0)))
    K = max(nseg / 1.056, 1.0)
    dof3 = _dof(K, 3)
    nbin = max(int(inb.sum()), 3)
    thr = (1 + CFG['eps_sys']) * gamma_upper(dof3, CFG['alpha_band'] / (nbin / 2.0))
    ratio = max(p3[i3] / floor_f, 1e-9)
    out.update(carrier_hz=float(f[i3]), carrier_z=float((ratio - 1.0) * math.sqrt(dof3)),
               carrier_over_thr_db=float(10 * math.log10(ratio / thr)))
    if tot > 0:
        p5 = np.convolve(ex, np.ones(5), mode='same')
        j = int(np.argmax(p5))
        out.update(carrier_frac=float(min(max(p5[j] / tot, 0.0), 1.0)), carrier_hz=float(f[j]))
    return out


def candidate_channels(band, prior, rate, df, cfg, line=None):
    """Channels to try, as (fc, bw, source).  The sweep gives a coarse prior (centre +-50 kHz, bandwidth >= its bin
    width); the PSD refines it.  Candidates: (1) hull of the PSD clusters inside the prior window, (2) the prior band
    itself when the hull is much narrower (wideband bursts whose average PSD is too weak to be flagged).
    Without a prior: the strongest PSD cluster."""
    cl = band['clusters']
    marg = cfg['chan_margin']
    blur = 2 * cfg['smooth_bins'] * df
    if prior is None:
        if not cl:
            return []
        return [(band['fc'], band['bw'] * marg + blur, 'psd')]
    pfc, pbw = prior
    lo_p, hi_p = pfc - pbw, pfc + pbw
    inside = [c for c in cl if c['f_hi'] >= lo_p and c['f_lo'] <= hi_p]
    out = []
    if inside:
        lo = min(c['f_lo'] for c in inside)
        hi = max(c['f_hi'] for c in inside)
        out.append((0.5 * (lo + hi), (hi - lo) * marg + blur, 'psd'))
        if (hi - lo) < 0.5 * pbw:
            out.append((pfc, pbw, 'prior'))
    else:
        out.append((pfc, pbw, 'prior'))
    return out


def analyze(src, rate, prior=None, oracle_band=None, oracle_on=None, cfg=None, keep_trace=False, refine_noise=True):
    """Full characterisation of one capture.  src: complex ndarray or ci8 path.

    prior        optional (fc_hz, bw_hz) = what the sweep knew (bandwidth >= its bin width, centre +- half a bin)
    oracle_band  (fc_hz, bw_hz) forces the channel (bench diagnostics: separates band-finder errors from the rest)
    oracle_on    function(tick_s, n_ticks, t0_tick) -> bool mask: forces the on-mask (diagnostic)
    """
    cfg = cfg or CFG
    R = dict(rate=rate)
    tm = {}
    t = time.perf_counter()
    p1 = pass1(src, rate, subtract_dc=cfg.get('subtract_dc', True))
    tm['pass1'] = time.perf_counter() - t
    t = time.perf_counter()
    band = find_band(p1, cfg)
    line = line_test(p1, band, cfg)
    tm['band'] = time.perf_counter() - t
    R.update(raw=p1['raw'], n=p1['n'], seconds=p1['n'] / rate, dc=p1['mean'], nseg=p1['nseg'], df=p1['df'],
             floor=band['floor'], floor_ok=band['floor_ok'], n_clusters=band['n_clusters'],
             line=line, band_excess_frac=band.get('band_excess_frac', 0.0),
             band_snr_db=band.get('band_snr_db', float('nan')),
             band_fc=band.get('fc', float('nan')), band_bw=band.get('bw', float('nan')))
    df = p1['df']
    if oracle_band is not None:
        cands = [(oracle_band[0], oracle_band[1], 'oracle')]
    else:
        cands = candidate_channels(band, prior, rate, df, cfg, line)
    if not cands:
        R['channel'] = None
        R['timing'] = tm
        return R
    best = None
    for (fc, bw_ch, chan_src) in cands:
        Rc = _analyze_channel(src, rate, p1, band, fc, bw_ch, chan_src, oracle_on, cfg, keep_trace, refine_noise)
        Rc['n_cand'] = len(cands)
        if best is None or Rc['det']['zbest'] > best['det']['zbest']:
            best = Rc
        tm.setdefault('pass2', 0.0)
        tm['pass2'] += Rc['timing']['pass2']
        tm['feats'] = tm.get('feats', 0.0) + Rc['timing']['feats']
    R.update({k: v for k, v in best.items() if k != 'timing'})
    R['timing'] = tm
    return R


def _analyze_channel(src, rate, p1, band, fc, bw_ch, chan_src, oracle_on, cfg, keep_trace, refine_noise):
    R = {}
    tm = {}
    df = p1['df']
    bw_raw = bw_ch
    bw_ch = max(bw_ch, cfg['min_chan_bins'] * df)
    bw_ch = min(bw_ch, 0.96 * rate / (1 + cfg['rho']))
    ch = Channel(rate, fc, bw_ch)
    noise = band['floor'] * rate * ch.noise_gain
    nout_est = p1['n'] * ch.M / ch.N
    g = max(1, int(round(cfg['tick_s'] * ch.rate_d)))
    while nout_est / g > cfg['max_ticks']:
        g *= 2
    t = time.perf_counter()
    fine = FinePSD(ch.rate_d, n_expected=nout_est)
    tk, nout = channelize_ticks(src, rate, ch, p1['mean'], g, fine=fine)
    tm['pass2'] = time.perf_counter() - t
    t = time.perf_counter()
    tick_s = g / ch.rate_d
    d0 = int(min(math.ceil(cfg['ring_trans'] / ch.trans_hz / tick_s), 0.05 * len(tk['s2'])))
    sl = slice(d0, len(tk['s2']) - d0)
    for k in tk:
        tk[k] = tk[k][sl]
    T = len(tk['s2'])
    k_tick = ch.dof(g)
    p = tk['s2'] / g
    e_ticks = int(math.ceil(1.0 / ch.trans_hz / tick_s)) + 1
    kfun = lambda m: ch.dof(g * m)          # noqa: E731
    noise_psd = noise
    noise_src = 'psd'
    refined = False
    for attempt in range(2):
        pres = presence_scan(p, noise, kfun, cfg)
        on = np.zeros(T, bool)
        seg = dict(em_d=0.0, em_L=noise, em_gain=0.0, L=noise, llr_nats=0.0)
        if pres['any']:
            seg = segment_hmm(p, noise, k_tick, cfg)
            on = seg['on']
        if oracle_on is not None or not refine_noise or attempt == 1 or not on.any():
            break
        # one-sided noise refinement: the PSD floor can only be biased HIGH by unflagged signal skirts
        off = ~_dilate(on, e_ticks + 2)
        n_off = int(off.sum())
        if n_off < 300:
            break
        r = float(p[off].mean() / noise)
        if r < 1.0 - max(0.005, 3.0 / math.sqrt(n_off * k_tick)) and r > 0.6:
            noise = noise * r
            noise_src = 'off'
            refined = True
        else:
            break
    if oracle_on is not None:
        on = oracle_on(tick_s, T, d0)
    R.update(channel=dict(src=chan_src, fc=fc, bw=bw_ch, bw_raw=bw_raw, rate_d=ch.rate_d, M=ch.M, g=g, tick_s=tick_s,
                          T=T, d0=d0, k_tick=k_tick, noise=noise, noise_psd=noise_psd, noise_src=noise_src,
                          enbw=ch.enbw))
    allm = _moments(tk, np.ones(T, bool), g)
    R['all'] = stats_from_sums(allm, noise)
    R['p_mean_over_noise'] = float(p.mean() / noise)
    R['det'] = dict(zbest=pres['zbest'], mbest=pres['mbest'], any=pres['any'], em_d=seg['em_d'],
                    em_L_over_N=seg['em_L'] / noise, L_over_N=seg['L'] / noise, llr=seg.get('llr_nats', 0.0),
                    em_gain=seg['em_gain'], refined=float(refined))
    R['duty'] = float(on.mean())
    R['runs'] = run_stats(on, tick_s)
    # on-state (interior) statistics and off-state statistics
    inter = _erode_runs(on, e_ticks)
    idx_on = np.flatnonzero(inter)
    R['n_on_ticks'] = int(len(idx_on))
    if len(idx_on) >= 8:
        R['on'] = stats_from_sums(_moments(tk, idx_on, g), noise)
        R['on_se'] = jackknife_se(tk, idx_on, g, noise)
    else:
        R['on'], R['on_se'] = None, {}
    off = ~_dilate(on, e_ticks + 2) if on.any() else np.ones(T, bool)
    idx_off = np.flatnonzero(off)
    if len(idx_off) >= 8 and on.any():
        R['off'] = stats_from_sums(_moments(tk, idx_off, g), noise)
    else:
        R['off'] = stats_from_sums(allm, noise) if not on.any() else None
    R['noise_ratio_off'] = float(R['off']['power'] / noise) if R.get('off') else float('nan')
    # --- smoothed-envelope distribution: Otsu / bimodality ---------------------------------------
    w_o = max(1, int(math.ceil(8.0 / k_tick)))
    cs = _csum(p)
    ps = (cs[w_o:] - cs[:-w_o]) / w_o
    lp = np.log10(np.maximum(ps, 1e-12 * noise))
    thr, eta, wlow = otsu(lp)
    R['otsu'] = dict(thr_over_noise_db=float(10 * (thr - math.log10(noise))), eta=eta, duty=float(1 - wlow),
                     bc=bimodality_coef(lp), spread_db=float(10 * (np.percentile(lp, 99) - np.percentile(lp, 1))))
    R['chip'] = chip_estimate(on, tick_s)
    R['carrier'] = carrier_test(fine, ch, noise)
    tm['feats'] = time.perf_counter() - t
    R['timing'] = tm
    if keep_trace:
        R['trace'] = dict(p=p, on=on, inter=inter, tk=tk, noise=noise, d0=d0, tick_s=tick_s, k_tick=k_tick, det=pres)
    return R
