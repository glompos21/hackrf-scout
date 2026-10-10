"""Radar-like pulse detection from the envelope: width and PRI, blind (noise level from the capture PSD).

  envelope   p = |y|^2 at the capture rate (or after a PSD-guided channeliser)
  detection  multi-scale box-car energy detector (m = 1,2,4,...,64 samples); the threshold at scale m is the
             Erlang(m) quantile of the *estimated* noise power (PSD floor x rate), one false sample in ~5e7
  measure    per event: -3 dB width at the best scale's peak (linear interpolation), leading-edge time,
             peak power above noise, rough in-channel SNR
  PRI        robust PRI from the in-train spacing of the event starts, share of spacings that are integer
             multiples (missed pulses), relative jitter
  decision   'pulsed' needs >= n_min events with consistent width (MAD/median), a stable PRI, low duty
  intra-pulse chirp: slope of the lag-1 phase increments inside each pulse (frequency ramp) pooled over pulses
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------------------------
# Erlang quantiles
# ----------------------------------------------------------------------------------------------
def erlang_upper(m, p):
    """x such that P(mean of m unit exponentials > x) = p."""
    def tail(x):
        mx = m * x
        k = np.arange(m)
        lg = np.array([math.lgamma(i + 1) for i in k])
        return float(np.exp(-mx + k * math.log(mx) - lg).sum())
    lo, hi = 1.0, 2.0
    while tail(hi) > p:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tail(mid) > p:
            lo = mid
        else:
            hi = mid
    return hi


def _runs(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def boxcar(p, m):
    """Centered moving average of length m (m odd or even), same length, edges use the available samples."""
    if m <= 1:
        return p
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    n = len(p)
    i = np.arange(n)
    lo = np.clip(i - m // 2, 0, n)
    hi = np.clip(i - m // 2 + m, 0, n)
    return ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# event detection
# ----------------------------------------------------------------------------------------------
SCALES = (4, 16, 64, 256, 1024)   # box-car lengths (samples), factor 4 apart: <= ~2 dB mismatch loss
_THR_CACHE = {}


def thr_factor(m, pfa):
    k = (m, pfa)
    if k not in _THR_CACHE:
        _THR_CACHE[k] = erlang_upper(m, pfa)
    return _THR_CACHE[k]


def detect_chunk(p, noise, pfa=1e-8, scales=SCALES):
    """Candidate pulses in one chunk of the power envelope.  Returns an array of (centre, scale, z) rows:
    the box-car (stride m/2) maximum of every above-threshold cluster, duplicates across scales merged."""
    n = len(p)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    rows = []
    for m in scales:
        if n < 2 * m:
            continue
        stride = max(1, m // 2)
        i0 = np.arange(0, n - m + 1, stride)
        S = (c[i0 + m] - c[i0]) / m
        thr = noise * thr_factor(m, pfa)
        flag = S > thr
        if not flag.any():
            continue
        if len(flag) > 3:                       # close short holes so a long burst is one event
            flag = np.convolve(flag.astype(np.int8), np.ones(5, np.int8), mode='same') > 0
        st, en = _runs(flag)
        for a, b in zip(st, en):
            k = a + int(np.argmax(S[a:b]))
            rows.append((i0[k] + m / 2.0, m, float(S[k] / thr)))
    if not rows:
        return np.zeros((0, 3))
    rows = np.array(sorted(rows))
    # merge across scales: events whose centres are closer than the larger box-car -> keep the max z
    keep = []
    cur = rows[0]
    for r in rows[1:]:
        if r[0] - cur[0] <= max(cur[1], r[1]):
            if r[2] > cur[2]:
                cur = r
        else:
            keep.append(cur)
            cur = r
    keep.append(cur)
    return np.array(keep)


def refine(p, cand, noise, half=None):
    """Width, edges, peak level of each candidate from the raw envelope (vectorised).  Returns list of dicts."""
    out = []
    n = len(p)
    for cen, m, z in cand:
        w = int(max(48, 6 * m))
        a = int(max(0, round(cen) - w))
        b = int(min(n, round(cen) + w))
        seg = p[a:b].astype(np.float64)
        if len(seg) < 16:
            continue
        sm = np.convolve(seg, np.ones(3) / 3.0, mode='same')
        # plateau level: median of the top part around the maximum of a length-m average
        mm = int(min(max(m, 3), len(seg) // 2))
        avg = np.convolve(seg, np.ones(mm) / mm, mode='same')
        k = int(np.argmax(avg))
        lvl = float(avg[k])
        # plateau: samples within [k - mm/2, k + mm/2] -> their mean is lvl; refine with the 70 % region
        top = sm >= noise + 0.7 * (lvl - noise)
        s0, e0 = _runs(top)
        j = int(np.searchsorted(s0, k, 'right') - 1)
        if j >= 0 and e0[j] > k:
            lvl = float(seg[s0[j]:e0[j]].mean())
        peak = max(lvl - noise, 1e-30)
        halfl = noise + 0.5 * peak
        above = sm > halfl
        s1, e1 = _runs(above)
        if len(s1) == 0:
            continue
        j = int(np.searchsorted(s1, k, 'right') - 1)
        j = min(max(j, 0), len(s1) - 1)
        lo, hi = int(s1[j]), int(e1[j])
        if not (lo <= k < hi):
            d = np.abs((s1 + e1) / 2.0 - k)
            j = int(np.argmin(d))
            lo, hi = int(s1[j]), int(e1[j])

        def cross(i0, i1):
            d = sm[i1] - sm[i0]
            return i0 + (halfl - sm[i0]) / d if abs(d) > 1e-30 else float(i1)
        t_lo = cross(lo - 1, lo) if lo > 0 else float(lo)
        t_hi = cross(hi - 1, hi) if hi < len(sm) else float(hi)
        out.append(dict(t0=a + t_lo + 0.5, t1=a + t_hi + 0.5, width=t_hi - t_lo, peak=peak, snr=peak / noise,
                        scale=int(m), z=float(z), long=bool(hi - lo >= len(seg) - 4)))
    return out


class PulseDetector(object):
    """Incremental detector: feed() consecutive chunks of the power envelope, then finish().  Candidates are
    assigned to the chunk that holds their centre (512-sample overlap) so nothing is counted twice."""

    def __init__(self, noise, pfa=1e-8, overlap=512):
        self.noise, self.pfa, self.overlap = noise, pfa, overlap
        self.events = []
        self.offset = 0
        self.tail = np.zeros(0, np.float32)

    def feed(self, pc):
        p = np.concatenate([self.tail, pc]) if len(self.tail) else pc
        base = self.offset - len(self.tail)
        cand = detect_chunk(p, self.noise, self.pfa)
        if len(cand):
            lo = (len(self.tail) - self.overlap // 2) if len(self.tail) else -1
            hi = len(p) - self.overlap // 2
            cand = cand[(cand[:, 0] >= lo) & (cand[:, 0] < hi)]
            for ev in refine(p, cand, self.noise):
                ev['t0'] += base
                ev['t1'] += base
                self.events.append(ev)
        self.tail = p[-self.overlap:].copy()
        self.offset += len(pc)

    def finish(self):
        return self.events


def detect_pulses(chunks, noise, rate, pfa=1e-8, overlap=512):
    d = PulseDetector(noise, pfa, overlap)
    for pc in chunks:
        d.feed(pc)
    return d.finish()


def stack_profile(src, mean, events, noise, max_events=600, iters=2):
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


# ----------------------------------------------------------------------------------------------
# PRI
# ----------------------------------------------------------------------------------------------
def estimate_pri(t, rate, tol_rel=0.06, tol_abs_s=3e-6):
    """t: sorted event start times (s).  Returns dict(pri, n_in_train, frac_multiple, jitter, ok)."""
    t = np.asarray(t, np.float64)
    if len(t) < 4:
        return dict(pri=None, frac_multiple=0.0, jitter=None, n_pairs=0)
    d = np.diff(t)
    d = d[d > 1e-9]
    if len(d) < 3:
        return dict(pri=None, frac_multiple=0.0, jitter=None, n_pairs=0)
    # in-train spacing: the dominant short spacing (10th percentile cluster)
    d_sorted = np.sort(d)
    ref = d_sorted[max(0, int(0.1 * len(d_sorted)))]
    cl = d[(d > 0.6 * ref) & (d < 1.6 * ref)]
    if len(cl) < 3:
        return dict(pri=float(np.median(d)), frac_multiple=0.0, jitter=None, n_pairs=len(d))
    pri = float(np.median(cl))
    # refine: allow the cluster to be centred on the median
    cl = d[(d > 0.75 * pri) & (d < 1.25 * pri)]
    pri = float(np.median(cl))
    # share of all spacings below 8 PRI that are integer multiples (missed pulses allowed) of pri
    dd = d[d < 8.5 * pri]
    k = np.maximum(np.round(dd / pri), 1)
    res = np.abs(dd - k * pri)
    ok = res <= (tol_rel * pri * np.sqrt(k) + tol_abs_s)
    jit = float(np.std((cl - pri)) / pri)
    return dict(pri=pri, frac_multiple=float(ok.mean()), jitter=jit, n_pairs=int(len(dd)), n_cluster=int(len(cl)))


def completeness(t, pri, split=4.5):
    """Share of the pulses expected on the PRI grid inside each train (a train ends at a gap > split*PRI) that
    were actually detected.  Low values mean only the noise-boosted pulses were found (selection bias: the
    stacked width and the PRI are then optimistic)."""
    t = np.sort(np.asarray(t, np.float64))
    if len(t) < 2 or not pri:
        return 0.0
    gaps = np.diff(t)
    brk = np.flatnonzero(gaps > split * pri)
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk + 1, [len(t)]])
    exp = 0.0
    det = 0.0
    for a, b in zip(starts, ends):
        n = b - a
        if n < 2:
            continue
        exp += (t[b - 1] - t[a]) / pri + 1.0
        det += n
    return float(min(1.0, det / exp)) if exp > 0 else 0.0


def decide_pulsed(events, rate, pri, width_s, cfg):
    """Return (flag, info).  `width_s`: -3 dB width of the stacked pulse profile (or None).  The per-event width
    spread is only judged on the strong events (peak > 8 dB over the noise), where it is measurable."""
    n = len(events)
    info = dict(n_events=n)
    if n < cfg['n_min'] or not pri.get('pri'):
        info['why'] = 'too few events or no PRI'
        return False, info
    w = np.array([(e['t1'] - e['t0']) / rate for e in events])
    strong = np.array([e['snr'] >= 31.6 for e in events])
    wm = float(width_s) if width_s else float(np.median(w))
    duty = wm / pri['pri']
    info.update(width=wm, pri=pri['pri'], jitter=pri['jitter'], frac_multiple=pri['frac_multiple'], duty=duty,
                n_cluster=pri.get('n_cluster', 0), n_strong=int(strong.sum()))
    cv = None
    if strong.sum() >= 20:
        ws = w[strong]
        med = float(np.median(ws))
        cv = float(np.median(np.abs(ws - med)) / max(med, 1e-12) * 1.4826)
        info['width_cv'] = cv
    reasons = []
    if pri.get('n_cluster', 0) < cfg['n_cluster_min']:
        reasons.append('few in-train spacings')
    if pri['frac_multiple'] < cfg['multiple_min']:
        reasons.append('PRI not regular')
    if (pri['jitter'] or 0) > cfg['jitter_max']:
        reasons.append('PRI jitter')
    if duty > cfg['duty_max']:
        reasons.append('duty too high')
    if cv is not None and cv > cfg['cv_max']:
        reasons.append('width spread')
    info['why'] = ', '.join(reasons)
    return (len(reasons) == 0), info


# ----------------------------------------------------------------------------------------------
# intra-pulse chirp
# ----------------------------------------------------------------------------------------------
def intrapulse_slope(src, mean, events, rate, lead=0.15, min_samples=8, max_events=300, band_hz=None, fc_hz=0.0):
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
