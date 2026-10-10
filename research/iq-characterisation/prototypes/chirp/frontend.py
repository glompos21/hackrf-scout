"""Streaming front end for the chirp / spread-spectrum / pulsed prototype.

numpy only, Python >= 3.9, no 3.10+ syntax.  Everything works on a stream of complex64 chunks
(ci8 file read in pieces, or an in-memory array in the benchmark harness), never on the whole file.

  pass 1  (stream)   running mean (the DC spike) + Welch PSD of the whole capture
  band    (PSD)      noise floor, occupied region(s): centre, -3 dB bandwidth, flatness
  pass 2  (stream)   FFT channeliser: block FFT, keep the bins of one sub-band, small IFFT
                     = mix to baseband + brick-wall filter + decimation in one step

Sources
    ArraySource(iq, rate)        complex array already in memory
    Ci8FileSource(path, rate)    signed 8-bit interleaved I/Q (hackrf_transfer / SigMF ci8)
"""
from __future__ import annotations

import math
import os

import numpy as np

CHUNK = 1 << 20            # samples per chunk


# ----------------------------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------------------------
class ArraySource(object):
    def __init__(self, iq, rate):
        self.iq = iq
        self.rate = float(rate)
        self.n = len(iq)

    def chunks(self, size=CHUNK):
        for a in range(0, self.n, size):
            yield self.iq[a:a + size]

    def read(self, a, n):
        return self.iq[max(a, 0):max(a, 0) + n]


class Ci8FileSource(object):
    def __init__(self, path, rate):
        self.path = path
        self.rate = float(rate)
        self.n = os.path.getsize(path) // 2

    def read(self, a, n):
        a = max(int(a), 0)
        with open(self.path, 'rb') as fh:
            fh.seek(2 * a)
            buf = fh.read(2 * int(n))
        buf = buf[:len(buf) - (len(buf) & 1)]
        return np.frombuffer(buf, np.int8).astype(np.float32).view(np.complex64)

    def chunks(self, size=CHUNK):
        with open(self.path, 'rb') as fh:
            while True:
                buf = fh.read(size * 2)
                if not buf:
                    return
                if len(buf) & 1:
                    buf = buf[:-1]
                a = np.frombuffer(buf, np.int8).astype(np.float32)
                yield a.view(np.complex64)


def rechunk(src, size):
    """Yield exactly `size`-sample blocks (last one shorter is dropped); constant memory."""
    buf = None
    for c in src.chunks():
        buf = c if buf is None else np.concatenate([buf, c])
        while len(buf) >= size:
            yield buf[:size]
            buf = buf[size:]


# ----------------------------------------------------------------------------------------------
# pass 1: mean + Welch PSD
# ----------------------------------------------------------------------------------------------
def pick_nfft(rate):
    """~500 Hz bins at 2 Msps (4096), ~600 Hz at 10 Msps (16384)."""
    return 4096 if rate <= 4.5e6 else 16384


def pass1(src, nfft=None):
    """Return dict(mean, freqs, psd, nseg, nfft, n).  psd in power/Hz (|x|^2 units per Hz), raw (DC not removed)."""
    rate = src.rate
    nfft = nfft or pick_nfft(rate)
    hop = nfft // 2
    w = np.hanning(nfft).astype(np.float32)
    wn = float((w.astype(np.float64) ** 2).sum())
    acc = np.zeros(nfft, np.float64)
    nseg = 0
    s = np.zeros(1, np.complex128)
    n = 0
    carry = None
    for c in src.chunks(CHUNK):
        s += c.sum(dtype=np.complex128)
        n += len(c)
        x = c if carry is None else np.concatenate([carry, c])
        m = (len(x) - nfft) // hop + 1
        if m <= 0:
            carry = x
            continue
        idx = np.arange(m)[:, None] * hop + np.arange(nfft)[None, :]
        F = np.fft.fft(x[idx] * w[None, :], axis=1)
        acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        nseg += m
        carry = x[m * hop:]
    psd = acc / (max(nseg, 1) * rate * wn)
    return dict(mean=complex(s[0] / max(n, 1)), freqs=np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate)),
                psd=np.fft.fftshift(psd), nseg=nseg, nfft=nfft, n=n, rate=rate)


# ----------------------------------------------------------------------------------------------
# band finder
# ----------------------------------------------------------------------------------------------
def _runs(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def find_bands(p1, smooth_hz=6e3, sigma=5.0, min_excess_db=0.6, merge_hz=20e3):
    """Occupied regions of the (time-averaged) PSD.

    floor: 25th percentile of the smoothed PSD with the small chi-square bias removed (assumes < 75 % of
    the band is occupied).  region: smoothed PSD above floor*(1 + max(sigma/sqrt(K), min_excess)).
    bandwidth: width at half of the region's plateau height above the floor (-3 dB of the excess), which
    for a flat-topped signal is its occupied bandwidth.  Returns (floor, list of dicts, info)."""
    f, P = p1['freqs'], p1['psd'].copy()
    rate = p1['rate']
    df = f[1] - f[0]
    # remove the DC spike: replace |f| < 3 bins by the median of the neighbours
    dc = np.abs(f) < 3.5 * df
    nb = (np.abs(f) >= 3.5 * df) & (np.abs(f) < 20 * df)
    P[dc] = np.median(P[nb])
    s = max(1, int(round(smooth_hz / df)))
    K = max(1.0, p1['nseg'] / 1.8 * s)                        # effective averages per smoothed bin
    Ps = np.convolve(P, np.ones(s) / s, mode='same')
    edge = s // 2 + 1
    core = slice(edge, len(Ps) - edge)
    q = 25.0
    zq = -0.6745
    floor = float(np.percentile(Ps[core], q)) / (1.0 + zq / math.sqrt(K))
    rel = max(sigma / math.sqrt(K), 10 ** (min_excess_db / 10.0) - 1.0)
    thr = floor * (1.0 + rel)
    above = Ps > thr
    above[:edge] = False
    above[-edge:] = False
    st, en = _runs(above)
    # merge regions separated by less than merge_hz
    regs = []
    for a, b in zip(st, en):
        if regs and (a - regs[-1][1]) * df < merge_hz:
            regs[-1][1] = b
        else:
            regs.append([a, b])
    out = []
    for a, b in regs:
        seg = Ps[a:b] - floor
        if b - a < 2:
            continue
        plateau = float(np.percentile(seg, 85))
        half = seg > 0.5 * plateau
        hs, he = _runs(half)
        if len(hs) == 0:
            continue
        # outermost half-height crossings of the region
        lo, hi = a + hs[0], a + he[-1]
        bw = (hi - lo) * df
        exc = float(np.maximum(P[a:b] - floor, 0).sum() * df)          # excess power in the region
        w = np.maximum(Ps[a:b] - floor, 0)
        cen = float((f[a:b] * w).sum() / max(w.sum(), 1e-30))
        edges_mid = 0.5 * (f[lo] + f[min(hi, len(f) - 1)])
        out.append(dict(lo_hz=float(f[a]), hi_hz=float(f[min(b, len(f) - 1)]), centre_hz=float(edges_mid),
                        centroid_hz=cen, bw3_hz=float(bw), bw_full_hz=float((b - a) * df),
                        excess_power=exc, plateau_db=float(10 * math.log10(1 + plateau / floor)),
                        n_bins=int(b - a)))
    out.sort(key=lambda r: -r['excess_power'])
    return floor, out, dict(df=df, K=K, thr=thr, rel=rel, smooth_bins=s)


# ----------------------------------------------------------------------------------------------
# channeliser (pass 2)
# ----------------------------------------------------------------------------------------------
class Channeliser(object):
    """Select the sub-band [fc - bw/2, fc + bw/2) of every block of `nin` samples and output nin/D samples at
    rate bw = rate / D.  Brick-wall filter + mix + decimation in the frequency domain.  Bins outside the
    band are *dropped* (not aliased).  Output scaled so the noise power inside the band is preserved
    (E|y|^2 = N0*bw, the same LSB^2 units as the input)."""

    def __init__(self, rate, fc, bw, nin):
        D = int(round(rate / bw))
        self.D = max(1, D)
        self.rate = float(rate)
        self.out_rate = rate / self.D
        self.nin = int(nin)
        self.nout = self.nin // self.D
        kc = int(round(fc / rate * self.nin))
        j = np.arange(self.nout) - self.nout // 2
        self.idx = (kc + j) % self.nin
        self.fc = kc * rate / self.nin
        self.scale = 1.0 / self.D

    def apply(self, X):
        """X = fft of one input block (length nin) -> complex64 output block."""
        return (np.fft.ifft(np.fft.ifftshift(X[self.idx])) * self.scale).astype(np.complex64)

    def block(self, xblk):
        return self.apply(np.fft.fft(xblk))


def channelise_stream(src, mean, chans, nin_max=1 << 16):
    """One pass over `src`.  `chans` is a list of Channeliser objects that share the same nin; returns a list
    of concatenated complex64 outputs (one per channeliser).  DC (`mean`) is removed first."""
    nin = chans[0].nin
    for c in chans:
        assert c.nin == nin
    outs = [[] for _ in chans]
    m = np.complex64(mean)
    for blk in rechunk(src, nin):
        b = blk - m
        X = np.fft.fft(b)
        for i, c in enumerate(chans):
            outs[i].append(c.apply(X))
    return [np.concatenate(o) if o else np.zeros(0, np.complex64) for o in outs]


def common_nin(rate, bws, target=1 << 16):
    """Block length (samples) that is a whole number of output samples for every bw in `bws`."""
    ds = [int(round(rate / b)) for b in bws]
    lcm = 1
    for d in ds:
        lcm = lcm * d // math.gcd(lcm, d)
    nin = lcm * max(1, int(round(target / lcm)))
    return nin
