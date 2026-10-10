"""Front end for the symbol-rate / modulation-family prototype.

numpy only, Python >= 3.9.  Everything here is streaming: the capture is read in chunks
(pass 1: mean + Welch PSD over the whole file; pass 2: FFT channeliser that mixes the detected
signal to baseband, band-limits it, decimates it and keeps only blocks that contain signal, stopping
as soon as enough have been kept).  Only the decimated, selected record (a few MB) is ever held in memory.

Source classes
    ArraySource(iq, rate)           complex64 array already in memory (the benchmark harness)
    Ci8FileSource(path, rate)       signed 8-bit interleaved I/Q file (hackrf_transfer / SigMF ci8)
"""
from __future__ import annotations

import math

import numpy as np

CHUNK = 1 << 20          # input samples per chunk (8 MB as complex64)


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


class Ci8FileSource(object):
    def __init__(self, path, rate):
        self.path = path
        self.rate = float(rate)
        import os
        self.n = os.path.getsize(path) // 2

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


# ----------------------------------------------------------------------------------------------
# pass 1: mean + Welch PSD
# ----------------------------------------------------------------------------------------------
def pick_nfft(rate):
    # ~250 Hz bins at 2 Msps, ~600 Hz at 10 Msps
    return 8192 if rate <= 4.5e6 else 16384


class PsdPass(object):
    """Streaming Welch PSD (Hann, 50 % overlap) after removing a fixed DC estimate.

    psd units: power/Hz with sum(psd)*df == mean|x|^2."""

    def __init__(self, rate, nfft, dc):
        self.rate, self.nfft, self.dc = rate, nfft, np.complex64(dc)
        self.hop = nfft // 2
        self.w = np.hanning(nfft).astype(np.float32)
        self.acc = np.zeros(nfft, np.float64)
        self.nseg = 0
        self.tail = np.zeros(0, np.complex64)
        self.norm = rate * float((self.w.astype(np.float64) ** 2).sum())

    def feed(self, x):
        x = np.asarray(x, np.complex64) - self.dc
        if len(self.tail):
            x = np.concatenate([self.tail, x])
        nfft, hop = self.nfft, self.hop
        nseg = (len(x) - nfft) // hop + 1 if len(x) >= nfft else 0
        if nseg <= 0:
            self.tail = x
            return
        # strided view of overlapping segments, processed in batches
        step = max(1, (1 << 21) // nfft)
        for s in range(0, nseg, step):
            e = min(nseg, s + step)
            idx = (np.arange(s, e)[:, None] * hop) + np.arange(nfft)[None, :]
            seg = x[idx] * self.w
            F = np.fft.fft(seg, axis=1)
            self.acc += (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2).sum(0)
        self.nseg += nseg
        self.tail = x[(nseg - 1) * hop + hop:]

    def result(self):
        psd = self.acc / max(self.nseg, 1) / self.norm
        f = np.fft.fftfreq(self.nfft, 1.0 / self.rate)
        return np.fft.fftshift(f), np.fft.fftshift(psd), self.nseg


def estimate_dc(source, nchunks=2):
    """DC estimate from the first chunks (the HackRF DC spike is a stable offset)."""
    acc, n = 0j, 0
    for i, c in enumerate(source.chunks()):
        acc += complex(c.astype(np.complex128).sum())
        n += len(c)
        if i + 1 >= nchunks:
            break
    return acc / max(n, 1)


def scan_psd(source):
    dc = estimate_dc(source)
    nfft = pick_nfft(source.rate)
    p = PsdPass(source.rate, nfft, dc)
    for c in source.chunks():
        p.feed(c)
    f, psd, nseg = p.result()
    return dict(freqs=f, psd=psd, nseg=nseg, dc=dc, nfft=nfft, df=source.rate / nfft)


# ----------------------------------------------------------------------------------------------
# band detection
# ----------------------------------------------------------------------------------------------
def _box(v, w):
    if w <= 1:
        return v.copy()
    k = np.ones(w) / w
    pad = w // 2
    vp = np.concatenate([v[-pad:], v, v[:pad]]) if pad else v
    out = np.convolve(vp, k, mode='valid')
    return out[:len(v)]


def find_bands(scan, rate, search_hz=None, dc_guard_bins=2, min_excess=0.05, gap_hz=8e3, nsig=5.0):
    """Locate signal-like excess in the averaged PSD.

    search_hz   half-width of the region that is searched (default 0.45*rate); outside it the
                HackRF baseband-filter roll-off makes the floor unreliable.
    Returns dict(n0, sigma, bands=[...]) with bands sorted by excess power (strongest first).
    Each band: lo, hi, centroid, excess (power in LSB^2), peak_ratio."""
    f, psd, nseg, df = scan['freqs'], scan['psd'], scan['nseg'], scan['df']
    if search_hz is None:
        search_hz = 0.45 * rate
    sel = np.abs(f) <= search_hz
    # DC guard: the removed constant leaves only a few residual bins
    dc_bin = np.argmin(np.abs(f))
    guard = np.zeros(len(f), bool)
    guard[max(0, dc_bin - dc_guard_bins):dc_bin + dc_guard_bins + 1] = True
    ok = sel & ~guard
    n0 = float(np.median(psd[ok]))
    # a signal wider than half the search region lifts the median; use a low percentile instead if the
    # PSD is clearly not flat (white noise: median and 8th percentile agree within ~2 sigma)
    p8 = float(np.percentile(psd[ok], 8))
    sig_bin = 1.0 / math.sqrt(max(nseg * 0.5, 1.0))
    if n0 > p8 * (1.0 + 4.0 * sig_bin + 0.05):
        n0 = p8 / (1.0 - 1.4 * sig_bin)
    r = psd / n0
    w = 5
    rs = _box(r, w)
    # noise scatter of the smoothed ratio from the noise-only bins (MAD)
    vals = rs[ok]
    vals = vals[vals < 1.3]            # noise bins only (a wide signal must not inflate the scatter)
    mad = float(np.median(np.abs(vals - np.median(vals)))) * 1.4826 if len(vals) > 20 else 0.05
    sigma = max(mad, 1e-4)
    thr = 1.0 + max(nsig * sigma, min_excess)
    m = ok & (rs > thr)
    # groups with gap merging
    idx = np.flatnonzero(m)
    bands = []
    if len(idx):
        gap_bins = max(4, int(gap_hz / df))
        starts = [idx[0]]
        ends = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > gap_bins:
                ends.append(a)
                starts.append(b)
        ends.append(idx[-1])
        # merge neighbouring groups separated by a gap that is small compared with their widths
        # (a wide FSK/GFSK/OFDM spectrum has notches that must not split it into 'several signals')
        groups = list(zip(starts, ends))
        merged = True
        while merged and len(groups) > 1:
            merged = False
            for i in range(len(groups) - 1):
                (s1, e1), (s2, e2) = groups[i], groups[i + 1]
                gap = (s2 - e1) * df
                if gap < max(gap_hz, 0.15 * ((e1 - s1) + (e2 - s2)) * df):
                    groups[i:i + 2] = [(s1, e2)]
                    merged = True
                    break
        starts = [g[0] for g in groups]
        ends = [g[1] for g in groups]
        rs2 = _box(r, 13)
        thr2 = 1.0 + max(3.0 * sigma * math.sqrt(w / 13.0), 0.5 * min_excess)
        for s, e in zip(starts, ends):
            # extend each edge while the heavily smoothed ratio is still above the floor
            while s > 0 and ok[s - 1] and rs2[s - 1] > thr2:
                s -= 1
            while e < len(f) - 1 and ok[e + 1] and rs2[e + 1] > thr2:
                e += 1
            seg = slice(s, e + 1)
            exc = np.maximum(r[seg] - 1.0, 0.0)
            pw = float(exc.sum() * n0 * df)
            cen = float((f[seg] * exc).sum() / max(exc.sum(), 1e-30))
            bands.append(dict(lo=float(f[s] - df / 2), hi=float(f[e] + df / 2), centroid=cen,
                              excess=pw, peak_ratio=float(r[seg].max()),
                              mean_ratio=float(r[seg].mean())))
        bands.sort(key=lambda b: -b['excess'])
    return dict(n0=n0, sigma=sigma, bands=bands, df=df, thr=thr)


# ----------------------------------------------------------------------------------------------
# pass 2: FFT channeliser (mix + brick-wall-with-taper filter + decimate), streaming
# ----------------------------------------------------------------------------------------------
class Channelizer(object):
    """Overlap-save FFT channeliser.

    For every block of N = D*M input samples: FFT, cut the M bins around the centre bin k0,
    multiply by a raised-cosine passband W, IFFT (size M) -> M decimated baseband samples.
    Output sample rate = rate/D, amplitude preserved (a tone of amplitude A in the passband comes
    out with amplitude A).  The signal centre is k0*rate/N (resolution rate/N, error < rate/2N)."""

    def __init__(self, rate, fc, bw, D, M=1 << 14, trans_frac=0.25, dc=0.0):
        self.rate, self.D, self.M = float(rate), int(D), int(M)
        self.dc = np.complex64(dc)
        self.N = self.D * self.M
        self.k0 = int(round(fc * self.N / rate))
        self.fc = self.k0 * rate / self.N
        self.fs = rate / self.D
        self.G = self.N // 8                # guard samples dropped at each block end
        self.H = self.N - 2 * self.G        # hop
        j = np.arange(-self.M // 2, self.M // 2)
        fj = j * rate / self.N
        half = bw / 2.0
        tr = max(trans_frac * bw, 4 * rate / self.N)
        W = np.ones(len(j))
        edge = np.abs(fj) - half
        up = edge > 0
        W[up] = np.where(edge[up] < tr, 0.5 * (1 + np.cos(np.pi * edge[up] / tr)), 0.0)
        self.W = W
        self.bin_idx = (self.k0 + j) % self.N
        self.shift_idx = j % self.M
        self.neb = float((W ** 2).sum() * rate / self.N)      # noise equivalent bandwidth [Hz]
        self.bw = bw
        self.buf = np.zeros(self.G, np.complex64)     # zero history so the first block has no wrap-around
        self.pos = -self.G  # index (in input samples) of buf[0]
        self.out = []
        self.nout = 0

    def _block(self, x, s):
        """Process x (length N) that starts at absolute input sample s."""
        X = np.fft.fft(x)
        Z = np.zeros(self.M, np.complex64 if X.dtype == np.complex64 else np.complex128)
        Z[self.shift_idx] = X[self.bin_idx] * self.W
        z = np.fft.ifft(Z) / self.D
        ph = np.exp(-2j * math.pi * (self.k0 * (s % self.N)) / self.N)
        return (z * ph).astype(np.complex64)

    def feed(self, x, store=True):
        """Consume input samples; returns the list of newly produced decimated blocks."""
        x = np.asarray(x, np.complex64) - self.dc
        self.buf = np.concatenate([self.buf, x])
        N, H, G, D = self.N, self.H, self.G, self.D
        new = []
        while len(self.buf) >= N:
            blk = self._block(self.buf[:N], self.pos)
            new.append(blk[G // D:(N - G) // D])
            self.buf = self.buf[H:]
            self.pos += H
        if store:
            self.out.extend(new)
            self.nout += sum(len(b) for b in new)
        return new

    def finish(self, store=True):
        """Flush the tail by zero padding to a full block."""
        N, G, D = self.N, self.G, self.D
        new = []
        if len(self.buf) > G:
            n_valid = len(self.buf)
            x = np.concatenate([self.buf, np.zeros(N - n_valid, np.complex64)])
            blk = self._block(x, self.pos)
            hi = int(math.ceil(n_valid / D))
            new.append(blk[G // D:hi])
        self.buf = np.zeros(0, np.complex64)
        if store:
            self.out.extend(new)
            self.nout += sum(len(b) for b in new)
        return new

    def record(self):
        if not self.out:
            return np.zeros(0, np.complex64)
        return np.concatenate(self.out)


def choose_decimation(rate, bw_ch, lo=3.4):
    """Largest D (with small prime factors) such that fs_out = rate/D >= lo*bw_ch."""
    D_max = max(1, int(rate / (lo * bw_ch)))
    cands = []
    for a in range(0, 16):
        for b in range(0, 6):
            for c in range(0, 4):
                d = (2 ** a) * (3 ** b) * (5 ** c)
                if d <= D_max:
                    cands.append(d)
    return max(cands) if cands else 1


def _block_size(D):
    M = 1 << 14 if D * (1 << 14) <= (1 << 21) else max(256, (1 << 21) // D)
    return 1 << int(math.log2(M))


def channelize(source, fc, bw, D, dc=0.0, max_out=1 << 25):
    """Plain pass 2 (keeps everything) -> (record, fs, info).  Used by tests."""
    ch = Channelizer(source.rate, fc, bw, D, M=_block_size(D), dc=dc)
    for c in source.chunks():
        ch.feed(c)
        if ch.nout > max_out:
            break
    else:
        ch.finish()
    rec = ch.record()[:max_out]
    return rec, ch.fs, dict(fc=ch.fc, neb=ch.neb, D=D, truncated=(ch.nout > max_out))


def channelize_select(source, fc, bw, D, n0, dc=0.0, max_keep=1 << 21, margin_blocks=1, min_excess=0.03, nsig=5.0):
    """Streaming pass 2 with block-level activity selection.

    Every decimated block (about 0.75*M samples) whose mean power exceeds the noise power
    n0*NEB*(1 + max(nsig/sqrt(K), min_excess)) is kept together with `margin_blocks` quiet blocks on each
    side; the pass stops as soon as max_keep samples are kept (continuous signals: after a fraction of
    the file).  Returns (record, fs, info) ; info['blocks_seen'/'blocks_active'] give the block duty."""
    ch = Channelizer(source.rate, fc, bw, D, M=_block_size(D), dc=dc)
    n_ch = n0 * ch.neb
    st = dict(kept=[], nkept=0, prev=None, tail=0, seen=0, act=0)

    def handle(blk):
        st['seen'] += 1
        K = max(len(blk) * ch.neb / ch.fs, 1.0)
        p = float((blk.real.astype(np.float32) ** 2 + blk.imag.astype(np.float32) ** 2).mean())
        if p > n_ch * (1.0 + max(nsig / math.sqrt(K), min_excess)):
            st['act'] += 1
            if st['prev'] is not None:
                st['kept'].append(st['prev'])
                st['nkept'] += len(st['prev'])
                st['prev'] = None
            st['kept'].append(blk)
            st['nkept'] += len(blk)
            st['tail'] = margin_blocks
        elif st['tail'] > 0:
            st['kept'].append(blk)
            st['nkept'] += len(blk)
            st['tail'] -= 1
        else:
            st['prev'] = blk
        return st['nkept'] >= max_keep

    stopped = False
    for c in source.chunks():
        for blk in ch.feed(c, store=False):
            if handle(blk):
                stopped = True
                break
        if stopped:
            break
    if not stopped:
        for blk in ch.finish(store=False):
            handle(blk)
    rec = np.concatenate(st['kept']) if st['kept'] else np.zeros(0, np.complex64)
    return rec, ch.fs, dict(fc=ch.fc, neb=ch.neb, D=D, truncated=stopped, blocks_seen=st['seen'],
                            blocks_active=st['act'], n_ch=n_ch)
