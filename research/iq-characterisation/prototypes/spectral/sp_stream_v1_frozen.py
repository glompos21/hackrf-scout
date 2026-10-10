"""Streaming passes for the spectral-shape family (numpy only, Python >= 3.9).

Everything here consumes an iterator of complex64 chunks (a ci8 file read in pieces, or slices of an
in-memory array), so a 100 MB capture never has to be held as float.

  pass 1  mean / power            -> DC offset (complex mean of the whole capture)
  pass 2  fine Welch (Hann, 50 % overlap) + coarse spectrogram ("TF") in the SAME loop
"""
from __future__ import annotations

import numpy as np

CHUNK = 1 << 20                      # samples per chunk (2**20 complex = 8 MB as complex64)


# ----------------------------------------------------------------------------------------------
# chunk sources
# ----------------------------------------------------------------------------------------------
def iter_array(iq, chunk=CHUNK):
    for a in range(0, len(iq), chunk):
        yield iq[a:a + chunk]


def iter_ci8_file(path, chunk=CHUNK):
    """hackrf_transfer / SigMF ci8 file -> complex64 chunks (I + jQ, signed 8 bit)."""
    with open(path, 'rb') as f:
        while True:
            b = np.fromfile(f, dtype=np.int8, count=2 * chunk)
            if b.size < 2:
                return
            b = b[: (b.size // 2) * 2].reshape(-1, 2).astype(np.float32)
            yield b.view(np.complex64).ravel()


def pass1_dc(chunks):
    """Complex mean and mean power of the whole capture, float64 accumulation."""
    s = 0j
    p = 0.0
    n = 0
    for c in chunks:
        s += complex(c.sum(dtype=np.complex128))
        p += float(np.einsum('i,i->', c.real, c.real, dtype=np.float64)
                   + np.einsum('i,i->', c.imag, c.imag, dtype=np.float64))
        n += len(c)
    return s / n, p / n, n


# ----------------------------------------------------------------------------------------------
# pass 2 accumulators
# ----------------------------------------------------------------------------------------------
class Welch:
    """Hann-window Welch PSD in power/Hz (sum(psd)*df == mean |x|^2), 50 % overlap, streaming."""

    def __init__(self, nfft, rate, overlap=0.5):
        self.nfft = int(nfft)
        self.rate = float(rate)
        self.hop = max(1, int(round(self.nfft * (1.0 - overlap))))
        self.win = np.hanning(self.nfft).astype(np.float32)
        self.wss = float((self.win.astype(np.float64) ** 2).sum())
        self.acc = np.zeros(self.nfft, np.float64)
        self.k = 0
        self.carry = np.zeros(0, np.complex64)
        self.overlap = overlap

    def feed(self, x):
        buf = np.concatenate((self.carry, x)) if len(self.carry) else x
        n = self.nfft
        if len(buf) < n:
            self.carry = np.array(buf, copy=True)
            return
        m = (len(buf) - n) // self.hop + 1
        segs = np.lib.stride_tricks.sliding_window_view(buf, n)[::self.hop][:m]
        # process in blocks so temporaries stay small whatever nfft is
        blk = max(1, (1 << 22) // n)
        for a in range(0, m, blk):
            s = segs[a:a + blk] * self.win
            F = np.fft.fft(s, axis=1)
            self.acc += (F.real * F.real + F.imag * F.imag).sum(0, dtype=np.float64)
        self.k += m
        self.carry = np.array(buf[m * self.hop:], copy=True)

    def result(self):
        psd = self.acc / (max(self.k, 1) * self.rate * self.wss)
        f = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / self.rate))
        # effective number of independent averages (Hann, 50 % overlap: x0.947; no overlap: x1)
        k_eff = self.k / 1.056 if self.overlap >= 0.49 else float(self.k)
        return f, np.fft.fftshift(psd), self.k, k_eff


class TF:
    """Coarse spectrogram: one Hann FFT of `frame` samples per frame (no overlap), power/Hz averaged
    into `ngroups` equal frequency groups.  Output S[t, g] (float32), fftshifted order."""

    def __init__(self, frame, ngroups, rate):
        assert frame % ngroups == 0
        self.frame = int(frame)
        self.ng = int(ngroups)
        self.rate = float(rate)
        self.win = np.hanning(self.frame).astype(np.float32)
        self.wss = float((self.win.astype(np.float64) ** 2).sum())
        self.out = []
        self.carry = np.zeros(0, np.complex64)

    def feed(self, x):
        buf = np.concatenate((self.carry, x)) if len(self.carry) else x
        fr = self.frame
        m = len(buf) // fr
        if m == 0:
            self.carry = np.array(buf, copy=True)
            return
        blk = max(1, (1 << 21) // fr)
        for a in range(0, m, blk):
            b = min(m, a + blk)
            seg = buf[a * fr:b * fr].reshape(b - a, fr) * self.win
            F = np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)
            p = F.real * F.real + F.imag * F.imag
            g = p.reshape(b - a, self.ng, fr // self.ng).mean(2)
            self.out.append((g / (self.rate * self.wss)).astype(np.float32))
        self.carry = np.array(buf[m * fr:], copy=True)

    def result(self):
        S = np.concatenate(self.out, axis=0) if self.out else np.zeros((0, self.ng), np.float32)
        df = self.rate / self.frame
        bpg = self.frame // self.ng
        j = np.arange(self.ng)
        fg = ((j + 0.5) * bpg - 0.5 - self.frame / 2.0) * df
        return S, fg, self.frame / self.rate


def default_cfg(rate):
    """Analysis resolution as a function of the sample rate only (no knowledge of the signal)."""
    nfft = 1 << 15                                         # 61 Hz @2 Msps, 305 Hz @10 Msps, 610 Hz @20 Msps
    frame = 1 << int(round(np.log2(max(128.0, rate * 100e-6))))   # ~100 us frames (256 @2M, 1024 @10M)
    ngroups = 64
    return dict(nfft=nfft, frame=frame, ngroups=ngroups)


def analyse_stream(make_chunks, rate, remove_dc=True, nfft=None, frame=None, ngroups=None, want_tf=True, stride=1):
    """Two passes over `make_chunks()` (a callable returning a fresh chunk iterator).

    Returns the dict of raw spectra that `sp_feat.features` consumes."""
    cfg = default_cfg(rate)
    nfft = nfft or cfg['nfft']
    frame = frame or cfg['frame']
    ngroups = ngroups or cfg['ngroups']
    dc, p_raw, n = pass1_dc(make_chunks())
    sub = np.complex64(dc) if remove_dc else np.complex64(0)
    w = Welch(nfft, rate)
    tf = TF(frame, ngroups, rate) if want_tf else None
    for i, c in enumerate(make_chunks()):
        if i % stride:                       # sub-sampled in time: skip chunk, never join non-adjacent samples
            w.carry = w.carry[:0]
            if tf is not None:
                tf.carry = tf.carry[:0]
            continue
        c = c - sub if remove_dc else c
        w.feed(c)
        if tf is not None:
            tf.feed(c)
    f, P, k, k_eff = w.result()
    out = dict(rate=float(rate), n=int(n), dc=complex(dc), p_raw=p_raw, remove_dc=bool(remove_dc),
               nfft=int(nfft), f=f, P=P, k=int(k), k_eff=float(k_eff), win=w.win)
    if tf is not None:
        S, fg, dt = tf.result()
        out.update(S=S, tf_f=fg, tf_dt=dt, tf_frame=int(frame), tf_ng=int(ngroups))
    return out
