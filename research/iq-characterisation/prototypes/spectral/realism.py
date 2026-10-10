"""Receiver-front-end stress layer, applied to a generated capture (NOT part of the bench; my own model).

The bench noise is white over the whole sampled band and the DC offset is a constant.  A real HackRF has
  * a baseband low-pass (hackrf_transfer -b) that rolls off towards the band edge,
  * a DC spike at the tune centre that wanders a little,
  * I/Q gain/phase imbalance (image of every strong signal at -f).
This module adds crude versions of those three so the "floor is flat, edges are clean" assumption can be
broken on purpose.  Parameters are assumptions, not measurements of any particular HackRF.
"""
from __future__ import annotations

import numpy as np


def _fir_from_mag(mag_fn, rate, ntaps=255):
    f = np.fft.fftfreq(4096, 1.0 / rate)
    H = mag_fn(np.abs(f))
    h = np.fft.fftshift(np.fft.ifft(H).real)
    c = len(h) // 2
    h = h[c - ntaps // 2:c + ntaps // 2 + 1] * np.hamming(ntaps)
    return (h / h.sum()).astype(np.float32)


def _fir_filter(x, h):
    """FFT overlap-save convolution, group delay removed (same length as x)."""
    n, L = len(x), len(h)
    d = (L - 1) // 2
    nf = 1 << 19
    Bv = nf - L + 1
    Hf = np.fft.fft(h, nf)
    xp = np.concatenate((np.zeros(L - 1, np.complex64), x, np.zeros(nf, np.complex64)))
    c = np.empty(n + d, np.complex64)
    for s in range(0, n + d, Bv):
        y = np.fft.ifft(np.fft.fft(xp[s:s + nf], nf) * Hf)[L - 1:L - 1 + Bv]
        e = min(s + Bv, n + d)
        c[s:e] = y[:e - s]
    return c[d:d + n]


def apply(iq, rate, seed=0, filter_hz=None, order=4, dc_wander_lsb=0.0, dc_corner_hz=500.0,
          iq_gain_db=0.0, iq_phase_deg=0.0, requant=True):
    x = np.asarray(iq, np.complex64)
    rng = np.random.default_rng([int(seed), 77])
    if iq_gain_db or iq_phase_deg:
        g = 10 ** (iq_gain_db / 20.0)
        ph = np.deg2rad(iq_phase_deg)
        I, Q = x.real.copy(), x.imag.copy()
        Qn = g * (Q * np.cos(ph) + I * np.sin(ph))
        x = (I + 1j * Qn).astype(np.complex64)
    if filter_hz:
        fc = filter_hz / 2.0
        h = _fir_from_mag(lambda f: 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order)), rate)
        x = _fir_filter(x, h)
    if dc_wander_lsb > 0:
        n = len(x)
        w = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
        f = np.fft.fftfreq(n, 1.0 / rate)
        w *= 1.0 / np.sqrt(1.0 + (f / dc_corner_hz) ** 2)
        d = np.fft.ifft(w)
        d *= dc_wander_lsb / np.sqrt(np.mean(np.abs(d) ** 2) / 2.0)
        x = (x + d.astype(np.complex64)).astype(np.complex64)
    if requant:
        x = (np.clip(np.rint(x.real), -128, 127) + 1j * np.clip(np.rint(x.imag), -128, 127)).astype(np.complex64)
    return x
