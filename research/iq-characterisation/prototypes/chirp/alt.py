"""The two textbook alternatives to dechirp / lag-line, for comparison (instantaneous-frequency slope test and
spectrogram ridge test).  Both work on y at the critical rate BW for a hypothesised spreading factor.

IF-SLOPE   second difference of the phase of y: d2[n] = arg( z[n+1] conj(z[n]) ), z[n] = y[n+1] conj(y[n]).
           For a chirp at the critical rate d2 = 2*pi/M for EVERY n (also across the frequency wrap), for noise
           it is uniform.  Statistic: |mean(exp(j d2))| over windows of W samples (resultant length).
RIDGE      STFT (32 samples, hop M/16): per frame argmax frequency bin; a chirp moves the ridge by +2 bins per
           frame.  Statistic: number of frames in a window of 16 whose ridge step equals +2 (+-0 bins).
"""
from __future__ import annotations

import numpy as np


def if_slope_stat(y, sf, W=None):
    M = 1 << sf
    W = W or max(256, 2 * M)
    z = y[1:] * np.conj(y[:-1])
    u = z[1:] * np.conj(z[:-1])
    e = u / (np.abs(u) + 1e-30)
    n = len(e) // W
    if n < 1:
        return None
    e = e[:n * W].reshape(n, W)
    # keep only the expected angle 2*pi/M (+ tolerance): coherent sum with the matched phasor
    m = np.exp(-2j * np.pi / M * 1.0)
    s_match = np.abs((e * m).sum(1)) / W
    s_any = np.abs(e.sum(1)) / W
    return dict(matched=s_match, any=s_any, W=W)


def ridge_stat(y, sf, nper=32, frames=16):
    M = 1 << sf
    hop = max(1, M // 16)
    n = (len(y) - nper) // hop + 1
    if n < frames + 1:
        return None
    # frames x nper via sliding windows
    win = np.lib.stride_tricks.sliding_window_view(y, nper)[::hop][:n]
    w = np.hanning(nper).astype(np.float32)
    F = np.abs(np.fft.fft(win * w[None, :], axis=1)) ** 2
    k = F.argmax(1)
    dk = (np.diff(k) + nper // 2) % nper - nper // 2          # wrapped ridge step
    hit = (dk == 2)
    nw = len(hit) // frames
    if nw < 1:
        return None
    h = hit[:nw * frames].reshape(nw, frames).sum(1)
    h2 = np.maximum(h, (dk[:nw * frames] == -2).reshape(nw, frames).sum(1))
    return dict(count=h2, frames=frames)
