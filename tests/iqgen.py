"""Synthetic IQ for the watch-mode tests: noise plus FSK bursts at known frequencies and times."""

import numpy as np


def _tri(t, half_period):
    """Integral of a +-1 square wave with the given half period (a triangle wave)."""
    tau = np.mod(t, 2 * half_period)
    return half_period - np.abs(tau - half_period)


def chunks(rate, seconds, slice_samples, bursts=(), noise=6.0, dc=0.0, seed=1):
    """Yield `slice_samples` complex samples at a time as signed 8-bit interleaved I/Q bytes.

    bursts: (offset_hz from the tune centre, start_s, duration_s, amplitude, deviation_hz) 2-FSK at 10 kbaud.
    """
    rng = np.random.default_rng(seed)
    ramp = 0.0005
    for k in range(int(rate * seconds) // slice_samples):
        t = (k * slice_samples + np.arange(slice_samples)) / rate
        x = rng.normal(0, noise, slice_samples) + 1j * rng.normal(0, noise, slice_samples) + dc * (1 + 1j)
        for offset, start, duration, amp, dev in bursts:
            on = (t >= start) & (t < start + duration)
            if on.any():
                tt = t[on]
                env = np.minimum(1.0, np.minimum(tt - start, start + duration - tt) / ramp)
                x[on] += amp * env * np.exp(2j * np.pi * (offset * tt + dev * _tri(tt, 1.0 / 10e3)))
        out = np.empty(2 * slice_samples, dtype=np.int8)
        out[0::2] = np.clip(np.rint(x.real), -128, 127)
        out[1::2] = np.clip(np.rint(x.imag), -128, 127)
        yield out.tobytes()
