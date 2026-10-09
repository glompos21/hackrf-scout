"""Generate fake `hackrf_sweep` CSV output for testing without hardware.

Mimics the real interleaved chunk order (so sweep-boundary detection is
exercised) with a noise floor and a configurable set of synthetic signals.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# (centre Hz, bandwidth Hz, level dB, duty cycle 0..1)
DEFAULT_SIGNALS: List[Tuple[float, float, float, float]] = [
    (98.1e6, 180e3, -35.0, 1.0),  # FM broadcast
    (101.9e6, 180e3, -42.0, 1.0),  # FM broadcast (weaker)
    (162.025e6, 25e3, -50.0, 0.45),  # AIS, bursty
    (433.92e6, 40e3, -48.0, 0.6),  # ISM 433 sensor
    (868.3e6, 125e3, -52.0, 0.4),  # LoRa-ish, bursty
    (947.4e6, 200e3, -40.0, 1.0),  # GSM900 downlink
    (1090.0e6, 1.0e6, -50.0, 0.7),  # ADS-B region
    (2437e6, 18e6, -45.0, 0.9),  # Wi-Fi channel 6
]
NOISE_DB = -72.0
NOISE_SIGMA = 2.0


def _atten_db(freqs: np.ndarray, centre: float, bw: float, bin_hz: float = 100e3) -> np.ndarray:
    """Attenuation in dB: 0 inside the signal bandwidth, then ~20 dB per quarter bandwidth.

    Signals narrower than one FFT bin still light up the nearest bin (the receiver
    integrates all energy that falls inside a bin).
    """
    bw = max(bw, bin_hz)
    d = np.abs(freqs - centre) - bw / 2.0
    return np.where(d <= 0, 0.0, 80.0 * d / bw)


def generate(
    start_mhz: float = 1.0,
    stop_mhz: float = 3000.0,
    bin_hz: float = 100e3,
    sweeps: int = 10,
    signals: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    seed: int = 1,
    period_s: float = 1.0,
    t0: Optional[datetime] = None,
) -> Iterator[str]:
    sigs = list(DEFAULT_SIGNALS if signals is None else signals)
    rng = np.random.default_rng(seed)
    t0 = t0 or datetime.now().replace(microsecond=0)
    chunk_hz = 5e6
    bins_per_chunk = int(round(chunk_hz / bin_hz))
    lo0 = int(start_mhz * 1e6)
    n_tunings = int(np.ceil((stop_mhz * 1e6 - lo0) / 20e6))
    total_bins = n_tunings * 4 * bins_per_chunk
    freqs = lo0 + (np.arange(total_bins) + 0.5) * bin_hz
    for s in range(sweeps):
        ts = t0 + timedelta(seconds=s * period_s)
        p_lin = 10 ** ((NOISE_DB + rng.normal(0, NOISE_SIGMA, total_bins)) / 10.0)
        for centre, bw, level, duty in sigs:
            if rng.random() >= duty:
                continue
            att = _atten_db(freqs, centre, bw, bin_hz)
            active = att < 45.0
            if not active.any():
                continue
            lvl = level + rng.normal(0, 1.2, int(active.sum())) - att[active]
            p_lin[active] += 10 ** (lvl / 10.0)
        p_db = 10 * np.log10(p_lin)
        date, tm = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M:%S.%f")
        for k in range(n_tunings):
            base = lo0 + k * 20_000_000
            # real hackrf_sweep order within one tuning pair: A_low, A_high, B_low, B_high
            for off in (0, 10_000_000, 5_000_000, 15_000_000):
                lo = base + off
                i0 = int((lo - lo0) / bin_hz)
                vals = p_db[i0 : i0 + bins_per_chunk]
                if vals.size == 0:
                    continue
                yield (
                    f"{date}, {tm}, {lo}, {lo + int(chunk_hz)}, {bin_hz:.2f}, 8192, "
                    + ", ".join(f"{v:.2f}" for v in vals)
                    + "\n"
                )


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Print simulated hackrf_sweep CSV")
    ap.add_argument("-f", default="1:3000", help="start:stop in MHz")
    ap.add_argument("-w", type=float, default=100000, help="bin width Hz")
    ap.add_argument("-N", type=int, default=10, help="number of sweeps")
    ap.add_argument("--seed", type=int, default=1)
    ns, _ = ap.parse_known_args(argv)
    a, b = (float(x) for x in ns.f.split(":"))
    for line in generate(a, b, ns.w, ns.N, seed=ns.seed):
        sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
