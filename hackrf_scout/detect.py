"""Find signals in one assembled sweep.

Method
------
1. Estimate a local noise floor: the 30th percentile of every ~20 MHz block,
   linearly interpolated between block centres (robust against busy bands).
2. Mark bins more than `snr_db` above that floor.
3. Merge neighbouring marked bins into regions (allowing small gaps) and split
   regions where two peaks are separated by a deep valley (`valley_db`).
4. For each region report centre, occupied bandwidth, peak power and SNR.

`FloorTracker` smooths that floor over successive sweeps and notices sudden jumps,
which usually mean a strong transmitter is overloading the receiver.

Powers are the relative dB values printed by hackrf_sweep (not calibrated dBm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    center_hz: float
    bw_hz: float
    peak_hz: float
    peak_db: float
    mean_db: float
    snr_db: float
    floor_db: float
    lo_hz: float
    hi_hz: float


def _block_stat(power, mask, per_block, fn, min_count):
    """Apply fn to (optionally masked) values of consecutive blocks; return centres and values."""
    n = power.size
    nblocks = max(1, n // per_block)
    edges = np.linspace(0, n, nblocks + 1).astype(int)
    xs, vs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = power[a:b] if mask is None else power[a:b][mask[a:b]]
        if v.size >= min_count:
            xs.append((a + b - 1) / 2.0)
            vs.append(fn(v))
    return np.array(xs), np.array(vs)


def estimate_floor(power: np.ndarray, bin_hz: float, coarse_block_hz: float = 50e6, fine_block_hz: float = 20e6) -> np.ndarray:
    """Local noise-floor estimate that survives wide signals and crowded bands.

    Stage 1 (coarse): 10th percentile of every ~50 MHz block, then the minimum over
    +-4 neighbouring blocks, so even a 200+ MHz fully occupied band (e.g. DVB-T)
    cannot lift the floor. Stage 2 (fine): median of the bins that look like noise
    (within 6 dB of the coarse floor) per ~20 MHz block, giving a tight local floor.
    """
    n = power.size
    xs, vs = _block_stat(power, None, max(40, int(coarse_block_hz / bin_hz)), lambda v: np.percentile(v, 10), 5)
    if xs.size == 0:
        return np.full(n, float(np.percentile(power, 10)))
    if vs.size > 1:
        vs = np.array([vs[max(0, i - 4) : i + 5].min() for i in range(vs.size)])
    coarse = np.interp(np.arange(n), xs, vs)
    mask = power < coarse + 6.0
    fx, fv = _block_stat(power, mask, max(20, int(fine_block_hz / bin_hz)), np.median, 8)
    if fx.size == 0:
        return coarse
    return np.interp(np.arange(n), fx, fv)


class FloorTracker:
    """Smooths the per-bin noise floor over sweeps and flags sudden jumps.

    One sweep's floor estimate wobbles by a dB or so, which moves signals back and forth across the
    SNR threshold. An exponential moving average (in dB, per frequency) steadies it: each new sweep
    counts for `alpha` (1 = no smoothing).

    A jump of more than `jump_db` in the median floor is a different matter. Brief ones usually mean
    a strong transmitter is overloading the receiver (the whole floor rises and ghost signals appear),
    so `update` reports them as suspect and keeps the old floor. If the new level holds for
    `max_suspect` sweeps in a row it is real (new antenna, new gain) and becomes the baseline.
    """

    def __init__(self, alpha: float = 0.2, jump_db: float = 6.0, max_suspect: int = 3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.jump_db = jump_db
        self.max_suspect = max(1, max_suspect)
        self._freqs: Optional[np.ndarray] = None
        self._floor: Optional[np.ndarray] = None
        self._suspect = 0
        self.accepted_jump: Optional[float] = None  # set on the sweep where a persistent jump became the new baseline

    def _previous_on(self, freqs: np.ndarray) -> Optional[np.ndarray]:
        """The smoothed floor re-sampled onto this sweep's frequencies (None if the ranges barely overlap)."""
        if self._freqs is None or self._floor is None:
            return None
        if self._freqs.size == freqs.size and np.allclose(self._freqs, freqs, rtol=0.0, atol=1.0):
            return self._floor
        lo, hi = float(freqs[0]), float(freqs[-1])
        overlap = min(hi, self._freqs[-1]) - max(lo, self._freqs[0])
        if overlap < 0.5 * (hi - lo):
            return None
        return np.interp(freqs, self._freqs, self._floor)

    def update(self, freqs: np.ndarray, raw: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Returns (floor to use, jump_db). `jump_db` is None normally, or the size of a suspect jump,
        in which case the caller should skip this sweep."""
        self.accepted_jump = None
        prev = self._previous_on(freqs)
        if prev is None:
            self._freqs, self._floor, self._suspect = freqs.copy(), raw.copy(), 0
            return raw, None
        jump = float(np.median(raw - prev))
        if self.jump_db > 0 and abs(jump) > self.jump_db:
            self._suspect += 1
            if self._suspect < self.max_suspect:
                return prev, jump
            self._freqs, self._floor, self._suspect = freqs.copy(), raw.copy(), 0  # it stayed: take it as the new normal
            self.accepted_jump = jump
            return raw, None
        self._suspect = 0
        smoothed = self.alpha * raw + (1.0 - self.alpha) * prev
        self._freqs, self._floor = freqs.copy(), smoothed
        return smoothed, None


def _split_on_valleys(p: np.ndarray, valley_db: float) -> List[int]:
    """Return indices (relative to p) of valley minima where a region should be cut."""
    cuts: List[int] = []
    hi = p[0]
    lo = p[0]
    lo_i = 0
    going_up = True
    for i, v in enumerate(p):
        if going_up:
            if v > hi:
                hi = v
            elif hi - v >= valley_db:
                going_up = False
                lo, lo_i = v, i
        else:
            if v < lo:
                lo, lo_i = v, i
            elif v - lo >= valley_db:
                cuts.append(lo_i)
                going_up = True
                hi = v
    return cuts


def detect_signals(
    freqs: np.ndarray,
    power: np.ndarray,
    bin_hz: float,
    snr_db: float = 10.0,
    merge_gap_bins: int = 1,
    valley_db: float = 10.0,
    ignore: Sequence[Tuple[float, float]] = (),
    floor: Optional[np.ndarray] = None,
) -> Tuple[List[Detection], float]:
    """Detect signals in a sweep. Returns (detections, median_floor_db).

    `floor` is the per-bin noise floor; it is estimated from this sweep when not given."""
    if power.size == 0:
        return [], float("nan")
    if floor is None:
        floor = estimate_floor(power, bin_hz)
    work = power.copy()
    for lo, hi in ignore:
        work[(freqs >= lo) & (freqs <= hi)] = floor[(freqs >= lo) & (freqs <= hi)]

    marked = work > (floor + snr_db)
    idx = np.flatnonzero(marked)
    if idx.size == 0:
        return [], float(np.median(floor))

    # contiguous runs, tolerating `merge_gap_bins` unmarked bins and frequency holes
    gaps = np.flatnonzero((np.diff(idx) > merge_gap_bins + 1) | (np.diff(freqs[idx]) > (merge_gap_bins + 1.5) * bin_hz))
    run_starts = np.r_[idx[0], idx[gaps + 1]]
    run_ends = np.r_[idx[gaps], idx[-1]]

    # second pass: a wide signal's skirt often breaks into fragments; re-join fragments whose
    # gap is small compared with their size (20 % of the larger piece)
    runs: List[Tuple[int, int]] = []
    for a, b in zip(run_starts.tolist(), run_ends.tolist()):
        if runs:
            pa, pb = runs[-1]
            allow = max(merge_gap_bins, int(0.2 * max(pb - pa + 1, b - a + 1)))
            if a - pb - 1 <= allow and freqs[a] - freqs[pb] <= (allow + 1.5) * bin_hz:
                runs[-1] = (pa, b)
                continue
        runs.append((a, b))

    dets: List[Detection] = []
    prev_end = -1
    for a, b in runs:
        seg = work[a : b + 1]
        if seg.size >= 12:
            # wide region: smooth over 3 bins so noise ripple is not mistaken for separate signals
            lin = np.pad(10.0 ** (seg / 10.0), (1, 1), mode="edge")
            smooth = 10.0 * np.log10(np.convolve(lin, np.ones(3) / 3.0, mode="valid"))
        else:
            smooth = seg
        raw_cuts = _split_on_valleys(smooth, valley_db)
        kept: List[int] = []
        last = 0
        for c in raw_cuts:  # peaks closer than 3 bins are not separable at this resolution
            if c - last >= 3 and (seg.size - 1 - c) >= 3:
                kept.append(c)
                last = c
        cuts = [a + c for c in kept]
        bounds = [a] + [c for c in cuts] + [b + 1]
        # sub-regions: [bounds[k], bounds[k+1]) ; valley bin belongs to the right side
        subs = [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1) if bounds[k + 1] - 1 >= bounds[k]]
        for si, (sa, sb) in enumerate(subs):
            region = work[sa : sb + 1]
            pk = int(np.argmax(region)) + sa
            peak_db = float(work[pk])
            fl = float(floor[pk])
            edge = max(fl + 3.0, peak_db - 20.0)
            # extend to the skirts (stop at neighbouring sub-regions / previous detection)
            left_lim = subs[si - 1][1] + 1 if si > 0 else max(0, prev_end + 1)
            right_lim = subs[si + 1][0] - 1 if si + 1 < len(subs) else work.size - 1
            ea, eb = sa, sb
            while ea - 1 >= left_lim and work[ea - 1] > edge:
                ea -= 1
            while eb + 1 <= right_lim and work[eb + 1] > edge:
                eb += 1
            sl = slice(ea, eb + 1)
            lin = 10.0 ** (work[sl] / 10.0)
            center = float(np.sum(freqs[sl] * lin) / np.sum(lin))
            dets.append(
                Detection(
                    center_hz=center,
                    bw_hz=float((eb - ea + 1) * bin_hz),
                    peak_hz=float(freqs[pk]),
                    peak_db=peak_db,
                    mean_db=float(10.0 * np.log10(np.mean(lin))),
                    snr_db=peak_db - fl,
                    floor_db=fl,
                    lo_hz=float(freqs[ea] - bin_hz / 2),
                    hi_hz=float(freqs[eb] + bin_hz / 2),
                )
            )
            prev_end = max(prev_end, eb)
    return dets, float(np.median(floor))
