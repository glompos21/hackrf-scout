"""Watch one band: stay tuned to it and log every burst with its timing.

`hackrf_sweep` hops across the spectrum and gets back to a frequency every 1-3 s, so a transmission that lasts
50 ms (a key fob, a weather sensor, a LoRa packet) is usually missed. Here the HackRF stays tuned to one
window of at most about 15 MHz and streams IQ from `hackrf_transfer`. The stream is cut into short slices
(20 ms by default); each slice becomes a power spectrum, the usual signal detector runs on it, and detections
that follow one another from slice to slice are joined into *bursts* with an onset time and a duration.

Times come from the sample clock (start time + samples seen / sample rate), so they are exact however fast or
slow the host processes the stream. Each burst is linked to a signal in the database (a known one near that
frequency, or a new one once it has been seen `min_bursts` times, identified like any sweep detection) and
stored in the `bursts` table.

The HackRF's DC spike sits at the tune centre. When the sample rate allows, the window is tuned so that the
spike falls just outside the band; otherwise the three bins at the centre are ignored and the log says where.
"""

from __future__ import annotations

import collections
import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .detect import Detection, FloorTracker, detect_signals, estimate_floor
from .identify import Identifier
from .store import Store

MIN_RATE = 2e6
MAX_RATE = 20e6
USABLE = 0.75  # share of the sample rate the HackRF's baseband filter passes cleanly
EDGE_GUARD_HZ = 50e3  # analysed beyond each band edge, so a signal on the edge is whole
DC_GUARD_HZ = 100e3  # distance kept between the DC spike and the band when the rate allows it
DC_AVOID_MAX_RATE = 8e6  # above this, accept the DC spike inside the band rather than burn the CPU
TARGET_BIN_HZ = 12.5e3


# ------------------------------------------------------------------------------------------ band plan
@dataclass(frozen=True)
class BandPlan:
    lo_hz: float
    hi_hz: float
    center_hz: float
    rate: float
    nfft: int
    dc_in_band: bool

    @property
    def bin_hz(self) -> float:
        return self.rate / self.nfft

    def describe(self) -> str:
        dc = (f"the DC spike is ignored ({3 * self.bin_hz / 1e3:.0f} kHz blind spot at {self.center_hz / 1e6:.3f} MHz)"
              if self.dc_in_band else "the DC spike is outside the band")
        return (f"watching {self.lo_hz / 1e6:.3f}-{self.hi_hz / 1e6:.3f} MHz: tuned to {self.center_hz / 1e6:.3f} MHz at "
                f"{self.rate / 1e6:g} Msps, {self.nfft}-point FFT ({self.bin_hz / 1e3:.1f} kHz bins); {dc}")


def _nfft_for(rate: float) -> int:
    return int(min(4096, max(64, 2 ** round(math.log2(rate / TARGET_BIN_HZ)))))


def _ceil_mhz(hz: float) -> float:
    return math.ceil(hz / 1e6 - 1e-9) * 1e6


def plan_band(lo_hz: float, hi_hz: float, rate: Optional[float] = None) -> BandPlan:
    """Choose sample rate, tune centre and FFT size for watching [lo_hz, hi_hz]."""
    if not lo_hz < hi_hz:
        raise ValueError("the band needs a start below its end")
    span = hi_hz - lo_hz
    widest = (MAX_RATE * USABLE - 2 * EDGE_GUARD_HZ) / 1e6
    too_wide = (f"{span / 1e6:.1f} MHz is too wide to watch at once: a HackRF can only see about {widest:.0f} MHz. "
                "Watch a narrower range (for example one sub-band), or use 'scan' for wide ones")
    half = USABLE * rate / 2 if rate else None
    if rate is None:
        if (span + 2 * EDGE_GUARD_HZ) / USABLE > MAX_RATE:
            raise ValueError(too_wide)
        keep_dc_out = max(MIN_RATE, _ceil_mhz((span + EDGE_GUARD_HZ + DC_GUARD_HZ) / (USABLE / 2)))
        if keep_dc_out <= DC_AVOID_MAX_RATE:
            rate, dc_in_band = keep_dc_out, False
        else:
            rate, dc_in_band = max(MIN_RATE, _ceil_mhz((span + 2 * EDGE_GUARD_HZ) / USABLE)), True
    else:
        if not MIN_RATE <= rate <= MAX_RATE:
            raise ValueError(f"the sample rate must be between {MIN_RATE / 1e6:g} and {MAX_RATE / 1e6:g} Msps")
        if rate * USABLE < span + 2 * EDGE_GUARD_HZ:
            raise ValueError(f"{rate / 1e6:g} Msps is too slow for a {span / 1e6:.2f} MHz band (it needs at least "
                             f"{_ceil_mhz((span + 2 * EDGE_GUARD_HZ) / USABLE) / 1e6:g} Msps)")
        dc_in_band = not (half >= span + EDGE_GUARD_HZ + DC_GUARD_HZ)
    center = (lo_hz + hi_hz) / 2 if dc_in_band else lo_hz - DC_GUARD_HZ
    return BandPlan(lo_hz, hi_hz, float(round(center)), float(rate), _nfft_for(rate), dc_in_band)


def plan_window(center_hz: float, rate: float, lo_hz: Optional[float] = None, hi_hz: Optional[float] = None) -> BandPlan:
    """The plan for a recording that was already made: its centre and rate are fixed. Without a band,
    watch the clean part of the window."""
    half = USABLE * rate / 2 - EDGE_GUARD_HZ
    lo = center_hz - half if lo_hz is None else lo_hz
    hi = center_hz + half if hi_hz is None else hi_hz
    if lo < center_hz - USABLE * rate / 2 or hi > center_hz + USABLE * rate / 2:
        raise ValueError("that band does not fit inside the recording's window")
    return BandPlan(lo, hi, center_hz, float(rate), _nfft_for(rate), lo <= center_hz <= hi)


# ------------------------------------------------------------------------------------------ IQ -> power
class SliceAnalyzer:
    """Turns one slice of raw signed 8-bit IQ into dB power per frequency bin, over the band only."""

    def __init__(self, plan: BandPlan, slice_samples: int):
        self.nfft = plan.nfft
        self.segments = max(4, slice_samples // plan.nfft)
        self.samples = self.segments * plan.nfft
        self.window = np.hanning(plan.nfft).astype(np.float32)
        freqs = plan.center_hz + np.fft.fftshift(np.fft.fftfreq(plan.nfft, 1.0 / plan.rate))
        keep = np.flatnonzero((freqs >= plan.lo_hz - EDGE_GUARD_HZ) & (freqs <= plan.hi_hz + EDGE_GUARD_HZ))
        if keep.size < 8:
            raise ValueError("the band is too narrow for this sample rate: fewer than 8 frequency bins")
        self._keep = keep
        self.freqs = freqs[keep]
        self._norm = 20.0 * math.log10(127.0 * float(self.window.sum()))  # a full-scale tone reads about 0 dB

    def power_db(self, raw: bytes) -> np.ndarray:
        a = np.frombuffer(raw, dtype=np.int8, count=self.samples * 2).astype(np.float32)
        iq = np.empty(self.samples, dtype=np.complex64)
        iq.real, iq.imag = a[0::2], a[1::2]
        iq -= iq.mean()  # the converter's DC offset
        spec = np.fft.fft(iq.reshape(self.segments, self.nfft) * self.window, axis=1)
        p = (spec.real ** 2 + spec.imag ** 2).mean(axis=0)
        return 10.0 * np.log10(np.fft.fftshift(p)[self._keep] + 1e-9) - self._norm


# ------------------------------------------------------------------------------------------ bursts
@dataclass
class Burst:
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    snr_db: float
    onset_s: float  # seconds from the start of the stream
    duration_s: float
    slices: int  # slices in which it was detected (gaps inside it are not counted)
    truncated: bool = False  # cut at the maximum length: the transmission went on
    start_ts: str = ""
    duration_ms: float = 0.0


@dataclass
class _Det:
    """What the tracker needs from a cluster of detections in one slice."""

    center: float
    bw: float
    peak_db: float
    snr_db: float
    power: float  # linear power above the noise floor, for timing the edges


@dataclass
class _Open:
    center_sum: float
    bw: float
    peak: float
    snr: float
    start: int
    last: int
    n: int
    p_first: float
    p_last: float
    p_max: float

    @property
    def center(self) -> float:
        return self.center_sum / self.n


class BurstTracker:
    """Joins detections from consecutive slices into bursts.

    Detections within `merge_hz` of each other in one slice count as one transmission (an FSK signal shows
    up as two lobes). A burst ends when its frequency has been quiet for more than `gap_slices` slices, or
    when it has lasted `max_slices`. Onset and duration are estimated to a fraction of a slice: a burst that
    only fills part of its first or last slice shows proportionally less power there than in a full slice.
    """

    def __init__(self, bin_hz: float, slice_s: float, gap_slices: int = 3, max_slices: int = 500, merge_hz: float = 50e3):
        self.bin_hz, self.slice_s = bin_hz, slice_s
        self.gap_slices = max(1, gap_slices)
        self.max_slices = max(2, max_slices)
        self.merge_hz = merge_hz
        self._open: List[_Open] = []

    def _cluster(self, dets: Sequence[Detection]) -> List[_Det]:
        groups: List[List[Detection]] = []
        reach = float("-inf")
        for d in sorted(dets, key=lambda x: x.lo_hz):
            if groups and d.lo_hz - reach <= self.merge_hz:
                groups[-1].append(d)
                reach = max(reach, d.hi_hz)
            else:
                groups.append([d])
                reach = d.hi_hz
        out = []
        for g in groups:
            top = max(g, key=lambda x: x.peak_db)
            weights = [10.0 ** (x.mean_db / 10.0) * max(x.bw_hz, 1.0) for x in g]
            center = sum(x.center_hz * w for x, w in zip(g, weights)) / sum(weights)
            power = max(10.0 ** (top.peak_db / 10.0) - 10.0 ** (top.floor_db / 10.0), 1e-12)
            out.append(_Det(center, max(x.hi_hz for x in g) - min(x.lo_hz for x in g), top.peak_db, top.snr_db, power))
        return out

    def _close(self, o: _Open, truncated: bool = False) -> Burst:
        f_first = min(1.0, max(0.02, o.p_first / o.p_max))
        f_last = min(1.0, max(0.02, o.p_last / o.p_max))
        onset = (o.start + 1.0 - f_first) * self.slice_s
        offset = (o.last + f_last) * self.slice_s
        return Burst(o.center, o.bw, o.peak, o.snr, onset, offset - onset, o.n, truncated)

    def update(self, slice_no: int, dets: Sequence[Detection]) -> List[Burst]:
        taken: set = set()
        for d in self._cluster(dets):
            best, best_dist = None, 0.0
            for i, o in enumerate(self._open):
                if i in taken:
                    continue
                dist = abs(o.center - d.center)
                if dist <= max(2 * self.bin_hz, 0.5 * max(o.bw, d.bw), self.merge_hz) and (best is None or dist < best_dist):
                    best, best_dist = i, dist
            if best is None:
                self._open.append(_Open(d.center, d.bw, d.peak_db, d.snr_db, slice_no, slice_no, 1, d.power, d.power, d.power))
                taken.add(len(self._open) - 1)
            else:
                o = self._open[best]
                o.center_sum += d.center
                o.n += 1
                o.bw = max(o.bw, d.bw)
                o.peak = max(o.peak, d.peak_db)
                o.snr = max(o.snr, d.snr_db)
                o.last = slice_no
                o.p_last = d.power
                o.p_max = max(o.p_max, d.power)
                taken.add(best)
        closed: List[Burst] = []
        keep: List[_Open] = []
        for o in self._open:
            if slice_no - o.last > self.gap_slices:
                closed.append(self._close(o))
            elif o.last - o.start + 1 >= self.max_slices:
                closed.append(self._close(o, truncated=True))
            else:
                keep.append(o)
        self._open = keep
        return closed

    def flush(self) -> List[Burst]:
        out = [self._close(o) for o in self._open]
        self._open = []
        return out


@dataclass
class _Candidate:
    center: float
    bw: float
    last_s: float
    bursts: List[Burst] = field(default_factory=list)


# ------------------------------------------------------------------------------------------ the watcher
class Watcher:
    def __init__(
        self,
        store: Store,
        identifier: Optional[Identifier],
        plan: BandPlan,
        *,
        snr_db: float = 10.0,
        slice_ms: float = 20.0,
        gap_ms: float = 60.0,
        max_burst_s: float = 10.0,
        min_bursts: int = 2,
        candidate_s: float = 600.0,
        warmup_s: float = 0.5,
        floor_alpha: float = 0.05,
        merge_khz: float = 50.0,
        start: Optional[datetime] = None,
        live: bool = False,
        log: Callable[[str], None] = print,
        warn: Optional[Callable[[str], None]] = None,
        commit_every: float = 1.0,
        summary_every: float = 30.0,
    ):
        if slice_ms < 5:
            raise ValueError("slices shorter than 5 ms are not supported")
        self.store, self.identifier, self.plan = store, identifier, plan
        self.snr_db, self.min_bursts, self.candidate_s = snr_db, max(1, min_bursts), candidate_s
        self.analyzer = SliceAnalyzer(plan, int(round(plan.rate * slice_ms / 1000.0)))
        self.slice_bytes = self.analyzer.samples * 2
        self.slice_s = self.analyzer.samples / plan.rate
        self.freqs, self.bin_hz = self.analyzer.freqs, plan.bin_hz
        self.ignore = [(plan.center_hz - 1.5 * self.bin_hz, plan.center_hz + 1.5 * self.bin_hz)] if plan.dc_in_band else []
        self.tracker = BurstTracker(
            self.bin_hz, self.slice_s, max(1, round(gap_ms / (self.slice_s * 1000.0))), max(2, int(max_burst_s / self.slice_s)), merge_khz * 1e3
        )
        self.floor = FloorTracker(alpha=floor_alpha, jump_db=0.0)  # the sweep-style overload guard would hide the bursts we want
        self.warmup_slices = int(math.ceil(warmup_s / self.slice_s))
        self.start = start or datetime.now()
        self.live, self.log = live, log
        self.warn = warn or (lambda m: log("WARN  " + m))
        self.commit_every, self.summary_every = commit_every, summary_every
        self.slice_no = 0
        self.bursts_total = 0
        self._window_bursts = 0
        self._window_busy = 0.0
        self._floor_db = float("nan")
        self._last_commit = time.monotonic()
        self._last_summary_s = 0.0
        self._cands: List[_Candidate] = []
        self._known: List[Dict[str, float]] = []
        near = (plan.lo_hz - 1e6, plan.hi_hz + 1e6)
        for r in store.load_signals():
            if near[0] <= r["center_hz"] <= near[1]:
                self._known.append({"id": r["id"], "center": r["center_hz"], "bw": r["bandwidth_hz"]})

    # ---- stream --------------------------------------------------------
    def run(self, chunks: Iterable[bytes], duration_s: Optional[float] = None) -> None:
        """Process slices until the stream ends, `duration_s` of recording time has passed, or it is interrupted."""
        self.log(self.plan.describe())
        try:
            for raw in chunks:
                if len(raw) < self.slice_bytes:
                    break
                self.process_slice(raw)
                if duration_s and self.slice_no * self.slice_s >= duration_s:
                    break
        finally:
            self.finish()

    def process_slice(self, raw: bytes) -> None:
        began = time.perf_counter()
        n = self.slice_no
        self.slice_no += 1
        power = self.analyzer.power_db(raw)
        floor, _ = self.floor.update(self.freqs, estimate_floor(power, self.bin_hz))
        if n >= self.warmup_slices:  # the first moments only settle the noise floor
            dets, self._floor_db = detect_signals(self.freqs, power, self.bin_hz, snr_db=self.snr_db, ignore=self.ignore, floor=floor)
            for burst in self.tracker.update(n, dets):
                self._burst(burst)
        self._window_busy += time.perf_counter() - began
        now = time.monotonic()
        if now - self._last_commit >= self.commit_every:
            self.store.commit()
            self._last_commit = now
        if self.slice_no * self.slice_s - self._last_summary_s >= self.summary_every:
            self._summary()

    def finish(self) -> None:
        for burst in self.tracker.flush():
            self._burst(burst)
        self._summary(final=True)
        self.store.commit()

    # ---- bursts --------------------------------------------------------
    def _time_of(self, slice_no: float) -> datetime:
        return self.start + timedelta(seconds=slice_no * self.slice_s)

    def _burst(self, b: Burst) -> None:
        b.start_ts = (self.start + timedelta(seconds=b.onset_s)).isoformat(timespec="milliseconds")
        b.duration_ms = b.duration_s * 1000.0
        sid = self._known_signal(b)
        if sid is not None:
            self._record(sid, b)
            return
        cand = self._candidate(b)
        if cand is not None:
            self._create(cand)

    def _known_signal(self, b: Burst) -> Optional[int]:
        best, best_dist = None, 0.0
        for k in self._known:
            dist = abs(k["center"] - b.center_hz)
            if dist <= max(2 * self.bin_hz, 0.5 * max(k["bw"], b.bandwidth_hz)) and (best is None or dist < best_dist):
                best, best_dist = int(k["id"]), dist
        return best

    def _record(self, sid: int, b: Burst) -> None:
        self.store.touch_signal(sid, b.start_ts, b.peak_db, b.snr_db)
        self.store.add_burst(sid, b.start_ts, b.duration_ms, b.center_hz, b.bandwidth_hz, b.peak_db, b.snr_db, b.slices)
        self.bursts_total += 1
        self._window_bursts += 1

    def _candidate(self, b: Burst) -> Optional[_Candidate]:
        """A burst at a frequency nothing is known at: wait for more before believing it is a signal."""
        now = b.onset_s
        self._cands = [c for c in self._cands if now - c.last_s <= self.candidate_s]
        for c in self._cands:
            if abs(c.center - b.center_hz) <= max(2 * self.bin_hz, 0.5 * max(c.bw, b.bandwidth_hz)):
                c.bursts.append(b)
                c.last_s = now
                c.center = float(np.mean([x.center_hz for x in c.bursts]))
                c.bw = max(c.bw, b.bandwidth_hz)
                break
        else:
            c = _Candidate(b.center_hz, b.bandwidth_hz, now, [b])
            self._cands.append(c)
        if len(c.bursts) >= self.min_bursts:
            self._cands.remove(c)
            return c
        return None

    def _create(self, c: _Candidate) -> None:
        bursts = c.bursts
        strongest = max(bursts, key=lambda x: x.peak_db)
        center = float(np.mean([x.center_hz for x in bursts]))
        bw = float(np.median([x.bandwidth_hz for x in bursts]))
        sid = self.store.insert_signal(
            center_hz=center, bandwidth_hz=bw, peak_hz=strongest.center_hz, first_seen=bursts[0].start_ts, last_seen=bursts[-1].start_ts,
            first_sweep=self.store.sweep_count(), hits=len(bursts), max_db=strongest.peak_db,
            avg_db=float(np.mean([x.peak_db for x in bursts])), last_db=bursts[-1].peak_db,
            max_snr=max(x.snr_db for x in bursts), last_snr=bursts[-1].snr_db,
        )
        ident = self.identifier.identify(center, bw, self.bin_hz) if self.identifier is not None else None
        if ident is not None:
            self.store.set_ident(sid, ident, bursts[-1].start_ts)
        for x in bursts:
            self.store.add_burst(sid, x.start_ts, x.duration_ms, x.center_hz, x.bandwidth_hz, x.peak_db, x.snr_db, x.slices)
        self.bursts_total += len(bursts)
        self._window_bursts += len(bursts)
        self._known.append({"id": sid, "center": center, "bw": bw})
        label = ident["name"] if ident and ident.get("name") else "unidentified"
        if ident and ident.get("source") == "artemis":
            label += f"  [{ident['score']:.0f}% match]"
        self.log(
            f"NEW  #{sid:<4d} {center / 1e6:10.3f} MHz  bw {bw / 1e3:8.1f} kHz  snr {max(x.snr_db for x in bursts):5.1f} dB  "
            f"{label}  (bursts of ~{float(np.median([x.duration_ms for x in bursts])):.0f} ms)"
        )

    # ---- reporting -----------------------------------------------------
    def _summary(self, final: bool = False) -> None:
        elapsed = self.slice_no * self.slice_s - self._last_summary_s
        if elapsed <= 0:
            return
        load = self._window_busy / elapsed
        if not final or self._window_bursts:
            self.log(
                f"  watching: {self._window_bursts} burst(s) in the last {elapsed:.0f} s, noise floor {self._floor_db:.1f} dB, "
                f"{load:.0%} of real time used"
            )
        if self.live and load > 0.8:
            self.warn(f"cannot keep up ({load:.0%} of real time): bursts may be missed. Watch a narrower band or lower --rate")
        self.store.add_sweep_summary(self._time_of(self.slice_no).isoformat(timespec="seconds"), int(self.freqs.size), self._window_bursts, self._floor_db)
        self._last_summary_s = self.slice_no * self.slice_s
        self._window_bursts = 0
        self._window_busy = 0.0


# ------------------------------------------------------------------------------------------ sources
def build_transfer_command(exe: str, plan: BandPlan, lna: int, vga: int, amp: bool) -> List[str]:
    return [exe, "-r", "-", "-f", str(int(plan.center_hz)), "-s", str(int(plan.rate)), "-l", str(lna), "-g", str(vga), "-a", "1" if amp else "0"]


class IQProcess:
    """`hackrf_transfer -r -`: signed 8-bit IQ on stdout, status on stderr."""

    def __init__(self, cmd: List[str], on_stderr: Optional[Callable[[str], None]] = None):
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(f"'{cmd[0]}' not found. Install the HackRF tools (sudo apt install hackrf) and check that `hackrf_info` sees your device.")
        self.cmd, self.on_stderr = cmd, on_stderr
        self.stderr_tail: Deque[str] = collections.deque(maxlen=40)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 22)
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            self.stderr_tail.append(line)
            if self.on_stderr is not None:
                try:
                    self.on_stderr(line)
                except Exception:  # logging must never break the stream
                    pass

    def chunks(self, nbytes: int) -> Iterator[bytes]:
        assert self.proc.stdout is not None
        while True:
            data = self.proc.stdout.read(nbytes)
            if len(data) < nbytes:
                return
            yield data

    def close(self) -> int:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._t.join(timeout=1)
        return self.proc.returncode or 0

    def error_text(self) -> str:
        return "\n".join(list(self.stderr_tail)[-10:])


def file_chunks(path: str, nbytes: int) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            data = fh.read(nbytes)
            if len(data) < nbytes:
                return
            yield data
