"""Read `hackrf_sweep` CSV output and assemble complete sweeps.

hackrf_sweep prints one CSV line per 5 MHz chunk:

    date, time, hz_low, hz_high, hz_bin_width, num_samples, dB, dB, dB, ...

Chunks inside one sweep are NOT strictly ascending (tunings are interleaved),
so a new sweep is detected when hz_low jumps back by more than `wrap_hz`.
"""

from __future__ import annotations

import collections
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class Segment:
    ts: str
    lo: int
    hi: int
    bin_hz: float
    power: np.ndarray


@dataclass
class Sweep:
    ts: str  # ISO-ish timestamp of the first chunk (local time, as printed by hackrf_sweep)
    freqs: np.ndarray  # bin centre frequencies in Hz, ascending, unique
    power: np.ndarray  # dB (uncalibrated, relative)
    bin_hz: float


def parse_line(line: str) -> Optional[Segment]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    try:
        date, tm = parts[0], parts[1]
        lo = int(float(parts[2]))
        hi = int(float(parts[3]))
        bin_hz = float(parts[4])
        vals = np.array([float(x) for x in parts[6:] if x != ""], dtype=np.float64)
    except ValueError:
        return None
    if vals.size == 0 or bin_hz <= 0 or hi <= lo:
        return None
    return Segment(ts=f"{date}T{tm}", lo=lo, hi=hi, bin_hz=bin_hz, power=vals)


def _build(segs: List[Segment]) -> Sweep:
    bin_hz = segs[0].bin_hz
    freqs = np.concatenate([s.lo + (np.arange(s.power.size) + 0.5) * s.bin_hz for s in segs])
    power = np.concatenate([s.power for s in segs])
    # sort and merge duplicate bins (overlapping chunks) by taking the max
    key = np.round(freqs / (bin_hz / 2.0)).astype(np.int64)
    order = np.argsort(key, kind="stable")
    key, freqs, power = key[order], freqs[order], power[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return Sweep(
        ts=segs[0].ts,
        freqs=freqs[starts],
        power=np.maximum.reduceat(power, starts),
        bin_hz=bin_hz,
    )


def iter_sweeps(lines: Iterable[str], wrap_hz: float = 30e6) -> Iterator[Sweep]:
    """Group hackrf_sweep CSV lines into complete sweeps.

    A sweep is yielded when the *next* sweep starts (or the input ends), so a
    partially received final sweep is never returned if the caller stops early.
    """
    segs: List[Segment] = []
    max_lo = -1.0
    for line in lines:
        seg = parse_line(line)
        if seg is None:
            continue
        if segs and seg.lo < max_lo - wrap_hz:
            yield _build(segs)
            segs, max_lo = [], -1.0
        segs.append(seg)
        max_lo = max(max_lo, seg.lo)
    if segs:
        yield _build(segs)


def build_sweep_command(
    exe: str,
    ranges_mhz: List[str],
    bin_hz: int,
    lna: int,
    vga: int,
    amp: bool,
    sweeps: Optional[int],
) -> List[str]:
    cmd = [exe]
    for r in ranges_mhz:
        cmd += ["-f", r]
    cmd += ["-w", str(int(bin_hz)), "-l", str(lna), "-g", str(vga), "-a", "1" if amp else "0"]
    if sweeps:
        cmd += ["-N", str(sweeps)]
    return cmd


class SweepProcess:
    """Run hackrf_sweep and expose its stdout as an iterator of lines."""

    def __init__(self, cmd: List[str]):
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(
                f"'{cmd[0]}' not found. Install the HackRF tools "
                "(Debian/Ubuntu: sudo apt install hackrf; macOS: brew install hackrf) "
                "and check that `hackrf_info` sees your device."
            )
        self.cmd = cmd
        self.stderr_tail: collections.deque = collections.deque(maxlen=40)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # hackrf_sweep logs to stderr every second; drain it so the pipe never fills.
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr_tail.append(line.rstrip())

    def lines(self) -> Iterator[str]:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            yield line

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
