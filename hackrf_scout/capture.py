"""Record raw IQ of detected signals with hackrf_transfer.

hackrf_sweep and hackrf_transfer cannot use the HackRF at the same time, so
captures are made *between* scanning rounds (see the `run` command).
Each recording is a SigMF pair: `NAME.sigmf-data` (signed 8-bit interleaved I/Q, exactly what
hackrf_transfer writes) and `NAME.sigmf-meta` (see `sigmf.py`).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import sigmf


def pick_sample_rate(bw_hz: float) -> int:
    """Smallest whole-MHz rate >= 2.5x the signal bandwidth, clamped to 2..20 Msps."""
    rate = math.ceil(bw_hz * 2.5 / 1e6) * 1e6
    return int(min(20e6, max(2e6, rate)))


def capture_signal(
    signal: Any,
    out_dir: str,
    seconds: float = 5.0,
    lna: int = 24,
    vga: int = 20,
    amp: bool = False,
    exe: str = "hackrf_transfer",
) -> Dict[str, Any]:
    if shutil.which(exe) is None:
        raise RuntimeError(f"'{exe}' not found. Install the HackRF tools (sudo apt install hackrf).")
    os.makedirs(out_dir, exist_ok=True)
    center = float(signal["center_hz"])
    bw = float(signal["bandwidth_hz"])
    rate = pick_sample_rate(bw)
    n = int(rate * seconds)
    started = datetime.now(timezone.utc)
    stamp = started.astimezone().strftime("%Y%m%dT%H%M%S")
    base = f"sig{signal['id']:04d}_{center / 1e6:.3f}MHz_{stamp}"
    path = os.path.join(out_dir, base + sigmf.DATA_SUFFIX)
    filter_hz = int(min(max(bw * 1.5, 1.75e6), rate))
    cmd = [
        exe, "-r", path, "-f", str(int(round(center))), "-s", str(rate), "-n", str(n),
        "-l", str(lna), "-g", str(vga), "-a", "1" if amp else "0",
        "-b", str(filter_hz),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"hackrf_transfer failed ({res.returncode}): {res.stderr.strip()[-300:]}")
    name = signal["ident_name"] or "unidentified"
    meta_path = sigmf.write_meta(
        path,
        sample_rate=rate,
        center_hz=center,
        started_at=started,
        label=name,
        description=f"hackrf-scout capture of signal #{int(signal['id'])}: {name}",
        freq_lower_hz=center - bw / 2,
        freq_upper_hz=center + bw / 2,
        extra={
            "signal_id": int(signal["id"]),
            "estimated_bandwidth_hz": bw,
            "lna_gain_db": lna,
            "vga_gain_db": vga,
            "amp_enabled": bool(amp),
            "baseband_filter_hz": filter_hz,
        },
    )
    return {"path": path, "meta": meta_path, "rate": rate, "seconds": seconds, "center_hz": center}
