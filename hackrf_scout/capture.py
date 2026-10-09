"""Record raw IQ of detected signals with hackrf_transfer.

hackrf_sweep and hackrf_transfer cannot use the HackRF at the same time, so
captures are made *between* scanning rounds (see the `run` command).
Files are signed 8-bit interleaved I/Q (`.cs8`) with a JSON sidecar.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


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
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = f"sig{signal['id']:04d}_{center / 1e6:.3f}MHz_{stamp}"
    path = os.path.join(out_dir, base + ".cs8")
    cmd = [
        exe, "-r", path, "-f", str(int(round(center))), "-s", str(rate), "-n", str(n),
        "-l", str(lna), "-g", str(vga), "-a", "1" if amp else "0",
        "-b", str(int(min(max(bw * 1.5, 1.75e6), rate))),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"hackrf_transfer failed ({res.returncode}): {res.stderr.strip()[-300:]}")
    meta = {
        "file": os.path.basename(path),
        "datatype": "ci8 (signed 8-bit interleaved I/Q)",
        "center_hz": center,
        "sample_rate": rate,
        "seconds": seconds,
        "signal_id": int(signal["id"]),
        "estimated_bandwidth_hz": bw,
        "identified_as": signal["ident_name"],
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "hackrf-scout",
    }
    with open(path[:-4] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return {"path": path, "rate": rate, "seconds": seconds, "center_hz": center}
