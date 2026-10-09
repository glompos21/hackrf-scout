"""SigMF (https://sigmf.org) metadata for the IQ recordings.

A recording is two files side by side: `NAME.sigmf-data`, the raw samples, and `NAME.sigmf-meta`, a JSON
description. HackRF's native format is signed 8-bit interleaved I/Q, which SigMF calls `ci8`, so the data
file is exactly what `hackrf_transfer` writes and nothing is converted.

The metadata carries what a tool needs to open the file without being told anything: sample rate, centre
frequency, start time (UTC), and an annotation marking the signal we were after (its estimated frequency
edges and what we think it is). Fields specific to this tool use the `hackrf_scout:` prefix.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from . import __version__

SIGMF_VERSION = "1.2.2"
DATA_SUFFIX = ".sigmf-data"
META_SUFFIX = ".sigmf-meta"


def utc_timestamp(when: Optional[datetime] = None) -> str:
    """ISO 8601 in UTC with milliseconds and a Z, as SigMF's core:datetime wants."""
    when = (when or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%S.") + f"{when.microsecond // 1000:03d}Z"


def sha512_of(path: str) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_meta(
    data_path: str,
    *,
    sample_rate: float,
    center_hz: float,
    started_at: datetime,
    label: Optional[str] = None,
    description: Optional[str] = None,
    freq_lower_hz: Optional[float] = None,
    freq_upper_hz: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    checksum: bool = True,
) -> str:
    """Write the `.sigmf-meta` for an existing `.sigmf-data` file and return its path."""
    if not data_path.endswith(DATA_SUFFIX):
        raise ValueError(f"expected a {DATA_SUFFIX} file, got {data_path}")
    samples = os.path.getsize(data_path) // 2  # two bytes per complex ci8 sample
    meta: Dict[str, Any] = {
        "global": {
            "core:datatype": "ci8",
            "core:sample_rate": float(sample_rate),
            "core:version": SIGMF_VERSION,
            "core:recorder": f"hackrf-scout {__version__}",
            "core:hw": "HackRF",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": float(center_hz), "core:datetime": utc_timestamp(started_at)}],
        "annotations": [],
    }
    g = meta["global"]
    if description:
        g["core:description"] = description
    if checksum:
        g["core:sha512"] = sha512_of(data_path)
    for k, v in (extra or {}).items():
        g[f"hackrf_scout:{k}"] = v
    if samples and freq_lower_hz is not None and freq_upper_hz is not None:
        note: Dict[str, Any] = {
            "core:sample_start": 0, "core:sample_count": samples,
            "core:freq_lower_edge": float(freq_lower_hz), "core:freq_upper_edge": float(freq_upper_hz),
        }
        if label:
            note["core:label"] = label
        meta["annotations"].append(note)
    meta_path = data_path[: -len(DATA_SUFFIX)] + META_SUFFIX
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    return meta_path


def read_meta(meta_path: str) -> Dict[str, Any]:
    """What a replay needs from a `.sigmf-meta`: sample rate, centre frequency, and the data file's path."""
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    g = meta.get("global", {})
    if g.get("core:datatype") != "ci8":
        raise ValueError(f"{meta_path}: only ci8 (signed 8-bit I/Q) recordings are supported, not {g.get('core:datatype')!r}")
    caps = meta.get("captures") or [{}]
    if "core:sample_rate" not in g or "core:frequency" not in caps[0]:
        raise ValueError(f"{meta_path}: needs core:sample_rate and a capture with core:frequency")
    base = meta_path[: -len(META_SUFFIX)] if meta_path.endswith(META_SUFFIX) else os.path.splitext(meta_path)[0]
    return {
        "sample_rate": float(g["core:sample_rate"]),
        "center_hz": float(caps[0]["core:frequency"]),
        "data_path": base + DATA_SUFFIX,
        "datetime": caps[0].get("core:datetime"),
    }
