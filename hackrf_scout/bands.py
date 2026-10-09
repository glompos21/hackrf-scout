"""Named frequency bands for filtering the database and choosing what to scan.

A band is a name plus a frequency range. Presets are curated (EU / ITU Region 1
flavoured, like `bandplan`), and extra ones can be added in a JSON file:

    {"bands": [{"key": "garage", "name": "Garage remotes", "lo_mhz": 433.0, "hi_mhz": 435.0}]}

Anywhere a band is expected you can also give a plain range `START:STOP` in MHz.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

HACKRF_MIN_MHZ = 1
HACKRF_MAX_MHZ = 6000

_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*:\s*(\d+(?:\.\d+)?|\.\d+)\s*$")
_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")


@dataclass(frozen=True)
class Band:
    key: str
    name: str
    lo_hz: float
    hi_hz: float
    group: str = "Custom"

    def sweep_range(self) -> str:
        """`hackrf_sweep -f` wants whole MHz, so round outwards: 433.05-434.79 -> 433:435."""
        lo = max(HACKRF_MIN_MHZ, int(math.floor(self.lo_hz / 1e6 + 1e-9)))
        hi = min(HACKRF_MAX_MHZ, int(math.ceil(self.hi_hz / 1e6 - 1e-9)))
        if hi <= lo:
            hi = lo + 1
        return f"{lo}:{hi}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "group": self.group,
            "lo_hz": self.lo_hz,
            "hi_hz": self.hi_hz,
            "sweepable": self.hi_hz / 1e6 > HACKRF_MIN_MHZ and self.lo_hz / 1e6 < HACKRF_MAX_MHZ,
            "sweep_range": self.sweep_range() if self.hi_hz / 1e6 > HACKRF_MIN_MHZ and self.lo_hz / 1e6 < HACKRF_MAX_MHZ else None,
        }


def _b(key: str, name: str, lo_mhz: float, hi_mhz: float, group: str) -> Band:
    return Band(key, name, lo_mhz * 1e6, hi_mhz * 1e6, group)


# UHF/VHF in the ITU sense are huge (UHF alone is 300-3000 MHz), so they are offered as umbrella
# groups and the practical sub-bands below them are what you will normally want.
PRESETS: Tuple[Band, ...] = (
    _b("vhf", "VHF (30-300 MHz)", 30, 300, "Umbrella"),
    _b("uhf", "UHF (300-3000 MHz)", 300, 3000, "Umbrella"),
    _b("433", "ISM 433 MHz (sensors, remotes, TPMS)", 433.05, 434.79, "ISM / SRD"),
    _b("pmr446", "PMR446 walkie-talkies", 446.0, 446.2, "ISM / SRD"),
    _b("868", "SRD 868 MHz (LoRa, sensors)", 863, 870, "ISM / SRD"),
    _b("dect", "DECT cordless phones", 1880, 1900, "ISM / SRD"),
    _b("2.4ghz", "ISM 2.4 GHz (Wi-Fi, Bluetooth, drones)", 2400, 2483.5, "ISM / SRD"),
    _b("5.8ghz", "ISM 5.8 GHz (Wi-Fi, FPV video)", 5725, 5875, "ISM / SRD"),
    _b("fm", "FM broadcast", 87.5, 108, "VHF"),
    _b("airband", "Airband voice (AM)", 118, 137, "VHF"),
    _b("wxsat", "Weather satellites (NOAA / Meteor)", 137, 138, "VHF"),
    _b("ham2m", "Amateur 2 m", 144, 146, "VHF"),
    _b("marine", "Marine VHF and AIS", 156, 162.05, "VHF"),
    _b("vhf-mobile", "VHF land mobile / PMR", 162.05, 174, "VHF"),
    _b("dab", "VHF band III (DAB+ / TV)", 174, 230, "VHF"),
    _b("tetra", "TETRA (emergency services)", 380, 400, "UHF"),
    _b("ham70cm", "Amateur 70 cm", 430, 440, "UHF"),
    _b("uhf-tv", "UHF TV (DVB-T/T2)", 470, 694, "UHF"),
    _b("gsm900", "GSM900 / LTE 900", 880, 960, "Cellular"),
    _b("gsm1800", "GSM1800 / LTE 1800", 1710, 1880, "Cellular"),
    _b("lte2100", "UMTS / LTE 2100", 1920, 2170, "Cellular"),
    _b("lte2600", "LTE 2600", 2500, 2690, "Cellular"),
    _b("adsb", "ADS-B / Mode S (1090 MHz)", 1088, 1092, "Aero / Satellite"),
    _b("gnss", "GNSS L1 / E1 / GLONASS", 1559, 1610, "Aero / Satellite"),
    _b("iridium", "Iridium satellite", 1616, 1626.5, "Aero / Satellite"),
)

_ALIASES = {
    "433mhz": "433",
    "868mhz": "868",
    "2.4": "2.4ghz",
    "2.4g": "2.4ghz",
    "5.8": "5.8ghz",
    "5.8g": "5.8ghz",
    "gps": "gnss",
}


def _custom_range(spec: str) -> Optional[Band]:
    m = _RANGE.match(spec)
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if not (0 <= lo < hi <= 300000):
        raise ValueError(f"invalid range '{spec.strip()}': need 0 <= START < STOP (MHz)")
    return Band(f"{lo:g}:{hi:g}", f"{lo:g}-{hi:g} MHz", lo * 1e6, hi * 1e6, "Custom range")


class Registry:
    """Presets plus any user-defined bands; resolves a spec string to a Band."""

    def __init__(self, extra: Iterable[Band] = ()):
        self._by_key = {b.key: b for b in PRESETS}
        for b in extra:
            self._by_key[b.key] = b

    def all(self) -> List[Band]:
        return list(self._by_key.values())

    def resolve(self, spec: str) -> Band:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("empty band")
        rng = _custom_range(spec)
        if rng is not None:
            return rng
        key = re.sub(r"[\s_]+", "", spec.strip().lower())
        key = _ALIASES.get(key, key)
        band = self._by_key.get(key)
        if band is None:
            raise ValueError(f"unknown band '{spec.strip()}' (use a preset name such as 868, or START:STOP in MHz)")
        return band

    def resolve_many(self, specs: Iterable[str]) -> List[Band]:
        out: List[Band] = []
        seen = set()
        for raw in specs:
            for part in str(raw).split(","):
                if part.strip():
                    b = self.resolve(part)
                    if b.key not in seen:
                        seen.add(b.key)
                        out.append(b)
        return out


def default_bands_path() -> Optional[str]:
    env = os.environ.get("HACKRF_SCOUT_BANDS")
    if env:
        return env
    p = os.path.expanduser(os.path.join("~", ".hackrf-scout", "bands.json"))
    return p if os.path.isfile(p) else None


def load_custom(path: Optional[str]) -> Tuple[List[Band], List[str]]:
    """Read extra bands from a JSON file. Bad entries are skipped and reported, never fatal."""
    if not path:
        return [], []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        return [], [f"{path}: {exc}"]
    items = data.get("bands") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], [f"{path}: expected a list of bands or {{\"bands\": [...]}}"]
    bands: List[Band] = []
    warnings: List[str] = []
    for i, it in enumerate(items):
        try:
            key = str(it["key"]).strip().lower()
            if not _KEY.match(key):
                raise ValueError("key must be 1-32 chars of a-z 0-9 . _ -")
            lo, hi = float(it["lo_mhz"]), float(it["hi_mhz"])
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi <= 300000):
                raise ValueError("need 0 <= lo_mhz < hi_mhz")
            name = str(it.get("name") or key)[:80]
            group = str(it.get("group") or "Custom")[:40]
            bands.append(Band(key, name, lo * 1e6, hi * 1e6, group))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"{path}: entry {i}: {exc}")
    return bands, warnings


def merge_sweep_ranges(bands: Sequence[Band]) -> List[str]:
    """Whole-MHz `START:STOP` strings for hackrf_sweep, with overlapping/adjacent ranges merged."""
    spans = []
    for b in bands:
        if b.hi_hz / 1e6 <= HACKRF_MIN_MHZ or b.lo_hz / 1e6 >= HACKRF_MAX_MHZ:
            raise ValueError(f"band '{b.key}' is outside the HackRF range ({HACKRF_MIN_MHZ}-{HACKRF_MAX_MHZ} MHz)")
        lo, hi = (int(x) for x in b.sweep_range().split(":"))
        spans.append((lo, hi))
    spans.sort()
    merged: List[List[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [f"{lo}:{hi}" for lo, hi in merged]


def signal_clause(bands: Sequence[Band], mode: str = "overlap", margin_hz: float = 0.0, alias: str = "s") -> Tuple[str, list]:
    """SQL fragment (and parameters) selecting signals that fall in any of `bands`.

    `overlap` keeps a signal if any part of its occupied bandwidth touches the band, so a 20 MHz
    Wi-Fi channel centred just below 2400 MHz is still found. `center` only looks at the centre
    frequency. `margin_hz` (half the widest stored signal) lets SQLite use the centre index first.
    """
    if mode not in ("overlap", "center"):
        raise ValueError("mode must be 'overlap' or 'center'")
    if not bands:
        return "1=1", []
    parts, params = [], []
    c, w = f"{alias}.center_hz", f"{alias}.bandwidth_hz"
    for b in bands:
        if mode == "center":
            parts.append(f"({c} BETWEEN ? AND ?)")
            params += [b.lo_hz, b.hi_hz]
        else:
            parts.append(f"({c} BETWEEN ? AND ? AND {c} + {w} / 2.0 >= ? AND {c} - {w} / 2.0 <= ?)")
            params += [b.lo_hz - margin_hz, b.hi_hz + margin_hz, b.lo_hz, b.hi_hz]
    return "(" + " OR ".join(parts) + ")", params
