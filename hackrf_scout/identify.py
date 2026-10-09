"""Identify a detected signal.

Two sources are combined:

* the Artemis / SigID Wiki signal database (run `hackrf-scout update-db` once to
  download it; it is not bundled), scored on frequency, bandwidth and region;
* a small built-in bandplan, which says what service normally lives there.

Modulation is not estimated by the scanner (a sweep only measures power), so it
is left out of the score; the typical modulation of each candidate is listed
as a hint so you can check it by hand with a demodulator.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import bandplan

DEFAULT_REGION_KEYWORDS = ("worldwide", "global", "international", "europe", "eu")


def default_signals_paths() -> List[str]:
    paths = []
    env = os.environ.get("HACKRF_SCOUT_SIGNALS")
    if env:
        paths.append(env)
    paths.append(os.path.join("data", "artemis_signals.json"))
    paths.append(os.path.expanduser(os.path.join("~", ".hackrf-scout", "artemis_signals.json")))
    return paths


def _group_ranges(freqs: Sequence[float]) -> List[Tuple[float, float, bool]]:
    """Merge point frequencies within a 1.5x ratio into (lo, hi, is_point) ranges."""
    fs = sorted(f for f in freqs if f and f > 0)
    if not fs:
        return []
    groups: List[List[float]] = [[fs[0]]]
    for f in fs[1:]:
        if f / groups[-1][-1] < 1.5:
            groups[-1].append(f)
        else:
            groups.append([f])
    return [(g[0], g[-1], len(g) == 1 or g[0] == g[-1]) for g in groups]


@dataclass
class _Sig:
    name: str
    url: Optional[str]
    description: str
    categories: List[str]
    ranges: List[Tuple[float, float, bool]]
    bw_lo: Optional[float]
    bw_hi: Optional[float]
    modulations: List[str]
    locations: List[str]


class Identifier:
    def __init__(
        self,
        signals: Optional[List[Dict[str, Any]]] = None,
        region_keywords: Sequence[str] = DEFAULT_REGION_KEYWORDS,
        min_score: float = 55.0,
    ):
        self.min_score = min_score
        self.region_keywords = tuple(k.lower() for k in region_keywords)
        self.sigs: List[_Sig] = []
        for s in signals or []:
            ranges = _group_ranges(s.get("freqs_hz", []))
            if not ranges:
                continue
            bws = [b for b in s.get("bandwidths_hz", []) if b and b > 0]
            self.sigs.append(
                _Sig(
                    name=s["name"],
                    url=s.get("url"),
                    description=s.get("description", ""),
                    categories=s.get("categories", []),
                    ranges=ranges,
                    bw_lo=min(bws) if bws else None,
                    bw_hi=max(bws) if bws else None,
                    modulations=s.get("modulations", []),
                    locations=s.get("locations", []),
                )
            )

    @classmethod
    def load(cls, path: Optional[str] = None, **kw: Any) -> "Identifier":
        for p in [path] if path else default_signals_paths():
            if p and os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return cls(data.get("signals", data) if isinstance(data, dict) else data, **kw)
        return cls(None, **kw)

    @property
    def has_database(self) -> bool:
        return bool(self.sigs)

    # ---- scoring -------------------------------------------------------
    def _freq_points(self, s: _Sig, c: float, bw: float, bin_hz: float) -> Tuple[float, bool]:
        """Return (points, broad). `broad` means the match came from a wide listed range
        (the database cannot say whether that is a span or a list of discrete channels),
        so it is a weaker hint than a point match and is not used to name the signal when
        the bandplan already knows what lives there."""
        best, broad = 0.0, False

        def take(pts: float, is_broad: bool) -> None:
            nonlocal best, broad
            if pts > best or (pts == best and broad and not is_broad):
                best, broad = pts, is_broad

        for lo, hi, is_point in s.ranges:
            if is_point:
                tol = max(bw, 3 * bin_hz)
                d = abs(c - lo)
                if d <= tol:
                    take(40.0, False)
                elif d <= 3 * tol:
                    take(40.0 * (1.0 - (d - tol) / (2 * tol)), False)
            elif lo - bw / 2 - bin_hz <= c <= hi + bw / 2 + bin_hz:
                width = hi - lo
                take(40.0 if width <= 5e6 else 40.0 * (1.0 - 0.5 * min(1.0, math.log10(width / 5e6) / 2.0)), width > 5e6)
        return best, broad

    def _bw_points(self, s: _Sig, bw: float) -> float:
        if s.bw_lo is None or s.bw_hi is None:
            return 15.0  # unknown -> neutral
        lo, hi = s.bw_lo, s.bw_hi
        if lo * 0.7 <= bw <= hi * 1.5:
            return 30.0
        ratio = bw / hi if bw > hi else lo / max(bw, 1.0)
        return max(0.0, 30.0 * (1.0 - math.log10(max(ratio, 1.0))))

    def _region_points(self, s: _Sig) -> float:
        if not s.locations:
            return 5.0
        locs = {loc.strip().lower() for loc in s.locations}  # whole names, not substrings
        return 10.0 if locs & set(self.region_keywords) else 0.0

    def identify(self, center_hz: float, bw_hz: float, bin_hz: float = 100e3, top: int = 5) -> Dict[str, Any]:
        cands: List[Dict[str, Any]] = []
        for s in self.sigs:
            f, broad = self._freq_points(s, center_hz, bw_hz, bin_hz)
            if f <= 0:
                continue
            b = self._bw_points(s, bw_hz)
            r = self._region_points(s)
            score = 100.0 * (f + b + r) / 80.0
            cands.append(
                {
                    "name": s.name,
                    "score": round(min(score, 100.0), 1),
                    "source": "artemis",
                    "broad": broad,
                    "url": s.url,
                    "modulations": s.modulations,
                    "categories": s.categories,
                }
            )
        cands.sort(key=lambda c: -c["score"])
        cands = cands[:top]

        services = bandplan.lookup(center_hz, bw_hz)
        service = "; ".join(services[:3]) if services else None

        name = score = url = source = None
        good = [c for c in cands if c["score"] >= self.min_score]
        specific = [c for c in good if not c["broad"]]
        pick = specific[0] if specific else (good[0] if good and not services else None)
        if pick and services and sum(1 for c in specific if c["score"] >= pick["score"] - 5.0) > 3:
            pick = None  # many entries tie: naming one would be a guess, let the bandplan speak
        if pick:
            name, score, url, source = pick["name"], pick["score"], pick["url"], "artemis"
        elif services:
            name, score, source = services[0], None, "bandplan"
        return {
            "name": name,
            "score": score,
            "source": source,
            "url": url,
            "service": service,
            "candidates": cands,
        }
