"""Turn per-sweep detections into persistent, de-duplicated signals.

A detection becomes a stored signal only after it has been seen `min_hits`
times at (about) the same frequency. That removes random noise spikes while
still keeping bursty signals (use a lower --min-hits and a longer --expire).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .detect import Detection, detect_signals
from .identify import Identifier
from .store import Store
from .sweep import Sweep


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.now()


@dataclass
class Track:
    center: float
    bw: float
    peak_hz: float
    first_seen: str
    first_sweep: int
    last_seen: str = ""
    hits: int = 0
    max_db: float = -1e9
    avg_db: float = 0.0
    last_db: float = 0.0
    max_snr: float = 0.0
    last_snr: float = 0.0
    id: Optional[int] = None
    missed: int = 0
    buf: List[tuple] = field(default_factory=list)
    last_obs: Optional[datetime] = None


class Scanner:
    def __init__(
        self,
        store: Store,
        identifier: Optional[Identifier] = None,
        snr_db: float = 10.0,
        min_hits: int = 3,
        expire_sweeps: int = 20,
        obs_interval_s: float = 30.0,
        ignore: Sequence[Tuple[float, float]] = (),
        log: Callable[[str], None] = print,
    ):
        self.store = store
        self.identifier = identifier
        self.snr_db = snr_db
        self.min_hits = max(1, min_hits)
        self.expire = expire_sweeps
        self.obs_interval = obs_interval_s
        self.ignore = list(ignore)
        self.log = log
        self.tracks: List[Track] = []
        self._centers = np.zeros(0)
        self._dirty = False
        self.new_signals: List[int] = []
        self._last_summary: Optional[datetime] = None
        for r in store.load_signals():
            t = Track(
                center=r["center_hz"],
                bw=r["bandwidth_hz"],
                peak_hz=r["peak_hz"] or r["center_hz"],
                first_seen=r["first_seen"],
                first_sweep=r["first_sweep"],
                last_seen=r["last_seen"],
                hits=r["hits"],
                max_db=r["max_db"] or -1e9,
                avg_db=r["avg_db"] or 0.0,
                last_db=r["last_db"] or 0.0,
                max_snr=r["max_snr"] or 0.0,
                last_snr=r["last_snr"] or 0.0,
                id=r["id"],
            )
            self.tracks.append(t)
        self._dirty = True

    # ---- matching ------------------------------------------------------
    def _rebuild(self) -> None:
        self._centers = np.array([t.center for t in self.tracks]) if self.tracks else np.zeros(0)
        self._dirty = False

    def _match(self, d: Detection, bin_hz: float, used: set) -> Optional[Track]:
        if self._dirty:
            self._rebuild()
        if self._centers.size == 0:
            return None
        order = np.argsort(np.abs(self._centers - d.center_hz))[:3]
        for i in order:
            t = self.tracks[int(i)]
            tol = max(2 * bin_hz, 0.5 * max(t.bw, d.bw_hz))
            if abs(t.center - d.center_hz) <= tol and id(t) not in used:
                return t
            if abs(t.center - d.center_hz) > tol:
                break
        return None

    # ---- main entry ----------------------------------------------------
    def process(self, sweep: Sweep) -> dict:
        dets, floor = detect_signals(
            sweep.freqs, sweep.power, sweep.bin_hz, snr_db=self.snr_db, ignore=self.ignore
        )
        sweep_no = self.store.bump_sweep()
        now = _parse_ts(sweep.ts)
        used: set = set()
        confirmed_now: List[Track] = []

        for d in dets:
            t = self._match(d, sweep.bin_hz, used)
            if t is None:
                t = Track(
                    center=d.center_hz, bw=d.bw_hz, peak_hz=d.peak_hz, first_seen=sweep.ts, first_sweep=sweep_no
                )
                self.tracks.append(t)
                self._dirty = True
            used.add(id(t))
            t.hits += 1
            w = min(t.hits, 20)
            t.center += (d.center_hz - t.center) / w
            t.bw += (d.bw_hz - t.bw) / w
            t.peak_hz = d.peak_hz
            t.last_seen = sweep.ts
            t.last_db, t.last_snr = d.peak_db, d.snr_db
            t.max_db = max(t.max_db, d.peak_db)
            t.max_snr = max(t.max_snr, d.snr_db)
            t.avg_db += (d.peak_db - t.avg_db) / t.hits
            t.missed = 0
            obs = (sweep.ts, d.center_hz, d.bw_hz, d.peak_db, d.snr_db)
            if t.id is None:
                t.buf.append(obs)
                if t.hits >= self.min_hits:
                    self._confirm(t, sweep.bin_hz)
                    confirmed_now.append(t)
            else:
                self._update(t)
                if t.last_obs is None or (now - t.last_obs).total_seconds() >= self.obs_interval:
                    self.store.add_observation(t.id, *obs)
                    t.last_obs = now

        # age out unconfirmed tracks that stopped appearing
        keep = []
        for t in self.tracks:
            if t.id is None and id(t) not in used:
                t.missed += 1
                if t.missed > self.expire:
                    self._dirty = True
                    continue
            keep.append(t)
        self.tracks = keep

        if self._last_summary is None or (now - self._last_summary).total_seconds() >= self.obs_interval:
            self.store.add_sweep_summary(sweep.ts, int(sweep.power.size), len(dets), floor)
            self._last_summary = now
        self.store.commit()
        return {
            "sweep": sweep_no,
            "detections": len(dets),
            "confirmed_new": [t.id for t in confirmed_now],
            "floor_db": floor,
            "bins": int(sweep.power.size),
        }

    # ---- helpers -------------------------------------------------------
    def _fields(self, t: Track) -> dict:
        return dict(
            center_hz=t.center,
            bandwidth_hz=t.bw,
            peak_hz=t.peak_hz,
            first_seen=t.first_seen,
            last_seen=t.last_seen,
            first_sweep=t.first_sweep,
            hits=t.hits,
            max_db=t.max_db,
            avg_db=t.avg_db,
            last_db=t.last_db,
            max_snr=t.max_snr,
            last_snr=t.last_snr,
        )

    def _confirm(self, t: Track, bin_hz: float) -> None:
        t.id = self.store.insert_signal(**self._fields(t))
        for o in t.buf[-self.min_hits :]:
            self.store.add_observation(t.id, *o)
        t.last_obs = _parse_ts(t.last_seen)
        t.buf = []
        ident = None
        if self.identifier is not None:
            ident = self.identifier.identify(t.center, t.bw, bin_hz)
            self.store.set_ident(t.id, ident, t.last_seen)
        self.new_signals.append(t.id)
        label = ident["name"] if ident and ident.get("name") else "unidentified"
        if ident and ident.get("source") == "artemis":
            label += f"  [{ident['score']:.0f}% match]"
        self.log(
            f"NEW  #{t.id:<4d} {t.center / 1e6:10.3f} MHz  bw {t.bw / 1e3:8.1f} kHz  "
            f"snr {t.max_snr:5.1f} dB  {label}"
        )

    def _update(self, t: Track) -> None:
        f = self._fields(t)
        f["id"] = t.id
        self.store.update_signal(**f)
