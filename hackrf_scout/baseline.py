"""What is normal for each signal, and what is not.

Everything here is computed from data already in the database (`signals`, `observations`, `scans`),
with plain SELECTs, so it works on a read-only connection (the web UI) and on the scanner's own
connection (live alerts in the log) alike.

Four kinds of flag:

* ``louder`` / ``quieter``: the SNR of a signal's recent observations differs from its own baseline.
  The baseline is the median and median absolute deviation (MAD) of its older observations, both of
  which ignore the odd past burst, so one burst does not widen what counts as normal. A flag needs a real
  shift (robust z-score and a minimum number of dB) in the *median* of the last `recent_n`
  observations, so a single blip never counts: it takes a majority of them.
* ``gone_quiet``: a signal that used to appear regularly has not been seen for much longer than its
  usual gaps, counting only time when its frequency was actually being scanned.
* ``new_in_quiet``: a signal first seen recently where nothing else had ever been seen nearby, in a
  part of the spectrum that had been scanned long enough to make that meaningful.

"Scanned" comes from the `scans` table (one row per scan session with its frequency ranges). Without
it, scanning a different range would make every signal look new or silent.

Powers are relative dB from hackrf_sweep; SNR (relative to the local noise floor) is used rather than
raw power, so a change of gain between sessions does not look like a change in the signal.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

KINDS = ("louder", "quieter", "gone_quiet", "new_in_quiet")
SEVERITY = {"louder": "alert", "quieter": "notice", "gone_quiet": "notice", "new_in_quiet": "alert"}


@dataclass
class BaselineConfig:
    min_obs: int = 8  # observations needed before "normal" means anything
    max_obs: int = 200  # newest observations that make up the baseline
    recent_n: int = 5  # the latest observations; their median must be beyond the limit (so at least 3 of 5 are)
    z_threshold: float = 3.0  # robust z-score limit
    min_delta_db: float = 3.0  # and at least this many dB
    min_scale_db: float = 1.5  # floor for the spread, since SNR values are coarse
    active_window_s: float = 900.0  # louder/quieter are judged only for signals seen this recently
    quiet_min_s: float = 300.0  # gone_quiet: never flag a silence shorter than this
    quiet_gap_factor: float = 5.0  # ...or shorter than this many times the signal's longest usual gap
    gap_cap_s: float = 3600.0  # gaps longer than this are breaks between sessions, not habits
    memory_s: float = 7 * 86400.0  # gone_quiet looks at signals last seen within this long
    new_window_s: float = 3600.0  # new_in_quiet: first seen within this long of the reference time
    min_coverage_s: float = 1800.0  # ...and its frequency was scanned at least this long before that
    neighborhood_hz: float = 1e6  # "nothing nearby" means within this distance


# ------------------------------------------------------------------------------------------ helpers
def parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    text = str(ts)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromisoformat(text.split(".")[0])
        except ValueError:
            return None


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _robust_baseline(values: Sequence[float], cfg: BaselineConfig) -> Tuple[float, float, int]:
    """(median, spread, count) of `values`. The median and the median absolute deviation (scaled to match a
    standard deviation) shrug off a few past bursts, which is why no outlier trimming is needed."""
    med = statistics.median(values)
    scale = max(1.4826 * statistics.median(abs(v - med) for v in values), cfg.min_scale_db)
    return float(med), float(scale), len(values)


# ------------------------------------------------------------------------------------------ coverage
class Coverage:
    """When each frequency was being scanned, from the `scans` table."""

    def __init__(self, scans: List[Tuple[datetime, datetime, List[Tuple[float, float]]]]):
        self.scans = sorted(scans, key=lambda s: s[0])

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "Coverage":
        if not _has_table(conn, "scans"):
            return cls([])
        out = []
        for started, last, ranges in conn.execute("SELECT started_at, last_ts, ranges FROM scans"):
            a, b = parse_ts(started), parse_ts(last)
            try:
                rs = [(float(lo), float(hi)) for lo, hi in json.loads(ranges)]
            except (ValueError, TypeError):
                continue
            if a is not None and b is not None and b >= a and rs:
                out.append((a, b, rs))
        return cls(out)

    @property
    def latest_end(self) -> Optional[datetime]:
        return max((s[1] for s in self.scans), default=None)

    def total_seconds(self, until: Optional[datetime] = None) -> float:
        """Seconds of scanning of any range, up to `until` (or all of it)."""
        total = 0.0
        for a, b, _ in self.scans:
            end = b if until is None else min(b, until)
            if end > a:
                total += (end - a).total_seconds()
        return total

    def seconds(self, freq_hz: float, t0: Optional[datetime] = None, t1: Optional[datetime] = None) -> float:
        """Seconds within [t0, t1] (open-ended if None) during which `freq_hz` was inside a scanned range."""
        total = 0.0
        for a, b, ranges in self.scans:
            if not any(lo <= freq_hz <= hi for lo, hi in ranges):
                continue
            start = a if t0 is None else max(a, t0)
            end = b if t1 is None else min(b, t1)
            if end > start:
                total += (end - start).total_seconds()
        return total


def reference_time(conn: sqlite3.Connection, cov: Coverage) -> Optional[datetime]:
    """"Now" for the judgements: the last moment the scanner was looking. Using the wall clock would make
    every signal look silent as soon as the scanner is stopped."""
    ref = cov.latest_end
    if ref is not None:
        return ref
    for sql in ("SELECT MAX(ts) FROM sweeps", "SELECT MAX(last_seen) FROM signals"):
        try:
            row = conn.execute(sql).fetchone()
        except sqlite3.Error:
            continue
        if row and row[0]:
            return parse_ts(row[0])
    return None


# ------------------------------------------------------------------------------------------ per-signal data
def _newest_observations(conn: sqlite3.Connection, signal_filter: str, params: list, limit: int, with_snr: bool) -> Dict[int, List[Tuple[str, Any]]]:
    """The newest `limit` observations of every signal selected by `signal_filter` (SQL on `signals s`),
    as (ts, snr_db) pairs, oldest first within each signal. One query for all signals."""
    sql = (
        "SELECT signal_id, ts, val FROM ("
        " SELECT o.signal_id AS signal_id, o.ts AS ts, " + ("o.snr_db" if with_snr else "NULL") + " AS val,"
        "        ROW_NUMBER() OVER (PARTITION BY o.signal_id ORDER BY o.ts DESC, o.id DESC) AS rn"
        " FROM observations o WHERE o.signal_id IN (SELECT s.id FROM signals s WHERE " + signal_filter + ")"
        ") WHERE rn <= ? ORDER BY signal_id, ts"
    )
    out: Dict[int, List[Tuple[str, Any]]] = {}
    for sid, ts, val in conn.execute(sql, params + [limit]):
        out.setdefault(sid, []).append((ts, val))
    return out


def _ids_clause(ids: Optional[Sequence[int]]) -> Tuple[str, list]:
    if ids is None:
        return "", []
    ids = [int(i) for i in ids][:500]
    if not ids:
        return " AND 0", []
    return " AND s.id IN (" + ",".join("?" * len(ids)) + ")", ids


def _signal_info(conn: sqlite3.Connection, where: str, params: list) -> Dict[int, Dict[str, Any]]:
    rows = conn.execute(
        f"SELECT s.id, s.center_hz, s.bandwidth_hz, s.ident_name, s.first_seen, s.last_seen FROM signals s WHERE {where}", params
    ).fetchall()
    return {r[0]: {"signal_id": r[0], "center_hz": r[1], "bandwidth_hz": r[2], "ident_name": r[3], "first_seen": r[4], "last_seen": r[5]} for r in rows}


def _flag(info: Dict[str, Any], kind: str, message: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "signal_id": info["signal_id"], "center_hz": info["center_hz"], "bandwidth_hz": info["bandwidth_hz"],
        "ident_name": info["ident_name"], "kind": kind, "severity": SEVERITY[kind], "message": message, "details": details,
    }


# ------------------------------------------------------------------------------------------ louder / quieter
def judge_level(snrs: Sequence[float], cfg: BaselineConfig) -> Optional[Dict[str, Any]]:
    """Compare the newest `recent_n` SNR values with the older ones. Returns details if they differ
    persistently, else None."""
    if len(snrs) < cfg.recent_n + cfg.min_obs:
        return None
    base, recent = list(snrs[: -cfg.recent_n]), list(snrs[-cfg.recent_n :])
    med, scale, n_base = _robust_baseline(base, cfg)
    rmed = float(statistics.median(recent))
    shift = rmed - med
    if abs(shift) < cfg.min_delta_db or abs(shift) / scale < cfg.z_threshold:
        return None
    # |shift| is the median of the last `recent_n` minus the baseline, so passing both limits already means a
    # majority of those observations are beyond them: that is the persistence gate. `over` is only reported.
    sign = 1 if shift > 0 else -1
    limit = max(cfg.min_delta_db, cfg.z_threshold * scale)
    over = sum(1 for v in recent if (v - med) * sign >= limit)
    return {
        "baseline_db": round(med, 2), "recent_db": round(rmed, 2), "shift_db": round(shift, 2), "z": round(shift / scale, 2),
        "spread_db": round(scale, 2), "recent_over": over, "recent_n": cfg.recent_n, "baseline_n": n_base,
    }


def _level_flags(conn: sqlite3.Connection, ref: datetime, cfg: BaselineConfig, ids: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    id_sql, id_params = _ids_clause(ids)
    where = "s.last_seen >= ? AND s.last_seen <= ?" + id_sql
    params = [_iso(ref - timedelta(seconds=cfg.active_window_s)), _iso(ref + timedelta(seconds=1))] + id_params
    info = _signal_info(conn, where, params)
    if not info:
        return []
    obs = _newest_observations(conn, where, params, cfg.recent_n + cfg.max_obs, True)
    flags = []
    for sid, rows in obs.items():
        snrs = [float(v) for _, v in rows if v is not None]
        d = judge_level(snrs, cfg)
        if d is None:
            continue
        kind = "louder" if d["shift_db"] > 0 else "quieter"
        word = "louder" if kind == "louder" else "quieter"
        msg = (f"{abs(d['shift_db']):.1f} dB {word} than usual: SNR {d['recent_db']:.1f} dB now against a baseline of "
               f"{d['baseline_db']:.1f} dB ({d['recent_over']} of the last {d['recent_n']} observations)")
        flags.append(_flag(info[sid], kind, msg, d))
    return flags


# ------------------------------------------------------------------------------------------ gone quiet
def _quiet_flags(conn: sqlite3.Connection, ref: datetime, cov: Coverage, cfg: BaselineConfig, ids: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    id_sql, id_params = _ids_clause(ids)
    where = "s.last_seen < ? AND s.last_seen >= ? AND s.hits >= ?" + id_sql
    params = [_iso(ref - timedelta(seconds=cfg.quiet_min_s)), _iso(ref - timedelta(seconds=cfg.memory_s)), cfg.min_obs] + id_params
    info = _signal_info(conn, where, params)
    if not info:
        return []
    obs = _newest_observations(conn, where, params, cfg.max_obs, False)
    flags = []
    for sid, rows in obs.items():
        times = [t for t in (parse_ts(ts) for ts, _ in rows) if t is not None]
        if len(times) < cfg.min_obs:
            continue
        gaps = sorted(g for g in ((b - a).total_seconds() for a, b in zip(times, times[1:])) if 0 < g <= cfg.gap_cap_s)
        if len(gaps) < 5:
            continue  # too irregular, or too few sessions, to say what "usual" is
        usual = _quantile(gaps, 0.9)
        limit = max(cfg.quiet_min_s, cfg.quiet_gap_factor * usual)
        last_seen = parse_ts(info[sid]["last_seen"])
        if last_seen is None:
            continue
        silent = cov.seconds(info[sid]["center_hz"], last_seen, ref)
        if silent < limit:
            continue
        msg = f"not seen for {fmt_duration(silent)} of scanning; it normally shows up at least every {fmt_duration(usual)}"
        flags.append(_flag(info[sid], "gone_quiet", msg, {
            "silent_s": round(silent), "usual_gap_s": round(usual), "limit_s": round(limit), "last_seen": info[sid]["last_seen"],
        }))
    return flags


# ------------------------------------------------------------------------------------------ new in a quiet area
def _new_flags(conn: sqlite3.Connection, ref: datetime, cov: Coverage, cfg: BaselineConfig, ids: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    cutoff = ref - timedelta(seconds=cfg.new_window_s)
    id_sql, id_params = _ids_clause(ids)
    where = "s.first_seen >= ? AND s.first_seen <= ?" + id_sql
    info = _signal_info(conn, where, [_iso(cutoff), _iso(ref + timedelta(seconds=1))] + id_params)
    flags = []
    for sid, i in info.items():
        covered = cov.seconds(i["center_hz"], None, cutoff)
        if covered < cfg.min_coverage_s:
            continue  # this part of the spectrum was not watched long enough to call anything new
        reach = max(cfg.neighborhood_hz, i["bandwidth_hz"] or 0.0)
        near = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE first_seen < ? AND center_hz BETWEEN ? AND ? AND id != ?",
            (_iso(cutoff), i["center_hz"] - reach, i["center_hz"] + reach, sid),
        ).fetchone()[0]
        if near:
            continue
        msg = (f"new signal in a quiet area: nothing else was ever seen within {reach / 1e6:g} MHz in "
               f"{fmt_duration(covered)} of earlier scanning here")
        flags.append(_flag(i, "new_in_quiet", msg, {"covered_s": round(covered), "radius_hz": reach, "first_seen": i["first_seen"]}))
    return flags


# ------------------------------------------------------------------------------------------ entry points
def evaluate(
    conn: sqlite3.Connection,
    ref: Optional[datetime] = None,
    cfg: Optional[BaselineConfig] = None,
    kinds: Optional[Iterable[str]] = None,
    signal_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Current flags, as of the last moment the scanner was looking (or `ref`)."""
    cfg = cfg or BaselineConfig()
    want = set(kinds) if kinds else set(KINDS)
    unknown = want - set(KINDS)
    if unknown:
        raise ValueError(f"unknown kind(s): {', '.join(sorted(unknown))} (use {', '.join(KINDS)})")
    cov = Coverage.load(conn)
    ref = ref or reference_time(conn, cov)
    # "new" is judged against what was seen before the last `new_window_s`, so that is the history that counts
    history = cov.total_seconds(ref - timedelta(seconds=cfg.new_window_s)) if ref else 0.0
    out: Dict[str, Any] = {
        "ref": _iso(ref) if ref else None,
        "flags": [],
        "learning": {"history_s": round(history), "needed_s": round(cfg.min_coverage_s), "new_alerts_ready": history >= cfg.min_coverage_s},
    }
    if ref is None:
        return out
    flags: List[Dict[str, Any]] = []
    if want & {"louder", "quieter"}:
        flags += [f for f in _level_flags(conn, ref, cfg, signal_ids) if f["kind"] in want]
    if "gone_quiet" in want:
        flags += _quiet_flags(conn, ref, cov, cfg, signal_ids)
    if "new_in_quiet" in want:
        flags += _new_flags(conn, ref, cov, cfg, signal_ids)
    flags.sort(key=lambda f: (0 if f["severity"] == "alert" else 1, -abs(f["details"].get("z", 0)), f["signal_id"]))
    out["flags"] = flags
    return out


def signal_baseline(conn: sqlite3.Connection, signal_id: int, cfg: Optional[BaselineConfig] = None) -> Optional[Dict[str, Any]]:
    """What is normal for one signal, for its detail view."""
    cfg = cfg or BaselineConfig()
    row = conn.execute("SELECT id FROM signals WHERE id=?", (signal_id,)).fetchone()
    if row is None:
        return None
    rows = conn.execute(
        "SELECT ts, snr_db FROM observations WHERE signal_id=? AND snr_db IS NOT NULL ORDER BY ts DESC, id DESC LIMIT ?",
        (signal_id, cfg.recent_n + cfg.max_obs),
    ).fetchall()
    rows.reverse()
    snrs = [float(r[1]) for r in rows]
    out: Dict[str, Any] = {"observations": len(snrs), "needed": cfg.recent_n + cfg.min_obs, "learning": len(snrs) < cfg.recent_n + cfg.min_obs}
    if not out["learning"]:
        base, recent = snrs[: -cfg.recent_n], snrs[-cfg.recent_n :]
        med, scale, n_kept = _robust_baseline(base, cfg)
        ordered = sorted(base)
        out["baseline"] = {
            "median_db": round(med, 2), "spread_db": round(scale, 2), "p10_db": round(_quantile(ordered, 0.1), 2),
            "p90_db": round(_quantile(ordered, 0.9), 2), "n": n_kept,
        }
        out["recent"] = {"values": [round(v, 1) for v in recent], "median_db": round(float(statistics.median(recent)), 2)}
    times = [t for t in (parse_ts(r[0]) for r in rows) if t is not None]
    gaps = sorted(g for g in ((b - a).total_seconds() for a, b in zip(times, times[1:])) if 0 < g <= cfg.gap_cap_s)
    if len(gaps) >= 5:
        out["gaps"] = {"median_s": round(statistics.median(gaps)), "p90_s": round(_quantile(gaps, 0.9)), "n": len(gaps)}
    out["flags"] = evaluate(conn, cfg=cfg, signal_ids=[signal_id])["flags"]
    return out
