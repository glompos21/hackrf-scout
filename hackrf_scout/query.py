"""Read-only queries behind the web interface.

Everything here takes an open read-only connection (`connect_ro`) and plain parameters, builds
SQL only from fixed fragments plus `?` placeholders (sort columns and table names are looked up
in whitelists), and returns JSON-friendly dicts. No FastAPI import, so it is easy to test.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from . import bandplan
from . import bands as bandlib
from . import baseline

TABLES = ("signals", "observations", "sweeps", "captures", "scans", "log", "meta")
MAX_PAGE_SIZE = 500
MAX_ID_FILTER = 500  # most flagged signals one list request will filter on

SIGNAL_SORTS = {
    "id", "center_hz", "bandwidth_hz", "first_seen", "last_seen", "hits", "max_db", "max_snr", "last_snr",
    "avg_db", "ident_name", "ident_score",
}
DEFAULT_DESC = {"hits", "max_snr", "last_snr", "last_seen", "max_db", "avg_db", "ident_score"}
EXPORT_COLS = ["id", "center_hz", "bandwidth_hz", "first_seen", "last_seen", "hits", "max_db", "avg_db", "max_snr",
               "ident_name", "ident_score", "ident_source", "ident_url", "service", "label", "notes"]
LIST_COLS = [c for c in (
    "id", "center_hz", "bandwidth_hz", "peak_hz", "first_seen", "last_seen", "first_sweep", "hits", "max_db", "avg_db",
    "last_db", "max_snr", "last_snr", "ident_name", "ident_score", "ident_source", "ident_url", "service", "label",
    "notes", "captured")]


class DatabaseUnavailable(Exception):
    pass


def connect_ro(path: str) -> sqlite3.Connection:
    p = Path(path)
    if not p.is_file():
        raise DatabaseUnavailable(f"database not found: {path} (start a scan to create it)")
    conn = None
    try:
        conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")  # fails here, not later, if it is not a database
    except sqlite3.DatabaseError as exc:
        if conn is not None:
            conn.close()
        raise DatabaseUnavailable(f"cannot open {path} read-only: {exc}")
    return conn


# ---------------------------------------------------------------------------------- helpers
def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    return v


def _row(row: sqlite3.Row, cols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    keys = cols if cols is not None else row.keys()
    return {k: _clean(row[k]) for k in keys}


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _page(page: int, page_size: int) -> tuple:
    page = max(1, int(page))
    page_size = min(MAX_PAGE_SIZE, max(1, int(page_size)))
    return page, page_size, (page - 1) * page_size


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _margin(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT COALESCE(MAX(bandwidth_hz), 0) / 2.0 FROM signals").fetchone()
    return float(row[0] or 0.0)


def _meta_int(conn: sqlite3.Connection, key: str) -> int:
    if not _has_table(conn, "meta"):
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    try:
        return int(row[0]) if row else 0
    except (TypeError, ValueError):
        return 0


def signal_where(
    conn: sqlite3.Connection,
    bands: Sequence[bandlib.Band] = (),
    mode: str = "overlap",
    unidentified: bool = False,
    min_hits: int = 0,
    min_snr: Optional[float] = None,
    q: Optional[str] = None,
    captured: Optional[bool] = None,
    ids: Optional[Sequence[int]] = None,
) -> tuple:
    clause, params = bandlib.signal_clause(bands, mode, _margin(conn) if bands else 0.0)
    where = [clause, "s.hits >= ?"]
    params.append(int(min_hits))
    if unidentified:
        where.append("s.ident_name IS NULL")
    if min_snr is not None:
        where.append("s.max_snr >= ?")
        params.append(float(min_snr))
    if captured is not None:
        where.append("s.captured = ?")
        params.append(1 if captured else 0)
    if ids is not None:
        ids = [int(i) for i in ids][:MAX_ID_FILTER]
        where.append("s.id IN (" + ",".join("?" * len(ids)) + ")" if ids else "0")
        params += ids
    if q and q.strip():
        text = q.strip()[:80]
        like = _like(text)
        parts = ["s.ident_name LIKE ? ESCAPE '\\'", "s.label LIKE ? ESCAPE '\\'", "s.notes LIKE ? ESCAPE '\\'", "s.service LIKE ? ESCAPE '\\'"]
        params += [like, like, like, like]
        if text.isdigit():
            parts.append("s.id = ?")
            params.append(int(text))
        where.append("(" + " OR ".join(parts) + ")")
    return " AND ".join(where), params


# ---------------------------------------------------------------------------------- status
def status(conn: Optional[sqlite3.Connection], db_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"db": os.path.abspath(db_path), "db_exists": conn is not None}
    if conn is None:
        return out
    try:
        out["db_bytes"] = os.path.getsize(db_path)
    except OSError:
        pass
    counts = {}
    for t in ("signals", "observations", "sweeps", "captures", "log"):
        counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if _has_table(conn, t) else 0
    out["counts"] = counts
    out["sweep_count"] = _meta_int(conn, "sweep_count")
    if counts["sweeps"]:
        out["last_sweep_at"] = conn.execute("SELECT MAX(ts) FROM sweeps").fetchone()[0]
    if counts["log"]:
        row = conn.execute("SELECT id, ts FROM log ORDER BY id DESC LIMIT 1").fetchone()
        out["last_log_id"], out["last_log_at"] = row["id"], row["ts"]
    if counts["signals"]:
        out["last_signal_at"] = conn.execute("SELECT MAX(last_seen) FROM signals").fetchone()[0]
    return out


# ---------------------------------------------------------------------------------- signals
def _signal_item(row: sqlite3.Row, sweeps: int) -> Dict[str, Any]:
    d = _row(row, LIST_COLS)
    d["duty_pct"] = round(min(100.0, 100.0 * row["hits"] / max(1, sweeps - row["first_sweep"] + 1)), 1)
    return d


def list_signals(
    conn: sqlite3.Connection,
    bands: Sequence[bandlib.Band] = (),
    mode: str = "overlap",
    unidentified: bool = False,
    min_hits: int = 0,
    min_snr: Optional[float] = None,
    q: Optional[str] = None,
    captured: Optional[bool] = None,
    sort: str = "center_hz",
    order: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    ids: Optional[Sequence[int]] = None,
    flags: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    """`ids` limits the list to those signals; `flags` ({signal id: [kinds]}) is attached to each item."""
    if sort not in SIGNAL_SORTS:
        raise ValueError(f"sort must be one of {sorted(SIGNAL_SORTS)}")
    if order is None:
        order = "desc" if sort in DEFAULT_DESC else "asc"
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    page, page_size, offset = _page(page, page_size)
    where, params = signal_where(conn, bands, mode, unidentified, min_hits, min_snr, q, captured, ids)
    total = conn.execute(f"SELECT COUNT(*) FROM signals s WHERE {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT s.* FROM signals s WHERE {where} ORDER BY s.{sort} {order.upper()}, s.id ASC LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()
    sweeps = _meta_int(conn, "sweep_count")
    items = [_signal_item(r, sweeps) for r in rows]
    if flags is not None:
        for it in items:
            it["flags"] = flags.get(it["id"], [])
    return {"total": total, "page": page, "page_size": page_size, "sort": sort, "order": order, "items": items}


def signal_detail(conn: sqlite3.Connection, signal_id: int, registry: bandlib.Registry, obs_limit: int = 300) -> Optional[Dict[str, Any]]:
    r = conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()
    if r is None:
        return None
    d = _signal_item(r, _meta_int(conn, "sweep_count"))
    cands: List[Any] = []
    if r["ident_json"]:
        try:
            cands = json.loads(r["ident_json"])
        except ValueError:
            cands = []
    d["candidates"] = cands
    d["ident_at"] = r["ident_at"]
    d["services"] = bandplan.lookup(r["center_hz"], r["bandwidth_hz"])
    lo, hi = r["center_hz"] - r["bandwidth_hz"] / 2.0, r["center_hz"] + r["bandwidth_hz"] / 2.0
    d["bands"] = [b.key for b in registry.all() if b.lo_hz <= hi and b.hi_hz >= lo]
    d["observation_count"] = conn.execute("SELECT COUNT(*) FROM observations WHERE signal_id=?", (signal_id,)).fetchone()[0]
    obs = conn.execute("SELECT * FROM observations WHERE signal_id=? ORDER BY ts DESC, id DESC LIMIT ?", (signal_id, obs_limit)).fetchall()
    d["observations"] = [_row(o) for o in obs]
    d["captures"] = [_row(c) for c in conn.execute("SELECT * FROM captures WHERE signal_id=? ORDER BY ts DESC", (signal_id,)).fetchall()]
    d["baseline"] = baseline.signal_baseline(conn, signal_id)
    return d


def export_signals(conn: sqlite3.Connection, fmt: str, with_observations: bool = False, **filters: Any) -> Iterator[str]:
    """Yield the filtered signals as CSV or JSON text, same columns as `hackrf-scout export`."""
    where, params = signal_where(conn, **filters)
    rows = conn.execute(f"SELECT s.* FROM signals s WHERE {where} ORDER BY s.center_hz", params).fetchall()
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLS)
        for r in rows:
            w.writerow([_clean(r[c]) for c in EXPORT_COLS])
        yield buf.getvalue()
        return
    if fmt != "json":
        raise ValueError("format must be csv or json")
    data = []
    for r in rows:
        d = _row(r, EXPORT_COLS)
        try:
            d["candidates"] = json.loads(r["ident_json"]) if r["ident_json"] else []
        except ValueError:
            d["candidates"] = []
        if with_observations:
            d["observations"] = [_row(o) for o in conn.execute("SELECT * FROM observations WHERE signal_id=? ORDER BY ts", (r["id"],))]
        data.append(d)
    yield json.dumps(data, indent=2)


# ---------------------------------------------------------------------------------- observations, sweeps, captures
def list_observations(
    conn: sqlite3.Connection,
    signal_id: Optional[int] = None,
    bands: Sequence[bandlib.Band] = (),
    mode: str = "overlap",
    since: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
    order: str = "desc",
) -> Dict[str, Any]:
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    page, page_size, offset = _page(page, page_size)
    where, params = ["1=1"], []
    if signal_id is not None:
        where.append("o.signal_id = ?")
        params.append(int(signal_id))
    if bands:
        clause, bparams = bandlib.signal_clause(bands, mode, _margin(conn))
        where.append(f"o.signal_id IN (SELECT s.id FROM signals s WHERE {clause})")
        params += bparams
    if since:
        where.append("o.ts >= ?")
        params.append(since)
    w = " AND ".join(where)
    total = conn.execute(f"SELECT COUNT(*) FROM observations o WHERE {w}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT o.* FROM observations o WHERE {w} ORDER BY o.ts {order.upper()}, o.id {order.upper()} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()
    return {"total": total, "page": page, "page_size": page_size, "items": [_row(r) for r in rows]}


def _simple_list(conn: sqlite3.Connection, table: str, page: int, page_size: int) -> Dict[str, Any]:
    page, page_size, offset = _page(page, page_size)
    if not _has_table(conn, table):
        return {"total": 0, "page": page, "page_size": page_size, "items": []}
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ? OFFSET ?", (page_size, offset)).fetchall()
    return {"total": total, "page": page, "page_size": page_size, "items": [_row(r) for r in rows]}


def list_sweeps(conn: sqlite3.Connection, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    return _simple_list(conn, "sweeps", page, page_size)


def list_captures(conn: sqlite3.Connection, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    return _simple_list(conn, "captures", page, page_size)


# ---------------------------------------------------------------------------------- anomalies
def anomalies(conn: sqlite3.Connection, kinds: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Baseline flags, plus how old the data they are based on is."""
    res = baseline.evaluate(conn, kinds=kinds)
    ref = baseline.parse_ts(res["ref"]) if res["ref"] else None
    res["age_s"] = max(0, round((datetime.now() - ref).total_seconds())) if ref else None
    return res


def flag_map(result: Dict[str, Any]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for f in result["flags"]:
        out.setdefault(f["signal_id"], []).append(f["kind"])
    return out


# ---------------------------------------------------------------------------------- bands
def band_summary(conn: sqlite3.Connection, bands: Sequence[bandlib.Band], mode: str = "overlap", min_hits: int = 0) -> List[Dict[str, Any]]:
    margin = _margin(conn)
    out = []
    for b in bands:
        clause, params = bandlib.signal_clause([b], mode, margin)
        w = f"{clause} AND s.hits >= ?"
        params = params + [int(min_hits)]
        agg = conn.execute(
            f"SELECT COUNT(*) AS n, COALESCE(SUM(s.ident_name IS NULL), 0) AS unidentified, MAX(s.max_snr) AS best_snr, "
            f"MAX(s.last_seen) AS last_seen FROM signals s WHERE {w}", params,
        ).fetchone()
        item = b.to_dict()
        item.update(signals=agg["n"], unidentified=agg["unidentified"], best_snr=_clean(agg["best_snr"]), last_seen=agg["last_seen"])
        if agg["n"]:
            top = conn.execute(
                f"SELECT s.id, s.center_hz, s.ident_name FROM signals s WHERE {w} ORDER BY s.max_snr DESC, s.id LIMIT 1", params
            ).fetchone()
            item["strongest"] = _row(top)
        out.append(item)
    return out


_BUCKETS = {"minute": 16, "hour": 13, "day": 10}


def band_activity(conn: sqlite3.Connection, band: bandlib.Band, bucket: str = "hour", limit: int = 72, mode: str = "overlap") -> List[Dict[str, Any]]:
    if bucket not in _BUCKETS:
        raise ValueError("bucket must be minute, hour or day")
    limit = min(1000, max(1, int(limit)))
    clause, params = bandlib.signal_clause([band], mode, _margin(conn))
    n = _BUCKETS[bucket]
    rows = conn.execute(
        f"SELECT substr(o.ts, 1, {n}) AS t, COUNT(*) AS observations, COUNT(DISTINCT o.signal_id) AS signals, MAX(o.snr_db) AS best_snr "
        f"FROM observations o WHERE o.signal_id IN (SELECT s.id FROM signals s WHERE {clause}) "
        f"GROUP BY t ORDER BY t DESC LIMIT ?", params + [limit],
    ).fetchall()
    return [_row(r) for r in reversed(rows)]


# ---------------------------------------------------------------------------------- raw tables
def table_counts(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    return [{"name": t, "rows": conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]} for t in TABLES if _has_table(conn, t)]


def browse_table(conn: sqlite3.Connection, name: str, sort: Optional[str] = None, order: str = "desc", page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    if name not in TABLES or not _has_table(conn, name):
        raise ValueError(f"unknown table '{name}'")
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({name})")]
    if sort is None:
        sort = "id" if "id" in cols else cols[0]
    if sort not in cols:
        raise ValueError(f"unknown column '{sort}'")
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    page, page_size, offset = _page(page, page_size)
    total = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    rows = conn.execute(f'SELECT * FROM {name} ORDER BY "{sort}" {order.upper()} LIMIT ? OFFSET ?', (page_size, offset)).fetchall()
    return {"table": name, "columns": cols, "total": total, "page": page, "page_size": page_size, "sort": sort, "order": order,
            "rows": [[_clean(r[c]) for c in cols] for r in rows]}


# ---------------------------------------------------------------------------------- log
def read_log(conn: sqlite3.Connection, after_id: Optional[int] = None, tail: Optional[int] = None, limit: int = 500) -> Dict[str, Any]:
    """Log lines oldest first. `tail=N` gives the last N; `after_id` gives the rows newer than that id."""
    limit = min(2000, max(1, int(limit)))
    if not _has_table(conn, "log"):
        return {"items": [], "last_id": 0}
    if after_id is not None:
        rows = conn.execute("SELECT * FROM log WHERE id > ? ORDER BY id LIMIT ?", (int(after_id), limit)).fetchall()
    else:
        n = min(limit, int(tail) if tail else 200)
        rows = list(reversed(conn.execute("SELECT * FROM log ORDER BY id DESC LIMIT ?", (n,)).fetchall()))
    last = conn.execute("SELECT COALESCE(MAX(id), 0) FROM log").fetchone()[0]
    items = [_row(r) for r in rows]
    return {"items": items, "last_id": items[-1]["id"] if items else (after_id or 0), "max_id": last}
