"""SQLite storage for signals, observations, sweeps and IQ captures."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  center_hz REAL NOT NULL,
  bandwidth_hz REAL NOT NULL,
  peak_hz REAL,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  first_sweep INTEGER NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0,
  max_db REAL, avg_db REAL, last_db REAL,
  max_snr REAL, last_snr REAL,
  ident_name TEXT, ident_score REAL, ident_source TEXT, ident_url TEXT,
  service TEXT, ident_json TEXT, ident_at TEXT,
  label TEXT, notes TEXT,
  captured INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_signals_center ON signals(center_hz);
CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_id INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
  ts TEXT NOT NULL,
  center_hz REAL, bandwidth_hz REAL, peak_db REAL, snr_db REAL
);
CREATE INDEX IF NOT EXISTS idx_obs_signal ON observations(signal_id, ts);
CREATE TABLE IF NOT EXISTS sweeps(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, bins INTEGER, detections INTEGER, floor_db REAL
);
CREATE TABLE IF NOT EXISTS captures(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_id INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
  ts TEXT NOT NULL, path TEXT NOT NULL,
  center_hz REAL, sample_rate REAL, seconds REAL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ---- meta / sweeps -------------------------------------------------
    def bump_sweep(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='sweep_count'").fetchone()
        n = (int(row["value"]) if row else 0) + 1
        self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('sweep_count',?)", (str(n),))
        return n

    def sweep_count(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='sweep_count'").fetchone()
        return int(row["value"]) if row else 0

    def add_sweep_summary(self, ts: str, bins: int, detections: int, floor_db: float) -> None:
        self.conn.execute("INSERT INTO sweeps(ts,bins,detections,floor_db) VALUES(?,?,?,?)", (ts, bins, detections, floor_db))

    # ---- signals -------------------------------------------------------
    def insert_signal(self, **f: Any) -> int:
        cur = self.conn.execute(
            """INSERT INTO signals(center_hz,bandwidth_hz,peak_hz,first_seen,last_seen,first_sweep,hits,
                                   max_db,avg_db,last_db,max_snr,last_snr)
               VALUES(:center_hz,:bandwidth_hz,:peak_hz,:first_seen,:last_seen,:first_sweep,:hits,
                      :max_db,:avg_db,:last_db,:max_snr,:last_snr)""",
            f,
        )
        return int(cur.lastrowid)

    def update_signal(self, **f: Any) -> None:
        self.conn.execute(
            """UPDATE signals SET center_hz=:center_hz, bandwidth_hz=:bandwidth_hz, peak_hz=:peak_hz,
                   last_seen=:last_seen, hits=:hits, max_db=:max_db, avg_db=:avg_db, last_db=:last_db,
                   max_snr=:max_snr, last_snr=:last_snr WHERE id=:id""",
            f,
        )

    def add_observation(self, signal_id: int, ts: str, center_hz: float, bw_hz: float, peak_db: float, snr_db: float) -> None:
        self.conn.execute(
            "INSERT INTO observations(signal_id,ts,center_hz,bandwidth_hz,peak_db,snr_db) VALUES(?,?,?,?,?,?)",
            (signal_id, ts, center_hz, bw_hz, peak_db, snr_db),
        )

    def set_ident(self, signal_id: int, ident: Dict[str, Any], ts: str) -> None:
        self.conn.execute(
            """UPDATE signals SET ident_name=?, ident_score=?, ident_source=?, ident_url=?, service=?,
                   ident_json=?, ident_at=? WHERE id=?""",
            (
                ident.get("name"),
                ident.get("score"),
                ident.get("source"),
                ident.get("url"),
                ident.get("service"),
                json.dumps(ident.get("candidates", [])),
                ts,
                signal_id,
            ),
        )

    def load_signals(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM signals ORDER BY center_hz").fetchall()

    def get_signal(self, signal_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM signals WHERE id=?", (signal_id,)).fetchone()

    def query_signals(
        self,
        unidentified: bool = False,
        min_hits: int = 0,
        order: str = "center_hz",
        limit: Optional[int] = None,
        band_hz: Optional[tuple] = None,
    ) -> List[sqlite3.Row]:
        allowed = {"center_hz", "hits", "max_snr", "last_seen", "first_seen", "bandwidth_hz", "id", "max_db"}
        if order not in allowed:
            raise ValueError(f"order must be one of {sorted(allowed)}")
        q = "SELECT * FROM signals WHERE hits>=?"
        args: List[Any] = [min_hits]
        if unidentified:
            q += " AND (ident_name IS NULL)"
        if band_hz:
            q += " AND center_hz BETWEEN ? AND ?"
            args += [band_hz[0], band_hz[1]]
        desc = order in {"hits", "max_snr", "last_seen", "max_db"}
        q += f" ORDER BY {order} {'DESC' if desc else 'ASC'}"
        if limit:
            q += " LIMIT ?"
            args.append(limit)
        return self.conn.execute(q, args).fetchall()

    def mark_captured(self, signal_id: int) -> None:
        self.conn.execute("UPDATE signals SET captured=1 WHERE id=?", (signal_id,))

    def add_capture(self, signal_id: int, ts: str, path: str, center_hz: float, rate: float, seconds: float) -> None:
        self.conn.execute(
            "INSERT INTO captures(signal_id,ts,path,center_hz,sample_rate,seconds) VALUES(?,?,?,?,?,?)",
            (signal_id, ts, path, center_hz, rate, seconds),
        )

    def observations(self, signal_id: Optional[int] = None) -> List[sqlite3.Row]:
        if signal_id is None:
            return self.conn.execute("SELECT * FROM observations ORDER BY ts").fetchall()
        return self.conn.execute("SELECT * FROM observations WHERE signal_id=? ORDER BY ts", (signal_id,)).fetchall()

    def commit(self) -> None:
        self.conn.commit()
