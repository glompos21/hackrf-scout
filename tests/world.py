"""Builds small databases with exact timestamps for the baseline and alert tests."""

import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hackrf_scout.store import Store  # noqa: E402

T0 = datetime(2026, 10, 9, 8, 0, 0)
EVERYTHING = [(1.0, 6000.0)]


def at(minutes=0.0, seconds=0.0):
    """ISO timestamp `minutes` after T0 (the format the scanner writes)."""
    return (T0 + timedelta(minutes=minutes, seconds=seconds)).isoformat(timespec="seconds")


def wobble(n, level=20.0, amp=0.4):
    """n SNR values around `level`, alternating +-amp, so the spread is small but not zero."""
    return [level + (amp if i % 2 else -amp) for i in range(n)]


class World:
    def __init__(self, path):
        self.path = path
        self.store = Store(path)

    def close(self):
        self.store.conn.commit()
        self.store.conn.close()

    def signal(self, center_mhz, bw_khz=200.0, first=0.0, last=None, hits=50, snr=20.0):
        last = first if last is None else last
        return self.store.insert_signal(
            center_hz=center_mhz * 1e6, bandwidth_hz=bw_khz * 1e3, peak_hz=center_mhz * 1e6, first_seen=at(first), last_seen=at(last),
            first_sweep=1, hits=hits, max_db=-50.0, avg_db=-55.0, last_db=-52.0, max_snr=snr, last_snr=snr,
        )

    def observations(self, sid, values, start=0.0, step_s=30.0):
        """One observation per value, `step_s` apart from `start` (minutes); returns the time of the last one (minutes)."""
        row = self.store.conn.execute("SELECT center_hz, bandwidth_hz FROM signals WHERE id=?", (sid,)).fetchone()
        for i, v in enumerate(values):
            self.store.conn.execute(
                "INSERT INTO observations(signal_id,ts,center_hz,bandwidth_hz,peak_db,snr_db) VALUES(?,?,?,?,?,?)",
                (sid, at(start, i * step_s), row[0], row[1], -50.0, v),
            )
        end = start + (len(values) - 1) * step_s / 60.0
        self.store.conn.execute("UPDATE signals SET last_seen=? WHERE id=?", (at(end), sid))
        return end

    def seen(self, center_mhz, values, start=0.0, step_s=30.0, **kw):
        """A signal plus its observations in one go."""
        sid = self.signal(center_mhz, first=start, **kw)
        self.observations(sid, values, start, step_s)
        return sid

    def scan(self, start, end, ranges=EVERYTHING):
        """A scan session from `start` to `end` minutes after T0."""
        sid = self.store.begin_scan(ranges, "scan", at(start))
        self.store.conn.execute("UPDATE scans SET last_ts=? WHERE id=?", (at(end), sid))
        self.store.scan_id = None
        return sid

    def commit(self):
        self.store.conn.commit()
