"""Tests for the web interface and its building blocks: bands, read-only queries, the HackRF lock,
scanner start/stop (with fake hackrf_* tools) and the HTTP API."""

import asyncio
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hackrf_scout import bands as bandlib  # noqa: E402
from hackrf_scout import controller, query, simulate  # noqa: E402
from hackrf_scout import store as store_mod  # noqa: E402
from hackrf_scout.store import Store  # noqa: E402

try:
    from fastapi.testclient import TestClient

    from hackrf_scout import webapp

    HAVE_WEB = True
except ImportError:  # the web extra (fastapi, uvicorn, httpx) is optional
    HAVE_WEB = False

TS = "2026-10-09T10:00:00"


def make_db(path):
    """A small database with known signals, observations, log rows and a sweep counter."""
    st = Store(path)

    def add(center_mhz, bw_khz, hits=5, snr=20.0, name=None, source="artemis", score=80.0):
        sid = st.insert_signal(
            center_hz=center_mhz * 1e6, bandwidth_hz=bw_khz * 1e3, peak_hz=center_mhz * 1e6, first_seen=TS,
            last_seen="2026-10-09T10:05:00", first_sweep=1, hits=hits, max_db=-50.0, avg_db=-55.0, last_db=-52.0,
            max_snr=snr, last_snr=snr - 1,
        )
        if name:
            cand = {"name": name, "score": score, "broad": False, "modulations": ["FSK"], "url": "https://www.sigidwiki.com/wiki/X"}
            st.set_ident(sid, {"name": name, "score": score, "source": source, "url": None, "service": name, "candidates": [cand]}, TS)
        return sid

    ids = {
        "lora": add(868.3, 125, hits=9, snr=22, name="LoRa"),
        "sensor": add(433.92, 40, hits=4, snr=18, name="Weather sensor"),
        "wifi_edge": add(2395.0, 20000, hits=6, snr=30),  # 2385-2405 MHz: centre below 2.4 GHz, edge inside it
        "wifi": add(2437.0, 18000, hits=8, snr=28, name="Wi-Fi"),
        "fm": add(98.1, 180, hits=20, snr=38, name="FM broadcast"),
        "odd": add(150.0, 12.5, hits=1, snr=11, name="100% legit_name"),
    }
    for t, snr in (("2026-10-09T10:00:10", 20.0), ("2026-10-09T10:30:00", 22.0), ("2026-10-09T11:05:00", 21.0)):
        st.add_observation(ids["lora"], t, 868.3e6, 125e3, -50.0, snr)
    for _ in range(30):
        st.bump_sweep()
    st.add_sweep_summary(TS, 1000, 3, -72.0)
    st.log("first line", "info", "scan")
    st.log("careful", "warn", "scan")
    st.log("broken", "error", "hackrf_sweep")
    st.commit()
    st.close()
    return ids


def wait_for(fn, timeout=20.0, step=0.1, msg="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError(f"timed out waiting for {msg}")


class BandTests(unittest.TestCase):
    def setUp(self):
        self.reg = bandlib.Registry()

    def test_presets_and_aliases(self):
        b = self.reg.resolve("868")
        self.assertEqual((b.lo_hz, b.hi_hz), (863e6, 870e6))
        self.assertEqual(self.reg.resolve("868MHz").key, "868")
        self.assertEqual(self.reg.resolve(" 2.4GHz ").key, "2.4ghz")
        self.assertEqual(self.reg.resolve("2.4").key, "2.4ghz")
        self.assertEqual(self.reg.resolve("UHF").key, "uhf")

    def test_ranges(self):
        b = self.reg.resolve("430:440")
        self.assertEqual((b.lo_hz, b.hi_hz, b.key), (430e6, 440e6, "430:440"))
        for bad in ("440:430", "5:5", "abc", "", "1:2:3", "-5:10"):
            with self.assertRaises(ValueError, msg=bad):
                self.reg.resolve(bad)

    def test_resolve_many_dedupes_and_splits_commas(self):
        self.assertEqual([b.key for b in self.reg.resolve_many(["868,433", "868", "430:440"])], ["868", "433", "430:440"])

    def test_sweep_range_rounds_outwards(self):
        self.assertEqual(self.reg.resolve("433").sweep_range(), "433:435")
        self.assertEqual(self.reg.resolve("868").sweep_range(), "863:870")
        self.assertEqual(self.reg.resolve("pmr446").sweep_range(), "446:447")
        self.assertEqual(self.reg.resolve("2.4ghz").sweep_range(), "2400:2484")

    def test_merge_sweep_ranges(self):
        merged = bandlib.merge_sweep_ranges(self.reg.resolve_many(["433", "434:440", "868"]))
        self.assertEqual(merged, ["433:440", "863:870"])
        with self.assertRaises(ValueError):
            bandlib.merge_sweep_ranges([self.reg.resolve("7000:7100")])

    def test_every_preset_is_sane_and_scannable(self):
        keys = [b.key for b in bandlib.PRESETS]
        self.assertEqual(len(keys), len(set(keys)))
        for b in bandlib.PRESETS:
            self.assertLess(b.lo_hz, b.hi_hz, b.key)
            self.assertTrue(b.to_dict()["sweepable"], b.key)

    def test_custom_bands_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bands.json")
            with open(p, "w") as fh:
                json.dump({"bands": [
                    {"key": "garage", "name": "Garage", "lo_mhz": 433, "hi_mhz": 435},
                    {"key": "BAD KEY", "lo_mhz": 1, "hi_mhz": 2},
                    {"key": "inverted", "lo_mhz": 9, "hi_mhz": 3},
                    {"key": "868", "name": "My 868", "lo_mhz": 868, "hi_mhz": 869},
                    "nonsense",
                ]}, fh)
            extra, warnings = bandlib.load_custom(p)
            self.assertEqual([b.key for b in extra], ["garage", "868"])
            self.assertEqual(len(warnings), 3)
            reg = bandlib.Registry(extra)
            self.assertEqual(reg.resolve("garage").hi_hz, 435e6)
            self.assertEqual(reg.resolve("868").name, "My 868")  # user definitions win
            self.assertEqual(bandlib.load_custom(os.path.join(d, "missing.json"))[0], [])

    def test_sql_overlap_vs_centre(self):
        con = sqlite3.connect(":memory:")
        self.addCleanup(con.close)
        con.execute("CREATE TABLE signals(id INTEGER, center_hz REAL, bandwidth_hz REAL)")
        con.executemany("INSERT INTO signals VALUES(?,?,?)", [(1, 2395e6, 20e6), (2, 2437e6, 18e6), (3, 2300e6, 1e6)])
        band = [self.reg.resolve("2.4ghz")]
        margin = 10e6
        sql, params = bandlib.signal_clause(band, "overlap", margin)
        self.assertEqual([r[0] for r in con.execute(f"SELECT id FROM signals s WHERE {sql} ORDER BY id", params)], [1, 2])
        sql, params = bandlib.signal_clause(band, "center", margin)
        self.assertEqual([r[0] for r in con.execute(f"SELECT id FROM signals s WHERE {sql} ORDER BY id", params)], [2])
        with self.assertRaises(ValueError):
            bandlib.signal_clause(band, "weird")


class StoreLogTests(unittest.TestCase):
    def test_log_from_another_thread_is_written_on_commit(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "s.db"))
            t = threading.Thread(target=lambda: st.log("from thread", "warn", "hackrf_sweep"))
            t.start()
            t.join()
            self.assertEqual(st.conn.execute("SELECT COUNT(*) FROM log").fetchone()[0], 0)  # queued, not written
            st.commit()
            row = st.conn.execute("SELECT level, source, msg FROM log").fetchone()
            self.assertEqual(tuple(row), ("warn", "hackrf_sweep", "from thread"))
            st.close()

    def test_log_is_pruned(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(store_mod, "LOG_KEEP", 10), mock.patch.object(store_mod, "LOG_PRUNE_EVERY", 5):
            st = Store(os.path.join(d, "s.db"))
            for i in range(40):
                st.log(f"line {i}")
                st.commit()
            n = st.conn.execute("SELECT COUNT(*) FROM log").fetchone()[0]
            self.assertLessEqual(n, 15)
            self.assertEqual(st.conn.execute("SELECT msg FROM log ORDER BY id DESC LIMIT 1").fetchone()[0], "line 39")
            st.close()

    def test_old_database_gets_the_log_table(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.db")
            con = sqlite3.connect(p)
            con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
            con.commit()
            con.close()
            st = Store(p)
            st.log("hello")
            st.commit()
            self.assertEqual(st.conn.execute("SELECT COUNT(*) FROM log").fetchone()[0], 1)
            st.close()


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "q.db")
        self.ids = make_db(self.path)
        self.conn = query.connect_ro(self.path)
        self.addCleanup(self.conn.close)
        self.reg = bandlib.Registry()

    def bands(self, *specs):
        return self.reg.resolve_many(specs)

    def test_connection_is_read_only(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute("DELETE FROM signals")
        with self.assertRaises(query.DatabaseUnavailable):
            query.connect_ro(os.path.join(self.tmp.name, "nope.db"))
        junk = os.path.join(self.tmp.name, "junk.db")
        with open(junk, "wb") as fh:
            fh.write(b"this is not a database" * 50)
        with self.assertRaises(query.DatabaseUnavailable):
            query.connect_ro(junk)

    def test_band_filters(self):
        def ids(**kw):
            return {i["id"] for i in query.list_signals(self.conn, **kw)["items"]}

        self.assertEqual(ids(bands=self.bands("868")), {self.ids["lora"]})
        self.assertEqual(ids(bands=self.bands("868", "433")), {self.ids["lora"], self.ids["sensor"]})
        # the 20 MHz signal centred at 2395 MHz touches the 2.4 GHz band; centre mode ignores it
        self.assertEqual(ids(bands=self.bands("2.4ghz")), {self.ids["wifi"], self.ids["wifi_edge"]})
        self.assertEqual(ids(bands=self.bands("2.4ghz"), mode="center"), {self.ids["wifi"]})
        self.assertEqual(ids(bands=self.bands("100:200")), {self.ids["odd"]})
        self.assertEqual(len(ids()), 6)

    def test_other_filters(self):
        lst = lambda **kw: query.list_signals(self.conn, **kw)  # noqa: E731
        self.assertEqual([i["id"] for i in lst(unidentified=True)["items"]], [self.ids["wifi_edge"]])
        self.assertEqual(lst(min_hits=8)["total"], 3)
        self.assertEqual(lst(min_snr=30)["total"], 2)
        self.assertEqual(lst(q="lora")["total"], 1)
        self.assertEqual(lst(q=f"{self.ids['fm']}")["items"][0]["ident_name"], "FM broadcast")

    def test_search_treats_wildcards_literally(self):
        self.assertEqual(query.list_signals(self.conn, q="%")["total"], 1)
        self.assertEqual(query.list_signals(self.conn, q="_")["total"], 1)
        self.assertEqual(query.list_signals(self.conn, q="x' OR '1'='1")["total"], 0)

    def test_sorting_and_paging(self):
        r = query.list_signals(self.conn, sort="hits", page=1, page_size=2)
        self.assertEqual((r["total"], len(r["items"]), r["order"]), (6, 2, "desc"))
        self.assertEqual(r["items"][0]["hits"], 20)
        r2 = query.list_signals(self.conn, sort="hits", page=3, page_size=2)
        self.assertEqual([i["hits"] for i in r2["items"]], [4, 1])
        self.assertEqual(query.list_signals(self.conn, sort="center_hz")["items"][0]["center_hz"], 98.1e6)
        for bad in ("hits; DROP TABLE signals", "nope", "ident_json"):
            with self.assertRaises(ValueError):
                query.list_signals(self.conn, sort=bad)
        with self.assertRaises(ValueError):
            query.list_signals(self.conn, order="sideways")
        # duty cycle: 9 hits since sweep 1 of 30 sweeps
        lora = query.list_signals(self.conn, bands=self.bands("868"))["items"][0]
        self.assertEqual(lora["duty_pct"], 30.0)

    def test_signal_detail(self):
        d = query.signal_detail(self.conn, self.ids["lora"], self.reg)
        self.assertEqual(d["candidates"][0]["name"], "LoRa")
        self.assertIn("868", d["bands"])
        self.assertIn("uhf", d["bands"])
        self.assertEqual(d["observation_count"], 3)
        self.assertEqual([o["ts"][11:16] for o in d["observations"]], ["11:05", "10:30", "10:00"])  # newest first
        self.assertTrue(any("868" in s for s in d["services"]))
        self.assertIsNone(query.signal_detail(self.conn, 9999, self.reg))

    def test_observations_by_band(self):
        self.assertEqual(query.list_observations(self.conn, bands=self.bands("868"))["total"], 3)
        self.assertEqual(query.list_observations(self.conn, bands=self.bands("433"))["total"], 0)
        self.assertEqual(query.list_observations(self.conn, signal_id=self.ids["lora"], since="2026-10-09T10:20")["total"], 2)

    def test_band_summary_and_activity(self):
        s = {b["key"]: b for b in query.band_summary(self.conn, self.reg.all())}
        self.assertEqual((s["868"]["signals"], s["868"]["unidentified"]), (1, 0))
        self.assertEqual(s["868"]["strongest"]["id"], self.ids["lora"])
        self.assertEqual((s["2.4ghz"]["signals"], s["2.4ghz"]["unidentified"]), (2, 1))
        self.assertEqual(s["5.8ghz"]["signals"], 0)
        self.assertEqual(s["vhf"]["signals"], 2)  # FM + 150 MHz; the 433 MHz sensor is UHF
        act = query.band_activity(self.conn, self.reg.resolve("868"), "hour")
        self.assertEqual([(a["t"], a["observations"]) for a in act], [("2026-10-09T10", 2), ("2026-10-09T11", 1)])
        with self.assertRaises(ValueError):
            query.band_activity(self.conn, self.reg.resolve("868"), "century")

    def test_raw_table_browser_is_whitelisted(self):
        self.assertEqual({t["name"] for t in query.table_counts(self.conn)}, {"signals", "observations", "sweeps", "captures", "log", "meta"})
        r = query.browse_table(self.conn, "log", page_size=2)
        self.assertEqual((r["total"], len(r["rows"]), r["columns"][0]), (3, 2, "id"))
        for name in ("sqlite_master", "signals; DROP TABLE x", "../etc"):
            with self.assertRaises(ValueError):
                query.browse_table(self.conn, name)
        for sort in ("nope", 'id"; DROP TABLE log; --'):
            with self.assertRaises(ValueError):
                query.browse_table(self.conn, "log", sort=sort)

    def test_log_reading(self):
        r = query.read_log(self.conn, tail=2)
        self.assertEqual([i["msg"] for i in r["items"]], ["careful", "broken"])
        self.assertEqual(r["last_id"], 3)
        self.assertEqual([i["msg"] for i in query.read_log(self.conn, after_id=1)["items"]], ["careful", "broken"])
        self.assertEqual(query.read_log(self.conn, after_id=3)["items"], [])

    def test_missing_log_table_is_not_an_error(self):
        p = os.path.join(self.tmp.name, "legacy.db")
        con = sqlite3.connect(p)
        con.execute("CREATE TABLE signals(id INTEGER PRIMARY KEY)")
        con.commit()
        con.close()
        legacy = query.connect_ro(p)
        self.addCleanup(legacy.close)
        self.assertEqual(query.read_log(legacy, tail=5)["items"], [])
        self.assertEqual(query.status(legacy, p)["counts"]["log"], 0)

    def test_non_finite_numbers_become_null(self):
        w = sqlite3.connect(self.path)
        w.execute("UPDATE signals SET max_db = 1e999 WHERE id = ?", (self.ids["fm"],))
        w.commit()
        w.close()
        conn = query.connect_ro(self.path)
        self.addCleanup(conn.close)
        item = [i for i in query.list_signals(conn)["items"] if i["id"] == self.ids["fm"]][0]
        self.assertIsNone(item["max_db"])
        json.dumps(item, allow_nan=False)  # must be valid JSON

    def test_export(self):
        csv_text = "".join(query.export_signals(self.conn, "csv", bands=self.bands("868", "433")))
        lines = csv_text.strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("id,center_hz"))
        data = json.loads("".join(query.export_signals(self.conn, "json", with_observations=True, bands=self.bands("868"))))
        self.assertEqual(len(data[0]["observations"]), 3)
        self.assertEqual(data[0]["candidates"][0]["name"], "LoRa")
        with self.assertRaises(ValueError):
            list(query.export_signals(self.conn, "xml"))


@unittest.skipUnless(controller.fcntl is not None, "hardware lock needs POSIX flock")
class HardwareLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def test_single_owner(self):
        self.assertFalse(controller.read_status(self.dir)["running"])
        lock = controller.HardwareLock(self.dir).acquire(mode="scan", argv=["scan"])
        st = controller.read_status(self.dir)
        self.assertTrue(st["running"])
        self.assertEqual((st["pid"], st["mode"], st["starting"]), (os.getpid(), "scan", False))
        with self.assertRaises(controller.HardwareBusy) as cm:
            controller.HardwareLock(self.dir).acquire(mode="capture")
        self.assertIn(str(os.getpid()), str(cm.exception))
        lock.update(phase="capturing")
        self.assertEqual(controller.read_status(self.dir)["phase"], "capturing")
        lock.release()
        self.assertFalse(controller.read_status(self.dir)["running"])
        self.assertFalse(os.path.exists(os.path.join(self.dir, "scanner.json")))
        controller.HardwareLock(self.dir).acquire().release()  # free again

    def test_lock_dies_with_the_process(self):
        code = ("import sys, time; sys.path.insert(0, %r); from hackrf_scout import controller; "
                "controller.HardwareLock(%r).acquire(mode='scan'); print('held', flush=True); time.sleep(60)") % (ROOT, self.dir)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        self.addCleanup(p.stdout.close)
        self.addCleanup(p.kill)
        self.assertEqual(p.stdout.readline().strip(), "held")
        self.assertTrue(controller.read_status(self.dir)["running"])
        p.kill()  # no chance to clean up: scanner.json stays behind
        p.wait()
        self.assertTrue(os.path.exists(os.path.join(self.dir, "scanner.json")))
        self.assertFalse(controller.read_status(self.dir)["running"])
        controller.HardwareLock(self.dir).acquire().release()

    def test_held_lock_without_info_reads_as_starting(self):
        lock = controller.HardwareLock(self.dir).acquire()
        os.unlink(os.path.join(self.dir, "scanner.json"))
        st = controller.read_status(self.dir)
        self.assertEqual((st["running"], st["starting"]), (True, True))
        lock.release()


class BuildArgsTests(unittest.TestCase):
    def build(self, params, **kw):
        return controller.build_scan_args(params, db="/data/scout.db", **kw)

    def pairs(self, argv):
        return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("-")}

    def test_defaults(self):
        a = self.build({})
        self.assertEqual(a[:3], [sys.executable, "-m", "hackrf_scout"])
        self.assertEqual(a[3], "scan")
        self.assertIn("--quiet", a)
        self.assertNotIn("-f", a)  # whole range
        self.assertEqual(self.pairs(a)["--db"], "/data/scout.db")
        self.assertEqual(self.pairs(a)["-l"], "24")
        self.assertTrue(all(isinstance(x, str) for x in a))

    def test_bands_become_merged_sweep_ranges(self):
        a = self.build({"bands": ["433", "434:440", "868"], "amp": True, "lna": 32, "vga": 40, "snr": 12.5, "duration": 90})
        fs = [a[i + 1] for i, x in enumerate(a) if x == "-f"]
        self.assertEqual(fs, ["433:440", "863:870"])
        self.assertIn("-a", a)
        self.assertEqual(self.pairs(a)["--snr"], "12.5")
        self.assertEqual(self.pairs(a)["--duration"], "90")

    def test_run_mode(self):
        a = self.build({"mode": "run", "scan_seconds": 30, "capture_seconds": 2, "capture_which": "new"},
                       capture_dir="caps", hackrf_transfer="/opt/ht")
        p = self.pairs(a)
        self.assertEqual((a[3], p["--scan-seconds"], p["--capture-which"], p["--hackrf-transfer"]), ("run", "30", "new", "/opt/ht"))
        self.assertTrue(os.path.isabs(p["--capture-dir"]))

    def test_server_side_settings_are_not_overridable(self):
        a = self.build({}, hackrf_sweep="/opt/hs")
        self.assertEqual(self.pairs(a)["--hackrf-sweep"], "/opt/hs")
        for key in ("db", "hackrf_sweep", "hackrf_transfer", "capture_dir", "signals", "ignore", "--db", "extra_args"):
            with self.assertRaises(ValueError, msg=key):
                self.build({key: "/etc/passwd"})

    def test_run_only_keys_rejected_in_scan_mode(self):
        with self.assertRaises(ValueError):
            self.build({"capture_seconds": 5})
        self.build({"mode": "run", "capture_seconds": 5})

    def test_invalid_values(self):
        bad = [
            {"mode": "capture"}, {"lna": 7}, {"lna": 48}, {"lna": True}, {"lna": "abc"}, {"vga": 3}, {"vga": -2}, {"snr": 0},
            {"snr": float("nan")}, {"min_hits": 0}, {"min_hits": 2.5}, {"amp": "yes"}, {"bin_width": 10},
            {"region_keywords": "a;rm -rf /"}, {"region_keywords": "x" * 200}, {"bands": "868"}, {"bands": [868]},
            {"bands": ["nonsense"]}, {"bands": ["7000:7100"]}, {"bands": ["1:2"] * 30}, {"duration": 0.2}, {"duration": 10 ** 9},
        ]
        for params in bad:
            with self.assertRaises(ValueError, msg=str(params)):
                self.build(params)
        with self.assertRaises(ValueError):
            self.build({"mode": "run", "capture_which": "everything"})
        with self.assertRaises(ValueError):
            controller.build_scan_args([], db="x")

    def test_strings_stay_single_arguments(self):
        a = self.build({"region_keywords": "greece cyprus"})
        self.assertEqual(self.pairs(a)["--region-keywords"], "greece cyprus")

    def test_custom_registry_bands(self):
        reg = bandlib.Registry([bandlib.Band("garage", "Garage", 433e6, 435e6)])
        a = self.build({"bands": ["garage"]}, registry=reg)
        self.assertEqual(self.pairs(a)["-f"], "433:435")


FAKE_SWEEP_LOOP = """#!@PY@
import itertools, sys, time
from datetime import datetime
sys.path.insert(0, "@ROOT@")
from hackrf_scout import simulate
a = sys.argv
ranges = [tuple(float(x) for x in a[i + 1].split(":")) for i, v in enumerate(a) if v == "-f"] or [(1.0, 3000.0)]
lo, hi = min(r[0] for r in ranges), min(max(r[1] for r in ranges), 3000.0)
sys.stderr.write("call hackrf_set_sample_rate(20000000 Hz)\\n")
sys.stderr.flush()
for i in itertools.count():
    for line in simulate.generate(lo, hi, 100e3, 1, seed=i, t0=datetime.now()):
        sys.stdout.write(line)
    sys.stdout.flush()
    sys.stderr.write("%d total sweeps completed\\n" % (i + 1))
    sys.stderr.flush()
    time.sleep(0.2)
"""
FAKE_SWEEP_BROKEN = """#!@PY@
import sys
sys.stderr.write("hackrf_open() failed: HACKRF_ERROR_NOT_FOUND (-5)\\n")
sys.exit(1)
"""
FAKE_SWEEP_RECORD = """#!@PY@
import json, sys
sys.path.insert(0, "@ROOT@")
with open(sys.argv[0] + ".args", "w") as fh:
    json.dump(sys.argv[1:], fh)
from hackrf_scout import simulate
sys.exit(simulate.main(sys.argv[1:]))
"""


class FakeTools(unittest.TestCase):
    """Base class: a temp dir with fake hackrf_* executables, a state dir and a database path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = self.tmp.name
        self.bin = os.path.join(t, "bin")
        os.mkdir(self.bin)
        self.state = os.path.join(t, "state")
        self.db = os.path.join(t, "scout.db")
        self.env = dict(os.environ, PYTHONPATH=ROOT, HACKRF_SCOUT_STATE_DIR=self.state)

    def script(self, name, body):
        p = os.path.join(self.bin, name)
        with open(p, "w") as fh:
            fh.write(body.replace("@PY@", sys.executable).replace("@ROOT@", ROOT))
        os.chmod(p, 0o755)
        return p

    def cli(self, *args, check=False):
        r = subprocess.run([sys.executable, "-m", "hackrf_scout", *args], capture_output=True, text=True, env=self.env, cwd=self.tmp.name, timeout=60)
        if check:
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r


@unittest.skipUnless(controller.fcntl is not None, "scanner control needs POSIX")
class ScannerControlE2E(FakeTools):
    def setUp(self):
        super().setUp()
        self.sweep = self.script("hackrf_sweep", FAKE_SWEEP_LOOP)
        self.script("hackrf_info", "#!/bin/sh\necho 'Found HackRF'\n")
        self.ctl = controller.ScannerController(
            self.db, directory=self.state, hackrf_sweep=self.sweep, capture_dir=os.path.join(self.tmp.name, "caps"),
            startup_wait=20, stop_wait=15, term_wait=5,
        )
        self.addCleanup(self.stop_quietly)

    def stop_quietly(self):
        try:
            self.ctl.stop()
        except Exception:
            pass

    def log_rows(self):
        try:
            conn = query.connect_ro(self.db)
        except query.DatabaseUnavailable:
            return []
        try:
            return query.read_log(conn, tail=2000)["items"]
        except sqlite3.Error:
            return []
        finally:
            conn.close()

    def messages(self):
        return [r["msg"] for r in self.log_rows()]

    def test_start_log_and_stop(self):
        st = self.ctl.start({"bands": ["868"], "min_hits": 1})
        self.assertEqual(st["state"], "running")
        self.assertTrue(st["managed"])
        self.assertEqual(st["mode"], "scan")
        wait_for(lambda: any("total sweeps completed" in m for m in self.messages()), msg="hackrf_sweep stderr in the log")
        wait_for(lambda: any(m.startswith("NEW") for m in self.messages()), 30, msg="a signal to be found")
        with self.assertRaises(controller.ControlError) as cm:
            self.ctl.start({})
        self.assertEqual(cm.exception.status, 409)

        st = self.ctl.stop()
        self.assertEqual(st["state"], "stopped")
        self.assertEqual(st["last_exit"]["code"], 0)
        msgs = self.messages()
        self.assertIn("Stopping ...", msgs)
        self.assertTrue(any(m.startswith("scan ended") for m in msgs))
        self.assertFalse(controller.read_status(self.state)["running"])
        self.assertEqual(self.ctl.stop()["state"], "stopped")  # stopping a stopped scanner is fine
        conn = query.connect_ro(self.db)
        try:
            centers = [i["center_hz"] for i in query.list_signals(conn)["items"]]
        finally:
            conn.close()
        self.assertTrue(centers and all(863e6 <= c <= 870e6 for c in centers), centers)

    def test_hardware_failure_is_reported(self):
        self.ctl.hackrf_sweep = self.script("hackrf_sweep_broken", FAKE_SWEEP_BROKEN)
        try:
            self.ctl.start({})
        except controller.ControlError as exc:  # it may already have exited by the time start() looked
            self.assertEqual(exc.status, 500)
            self.assertIn("code 2", str(exc))
        st = wait_for(lambda: (lambda s: s if s["state"] == "stopped" else None)(self.ctl.status()), msg="scanner to exit")
        self.assertEqual(st["last_exit"]["code"], 2)
        self.assertIn("No sweep data", st["last_exit"]["output_tail"])
        rows = self.log_rows()
        self.assertTrue(any(r["level"] == "error" and "HACKRF_ERROR_NOT_FOUND" in r["msg"] for r in rows), rows)

    def test_stop_ends_run_mode_instead_of_starting_the_next_cycle(self):
        self.ctl.start({"mode": "run", "scan_seconds": 5, "capture_seconds": 0, "bands": ["868"]})
        wait_for(lambda: any(m.startswith("[cycle 1]") for m in self.messages()), msg="first cycle")
        wait_for(lambda: any("total sweeps completed" in m for m in self.messages()), msg="sweeps")
        t0 = time.monotonic()
        st = self.ctl.stop()
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual((st["state"], st["last_exit"]["code"]), ("stopped", 0))
        msgs = self.messages()
        self.assertIn("Stopped.", msgs)
        self.assertFalse(any(m.startswith("[cycle 2]") for m in msgs))

    def test_cli_respects_the_lock_but_replay_does_not_need_it(self):
        self.ctl.start({"bands": ["868"]})
        busy = self.cli("scan", "--db", os.path.join(self.tmp.name, "other.db"), "--sweeps", "1", "--hackrf-sweep", self.sweep)
        self.assertEqual(busy.returncode, 2)
        self.assertIn("already in use", busy.stderr)
        busy = self.cli("capture", "--db", self.db)
        self.assertEqual(busy.returncode, 2)
        sim = os.path.join(self.tmp.name, "sim.csv")
        with open(sim, "w") as fh:
            fh.writelines(simulate.generate(1, 1000, 100e3, 4, seed=1))
        replay = self.cli("scan", "--source", sim, "--db", os.path.join(self.tmp.name, "replay.db"), "--quiet")
        self.assertEqual(replay.returncode, 0, replay.stderr)

    def test_scanner_started_elsewhere_with_sigint_ignored_can_still_be_stopped(self):
        # a scanner launched in the background by a shell inherits "ignore SIGINT"; it must still stop cleanly
        p = subprocess.Popen(
            [sys.executable, "-m", "hackrf_scout", "scan", "--db", self.db, "--quiet", "-f", "863:870", "--hackrf-sweep", self.sweep],
            env=self.env, cwd=self.tmp.name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )
        self.addCleanup(p.kill)
        st = wait_for(lambda: (lambda s: s if s["state"] == "running" else None)(self.ctl.status()), msg="scanner to start")
        self.assertFalse(st["managed"])  # not started by this controller
        self.assertEqual(st["pid"], p.pid)
        self.assertEqual(self.ctl.stop()["state"], "stopped")
        self.assertEqual(p.wait(timeout=10), 0)

    def test_check_device(self):
        with mock.patch.dict(os.environ, {"PATH": self.bin + os.pathsep + os.environ["PATH"]}):
            r = self.ctl.check_device()
            self.assertEqual((r["ok"], r["busy"]), (True, False))
            self.assertIn("Found HackRF", r["output"])
            self.ctl.start({"bands": ["868"]})
            self.assertTrue(self.ctl.check_device()["busy"])
        self.ctl.stop()
        with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}):
            with self.assertRaises(controller.ControlError) as cm:
                self.ctl.check_device()
            self.assertEqual(cm.exception.status, 404)


class CliBandTests(FakeTools):
    def test_scan_accepts_band_names(self):
        sweep = self.script("hackrf_sweep", FAKE_SWEEP_RECORD)
        r = self.cli("scan", "--db", self.db, "-f", "868", "-f", "pmr446", "-f", "400:410", "--sweeps", "2", "--quiet", "--hackrf-sweep", sweep)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(sweep + ".args") as fh:
            args = json.load(fh)
        self.assertEqual([args[i + 1] for i, x in enumerate(args) if x == "-f"], ["863:870", "446:447", "400:410"])

    def test_unknown_band_is_a_clean_error(self):
        sweep = self.script("hackrf_sweep", FAKE_SWEEP_RECORD)
        r = self.cli("scan", "--db", self.db, "-f", "nonsense", "--sweeps", "1", "--hackrf-sweep", sweep)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown band", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_report_band_names(self):
        make_db(self.db)
        out = self.cli("report", "--db", self.db, "--band", "868", check=True).stdout
        self.assertIn("868.300", out)
        self.assertNotIn("433.920", out)
        self.assertIn("2437.000", self.cli("report", "--db", self.db, "--band", "2.4ghz", check=True).stdout)
        self.assertEqual(self.cli("report", "--db", self.db, "--band", "bogus").returncode, 2)


class Stub:
    """Stands in for ScannerController in HTTP tests, so they spawn no processes."""

    def __init__(self):
        self.calls = []
        self.error = None

    def status(self):
        return {"supported": True, "state": "stopped", "managed": False}

    def start(self, params):
        self.calls.append(("start", params))
        if self.error:
            raise self.error
        return {"supported": True, "state": "running", "managed": True, "pid": 123}

    def stop(self):
        self.calls.append(("stop", None))
        return self.status()

    def check_device(self):
        self.calls.append(("check", None))
        return {"ok": True, "busy": False, "output": "Found HackRF"}


@unittest.skipUnless(HAVE_WEB, "needs the web extra: pip install 'hackrf-scout[web]' httpx")
class WebApiTests(unittest.TestCase):
    TOKEN = "s3cret-token"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "w.db")
        self.ids = make_db(self.db)
        self.stub = Stub()

    def client(self, **kw):
        cfg = webapp.WebConfig(db_path=kw.pop("db", self.db), controller=self.stub, **kw)
        return TestClient(webapp.create_app(cfg), base_url="http://localhost")

    def auth(self):
        return {"Authorization": f"Bearer {self.TOKEN}"}

    # ---- pages and headers
    def test_page_headers_and_assets(self):
        c = self.client()
        r = c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("hackrf-scout", r.text)
        csp = r.headers["content-security-policy"]
        self.assertIn("default-src 'none'", csp)
        self.assertNotIn("unsafe", csp)
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["x-frame-options"], "DENY")
        for path, kind in (("/static/app.js", "javascript"), ("/static/style.css", "css")):
            a = c.get(path)
            self.assertEqual(a.status_code, 200, path)
            self.assertIn(kind, a.headers["content-type"])
        self.assertEqual(c.get("/api/status").headers["cache-control"], "no-store")
        for hidden in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(c.get(hidden).status_code, 404, hidden)

    def test_no_inline_script_or_style_in_the_page(self):
        html = self.client().get("/").text
        self.assertNotIn("<script>", html)
        self.assertNotIn(" style=", html)
        self.assertNotIn("onclick=", html)

    # ---- auth and request checks
    def test_token_is_enforced_on_the_api_only(self):
        c = self.client(token=self.TOKEN)
        self.assertEqual(c.get("/").status_code, 200)
        r = c.get("/api/status")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.headers["www-authenticate"], "Bearer")
        self.assertEqual(c.get("/api/signals").status_code, 401)
        self.assertEqual(c.get("/api/status", headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(c.get("/api/status", headers={"Authorization": f"Basic {self.TOKEN}"}).status_code, 401)
        self.assertEqual(c.get("/api/status", headers=self.auth()).status_code, 200)

    def test_host_header_must_be_local_when_bound_to_loopback(self):
        c = self.client()
        self.assertEqual(c.get("/api/status", headers={"host": "evil.example"}).status_code, 403)  # DNS rebinding
        self.assertEqual(c.get("/api/status", headers={"host": "127.0.0.1:8765"}).status_code, 200)
        self.assertEqual(c.get("/api/status", headers={"host": "[::1]:8765"}).status_code, 200)
        c2 = self.client(allowed_hosts={"scout.lan"})
        self.assertEqual(c2.get("/api/status", headers={"host": "scout.lan:8765"}).status_code, 200)

    def test_network_binding_does_not_pin_the_host_header(self):
        c = self.client(host="0.0.0.0", token=self.TOKEN)
        self.assertEqual(c.get("/api/status", headers={**self.auth(), "host": "192.168.1.20:8765"}).status_code, 200)

    def test_cross_origin_posts_are_rejected(self):
        c = self.client(allow_control=True, token=self.TOKEN)
        r = c.post("/api/scanner/stop", headers={**self.auth(), "Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.stub.calls, [])
        ok = c.post("/api/scanner/stop", headers={**self.auth(), "Origin": "http://localhost"})
        self.assertEqual(ok.status_code, 200)

    # ---- read API
    def test_status(self):
        s = self.client().get("/api/status").json()
        self.assertEqual(s["counts"], {"signals": 6, "observations": 3, "sweeps": 1, "captures": 0, "log": 3})
        self.assertEqual((s["sweep_count"], s["db_exists"], s["control_enabled"], s["token_required"]), (30, True, False, False))
        self.assertEqual(s["scanner"]["state"], "stopped")

    def test_signals_endpoint(self):
        c = self.client()
        r = c.get("/api/signals", params={"band": ["868", "433"], "sort": "center_hz"}).json()
        self.assertEqual([i["id"] for i in r["items"]], [self.ids["sensor"], self.ids["lora"]])
        self.assertEqual(c.get("/api/signals", params={"band": "2.4ghz"}).json()["total"], 2)
        self.assertEqual(c.get("/api/signals", params={"band": "2.4ghz", "mode": "center"}).json()["total"], 1)
        self.assertEqual(c.get("/api/signals", params={"unidentified": "true"}).json()["total"], 1)
        self.assertEqual(c.get("/api/signals", params={"band": "430:440,868"}).json()["total"], 2)
        self.assertEqual(c.get("/api/signals", params={"page": 2, "page_size": 4}).json()["items"].__len__(), 2)

    def test_bad_input_gives_400_or_422_not_500(self):
        c = self.client()
        for params in ({"band": "nonsense"}, {"sort": "x; DROP TABLE signals"}, {"order": "up"}, {"mode": "sideways"}, {"band": "5:1"}):
            self.assertEqual(c.get("/api/signals", params=params).status_code, 400, params)
        for params in ({"page": 0}, {"page_size": 501}, {"min_hits": -1}, {"min_snr": "abc"}):
            self.assertEqual(c.get("/api/signals", params=params).status_code, 422, params)
        self.assertEqual(c.get("/api/tables/sqlite_master").status_code, 400)
        self.assertEqual(c.get("/api/tables/log", params={"sort": "id; DROP TABLE log"}).status_code, 400)
        self.assertEqual(c.get("/api/bands/activity", params={"band": "868", "bucket": "year"}).status_code, 400)
        self.assertEqual(c.get("/api/signals/abc").status_code, 422)

    def test_signal_detail_and_missing(self):
        c = self.client()
        d = c.get(f"/api/signals/{self.ids['lora']}").json()
        self.assertEqual(d["candidates"][0]["name"], "LoRa")
        self.assertEqual(d["observation_count"], 3)
        self.assertEqual(c.get("/api/signals/99999").status_code, 404)

    def test_export_downloads(self):
        c = self.client()
        r = c.get("/api/signals/export", params={"format": "csv", "band": "868"})
        self.assertIn("text/csv", r.headers["content-type"])
        self.assertIn('filename="signals.csv"', r.headers["content-disposition"])
        self.assertEqual(len(r.text.strip().splitlines()), 2)
        j = c.get("/api/signals/export", params={"format": "json", "observations": "true"}).json()
        self.assertEqual(len(j), 6)
        self.assertEqual(c.get("/api/signals/export", params={"format": "pdf"}).status_code, 400)

    def test_bands_endpoints(self):
        c = self.client()
        names = [b["key"] for b in c.get("/api/bands").json()["bands"]]
        self.assertIn("868", names)
        self.assertIn("2.4ghz", names)
        summ = {b["key"]: b for b in c.get("/api/bands/summary").json()["bands"]}
        self.assertEqual(summ["868"]["signals"], 1)
        only = c.get("/api/bands/summary", params={"band": ["433", "868"]}).json()["bands"]
        self.assertEqual([b["key"] for b in only], ["433", "868"])
        act = c.get("/api/bands/activity", params={"band": "868", "bucket": "hour"}).json()
        self.assertEqual(sum(i["observations"] for i in act["items"]), 3)

    def test_tables_log_and_observations(self):
        c = self.client()
        self.assertIn("signals", [t["name"] for t in c.get("/api/tables").json()["tables"]])
        t = c.get("/api/tables/signals", params={"sort": "hits", "page_size": 2}).json()
        self.assertEqual((t["total"], len(t["rows"]), t["sort"]), (6, 2, "hits"))
        self.assertEqual([i["msg"] for i in c.get("/api/log", params={"tail": 2}).json()["items"]], ["careful", "broken"])
        self.assertEqual([i["msg"] for i in c.get("/api/log", params={"after": 2}).json()["items"]], ["broken"])
        self.assertEqual(c.get("/api/observations", params={"band": "868"}).json()["total"], 3)
        self.assertEqual(c.get("/api/sweeps").json()["total"], 1)
        self.assertEqual(c.get("/api/captures").json()["total"], 0)

    def test_missing_database(self):
        c = self.client(db=os.path.join(self.tmp.name, "absent.db"))
        s = c.get("/api/status").json()
        self.assertFalse(s["db_exists"])
        self.assertEqual(c.get("/api/log").json()["items"], [])
        r = c.get("/api/signals")
        self.assertEqual(r.status_code, 503)
        self.assertIn("not found", r.json()["detail"])

    # ---- scanner control
    def test_control_is_off_by_default(self):
        c = self.client()
        for path in ("start", "stop", "check"):
            r = c.post(f"/api/scanner/{path}", json={})
            self.assertEqual(r.status_code, 403, path)
            self.assertIn("--allow-control", r.json()["detail"])
        self.assertEqual(self.stub.calls, [])
        r = c.get("/api/scanner").json()
        self.assertEqual((r["state"], r["control_enabled"]), ("stopped", False))

    def test_control_start_stop_check(self):
        c = self.client(allow_control=True, token=self.TOKEN)
        self.assertEqual(c.post("/api/scanner/start", json={}).status_code, 401)  # token still needed
        r = c.post("/api/scanner/start", json={"bands": ["868"], "snr": 12}, headers=self.auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"], "running")
        self.assertEqual(self.stub.calls[-1], ("start", {"bands": ["868"], "snr": 12}))
        self.assertEqual(c.post("/api/scanner/stop", headers=self.auth()).json()["state"], "stopped")
        self.assertEqual(c.post("/api/scanner/check", headers=self.auth()).json()["output"], "Found HackRF")

    def test_control_errors_map_to_http_status(self):
        c = self.client(allow_control=True)
        self.stub.error = controller.ControlError("a scanner is already running", 409)
        r = c.post("/api/scanner/start", json={})
        self.assertEqual((r.status_code, r.json()["detail"]), (409, "a scanner is already running"))
        self.stub.error = controller.ControlError("mode must be 'scan' or 'run'", 400)
        self.assertEqual(c.post("/api/scanner/start", json={"mode": "x"}).status_code, 400)

    # ---- live log stream
    def test_log_stream_delivers_new_rows_and_heartbeats(self):
        last = 3

        async def scenario():
            gen = webapp.log_event_stream(self.db, last, poll=0.05, heartbeat=0.3)
            self.assertTrue((await asyncio.wait_for(gen.__anext__(), 2)).startswith("retry:"))
            st = Store(self.db)
            st.log("fresh line", "warn", "test")
            st.commit()
            st.close()
            chunk = await asyncio.wait_for(gen.__anext__(), 3)
            self.assertTrue(chunk.startswith("id: 4\ndata: "), chunk)
            self.assertEqual(json.loads(chunk.split("data: ", 1)[1])["msg"], "fresh line")
            beat = await asyncio.wait_for(gen.__anext__(), 3)
            self.assertTrue(beat.startswith(":"), beat)
            await gen.aclose()

        asyncio.run(scenario())

    def test_log_stream_survives_a_reset_database(self):
        items, last = webapp._poll_log(self.db, 500)  # client remembers id 500, the log only goes to 3
        self.assertEqual((items, last), ([], 3))
        self.assertEqual(webapp._poll_log(os.path.join(self.tmp.name, "gone.db"), 7), ([], 7))

    def test_log_stream_endpoint_resumes_from_last_event_id(self):
        # the generator is infinite, so call the route's helper path through the app with a stub generator
        seen = {}

        async def fake_stream(db_path, after, **kw):
            seen["after"] = after
            yield "retry: 1\n\n"

        with mock.patch.object(webapp, "log_event_stream", fake_stream):
            c = self.client()
            with c.stream("GET", "/api/log/stream", params={"after": 1}) as r:
                self.assertEqual(r.headers["content-type"], "text/event-stream; charset=utf-8")
                list(r.iter_lines())
            self.assertEqual(seen["after"], 1)
            with c.stream("GET", "/api/log/stream", params={"after": 1}, headers={"Last-Event-ID": "2"}) as r:
                list(r.iter_lines())
            self.assertEqual(seen["after"], 2)
            with c.stream("GET", "/api/log/stream") as r:  # nothing given: follow from now
                list(r.iter_lines())
            self.assertEqual(seen["after"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
