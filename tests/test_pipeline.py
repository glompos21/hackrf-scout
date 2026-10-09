"""End-to-end tests that need no HackRF: simulated sweeps and fake hackrf_* executables."""

import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hackrf_scout import artemisdb, simulate  # noqa: E402
from hackrf_scout.detect import detect_signals, estimate_floor  # noqa: E402
from hackrf_scout.identify import Identifier  # noqa: E402
from hackrf_scout.scanner import Scanner  # noqa: E402
from hackrf_scout.store import Store  # noqa: E402
from hackrf_scout.sweep import iter_sweeps  # noqa: E402

EXPECTED = [s[0] for s in simulate.DEFAULT_SIGNALS]


def _scan_sim(db_path, seed=1, sweeps=12, signals=None, start=1.0, stop=3000.0, ident=None):
    store = Store(db_path)
    sc = Scanner(store, ident, obs_interval_s=2, log=lambda m: None)
    lines = simulate.generate(start, stop, 100e3, sweeps, signals=signals, seed=seed)
    n = 0
    for sw in iter_sweeps(lines):
        sc.process(sw)
        n += 1
    rows = store.load_signals()
    store.close()
    return n, rows


class SweepParsing(unittest.TestCase):
    def test_interleaved_chunks_form_whole_sweeps(self):
        sweeps = list(iter_sweeps(simulate.generate(100, 200, 100e3, 3, signals=[])))
        self.assertEqual(len(sweeps), 3)
        for s in sweeps:
            self.assertEqual(s.freqs.size, 1000)
            self.assertTrue(np.all(np.diff(s.freqs) > 0))

    def test_narrow_scans_split_into_sweeps_too(self):
        # a span under 30 MHz never "wraps"; the repeated chunks are what mark the next sweep
        for a, b in ((863, 870), (433, 435)):
            sweeps = list(iter_sweeps(simulate.generate(a, b, 100e3, 4, signals=[])))
            self.assertEqual(len(sweeps), 4, (a, b))
            self.assertTrue(all(s.freqs.size == sweeps[0].freqs.size for s in sweeps))

    def test_narrow_scan_still_detects_signals(self):
        with tempfile.TemporaryDirectory() as d:
            _, rows = _scan_sim(os.path.join(d, "n.db"), sweeps=14, start=863.0, stop=870.0)
        self.assertEqual([round(r["center_hz"] / 1e6) for r in rows], [868])

    def test_garbage_lines_ignored(self):
        lines = ["# comment\n", "not,a,csv\n", "\n"] + list(simulate.generate(100, 120, 100e3, 1, signals=[]))
        self.assertEqual(len(list(iter_sweeps(lines))), 1)


class Detection(unittest.TestCase):
    def test_finds_all_known_signals_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as d:
            n, rows = _scan_sim(os.path.join(d, "t.db"))
        self.assertEqual(n, 12)
        centers = sorted(r["center_hz"] for r in rows)
        self.assertEqual(len(rows), len(EXPECTED), f"unexpected detections: {[c / 1e6 for c in centers]}")
        for exp in EXPECTED:
            nearest = min(centers, key=lambda c: abs(c - exp))
            tol = 150e3 if exp < 2e9 else 3e6
            self.assertLess(abs(nearest - exp), tol, f"{exp / 1e6} MHz not found")

    def test_noise_only_gives_no_signals(self):
        with tempfile.TemporaryDirectory() as d:
            _, rows = _scan_sim(os.path.join(d, "t.db"), signals=[], sweeps=15)
        self.assertEqual(rows, [])

    def test_rescan_updates_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            _, first = _scan_sim(p, seed=1)
            _, second = _scan_sim(p, seed=2)
        self.assertEqual(len(first), len(second))
        self.assertGreater(sum(r["hits"] for r in second), sum(r["hits"] for r in first))

    def test_wide_occupied_band_does_not_hide_signals(self):
        # eight adjacent 8 MHz channels (like DVB-T) = one 64 MHz block; floor must not rise into it
        sigs = [(474e6 + 8e6 * k + 4e6, 8e6, -45.0, 1.0) for k in range(8)]
        sweeps = list(iter_sweeps(simulate.generate(400, 800, 100e3, 1, signals=sigs, seed=3)))
        dets, floor = detect_signals(sweeps[0].freqs, sweeps[0].power, sweeps[0].bin_hz)
        self.assertEqual(len(dets), 1)
        self.assertGreater(dets[0].bw_hz, 60e6)
        self.assertLess(abs(dets[0].center_hz - 506e6), 1e6)  # block spans 474-538 MHz
        self.assertLess(floor, -65)

    def test_floor_tracks_noise_not_signals(self):
        sw = next(iter_sweeps(simulate.generate(2300, 2600, 100e3, 1, seed=5)))
        fl = estimate_floor(sw.power, sw.bin_hz)
        self.assertLess(abs(float(np.median(fl)) - simulate.NOISE_DB), 2.0)


class Identification(unittest.TestCase):
    DB = [
        {"name": "Synthetic ISM sensor", "url": "https://www.sigidwiki.com/wiki/x", "freqs_hz": [433920000],
         "bandwidths_hz": [40000], "modulations": ["FSK"], "locations": ["Worldwide"], "categories": ["ism"]},
        {"name": "Synthetic FM broadcast", "freqs_hz": [87500000, 108000000], "bandwidths_hz": [200000],
         "modulations": ["WFM"], "locations": ["Worldwide"], "categories": []},
        {"name": "US-only thing", "freqs_hz": [433920000], "bandwidths_hz": [40000], "locations": ["USA"]},
    ]

    def setUp(self):
        self.ident = Identifier(self.DB)

    def test_point_match(self):
        r = self.ident.identify(433.93e6, 50e3)
        self.assertEqual(r["name"], "Synthetic ISM sensor")
        self.assertEqual(r["source"], "artemis")
        self.assertGreaterEqual(r["score"], 90)
        self.assertIn("ISM 433", r["service"])

    def test_range_match(self):
        r = self.ident.identify(98.1e6, 200e3)
        self.assertEqual(r["candidates"][0]["name"], "Synthetic FM broadcast")
        self.assertTrue(r["candidates"][0]["broad"])  # 20 MHz span -> bandplan names it
        self.assertEqual(r["name"], "FM broadcast")
        narrow = Identifier([{"name": "Narrow", "freqs_hz": [150000000, 152000000], "bandwidths_hz": [12500],
                              "locations": ["Worldwide"]}]).identify(151e6, 12.5e3)
        self.assertEqual(narrow["name"], "Narrow")

    def test_region_penalty_ranks_local_first(self):
        r = self.ident.identify(433.92e6, 40e3)
        self.assertEqual(r["candidates"][0]["name"], "Synthetic ISM sensor")
        self.assertGreater(r["candidates"][0]["score"], r["candidates"][1]["score"])

    def test_bandwidth_mismatch_lowers_score(self):
        good = self.ident.identify(433.92e6, 40e3)["score"]
        bad = self.ident.identify(433.92e6, 8e6)["score"]
        self.assertLess(bad, good)

    def test_falls_back_to_bandplan(self):
        r = self.ident.identify(2437e6, 20e6)
        self.assertEqual(r["source"], "bandplan")
        self.assertIn("2.4 GHz", r["name"])

    def test_unknown_frequency(self):
        r = self.ident.identify(5.0e9, 1e6)
        self.assertIsNone(r["name"])

    def test_broad_range_does_not_beat_bandplan(self):
        db = [{"name": "LTE-like", "freqs_hz": [700000000, 900000000], "bandwidths_hz": [1400000, 20000000],
               "locations": ["Worldwide"]}]
        r = Identifier(db).identify(800e6, 5e6)
        self.assertEqual(r["candidates"][0]["name"], "LTE-like")  # still listed as a candidate
        self.assertTrue(r["candidates"][0]["broad"])
        self.assertEqual(r["source"], "bandplan")

    def test_region_keyword_is_whole_name(self):
        db = [{"name": "X", "freqs_hz": [433920000], "bandwidths_hz": [40000], "locations": ["Europe"]}]
        eu = Identifier(db, region_keywords=("eu",)).identify(433.92e6, 40e3)["score"]
        self.assertLess(eu, Identifier(db, region_keywords=("europe",)).identify(433.92e6, 40e3)["score"])


class ArtemisConversion(unittest.TestCase):
    def test_convert_real_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "data.sqlite")
            con = sqlite3.connect(p)
            con.executescript(
                """
                CREATE TABLE signals (SIG_ID INTEGER PRIMARY KEY, NAME TEXT, DESCRIPTION TEXT, URL TEXT);
                CREATE TABLE frequency (SIG_ID INT, VALUE TEXT);
                CREATE TABLE bandwidth (SIG_ID INT, VALUE TEXT);
                CREATE TABLE modulation (SIG_ID INT, VALUE TEXT);
                CREATE TABLE location (SIG_ID INT, VALUE TEXT);
                CREATE TABLE category (SIG_ID INT, CLB_ID INT);
                CREATE TABLE categorylabel (CLB_ID INTEGER PRIMARY KEY, VALUE TEXT);
                INSERT INTO signals VALUES (1, 'LoRa', 'desc', 'https://www.sigidwiki.com/wiki/LoRa'),
                                           (2, 'No frequency', '', NULL);
                INSERT INTO frequency VALUES (1, '433000000'), (1, '868000000');
                INSERT INTO bandwidth VALUES (1, '250000');
                INSERT INTO modulation VALUES (1, 'css');
                INSERT INTO location VALUES (1, 'Worldwide');
                INSERT INTO categorylabel VALUES (7, 'Digital');
                INSERT INTO category VALUES (1, 7);
                """
            )
            con.commit()
            con.close()
            out = artemisdb.convert(p)
        self.assertEqual(len(out), 1)  # signals without a frequency are skipped
        self.assertEqual(out[0]["freqs_hz"], [433000000, 868000000])
        self.assertEqual(out[0]["modulations"], ["CSS"])
        self.assertEqual(out[0]["categories"], ["digital"])


FAKE_SWEEP = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import sys
    from hackrf_scout import simulate
    sys.exit(simulate.main(sys.argv[1:]))
    """
)
FAKE_TRANSFER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, sys
    a = sys.argv
    path = a[a.index("-r") + 1]
    n = int(a[a.index("-n") + 1])
    with open(path, "wb") as fh:
        fh.write(os.urandom(2 * n))
    """
)


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = os.path.join(self.tmp.name, "bin")
        os.mkdir(self.bin)
        for name, body in (("hackrf_sweep", FAKE_SWEEP), ("hackrf_transfer", FAKE_TRANSFER)):
            p = os.path.join(self.bin, name)
            with open(p, "w") as fh:
                fh.write(body)
            os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
        self.env = dict(os.environ, PATH=self.bin + os.pathsep + os.environ["PATH"], PYTHONPATH=ROOT,
                        HACKRF_SCOUT_STATE_DIR=os.path.join(self.tmp.name, "state"))
        self.db = os.path.join(self.tmp.name, "s.db")

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args, check=True):
        r = subprocess.run([sys.executable, "-m", "hackrf_scout", *args], capture_output=True, text=True, env=self.env, cwd=self.tmp.name)
        if check:
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return r

    def test_live_scan_through_subprocess(self):
        r = self.cli("scan", "--db", self.db, "-f", "1:3000", "--sweeps", "12", "--quiet", "--obs-interval", "2")
        self.assertIn("8 new signals", r.stdout)
        rep = self.cli("report", "--db", self.db).stdout
        self.assertIn("FM broadcast", rep)
        self.assertIn("ADS-B", rep)

    def test_run_loop_captures_iq(self):
        cap = os.path.join(self.tmp.name, "caps")
        self.cli("run", "--db", self.db, "-f", "1:3000", "--scan-seconds", "30", "--cycles", "1",
                 "--capture-seconds", "1", "--capture-max", "2", "--capture-dir", cap, "--quiet", "--obs-interval", "2")
        files = sorted(os.listdir(cap))
        data = [f for f in files if f.endswith(".sigmf-data")]
        metas = [f for f in files if f.endswith(".sigmf-meta")]
        self.assertEqual(len(data), 2, files)
        self.assertEqual(len(metas), 2, files)
        self.assertEqual(len(files), 4, files)  # a SigMF pair per capture, nothing else
        with open(os.path.join(cap, metas[0])) as fh:
            meta = json.load(fh)
        self.assertEqual(meta["global"]["core:datatype"], "ci8")
        self.assertIn("core:frequency", meta["captures"][0])
        con = sqlite3.connect(self.db)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM signals WHERE captured=1").fetchone()[0], 2)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM captures").fetchone()[0], 2)
        paths = [r[0] for r in con.execute("SELECT path FROM captures")]
        self.assertTrue(all(p.endswith(".sigmf-data") and os.path.exists(p) for p in paths), paths)
        size = os.path.getsize(os.path.join(cap, data[0]))
        self.assertGreaterEqual(size, 2_000_000)  # >= 1 s at 1 Msps... rate is at least 2 Msps
        con.close()

    def test_export_csv_and_json(self):
        self.cli("scan", "--db", self.db, "-f", "1:3000", "--sweeps", "12", "--quiet", "--obs-interval", "2")
        out = self.cli("export", "--db", self.db, "--format", "csv").stdout.strip().splitlines()
        self.assertEqual(len(out), 1 + 8)
        js = json.loads(self.cli("export", "--db", self.db, "--format", "json", "--observations").stdout)
        self.assertEqual(len(js), 8)
        self.assertTrue(all(j["observations"] for j in js))

    def test_missing_hackrf_gives_helpful_error(self):
        env = dict(self.env, PATH="/nonexistent")
        r = subprocess.run([sys.executable, "-m", "hackrf_scout", "scan", "--db", self.db, "--sweeps", "1"],
                           capture_output=True, text=True, env=env, cwd=self.tmp.name)
        self.assertEqual(r.returncode, 2)
        self.assertIn("hackrf_sweep", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
