"""Watch mode: band planning, IQ analysis, burst tracking, the database records, the CLI and the web API.

The IQ is synthetic (tests/iqgen.py: noise plus FSK bursts at known frequencies and times), and the CLI tests
use a fake `hackrf_transfer` that streams the same, so no HackRF is needed."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import iqgen  # noqa: E402
from hackrf_scout import controller, query, sigmf, watch  # noqa: E402
from hackrf_scout.detect import Detection  # noqa: E402
from hackrf_scout.identify import Identifier  # noqa: E402
from hackrf_scout.store import Store  # noqa: E402

try:
    from fastapi.testclient import TestClient

    from hackrf_scout import webapp

    HAVE_WEB = True
except ImportError:
    HAVE_WEB = False

START = datetime(2026, 10, 9, 12, 0, 0)
B433 = (433.05e6, 434.79e6)
B868 = (863e6, 870e6)


def burst(plan, freq_hz, start_s, duration_s, amp=40.0, dev=25e3):
    return (freq_hz - plan.center_hz, start_s, duration_s, amp, dev)


class Run:
    """Watch a synthetic stream and keep what it wrote."""

    def __init__(self, test, bursts_at, seconds, band=B433, rate=None, noise=6.0, dc=0.0, seed=1, setup=None, ident=None, **kw):
        self.plan = watch.plan_band(band[0], band[1], rate)
        self.dir = tempfile.TemporaryDirectory()
        test.addCleanup(self.dir.cleanup)
        self.db = os.path.join(self.dir.name, "w.db")
        self.store = Store(self.db)
        if setup:
            setup(self.store)
        self.logs = []
        kw.setdefault("start", START)
        self.watcher = watch.Watcher(self.store, ident or Identifier(), self.plan, log=self.logs.append, **kw)
        bursts = [burst(self.plan, *b) for b in bursts_at]
        self.watcher.run(iqgen.chunks(self.plan.rate, seconds, self.watcher.analyzer.samples, bursts, noise=noise, dc=dc, seed=seed))
        self.store.commit()
        self.conn = sqlite3.connect(self.db)
        self.conn.row_factory = sqlite3.Row
        test.addCleanup(self.conn.close)
        test.addCleanup(self.store.close)
        self.signals = self.conn.execute("SELECT * FROM signals ORDER BY center_hz").fetchall()
        self.bursts = self.conn.execute("SELECT * FROM bursts ORDER BY start_ts").fetchall()

    def offsets_ms(self):
        """Start of each logged burst, in ms after START."""
        return [(datetime.fromisoformat(b["start_ts"]) - START).total_seconds() * 1000 for b in self.bursts]


class PlanTests(unittest.TestCase):
    def test_433_keeps_the_dc_spike_out_of_the_band(self):
        p = watch.plan_band(*B433)
        self.assertEqual((p.rate, p.nfft, p.dc_in_band), (6e6, 512, False))
        self.assertLess(p.center_hz, p.lo_hz - watch.EDGE_GUARD_HZ)
        self.assertLess(p.bin_hz, 12.5e3)

    def test_868_has_to_ignore_the_dc_spike(self):
        p = watch.plan_band(*B868)
        self.assertEqual((p.rate, p.nfft, p.dc_in_band), (10e6, 1024, True))
        self.assertEqual(p.center_hz, 866.5e6)
        self.assertIn("ignored", p.describe())

    def test_every_plan_covers_the_band_inside_the_clean_part_of_the_window(self):
        for lo, hi in ((433.05e6, 434.79e6), (863e6, 870e6), (2400e6, 2410e6), (144e6, 146e6), (902e6, 914e6), (430e6, 440e6)):
            p = watch.plan_band(lo, hi)
            usable = watch.USABLE * p.rate / 2
            self.assertGreaterEqual(p.lo_hz - watch.EDGE_GUARD_HZ, p.center_hz - usable, (lo, hi))
            self.assertLessEqual(p.hi_hz + watch.EDGE_GUARD_HZ, p.center_hz + usable, (lo, hi))
            self.assertEqual(p.dc_in_band, p.lo_hz <= p.center_hz <= p.hi_hz, (lo, hi))

    def test_too_wide_a_band_is_explained(self):
        with self.assertRaisesRegex(ValueError, "too wide"):
            watch.plan_band(400e6, 430e6)
        with self.assertRaisesRegex(ValueError, "start below"):
            watch.plan_band(434e6, 433e6)

    def test_explicit_rate(self):
        self.assertTrue(watch.plan_band(*B433, rate=4e6).dc_in_band)
        self.assertEqual(watch.plan_band(*B433, rate=20e6).rate, 20e6)
        with self.assertRaisesRegex(ValueError, "too slow"):
            watch.plan_band(*B868, rate=4e6)
        with self.assertRaisesRegex(ValueError, "between"):
            watch.plan_band(*B433, rate=1e6)

    def test_plan_for_a_finished_recording(self):
        p = watch.plan_window(433.92e6, 8e6)
        self.assertTrue(p.dc_in_band)
        self.assertAlmostEqual((p.lo_hz + p.hi_hz) / 2, 433.92e6)
        narrow = watch.plan_window(433.92e6, 8e6, 433.5e6, 434.3e6)
        self.assertEqual((narrow.lo_hz, narrow.hi_hz), (433.5e6, 434.3e6))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            watch.plan_window(433.92e6, 4e6, 430e6, 440e6)


class AnalyzerTests(unittest.TestCase):
    def raw(self, plan, an, bursts=(), dc=0.0):
        return next(iqgen.chunks(plan.rate, 0.05, an.samples, bursts, dc=dc))

    def test_a_tone_shows_up_at_its_frequency(self):
        plan = watch.plan_band(*B433)
        an = watch.SliceAnalyzer(plan, int(plan.rate * 0.02))
        f = 433.92e6
        p = an.power_db(self.raw(plan, an, [burst(plan, f, 0.0, 1.0, amp=30.0, dev=0.0)]))
        self.assertLess(abs(an.freqs[int(np.argmax(p))] - f), plan.bin_hz)
        self.assertGreater(p.max() - np.median(p), 25)

    def test_a_dc_offset_leaves_no_spike_at_the_centre(self):
        plan = watch.plan_band(*B868)
        an = watch.SliceAnalyzer(plan, int(plan.rate * 0.02))
        p = an.power_db(self.raw(plan, an, dc=25.0))
        centre = int(np.argmin(np.abs(an.freqs - plan.center_hz)))
        self.assertLess(p[centre] - np.median(p), 6.0)

    def test_only_the_band_plus_its_guard_is_analysed(self):
        plan = watch.plan_band(*B433)
        an = watch.SliceAnalyzer(plan, int(plan.rate * 0.02))
        self.assertGreaterEqual(an.freqs[0], plan.lo_hz - watch.EDGE_GUARD_HZ - plan.bin_hz)
        self.assertLessEqual(an.freqs[-1], plan.hi_hz + watch.EDGE_GUARD_HZ + plan.bin_hz)
        self.assertGreater(an.freqs.size, 100)


def det(f, power=-60.0, floor=-90.0, width=20e3):
    return Detection(f, width, f, power, power - 3, power - floor, floor, f - width / 2, f + width / 2)


class TrackerTests(unittest.TestCase):
    S = 0.02

    def tracker(self, **kw):
        return watch.BurstTracker(12.5e3, self.S, **dict(dict(gap_slices=3, max_slices=500, merge_hz=50e3), **kw))

    def feed(self, tr, plan):
        """plan: {slice_no: [detections]}; returns every burst closed, then whatever is left at the end."""
        out = []
        for n in range(max(plan) + 1):
            out += tr.update(n, plan.get(n, []))
        return out + tr.flush()

    def test_consecutive_slices_make_one_burst(self):
        out = self.feed(self.tracker(), {2: [det(433.9e6)], 3: [det(433.9e6)], 4: [det(433.9e6)]})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].slices, 3)
        self.assertAlmostEqual(out[0].onset_s, 2 * self.S)
        self.assertAlmostEqual(out[0].duration_s, 3 * self.S)

    def test_a_short_gap_is_inside_the_burst_a_long_one_ends_it(self):
        out = self.feed(self.tracker(), {0: [det(433.9e6)], 2: [det(433.9e6)]})
        self.assertEqual([(b.slices, round(b.duration_s / self.S)) for b in out], [(2, 3)])
        out = self.feed(self.tracker(), {0: [det(433.9e6)], 10: [det(433.9e6)]})
        self.assertEqual(len(out), 2)

    def test_partial_first_and_last_slices_are_timed_to_a_fraction(self):
        out = self.feed(self.tracker(), {
            0: [det(433.9e6, -66.0)],  # a quarter of the power of a full slice: the burst started 3/4 of the way in
            1: [det(433.9e6, -60.0)], 2: [det(433.9e6, -60.0)],
            3: [det(433.9e6, -63.0)],  # half of it: the burst stopped half way
        })
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].onset_s / self.S, 0.75, delta=0.02)
        self.assertAlmostEqual(out[0].duration_s / self.S, 2.75, delta=0.03)

    def test_a_long_transmission_is_cut_at_the_maximum_and_goes_on(self):
        out = self.feed(self.tracker(max_slices=4), {n: [det(433.9e6)] for n in range(10)})
        self.assertEqual([b.slices for b in out], [4, 4, 2])
        self.assertEqual([b.truncated for b in out], [True, True, False])

    def test_the_two_lobes_of_an_fsk_signal_are_one_burst(self):
        both = [det(433.89e6), det(433.92e6)]
        out = self.feed(self.tracker(), {1: both, 2: both})
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0].bandwidth_hz, 40e3)
        self.assertAlmostEqual(out[0].center_hz, 433.905e6, delta=20e3)

    def test_different_frequencies_are_tracked_separately(self):
        out = self.feed(self.tracker(), {0: [det(433.2e6)], 1: [det(433.2e6), det(434.4e6)], 2: [det(433.2e6), det(434.4e6)], 3: [det(434.4e6)]})
        by_freq = {round(b.center_hz / 1e5): b for b in out}
        self.assertEqual(set(by_freq), {4332, 4344})
        self.assertEqual((by_freq[4332].slices, by_freq[4344].slices), (3, 3))


class WatcherTests(unittest.TestCase):
    def test_bursts_are_timed_and_a_signal_appears_after_the_second(self):
        r = Run(self, [(433.92e6, 1.0, 0.060), (433.92e6, 2.0, 0.100), (433.92e6, 3.0, 0.045)], 4.2)
        self.assertEqual(len(r.signals), 1)
        s = r.signals[0]
        self.assertAlmostEqual(s["center_hz"], 433.92e6, delta=15e3)
        self.assertEqual(s["hits"], 3)
        self.assertEqual(len(r.bursts), 3)
        self.assertTrue(all(b["signal_id"] == s["id"] for b in r.bursts))
        for got, want in zip(r.offsets_ms(), (1000, 2000, 3000)):
            self.assertAlmostEqual(got, want, delta=8)
        for b, want in zip(r.bursts, (60, 100, 45)):
            self.assertAlmostEqual(b["duration_ms"], want, delta=8)
        self.assertEqual(sum("NEW " in line for line in r.logs), 1)
        self.assertIn("ISM", s["ident_name"] or "")

    def test_a_single_burst_is_not_enough_for_a_new_signal(self):
        r = Run(self, [(433.92e6, 1.0, 0.060)], 2.0)
        self.assertEqual((r.signals, r.bursts, r.watcher.bursts_total), ([], [], 0))
        r = Run(self, [(433.92e6, 1.0, 0.060)], 2.0, min_bursts=1)
        self.assertEqual((len(r.signals), len(r.bursts)), (1, 1))

    def test_a_known_signal_counts_its_first_burst(self):
        def setup(store):
            store.insert_signal(center_hz=433.92e6, bandwidth_hz=40e3, peak_hz=433.92e6, first_seen="2026-10-09T08:00:00", last_seen="2026-10-09T08:01:00",
                                first_sweep=1, hits=7, max_db=-60.0, avg_db=-65.0, last_db=-65.0, max_snr=20.0, last_snr=20.0)

        r = Run(self, [(433.93e6, 1.0, 0.060)], 2.0, setup=setup)
        self.assertEqual(len(r.signals), 1)
        self.assertEqual(r.signals[0]["hits"], 8)
        self.assertEqual(len(r.bursts), 1)
        self.assertEqual(r.signals[0]["last_seen"], r.bursts[0]["start_ts"])
        self.assertFalse(any("NEW " in line for line in r.logs))

    def test_noise_and_a_dc_offset_alone_find_nothing(self):
        for band in (B433, B868):
            r = Run(self, [], 2.0, band=band, dc=20.0, seed=3)
            self.assertEqual((r.signals, r.bursts), ([], []), band)

    def test_a_spike_at_the_tune_centre_is_ignored_but_a_signal_beside_it_is_not(self):
        centre = watch.plan_band(*B868).center_hz
        leak = (centre + 2e3, 0.0, 10.0, 10.0, 0.0)  # a steady narrow tone at the centre: the HackRF's own LO leakage
        r = Run(self, [leak], 2.0, band=B868, min_bursts=1)
        self.assertEqual((r.signals, r.bursts), ([], []))
        r = Run(self, [leak, (centre + 150e3, 1.0, 0.060)], 2.2, band=B868, min_bursts=1)
        self.assertEqual(len(r.bursts), 1)
        self.assertAlmostEqual(r.bursts[0]["center_hz"], centre + 150e3, delta=20e3)

    def test_simultaneous_transmissions_on_different_frequencies(self):
        r = Run(self, [(433.20e6, 1.0, 0.060), (434.40e6, 1.0, 0.080)], 2.2, min_bursts=1)
        self.assertEqual([round(s["center_hz"] / 1e5) for s in r.signals], [4332, 4344])
        self.assertEqual(len(r.bursts), 2)
        self.assertAlmostEqual(r.offsets_ms()[0], r.offsets_ms()[1], delta=10)

    def test_the_noise_floor_settles_before_anything_is_reported(self):
        r = Run(self, [(433.92e6, 0.1, 0.050), (433.92e6, 1.0, 0.060)], 2.0, min_bursts=1)
        self.assertEqual(len(r.bursts), 1)
        self.assertAlmostEqual(r.offsets_ms()[0], 1000, delta=8)

    def test_a_continuous_carrier_is_cut_into_bursts(self):
        r = Run(self, [(433.92e6, 0.8, 3.4)], 4.6, min_bursts=1, max_burst_s=1.0)
        self.assertGreaterEqual(len(r.bursts), 3)
        self.assertLessEqual(len(r.bursts), 5)
        self.assertTrue(all(b["duration_ms"] <= 1100 for b in r.bursts))
        self.assertAlmostEqual(sum(b["duration_ms"] for b in r.bursts), 3400, delta=250)

    def test_a_very_short_burst_is_still_caught_and_timed_to_within_a_slice(self):
        r = Run(self, [(433.92e6, 1.0, 0.025), (433.92e6, 2.0, 0.025)], 3.0)
        self.assertEqual(len(r.bursts), 2)
        for got, want in zip(r.offsets_ms(), (1000, 2000)):
            self.assertAlmostEqual(got, want, delta=25)
        self.assertTrue(all(b["duration_ms"] < 60 for b in r.bursts))

    def test_burst_rows_summary_and_scan_state(self):
        r = Run(self, [(433.92e6, 1.0, 0.060), (433.92e6, 2.0, 0.060)], 3.0, summary_every=1.0)
        self.assertTrue(any("watching:" in line for line in r.logs))
        self.assertGreaterEqual(r.conn.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0], 2)
        self.assertEqual(r.watcher.bursts_total, 2)

    def test_it_stops_after_the_requested_duration(self):
        plan = watch.plan_band(*B433)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(os.path.join(tmp.name, "d.db"))
        self.addCleanup(store.close)
        w = watch.Watcher(store, Identifier(), plan, start=START, log=lambda m: None)
        w.run(iqgen.chunks(plan.rate, 5.0, w.analyzer.samples, []), duration_s=1.0)
        self.assertAlmostEqual(w.slice_no * w.slice_s, 1.0, delta=0.05)

    def test_bad_settings_are_refused(self):
        store = Store(":memory:")
        self.addCleanup(store.close)
        with self.assertRaisesRegex(ValueError, "5 ms"):
            watch.Watcher(store, Identifier(), watch.plan_band(*B433), slice_ms=2)


class FakeIQ(unittest.TestCase):
    """A temp dir with a fake `hackrf_transfer` that streams synthetic IQ."""

    SCRIPT = """#!@PY@
import os, sys
sys.path.insert(0, "@ROOT@/tests")
import iqgen
a = sys.argv
with open(os.environ["FAKE_ARGS"], "w") as fh:
    fh.write(" ".join(a[1:]))
rate = float(a[a.index("-s") + 1]); center = float(a[a.index("-f") + 1])
secs = float(os.environ.get("FAKE_SECONDS", "6"))
if secs <= 0:
    sys.stderr.write("hackrf_open() failed: HACKRF_ERROR_NOT_FOUND\\n")
    sys.exit(1)
sys.stderr.write("call hackrf_set_sample_rate(%d)\\n" % rate)
f0 = float(os.environ.get("FAKE_FREQ", "433.92e6"))
bursts = [(f0 - center, 1.0 + k, 0.06, 40.0, 25e3) for k in range(int(secs) - 1)]
n = int(rate * 0.02) // 512 * 512
out = sys.stdout.buffer
try:
    for chunk in iqgen.chunks(rate, secs, n, bursts):
        out.write(chunk)
    out.flush()
except BrokenPipeError:
    pass
"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = self.tmp.name
        self.exe = os.path.join(t, "hackrf_transfer")
        with open(self.exe, "w") as fh:
            fh.write(self.SCRIPT.replace("@PY@", sys.executable).replace("@ROOT@", ROOT))
        os.chmod(self.exe, 0o755)
        self.db = os.path.join(t, "scout.db")
        self.args_file = os.path.join(t, "args.txt")
        self.env = dict(os.environ, PYTHONPATH=ROOT, HACKRF_SCOUT_STATE_DIR=os.path.join(t, "state"), FAKE_ARGS=self.args_file, HOME=t)

    def cli(self, *args, **env):
        e = dict(self.env, **env)
        return subprocess.run([sys.executable, "-m", "hackrf_scout", *args], capture_output=True, text=True, env=e, cwd=self.tmp.name, timeout=90)

    def rows(self, sql, *params):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()


class WatchCliTests(FakeIQ):
    def test_watching_433_from_the_command_line(self):
        r = self.cli("watch", "--band", "433", "--db", self.db, "--hackrf-transfer", self.exe, "-l", "32", "-g", "30", "-a")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Done: watched", r.stdout)
        with open(self.args_file) as fh:
            argv = fh.read().split()
        self.assertEqual(argv[:2], ["-r", "-"])
        pairs = dict(zip(argv[2::2], argv[3::2]))
        self.assertEqual((pairs["-s"], pairs["-l"], pairs["-g"], pairs["-a"]), ("6000000", "32", "30", "1"))
        self.assertAlmostEqual(float(pairs["-f"]), 432.95e6, delta=1)
        sig = self.rows("SELECT * FROM signals")
        self.assertEqual(len(sig), 1)
        self.assertAlmostEqual(sig[0]["center_hz"], 433.92e6, delta=15e3)
        self.assertGreaterEqual(len(self.rows("SELECT * FROM bursts")), 4)
        scan = self.rows("SELECT mode, ranges FROM scans")
        self.assertEqual([s["mode"] for s in scan], ["watch"])
        self.assertAlmostEqual(json.loads(scan[0]["ranges"])[0][0], 433.05e6, delta=1)
        log = " ".join(r["msg"] for r in self.rows("SELECT msg FROM log"))
        self.assertIn("watching 433.050-434.790 MHz", log)
        self.assertIn("NEW", log)

    def test_the_report_shows_bursts_instead_of_a_duty_cycle(self):
        self.cli("watch", "--band", "433", "--db", self.db, "--hackrf-transfer", self.exe, "--duration", "5")
        r = self.cli("report", "--db", self.db)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("433.9", r.stdout)

    def test_replaying_a_sigmf_recording(self):
        plan = watch.plan_window(433.92e6, 8e6)
        data = os.path.join(self.tmp.name, "rec.sigmf-data")
        store = Store(":memory:")
        self.addCleanup(store.close)
        w = watch.Watcher(store, Identifier(), plan)
        with open(data, "wb") as fh:
            for c in iqgen.chunks(plan.rate, 3.5, w.analyzer.samples, [(0.0, 1.0, 0.05, 40.0, 25e3), (0.0, 2.0, 0.05, 40.0, 25e3)]):
                fh.write(c)
        meta = sigmf.write_meta(data, sample_rate=plan.rate, center_hz=plan.center_hz, started_at=datetime(2026, 10, 9, 12, 0, 0).astimezone(), checksum=False)
        r = self.cli("watch", "--source", meta, "--db", self.db)
        self.assertEqual(r.returncode, 0, r.stderr)
        bursts = self.rows("SELECT start_ts FROM bursts ORDER BY start_ts")
        self.assertEqual(len(bursts), 2)
        want = datetime(2026, 10, 9, 12, 0, 0).astimezone().replace(tzinfo=None) + timedelta(seconds=1)
        self.assertAlmostEqual((datetime.fromisoformat(bursts[0]["start_ts"]) - want).total_seconds(), 0, delta=0.02)
        self.assertEqual(self.rows("SELECT COUNT(*) AS n FROM scans")[0]["n"], 0)  # a recording is not a live scan

    def test_mistakes_are_explained(self):
        cases = [
            ((), "--band is required"),
            (("--band", "400:430"), "too wide"),
            (("--band", "433", "--rate", "50e6"), "sample rate"),
            (("--band", "433", "--slice-ms", "1"), "--slice-ms"),
            (("--source", "x.cs8"), "SigMF"),
            (("--band", "nonsense"), "band"),
        ]
        for extra, text in cases:
            r = self.cli("watch", "--db", self.db, "--hackrf-transfer", self.exe, *extra)
            self.assertNotEqual(r.returncode, 0, extra)
            self.assertIn(text, r.stderr, extra)

    def test_no_data_from_the_hackrf_is_an_error_with_the_reason(self):
        r = self.cli("watch", "--band", "433", "--db", self.db, "--hackrf-transfer", self.exe, FAKE_SECONDS="0")
        self.assertEqual(r.returncode, 2)
        self.assertIn("No IQ data", r.stderr)
        self.assertIn("HACKRF_ERROR_NOT_FOUND", r.stderr)

    def test_a_missing_hackrf_transfer_is_explained(self):
        r = self.cli("watch", "--band", "433", "--db", self.db, "--hackrf-transfer", "/nonexistent/hackrf_transfer")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not found", r.stderr)


class BuildWatchArgsTests(unittest.TestCase):
    def build(self, params):
        return controller.build_scan_args(params, db="/data/scout.db", hackrf_transfer="/opt/hackrf_transfer")

    def pairs(self, argv):
        return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("-")}

    def test_a_watch_request_becomes_a_watch_command(self):
        a = self.build({"mode": "watch", "bands": ["433"], "lna": 32, "vga": 30, "amp": True, "snr": 12, "slice_ms": 10, "min_bursts": 3, "duration": 600})
        self.assertEqual(a[3], "watch")
        p = self.pairs(a)
        self.assertEqual((p["-l"], p["-g"], p["--snr"], p["--slice-ms"], p["--min-bursts"], p["--duration"]), ("32", "30", "12", "10", "3", "600"))
        self.assertIn("-a", a)
        self.assertIn("--quiet", a)
        self.assertEqual(p["--hackrf-transfer"], "/opt/hackrf_transfer")
        lo, hi = (float(x) for x in p["--band"].split(":"))
        self.assertAlmostEqual(lo, 433.05, places=3)
        self.assertGreater(hi, lo)
        self.assertNotIn("-f", a)  # the band goes in --band; there is nothing to sweep

    def test_exactly_one_band_is_needed(self):
        for bands in ([], ["433", "868"]):
            with self.assertRaisesRegex(ValueError, "exactly one band"):
                self.build({"mode": "watch", "bands": bands})
        with self.assertRaisesRegex(ValueError, "exactly one band"):
            self.build({"mode": "watch"})

    def test_a_band_that_is_too_wide_is_refused(self):
        with self.assertRaisesRegex(ValueError, "too wide"):
            self.build({"mode": "watch", "bands": ["400:430"]})

    def test_sweep_and_capture_settings_do_not_apply(self):
        for key in ("min_hits", "scan_seconds", "capture_max", "bin_width", "hysteresis"):
            with self.assertRaisesRegex(ValueError, "unknown or unsupported"):
                self.build({"mode": "watch", "bands": ["433"], key: 1})

    def test_values_are_range_checked(self):
        for params in ({"slice_ms": 1}, {"slice_ms": 5000}, {"min_bursts": 0}, {"lna": 7}, {"amp": "yes"}, {"region_keywords": "a;b"}):
            with self.assertRaises(ValueError, msg=params):
                self.build(dict(params, mode="watch", bands=["433"]))


class _NoScanner:
    def status(self):
        return {"supported": True, "state": "stopped", "managed": False}


def make_burst_db(path):
    st = Store(path)
    quiet = st.insert_signal(center_hz=868.3e6, bandwidth_hz=100e3, peak_hz=868.3e6, first_seen="2026-10-09T10:00:00", last_seen="2026-10-09T10:00:00",
                             first_sweep=1, hits=5, max_db=-50.0, avg_db=-55.0, last_db=-52.0, max_snr=20.0, last_snr=20.0)
    bursty = st.insert_signal(center_hz=433.92e6, bandwidth_hz=40e3, peak_hz=433.92e6, first_seen="2026-10-09T10:00:00", last_seen="2026-10-09T10:00:50",
                              first_sweep=1, hits=6, max_db=-50.0, avg_db=-55.0, last_db=-52.0, max_snr=20.0, last_snr=20.0)
    for i in range(6):  # every 10 s, 60-ish ms long
        st.add_burst(bursty, f"2026-10-09T10:00:{10 * i:02d}.250", 58.0 + i, 433.92e6, 40e3, -50.0, 20.0, 3)
    st.bump_sweep()
    st.commit()
    st.close()
    return quiet, bursty


@unittest.skipUnless(HAVE_WEB, "needs the web extra: pip install 'hackrf-scout[web]' httpx")
class BurstApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "b.db")
        self.quiet, self.bursty = make_burst_db(self.db)
        self.c = TestClient(webapp.create_app(webapp.WebConfig(db_path=self.db, controller=_NoScanner())), base_url="http://localhost")

    def test_burst_list(self):
        r = self.c.get("/api/bursts", params={"signal_id": self.bursty, "order": "asc"}).json()
        self.assertEqual(r["total"], 6)
        self.assertEqual(r["items"][0]["start_ts"], "2026-10-09T10:00:00.250")
        self.assertEqual(len(self.c.get("/api/bursts").json()["items"]), 6)
        self.assertEqual(self.c.get("/api/bursts", params={"signal_id": self.quiet}).json()["total"], 0)
        self.assertEqual(self.c.get("/api/bursts", params={"order": "sideways"}).status_code, 400)

    def test_signals_with_bursts_have_a_count_and_no_duty_cycle(self):
        items = {s["id"]: s for s in self.c.get("/api/signals").json()["items"]}
        self.assertEqual(items[self.bursty]["burst_count"], 6)
        self.assertIsNone(items[self.bursty]["duty_pct"])
        self.assertEqual(items[self.quiet]["burst_count"], 0)
        self.assertIsNotNone(items[self.quiet]["duty_pct"])

    def test_signal_detail_summarises_the_bursts(self):
        d = self.c.get(f"/api/signals/{self.bursty}").json()
        b = d["burst_summary"]
        self.assertEqual(b["count"], 6)
        self.assertEqual(len(b["recent"]), 6)
        self.assertEqual(b["recent"][0]["start_ts"], "2026-10-09T10:00:50.250")  # newest first
        self.assertAlmostEqual(b["duration_ms"]["median"], 61.0, delta=3)
        self.assertEqual(b["interval_s"]["median"], 10.0)
        self.assertTrue(b["interval_s"]["regular"])
        self.assertIsNone(self.c.get(f"/api/signals/{self.quiet}").json()["burst_summary"])

    def test_status_and_raw_tables_include_bursts(self):
        self.assertEqual(self.c.get("/api/status").json()["counts"]["bursts"], 6)
        t = self.c.get("/api/tables/bursts").json()
        self.assertEqual(t["total"], 6)


if __name__ == "__main__":
    unittest.main()
