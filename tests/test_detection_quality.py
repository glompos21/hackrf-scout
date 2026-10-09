"""Steadier detection: smoothed noise floor, hysteresis for confirmed signals, and the overload guard.

The sweeps here are built by hand (flat noise plus rectangular bumps of a chosen SNR) so every threshold
is exact instead of statistical.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hackrf_scout import cli, controller, simulate  # noqa: E402
from hackrf_scout.detect import FloorTracker  # noqa: E402
from hackrf_scout.scanner import Scanner  # noqa: E402
from hackrf_scout.store import Store  # noqa: E402
from hackrf_scout.sweep import Sweep  # noqa: E402

BIN_HZ = 100e3
N_BINS = 3000  # 300 MHz from 100 MHz
START = 100e6
T0 = datetime(2026, 10, 9, 10, 0, 0)


def bump_index(center_mhz):
    return int(round((center_mhz * 1e6 - START) / BIN_HZ))


def make_sweep(i, bumps=(), floor=-70.0, noise=0.15, seed=None):
    """Sweep number `i`: flat noise at `floor` dB plus (centre MHz, SNR dB) bumps, 3 bins wide."""
    rng = np.random.default_rng(i if seed is None else seed)
    freqs = START + (np.arange(N_BINS) + 0.5) * BIN_HZ
    power = floor + rng.normal(0.0, noise, N_BINS)
    for centre_mhz, snr in bumps:
        k = bump_index(centre_mhz)
        power[k - 1 : k + 2] = floor + snr + rng.normal(0.0, noise, 3)
    ts = (T0 + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.000000")
    return Sweep(ts=ts, freqs=freqs, power=power, bin_hz=BIN_HZ)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(os.path.join(self.tmp.name, "q.db"))
        self.addCleanup(self.store.close)
        self.warnings = []

    def scanner(self, **kw):
        kw.setdefault("min_hits", 2)
        return Scanner(self.store, None, snr_db=10.0, log=lambda m: None, warn=self.warnings.append, obs_interval_s=1, **kw)

    def run_sweeps(self, scanner, start, specs):
        """specs: list of kwargs for make_sweep; returns the process() results."""
        return [scanner.process(make_sweep(start + k, **spec)) for k, spec in enumerate(specs)]


class FloorTrackerTests(unittest.TestCase):
    freqs = START + (np.arange(500) + 0.5) * BIN_HZ

    def raw(self, level, seed):
        return level + np.random.default_rng(seed).normal(0, 1.0, self.freqs.size)

    def test_first_sweep_seeds_the_floor(self):
        t = FloorTracker(alpha=0.2)
        raw = self.raw(-70, 1)
        floor, jump = t.update(self.freqs, raw)
        self.assertIsNone(jump)
        np.testing.assert_array_equal(floor, raw)

    def test_smoothing_steadies_the_floor(self):
        t = FloorTracker(alpha=0.2)
        raws = [self.raw(-70, k) for k in range(60)]
        smoothed = [t.update(self.freqs, r)[0] for r in raws]
        # the median floor of each sweep wobbles less after smoothing
        spread_raw = np.std([np.median(r) for r in raws[10:]])
        spread_smooth = np.std([np.median(s) for s in smoothed[10:]])
        self.assertLess(spread_smooth, 0.6 * spread_raw)

    def test_alpha_one_means_no_smoothing(self):
        t = FloorTracker(alpha=1.0)
        t.update(self.freqs, self.raw(-70, 1))
        raw2 = self.raw(-70, 2)
        floor, _ = t.update(self.freqs, raw2)
        np.testing.assert_allclose(floor, raw2)

    def test_small_drift_is_followed_not_flagged(self):
        t = FloorTracker(alpha=0.5, jump_db=6.0)
        t.update(self.freqs, self.raw(-70, 1))
        for k in range(10):
            floor, jump = t.update(self.freqs, self.raw(-67, k))
            self.assertIsNone(jump)
        self.assertAlmostEqual(float(np.median(floor)), -67.0, delta=0.5)

    def test_a_brief_jump_is_suspect_and_a_lasting_one_is_accepted(self):
        t = FloorTracker(alpha=0.2, jump_db=6.0, max_suspect=3)
        for k in range(5):
            t.update(self.freqs, self.raw(-70, k))
        jumps = []
        for k in range(3):
            floor, jump = t.update(self.freqs, self.raw(-61, 10 + k))
            jumps.append(jump)
            if jump is not None:
                self.assertAlmostEqual(float(np.median(floor)), -70.0, delta=0.6)  # old floor kept
        self.assertAlmostEqual(jumps[0], 9.0, delta=0.5)
        self.assertIsNotNone(jumps[1])
        self.assertIsNone(jumps[2])  # third in a row: accepted
        self.assertAlmostEqual(t.accepted_jump, 9.0, delta=0.5)
        self.assertAlmostEqual(float(np.median(floor)), -61.0, delta=0.6)
        _, jump = t.update(self.freqs, self.raw(-61, 99))
        self.assertIsNone(jump)
        self.assertIsNone(t.accepted_jump)

    def test_a_single_glitch_does_not_disturb_the_baseline(self):
        t = FloorTracker(alpha=0.2, jump_db=6.0)
        for k in range(5):
            t.update(self.freqs, self.raw(-70, k))
        _, jump = t.update(self.freqs, self.raw(-58, 50))
        self.assertIsNotNone(jump)
        floor, jump = t.update(self.freqs, self.raw(-70, 51))
        self.assertIsNone(jump)
        self.assertAlmostEqual(float(np.median(floor)), -70.0, delta=0.5)

    def test_jump_check_can_be_switched_off(self):
        t = FloorTracker(alpha=1.0, jump_db=0)
        t.update(self.freqs, self.raw(-70, 1))
        floor, jump = t.update(self.freqs, self.raw(-50, 2))
        self.assertIsNone(jump)
        self.assertAlmostEqual(float(np.median(floor)), -50.0, delta=0.5)

    def test_a_sweep_with_missing_bins_is_resampled(self):
        t = FloorTracker(alpha=0.5)
        t.update(self.freqs, self.raw(-70, 1))
        keep = np.r_[0:200, 230:500]  # a dropped chunk
        floor, jump = t.update(self.freqs[keep], self.raw(-70, 2)[keep])
        self.assertIsNone(jump)
        self.assertEqual(floor.size, keep.size)

    def test_a_different_range_starts_over(self):
        t = FloorTracker(alpha=0.5, jump_db=6.0)
        t.update(self.freqs, self.raw(-70, 1))
        other = self.freqs + 2e9
        floor, jump = t.update(other, self.raw(-50, 2))  # no overlap: a new scan, not a jump
        self.assertIsNone(jump)
        self.assertAlmostEqual(float(np.median(floor)), -50.0, delta=0.5)

    def test_invalid_alpha(self):
        for a in (0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                FloorTracker(alpha=a)


class HysteresisTests(Base):
    CONFIRM = [dict(bumps=[(150.0, 15.0)])] * 3  # strong: confirms the signal
    WEAK = dict(bumps=[(150.0, 8.0)])  # between snr - hysteresis (7) and snr (10)

    def hits(self):
        rows = self.store.load_signals()
        self.assertEqual(len(rows), 1)
        return rows[0]["hits"]

    def test_confirmed_signal_survives_dips_below_the_threshold(self):
        sc = self.scanner(hysteresis_db=3.0)
        self.run_sweeps(sc, 0, self.CONFIRM + [self.WEAK] * 4)
        self.assertEqual(self.hits(), 7)

    def test_without_hysteresis_the_dips_are_misses(self):
        sc = self.scanner(hysteresis_db=0.0)
        self.run_sweeps(sc, 0, self.CONFIRM + [self.WEAK] * 4)
        self.assertEqual(self.hits(), 3)

    def test_a_dip_below_the_lower_threshold_is_still_a_miss(self):
        sc = self.scanner(hysteresis_db=3.0)
        too_weak = dict(bumps=[(150.0, 5.5)])
        self.run_sweeps(sc, 0, self.CONFIRM + [too_weak] * 3)
        self.assertEqual(self.hits(), 3)

    def test_a_signal_that_is_only_ever_weak_never_starts(self):
        sc = self.scanner(hysteresis_db=3.0)
        self.run_sweeps(sc, 0, [self.WEAK] * 10)
        self.assertEqual(self.store.load_signals(), [])
        self.assertEqual(sc.tracks, [])

    def test_weak_neighbours_do_not_start_new_tracks(self):
        sc = self.scanner(hysteresis_db=3.0)
        self.run_sweeps(sc, 0, self.CONFIRM)
        self.run_sweeps(sc, 10, [dict(bumps=[(150.0, 15.0), (200.0, 8.0)])] * 5)
        self.assertEqual(len(sc.tracks), 1)
        self.assertEqual(len(self.store.load_signals()), 1)

    def test_weak_hits_do_not_raise_the_maximum_snr(self):
        sc = self.scanner(hysteresis_db=3.0)
        self.run_sweeps(sc, 0, self.CONFIRM + [self.WEAK] * 3)
        row = self.store.load_signals()[0]
        self.assertGreater(row["max_snr"], 14.0)
        self.assertLess(row["last_snr"], 9.5)


class OverloadGuardTests(Base):
    QUIET = dict(bumps=[(150.0, 15.0), (250.0, 15.0)])

    def test_a_floor_jump_skips_sweeps_until_it_proves_lasting(self):
        sc = self.scanner()
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        before = self.store.sweep_count()
        hits_before = self.store.load_signals()[0]["hits"]
        jumped = dict(bumps=[(150.0, 15.0), (250.0, 15.0)], floor=-61.0)
        results = self.run_sweeps(sc, 10, [jumped] * 3)
        self.assertEqual([bool(r["skipped"]) for r in results], [True, True, False])
        self.assertIn("noise floor moved", results[0]["skipped"])
        self.assertEqual(sc.skipped_sweeps, 2)
        self.assertEqual(self.store.sweep_count(), before + 1)  # only the accepted sweep was counted
        self.assertEqual(self.store.load_signals()[0]["hits"], hits_before + 1)  # skipped ones gave no hits
        self.assertTrue(any("possible receiver overload" in w for w in self.warnings))
        self.assertTrue(any("new baseline" in w for w in self.warnings))

    def test_a_single_glitch_costs_one_sweep_and_nothing_else(self):
        sc = self.scanner()
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        results = self.run_sweeps(sc, 10, [dict(bumps=[], floor=-58.0), self.QUIET, self.QUIET])
        self.assertEqual([bool(r["skipped"]) for r in results], [True, False, False])
        self.assertEqual(len(self.store.load_signals()), 2)

    def test_skipped_sweeps_are_logged_as_warnings_for_the_web_log(self):
        sc = Scanner(self.store, None, snr_db=10.0, min_hits=2, log=lambda m: None, warn=cli._warn_logger(self.store))
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        self.run_sweeps(sc, 10, [dict(bumps=[], floor=-58.0)])
        rows = self.store.conn.execute("SELECT level, source, msg FROM log").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["level"], rows[0]["source"]), ("warn", "detect"))
        self.assertIn("overload", rows[0]["msg"])

    def test_a_flood_of_detections_is_skipped(self):
        sc = self.scanner()
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        flood = dict(bumps=[(101.0 + 1.0 * k, 15.0) for k in range(60)])
        results = self.run_sweeps(sc, 10, [flood])
        self.assertIn("detections at once", results[0]["skipped"])
        self.assertEqual(len(self.store.load_signals()), 2)

    def test_a_lasting_flood_becomes_the_new_normal(self):
        sc = self.scanner()
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        flood = dict(bumps=[(101.0 + 1.0 * k, 15.0) for k in range(60)])
        results = self.run_sweeps(sc, 10, [flood] * 3)
        self.assertEqual([bool(r["skipped"]) for r in results], [True, True, False])

    def test_guard_off_means_nothing_is_skipped(self):
        sc = self.scanner(overload_db=0.0)
        self.run_sweeps(sc, 0, [self.QUIET] * 6)
        flood = dict(bumps=[(101.0 + 1.0 * k, 15.0) for k in range(60)], floor=-55.0)
        results = self.run_sweeps(sc, 10, [flood])
        self.assertIsNone(results[0]["skipped"])
        self.assertEqual(sc.skipped_sweeps, 0)

    def test_no_guard_decision_before_there_is_history(self):
        sc = self.scanner()
        flood = dict(bumps=[(101.0 + 1.0 * k, 15.0) for k in range(60)])
        results = self.run_sweeps(sc, 0, [flood] * 2)  # the very first sweeps set the norm
        self.assertEqual([r["skipped"] for r in results], [None, None])


class WholePipelineTests(Base):
    def test_simulated_scan_still_finds_every_signal_with_the_new_defaults(self):
        sc = Scanner(self.store, None, log=lambda m: None, obs_interval_s=2)
        from hackrf_scout.sweep import iter_sweeps

        for sweep in iter_sweeps(simulate.generate(1.0, 3000.0, 100e3, 14, seed=3)):
            res = sc.process(sweep)
            self.assertIsNone(res["skipped"])
        self.assertEqual(len(self.store.load_signals()), len(simulate.DEFAULT_SIGNALS))


class CommandLineTests(unittest.TestCase):
    def test_bad_values_are_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            sim = os.path.join(d, "s.csv")
            with open(sim, "w") as fh:
                fh.writelines(simulate.generate(1, 200, 100e3, 3))
            for flag, value in (("--floor-alpha", "5"), ("--floor-alpha", "0"), ("--hysteresis", "-1"), ("--overload-db", "999")):
                r = subprocess.run(
                    [sys.executable, "-m", "hackrf_scout", "scan", "--source", sim, "--db", os.path.join(d, "x.db"), "--quiet", flag, value],
                    capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT), cwd=d,
                )
                self.assertEqual(r.returncode, 2, (flag, value, r.stderr))
                self.assertIn("must be between", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_options_reach_the_browser_started_scanner(self):
        argv = controller.build_scan_args({"floor_alpha": 0.5, "hysteresis": 0, "overload_db": 9}, db="/x.db")
        pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)}
        self.assertEqual((pairs["--floor-alpha"], pairs["--hysteresis"], pairs["--overload-db"]), ("0.5", "0", "9"))
        defaults = controller.build_scan_args({}, db="/x.db")
        self.assertIn("--floor-alpha", defaults)
        for bad in ({"floor_alpha": 0}, {"floor_alpha": 2}, {"hysteresis": -1}, {"hysteresis": 99}, {"overload_db": -3}, {"overload_db": "x"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                controller.build_scan_args(bad, db="/x.db")


if __name__ == "__main__":
    unittest.main(verbosity=2)
