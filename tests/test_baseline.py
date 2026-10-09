"""Baselines and anomaly flags: louder, quieter, gone quiet, new in a quiet area."""

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from world import EVERYTHING, T0, World, at, wobble  # noqa: E402

from hackrf_scout import baseline, query  # noqa: E402
from hackrf_scout.baseline import BaselineConfig, Coverage, evaluate, judge_level, signal_baseline  # noqa: E402
from hackrf_scout.store import Store  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "b.db")
        self.world = World(self.path)
        self.addCleanup(self.world.close)

    def flags(self, **kw):
        self.world.commit()
        return evaluate(self.world.store.conn, **kw)["flags"]

    def kinds(self, **kw):
        return sorted((f["signal_id"], f["kind"]) for f in self.flags(**kw))


class LevelTests(Base):
    def test_a_steady_signal_has_no_flags(self):
        self.world.seen(868.3, wobble(60))
        self.world.scan(0, 31)
        self.assertEqual(self.flags(), [])

    def test_louder_needs_a_persistent_shift(self):
        a = self.world.seen(868.3, wobble(40) + [28, 28, 20, 20, 20])  # a blip: median of the last five is back to normal
        b = self.world.seen(433.9, wobble(40) + [28, 28, 28, 20, 20])  # three of five stay high
        self.world.scan(0, 23)
        self.assertEqual(self.kinds(), [(b, "louder")])
        self.assertNotIn(a, [f["signal_id"] for f in self.flags()])

    def test_louder_details_and_message(self):
        sid = self.world.seen(868.3, wobble(40) + [29, 28, 30, 28, 29])
        self.world.scan(0, 23)
        (f,) = self.flags()
        self.assertEqual((f["signal_id"], f["kind"], f["severity"]), (sid, "louder", "alert"))
        d = f["details"]
        self.assertAlmostEqual(d["baseline_db"], 20.0, delta=0.5)
        self.assertAlmostEqual(d["shift_db"], 9.0, delta=0.6)
        self.assertEqual((d["recent_over"], d["recent_n"]), (5, 5))
        self.assertIn("louder than usual", f["message"])
        self.assertIn("5 of the last 5", f["message"])

    def test_quieter(self):
        sid = self.world.seen(868.3, wobble(40) + [12, 12, 13, 12, 12])
        self.world.scan(0, 23)
        (f,) = self.flags()
        self.assertEqual((f["signal_id"], f["kind"], f["severity"]), (sid, "quieter", "notice"))
        self.assertLess(f["details"]["shift_db"], -7)

    def test_small_shifts_are_not_flagged(self):
        # the spread of SNR values is never trusted below 1.5 dB, so 3 sigma is 4.5 dB
        self.world.seen(868.3, wobble(40, amp=0.1) + [24.0] * 5)  # +4 dB
        self.world.scan(0, 23)
        self.assertEqual(self.flags(), [])
        sid = self.world.seen(433.9, wobble(40, amp=0.1) + [25.5] * 5)  # +5.5 dB
        self.world.scan(0, 23)
        self.assertEqual(self.kinds(), [(sid, "louder")])

    def test_a_noisy_signal_needs_a_bigger_shift(self):
        noisy = [14, 26, 16, 24, 18, 22, 15, 25, 17, 23] * 4  # spread ~ 7 dB
        self.world.seen(868.3, noisy + [27, 27, 27, 27, 27])  # +7 dB is not unusual for this signal
        self.world.scan(0, 23)
        self.assertEqual(self.flags(), [])

    def test_past_spikes_do_not_hide_a_real_change_or_make_a_false_one(self):
        spikes = wobble(37) + [38, 39, 38]  # three old bursts inside the baseline
        quiet = self.world.seen(868.3, spikes + wobble(5))
        real = self.world.seen(433.9, spikes + [29, 28, 29, 30, 29])
        self.world.scan(0, 23)
        self.assertEqual(self.kinds(), [(real, "louder")])
        f = [x for x in self.flags() if x["signal_id"] == real][0]
        self.assertAlmostEqual(f["details"]["baseline_db"], 20.0, delta=0.6)  # not dragged up by the spikes
        self.assertNotIn(quiet, [x["signal_id"] for x in self.flags()])

    def test_not_enough_history_means_no_judgement(self):
        sid = self.world.seen(868.3, wobble(8) + [30] * 4)  # 12 observations, 13 needed
        self.world.scan(0, 8)
        self.assertEqual(self.flags(), [])
        info = signal_baseline(self.world.store.conn, sid)
        self.assertTrue(info["learning"])
        self.assertEqual((info["observations"], info["needed"]), (12, 13))
        self.assertNotIn("baseline", info)

    def test_only_signals_seen_recently_are_judged(self):
        self.world.seen(868.3, wobble(40) + [30] * 5, start=0)  # ends at 22.5 min
        self.world.scan(0, 90)  # the scanner kept looking for another hour
        self.assertEqual(self.kinds(kinds=["louder", "quieter"]), [])

    def test_observations_without_snr_are_ignored(self):
        sid = self.world.seen(868.3, wobble(40))
        self.world.store.conn.execute("UPDATE observations SET snr_db=NULL WHERE signal_id=? AND id%2=0", (sid,))
        self.world.scan(0, 21)
        self.assertEqual(self.flags(), [])

    def test_judge_level_is_pure_and_symmetric(self):
        cfg = BaselineConfig()
        up = judge_level(wobble(30) + [30] * 5, cfg)
        down = judge_level(wobble(30) + [10] * 5, cfg)
        self.assertGreater(up["shift_db"], 0)
        self.assertLess(down["shift_db"], 0)
        self.assertIsNone(judge_level(wobble(30) + [20] * 5, cfg))
        self.assertIsNone(judge_level([20.0] * 5, cfg))


class GoneQuietTests(Base):
    def regular(self, minutes=60, center=868.3):
        return self.world.seen(center, wobble(minutes * 2), start=0)  # every 30 s

    def test_a_regular_signal_that_disappears(self):
        sid = self.regular()
        self.world.scan(0, 100)
        (f,) = self.flags()
        self.assertEqual((f["signal_id"], f["kind"], f["severity"]), (sid, "gone_quiet", "notice"))
        self.assertAlmostEqual(f["details"]["silent_s"], 40 * 60, delta=61)
        self.assertEqual(f["details"]["usual_gap_s"], 30)
        self.assertIn("not seen for 40 min of scanning", f["message"])

    def test_silence_while_the_frequency_was_not_scanned_does_not_count(self):
        self.regular()
        self.world.scan(0, 60)
        self.world.scan(60, 100, ranges=[(400.0, 500.0)])  # looking elsewhere
        self.assertEqual(self.flags(), [])

    def test_only_the_scanned_part_of_the_silence_counts(self):
        self.regular()
        self.world.scan(0, 60)
        self.world.scan(60, 90, ranges=[(400.0, 500.0)])
        self.world.scan(90, 100, ranges=EVERYTHING)  # 10 min of looking again: still above the 5 min minimum
        (f,) = self.flags()
        self.assertAlmostEqual(f["details"]["silent_s"], 10 * 60, delta=61)

    def test_a_short_silence_is_ignored(self):
        self.regular()
        self.world.scan(0, 63)
        self.assertEqual(self.flags(), [])

    def test_stopping_the_scanner_does_not_make_everything_silent(self):
        self.regular()
        self.world.scan(0, 60)  # the scan simply ends; "now" for the judgement is the end of the scan
        self.assertEqual(self.flags(), [])
        self.assertEqual(evaluate(self.world.store.conn)["ref"], at(60))

    def test_an_irregular_signal_is_judged_by_its_own_habits(self):
        sid = self.world.signal(433.9, first=0)
        starts = [0, 12, 30, 42, 66, 80, 100, 118]  # appears every 12-24 minutes
        for m in starts:
            self.world.observations(sid, [20.0], start=m)
        self.world.store.conn.execute("UPDATE signals SET last_seen=?, hits=60 WHERE id=?", (at(118), sid))
        self.world.scan(0, 160)  # 42 minutes without it, but its usual gaps are up to ~24
        self.assertEqual(self.flags(), [])

    def test_too_few_observations_to_know_what_usual_is(self):
        self.world.seen(868.3, wobble(6), start=0)
        self.world.store.conn.execute("UPDATE signals SET hits=60")
        self.world.scan(0, 100)
        self.assertEqual(self.flags(), [])

    def test_signals_gone_for_longer_than_the_memory_are_dropped(self):
        self.regular()
        self.world.scan(0, 60)
        self.world.scan(60 + 8 * 24 * 60, 60 + 8 * 24 * 60 + 30)  # 8 days later
        self.assertEqual(self.flags(), [])


class NewInQuietAreaTests(Base):
    def history(self):
        """Three hours of scanning everything, a signal at 100 MHz seen early on, and the scan ran on to hour 4."""
        self.world.signal(100.0, first=5, last=170)
        self.world.scan(0, 240)

    def test_a_new_signal_with_nothing_around_it(self):
        self.history()
        sid = self.world.signal(433.92, first=200)
        (f,) = self.flags()
        self.assertEqual((f["signal_id"], f["kind"], f["severity"]), (sid, "new_in_quiet", "alert"))
        self.assertEqual(f["details"]["covered_s"], 180 * 60)
        self.assertIn("new signal in a quiet area", f["message"])

    def test_a_neighbour_known_from_before_means_not_quiet(self):
        self.history()
        self.world.signal(433.5, first=10)  # 0.42 MHz away, long known
        self.world.signal(433.92, first=200)
        self.assertEqual(self.flags(), [])

    def test_a_distant_neighbour_does_not_matter(self):
        self.history()
        self.world.signal(436.0, first=10)  # 2 MHz away
        sid = self.world.signal(433.92, first=200)
        self.assertEqual(self.kinds(), [(sid, "new_in_quiet")])

    def test_neighbours_that_are_new_themselves_are_not_history(self):
        self.history()
        a = self.world.signal(433.92, first=200)
        b = self.world.signal(433.40, first=210)
        self.assertEqual(self.kinds(), [(a, "new_in_quiet"), (b, "new_in_quiet")])

    def test_wide_signals_widen_the_neighbourhood(self):
        self.history()
        self.world.signal(430.0, bw_khz=1000, first=10)
        self.world.signal(432.1, bw_khz=1000, first=10)
        sid = self.world.signal(433.92, bw_khz=3000, first=200)  # +-1.5 MHz reaches the one at 432.1
        self.assertEqual(self.flags(), [])
        self.assertIsNotNone(sid)

    def test_signals_first_seen_long_ago_are_not_new(self):
        self.history()
        self.world.signal(433.92, first=100)
        self.assertEqual(self.flags(), [])

    def test_a_range_that_was_not_scanned_before_is_not_judged(self):
        self.world.scan(0, 180, ranges=[(100.0, 200.0)])  # earlier scans looked elsewhere
        self.world.scan(180, 240)
        self.world.signal(433.92, first=200)
        self.assertEqual(self.flags(), [])

    def test_too_little_earlier_scanning_is_not_enough(self):
        self.world.scan(160, 240)  # only 20 minutes before the 1 hour window
        self.world.signal(433.92, first=200)
        self.assertEqual(self.flags(), [])
        info = evaluate(self.world.store.conn)
        self.assertEqual((info["learning"]["history_s"], info["learning"]["new_alerts_ready"]), (20 * 60, False))

    def test_without_scan_sessions_nothing_is_called_new(self):
        self.world.signal(433.92, first=200)
        self.world.store.conn.execute("INSERT INTO sweeps(ts,bins,detections,floor_db) VALUES(?,?,?,?)", (at(240), 100, 1, -70.0))
        self.assertEqual(self.flags(), [])


class EvaluateTests(Base):
    def test_empty_database(self):
        out = evaluate(self.world.store.conn)
        self.assertEqual((out["ref"], out["flags"]), (None, []))

    def test_old_database_without_scans_table_still_works(self):
        sid = self.world.seen(868.3, wobble(40) + [30] * 5)
        self.world.store.conn.execute("INSERT INTO sweeps(ts,bins,detections,floor_db) VALUES(?,?,?,?)", (at(23), 100, 1, -70.0))
        self.world.commit()
        self.world.store.conn.execute("DROP TABLE scans")
        out = evaluate(self.world.store.conn)
        self.assertEqual(out["ref"], at(23))
        self.assertEqual([(f["signal_id"], f["kind"]) for f in out["flags"]], [(sid, "louder")])

    def test_kind_filter(self):
        sid = self.world.seen(868.3, wobble(40) + [30] * 5, start=215)  # still active when the scan ends at 240
        self.world.signal(433.92, first=200)
        self.world.scan(0, 240)
        self.world.signal(100.0, first=5)
        self.assertEqual({f["kind"] for f in self.flags()}, {"louder", "new_in_quiet"})
        self.assertEqual(self.kinds(kinds=["louder"]), [(sid, "louder")])
        self.assertTrue(all(f["kind"] == "new_in_quiet" for f in self.flags(kinds=["new_in_quiet"])))
        with self.assertRaises(ValueError):
            evaluate(self.world.store.conn, kinds=["nonsense"])

    def test_alerts_come_before_notices(self):
        self.world.seen(868.3, wobble(40) + [12] * 5)  # quieter: a notice
        self.world.seen(433.9, wobble(40) + [30] * 5)  # louder: an alert
        self.world.scan(0, 23)
        self.assertEqual([f["kind"] for f in self.flags()], ["louder", "quieter"])

    def test_works_on_a_read_only_connection(self):
        sid = self.world.seen(868.3, wobble(40) + [30] * 5)
        self.world.scan(0, 23)
        self.world.commit()
        conn = query.connect_ro(self.path)
        self.addCleanup(conn.close)
        self.assertEqual([f["signal_id"] for f in evaluate(conn)["flags"]], [sid])

    def test_restricting_to_given_signals(self):
        a = self.world.seen(868.3, wobble(40) + [30] * 5)
        self.world.seen(433.9, wobble(40) + [30] * 5)
        self.world.scan(0, 23)
        self.world.commit()
        self.assertEqual([f["signal_id"] for f in evaluate(self.world.store.conn, signal_ids=[a])["flags"]], [a])
        self.assertEqual(evaluate(self.world.store.conn, signal_ids=[])["flags"], [])

    def test_flags_are_json_safe(self):
        import json

        self.world.seen(868.3, wobble(40) + [30] * 5)
        self.world.scan(0, 23)
        self.world.commit()
        json.dumps(evaluate(self.world.store.conn), allow_nan=False)

    def test_signal_baseline_describes_what_is_normal(self):
        sid = self.world.seen(868.3, wobble(60), step_s=30)
        self.world.scan(0, 30)
        self.world.commit()
        info = signal_baseline(self.world.store.conn, sid)
        self.assertFalse(info["learning"])
        self.assertAlmostEqual(info["baseline"]["median_db"], 20.0, delta=0.5)
        self.assertEqual(info["baseline"]["n"], 55)
        self.assertEqual(len(info["recent"]["values"]), 5)
        self.assertEqual((info["gaps"]["median_s"], info["gaps"]["p90_s"]), (30, 30))
        self.assertEqual(info["flags"], [])
        self.assertIsNone(signal_baseline(self.world.store.conn, 9999))


class HelperTests(unittest.TestCase):
    def test_parse_ts_accepts_what_the_scanner_writes(self):
        self.assertEqual(baseline.parse_ts("2026-10-09T08:00:00"), T0)
        self.assertEqual(baseline.parse_ts("2026-10-09T08:00:00.000000"), T0)
        self.assertEqual(baseline.parse_ts("2026-10-09T08:00:00.250"), datetime(2026, 10, 9, 8, 0, 0, 250000))
        self.assertIsNone(baseline.parse_ts("yesterday"))
        self.assertIsNone(baseline.parse_ts(None))

    def test_fmt_duration(self):
        self.assertEqual(baseline.fmt_duration(45), "45 s")
        self.assertEqual(baseline.fmt_duration(600), "10 min")
        self.assertEqual(baseline.fmt_duration(7200), "2.0 h")
        self.assertEqual(baseline.fmt_duration(3 * 86400), "3.0 days")

    def test_coverage_counts_only_scanned_time_at_that_frequency(self):
        cov = Coverage([
            (T0, datetime(2026, 10, 9, 9, 0, 0), [(100e6, 200e6)]),
            (datetime(2026, 10, 9, 9, 0, 0), datetime(2026, 10, 9, 10, 0, 0), [(150e6, 300e6)]),
        ])
        self.assertEqual(cov.seconds(120e6), 3600)
        self.assertEqual(cov.seconds(250e6), 3600)
        self.assertEqual(cov.seconds(170e6), 7200)
        self.assertEqual(cov.seconds(400e6), 0)
        self.assertEqual(cov.seconds(170e6, datetime(2026, 10, 9, 8, 30, 0), datetime(2026, 10, 9, 9, 30, 0)), 3600)
        self.assertEqual(cov.total_seconds(), 7200)


class ScanSessionStoreTests(unittest.TestCase):
    def test_begin_touch_and_close_record_the_session(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "s.db"))
            sid = st.begin_scan([(863, 870), (433, 435)], "run", started_at="2026-10-09T10:00:00")
            st.add_sweep_summary("2026-10-09T10:00:30", 100, 2, -70.0)
            st.add_sweep_summary("2026-10-09T10:00:20", 100, 2, -70.0)  # an older time never moves it back
            row = st.conn.execute("SELECT * FROM scans WHERE id=?", (sid,)).fetchone()
            self.assertEqual((row["started_at"], row["last_ts"], row["mode"]), ("2026-10-09T10:00:00", "2026-10-09T10:00:30", "run"))
            self.assertEqual(row["ranges"], "[[863000000.0, 870000000.0], [433000000.0, 435000000.0]]")
            st.close()
            con = sqlite3.connect(os.path.join(d, "s.db"))
            self.addCleanup(con.close)
            last = con.execute("SELECT last_ts FROM scans").fetchone()[0]
            self.assertGreater(last, "2026-10-09T10:00:30")  # close() stamps the real end

    def test_sweep_summaries_without_a_session_are_fine(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(os.path.join(d, "s.db"))
            st.add_sweep_summary("2026-10-09T10:00:30", 100, 2, -70.0)
            self.assertEqual(st.conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0], 0)
            st.close()

    def test_old_database_gets_the_table(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.db")
            con = sqlite3.connect(p)
            con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
            con.commit()
            con.close()
            st = Store(p)
            self.assertEqual(st.begin_scan([(1, 2)]), 1)
            st.close()


class AlertWatcherTests(Base):
    def setUp(self):
        super().setUp()
        from hackrf_scout import cli

        self.cli = cli
        self.sid = self.world.signal(868.3, first=0)
        self.world.observations(self.sid, wobble(40) + [29, 28, 30, 28, 29], start=215)
        self.world.scan(0, 240)
        self.world.commit()

    def logged(self):
        self.world.store.commit()
        return [tuple(r) for r in self.world.store.conn.execute("SELECT level, source, msg FROM log ORDER BY id")]

    def test_a_new_flag_is_logged_once(self):
        watcher = self.cli.AlertWatcher(self.world.store, interval=0)
        watcher.poll(quiet=True)
        watcher.poll(quiet=True)
        rows = self.logged()
        self.assertEqual(len(rows), 1)
        level, source, msg = rows[0]
        self.assertEqual((level, source), ("warn", "baseline"))
        self.assertTrue(msg.startswith(f"ALERT  #{self.sid} 868.300 MHz"), msg)
        self.assertIn("louder than usual", msg)

    def test_a_flag_that_clears_and_returns_is_logged_again(self):
        watcher = self.cli.AlertWatcher(self.world.store, interval=0)
        watcher.poll(quiet=True)
        self.world.store.conn.execute("UPDATE observations SET snr_db=20 WHERE signal_id=?", (self.sid,))
        watcher.poll(quiet=True)  # normal again
        self.world.store.conn.execute("UPDATE observations SET snr_db=40 WHERE signal_id=? AND ts >= ?", (self.sid, at(236)))  # three of the last five
        watcher.poll(quiet=True)
        self.assertEqual(len(self.logged()), 2)

    def test_notices_are_info_and_alerts_are_warnings(self):
        sid = self.world.signal(433.9, first=0)
        self.world.observations(sid, wobble(40) + [12] * 5, start=215)
        watcher = self.cli.AlertWatcher(self.world.store, interval=0)
        watcher.poll(quiet=True)
        levels = {msg.split()[0]: level for level, _, msg in self.logged()}
        self.assertEqual(levels, {"ALERT": "warn", "notice": "info"})

    def test_the_interval_throttles_and_force_overrides(self):
        watcher = self.cli.AlertWatcher(self.world.store, interval=3600)
        watcher.poll(quiet=True)
        self.assertEqual(self.logged(), [])
        watcher.poll(quiet=True, force=True)
        self.assertEqual(len(self.logged()), 1)

    def test_a_flood_of_flags_is_summarised(self):
        for k in range(5):
            sid = self.world.signal(433.0 + k, first=0)
            self.world.observations(sid, wobble(40) + [29] * 5, start=215)
        watcher = self.cli.AlertWatcher(self.world.store, interval=0, max_per_poll=3)
        watcher.poll(quiet=True)
        rows = self.logged()
        self.assertEqual(len(rows), 4)
        self.assertIn("and 3 more flag(s)", rows[-1][2])

    def test_a_failing_evaluation_never_stops_the_scan(self):
        from unittest import mock

        watcher = self.cli.AlertWatcher(self.world.store, interval=0)
        with mock.patch.object(baseline, "evaluate", side_effect=sqlite3.OperationalError("database is locked")):
            watcher.poll(quiet=True)
            watcher.poll(quiet=True)
        rows = self.logged()
        self.assertEqual(len(rows), 1)  # complained once, not on every poll
        self.assertIn("alerts unavailable", rows[0][2])


class PerformanceTests(unittest.TestCase):
    def test_a_big_database_is_judged_quickly(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(os.path.join(d, "big.db"))
            for k in range(3000):
                sid = w.signal(100.0 + k * 0.5, first=0)
                values = wobble(60) if k % 50 else wobble(55) + [30] * 5
                w.observations(sid, values, start=0)
            for m in range(0, 600, 60):
                w.scan(m, m + 60)
            w.commit()
            t = time.perf_counter()
            out = evaluate(w.store.conn, ref=T0 + timedelta(minutes=30))
            elapsed = time.perf_counter() - t
            w.close()
        self.assertEqual(sum(1 for f in out["flags"] if f["kind"] == "louder"), 60)
        self.assertLess(elapsed, 8.0, f"evaluate took {elapsed:.2f} s for 3000 signals")
        print(f"\n    evaluate(): 3000 signals x 60 observations in {elapsed:.2f} s", file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
