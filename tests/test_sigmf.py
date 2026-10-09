"""SigMF recordings: metadata content, the reader, and the capture command's output."""

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hackrf_scout import sigmf  # noqa: E402
from hackrf_scout.capture import capture_signal, pick_sample_rate  # noqa: E402

try:
    from sigmf import sigmffile  # the reference implementation, only used to cross-check us

    HAVE_REFERENCE = True
except ImportError:
    HAVE_REFERENCE = False

FAKE_TRANSFER = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import sys
    import numpy as np
    a = sys.argv
    n = int(a[a.index("-n") + 1])
    x = np.clip(np.rint(np.random.default_rng(3).normal(0, 12, 2 * n)), -128, 127).astype(np.int8)
    open(a[a.index("-r") + 1], "wb").write(x.tobytes())
    """
)
SIGNAL = {"id": 2202, "center_hz": 2435697003.5, "bandwidth_hz": 3548835.6, "ident_name": "ISM 2.4 GHz (Wi-Fi, Bluetooth, drones, Zigbee)"}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bin = os.path.join(self.tmp.name, "hackrf_transfer")
        with open(self.bin, "w") as fh:
            fh.write(FAKE_TRANSFER)
        os.chmod(self.bin, os.stat(self.bin).st_mode | stat.S_IEXEC)

    def capture(self, signal=None, seconds=0.5, **kw):
        return capture_signal(signal or SIGNAL, os.path.join(self.tmp.name, "caps"), seconds=seconds, exe=self.bin, **kw)

    def meta(self, info):
        with open(info["meta"]) as fh:
            return json.load(fh)


class CaptureOutputTests(Base):
    def test_a_capture_is_a_sigmf_pair_and_nothing_else(self):
        info = self.capture()
        self.assertEqual(sorted(os.listdir(os.path.dirname(info["path"]))), [os.path.basename(info["path"]), os.path.basename(info["meta"])])
        self.assertTrue(info["path"].endswith(".sigmf-data"))
        self.assertTrue(info["meta"].endswith(".sigmf-meta"))
        self.assertEqual(info["path"][: -len(".sigmf-data")], info["meta"][: -len(".sigmf-meta")])
        self.assertRegex(os.path.basename(info["path"]), r"^sig2202_2435\.697MHz_\d{8}T\d{6}\.sigmf-data$")

    def test_the_data_file_is_what_hackrf_transfer_wrote(self):
        info = self.capture()
        self.assertEqual(os.path.getsize(info["path"]), 9_000_000 // 2 * 2 * 1)  # 9 Msps x 0.5 s x 2 bytes
        self.assertEqual(info["rate"], pick_sample_rate(SIGNAL["bandwidth_hz"]))

    def test_global_fields(self):
        info = self.capture(lna=32, vga=40, amp=True)
        g = self.meta(info)["global"]
        self.assertEqual(g["core:datatype"], "ci8")  # SigMF's name for signed 8-bit I/Q: no endianness suffix
        self.assertEqual(g["core:sample_rate"], 9_000_000)
        self.assertEqual(g["core:version"], sigmf.SIGMF_VERSION)
        self.assertTrue(g["core:recorder"].startswith("hackrf-scout "))
        self.assertIn("signal #2202", g["core:description"])
        self.assertEqual(g["hackrf_scout:signal_id"], 2202)
        self.assertEqual((g["hackrf_scout:lna_gain_db"], g["hackrf_scout:vga_gain_db"], g["hackrf_scout:amp_enabled"]), (32, 40, True))
        self.assertAlmostEqual(g["hackrf_scout:estimated_bandwidth_hz"], 3548835.6)
        self.assertTrue(all(k.startswith(("core:", "hackrf_scout:")) for k in g))

    def test_capture_segment_and_annotation(self):
        before = datetime.now(timezone.utc)
        info = self.capture()
        meta = self.meta(info)
        (cap,) = meta["captures"]
        self.assertEqual((cap["core:sample_start"], cap["core:frequency"]), (0, SIGNAL["center_hz"]))
        self.assertRegex(cap["core:datetime"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")  # UTC, milliseconds, Z
        when = datetime.strptime(cap["core:datetime"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        self.assertLess(abs((when - before).total_seconds()), 30)
        (note,) = meta["annotations"]
        self.assertEqual((note["core:sample_start"], note["core:sample_count"]), (0, 4_500_000))
        self.assertAlmostEqual(note["core:freq_lower_edge"], SIGNAL["center_hz"] - SIGNAL["bandwidth_hz"] / 2)
        self.assertAlmostEqual(note["core:freq_upper_edge"], SIGNAL["center_hz"] + SIGNAL["bandwidth_hz"] / 2)
        self.assertEqual(note["core:label"], SIGNAL["ident_name"])

    def test_unidentified_signals_are_labelled_as_such(self):
        info = self.capture(dict(SIGNAL, ident_name=None))
        self.assertEqual(self.meta(info)["annotations"][0]["core:label"], "unidentified")

    def test_checksum_matches_the_data(self):
        info = self.capture()
        with open(info["path"], "rb") as fh:
            self.assertEqual(self.meta(info)["global"]["core:sha512"], hashlib.sha512(fh.read()).hexdigest())

    def test_failed_capture_leaves_nothing_half_written(self):
        with open(self.bin, "w") as fh:
            fh.write("#!/bin/sh\necho 'hackrf_open() failed' >&2\nexit 1\n")
        with self.assertRaises(RuntimeError) as cm:
            self.capture()
        self.assertIn("hackrf_open() failed", str(cm.exception))
        caps = os.path.join(self.tmp.name, "caps")
        self.assertEqual(os.listdir(caps) if os.path.isdir(caps) else [], [])

    @unittest.skipUnless(HAVE_REFERENCE, "the reference sigmf package is not installed")
    def test_the_reference_implementation_accepts_it_and_reads_the_same_samples(self):
        import numpy as np

        info = self.capture()
        handle = sigmffile.fromfile(info["meta"])
        handle.validate()
        self.assertEqual((handle.sample_count, handle.sample_rate), (4_500_000, 9_000_000))
        samples = handle.read_samples(count=1000)
        with open(info["path"], "rb") as fh:
            raw = np.frombuffer(fh.read(2000), dtype=np.int8)
        self.assertTrue(np.array_equal(np.rint(samples.real * 128).astype(np.int8), raw[0::2]))
        self.assertTrue(np.array_equal(np.rint(samples.imag * 128).astype(np.int8), raw[1::2]))


class MetaWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = os.path.join(self.tmp.name, "x.sigmf-data")
        with open(self.data, "wb") as fh:
            fh.write(bytes(range(200)))

    def test_minimal_metadata(self):
        path = sigmf.write_meta(self.data, sample_rate=2e6, center_hz=433.92e6, started_at=datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc), checksum=False)
        self.assertEqual(path, os.path.join(self.tmp.name, "x.sigmf-meta"))
        with open(path) as fh:
            meta = json.load(fh)
        self.assertEqual(meta["annotations"], [])  # no annotation without frequency edges
        self.assertNotIn("core:sha512", meta["global"])
        self.assertEqual(meta["captures"][0]["core:datetime"], "2026-10-09T12:00:00.000Z")

    def test_wrong_suffix_is_refused(self):
        with self.assertRaises(ValueError):
            sigmf.write_meta(os.path.join(self.tmp.name, "x.cs8"), sample_rate=1, center_hz=1, started_at=datetime.now(timezone.utc))

    def test_timestamps_are_converted_to_utc(self):
        from datetime import timedelta

        local = datetime(2026, 10, 9, 15, 30, 0, 123000, tzinfo=timezone(timedelta(hours=3)))
        self.assertEqual(sigmf.utc_timestamp(local), "2026-10-09T12:30:00.123Z")
        self.assertRegex(sigmf.utc_timestamp(), r"Z$")

    def test_read_meta_round_trip(self):
        path = sigmf.write_meta(self.data, sample_rate=3e6, center_hz=433.92e6, started_at=datetime.now(timezone.utc))
        info = sigmf.read_meta(path)
        self.assertEqual((info["sample_rate"], info["center_hz"], info["data_path"]), (3e6, 433.92e6, self.data))

    def test_read_meta_refuses_what_it_cannot_replay(self):
        def write(meta):
            p = os.path.join(self.tmp.name, "bad.sigmf-meta")
            with open(p, "w") as fh:
                json.dump(meta, fh)
            return p

        with self.assertRaises(ValueError) as cm:
            sigmf.read_meta(write({"global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6}, "captures": [{"core:frequency": 1e8}]}))
        self.assertIn("ci8", str(cm.exception))
        with self.assertRaises(ValueError):
            sigmf.read_meta(write({"global": {"core:datatype": "ci8"}, "captures": [{}]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
