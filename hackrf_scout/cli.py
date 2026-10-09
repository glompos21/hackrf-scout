"""Command line interface: scan, run, report, identify, export, capture, update-db, simulate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from typing import Iterator, List, Optional, Sequence, Tuple

from . import __version__
from .capture import capture_signal
from .identify import Identifier
from .scanner import Scanner
from .store import Store
from .sweep import SweepProcess, build_sweep_command, iter_sweeps


# ----------------------------------------------------------------- helpers
def _ranges(values: Optional[Sequence[str]]) -> List[Tuple[float, float]]:
    out = []
    for v in values or []:
        a, b = v.split(":")
        out.append((float(a) * 1e6, float(b) * 1e6))
    return out


def _fmt_ident(r) -> str:
    name = r["ident_name"]
    if not name:
        return "-"
    if r["ident_source"] == "artemis" and r["ident_score"] is not None:
        return f"{name} [{r['ident_score']:.0f}%]"
    return f"{name} (bandplan)"


def _add_scan_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("scanning")
    g.add_argument("-f", "--freq", action="append", metavar="START:STOP",
                   help="range in MHz (repeatable). Default 1:6000, the full HackRF range")
    g.add_argument("-w", "--bin-width", type=int, default=100000, help="FFT bin width in Hz (default 100000)")
    g.add_argument("-l", "--lna", type=int, default=24, help="LNA gain 0-40 dB, 8 dB steps (default 24)")
    g.add_argument("-g", "--vga", type=int, default=20, help="VGA gain 0-62 dB, 2 dB steps (default 20)")
    g.add_argument("-a", "--amp", action="store_true", help="enable the +14 dB RF amp (off by default)")
    g.add_argument("--hackrf-sweep", default="hackrf_sweep", help="path to hackrf_sweep")
    d = p.add_argument_group("detection")
    d.add_argument("--snr", type=float, default=10.0, help="dB above local noise floor to count as a signal (default 10)")
    d.add_argument("--min-hits", type=int, default=3, help="sweeps a signal must appear in before it is stored (default 3)")
    d.add_argument("--expire", type=int, default=20, help="drop unconfirmed candidates after this many sweeps without a hit")
    d.add_argument("--ignore", action="append", metavar="START:STOP", help="MHz range to ignore (repeatable), e.g. known spurs")
    d.add_argument("--obs-interval", type=float, default=30.0, help="seconds between stored observations per signal (default 30)")
    i = p.add_argument_group("identification")
    i.add_argument("--signals", help="path to artemis_signals.json (see update-db)")
    i.add_argument("--region-keywords", default="", help="extra comma-separated region words, e.g. greece,cyprus")
    i.add_argument("--min-score", type=float, default=55.0, help="minimum Artemis match score to accept (default 55)")


def _identifier(args) -> Identifier:
    extra = [k.strip() for k in (args.region_keywords or "").split(",") if k.strip()]
    from .identify import DEFAULT_REGION_KEYWORDS

    return Identifier.load(args.signals, region_keywords=tuple(DEFAULT_REGION_KEYWORDS) + tuple(extra), min_score=args.min_score)


def _line_source(args, sweeps: Optional[int]):
    """Returns (iterator_of_lines, closer, process_or_None)."""
    src = getattr(args, "source", None)
    if src:
        if src == "-":
            return iter(sys.stdin), (lambda: None), None
        fh = open(src, "r", encoding="utf-8", errors="replace")
        return iter(fh), fh.close, None
    ranges = args.freq or ["1:6000"]
    cmd = build_sweep_command(args.hackrf_sweep, ranges, args.bin_width, args.lna, args.vga, args.amp, sweeps)
    proc = SweepProcess(cmd)
    return proc.lines(), proc.close, proc


def do_scan(args, store: Store, scanner: Scanner, duration: Optional[float], sweeps: Optional[int], quiet: bool) -> int:
    lines, close, proc = _line_source(args, sweeps)
    t_end = time.monotonic() + duration if duration else None
    n = 0
    try:
        for sweep in iter_sweeps(lines):
            res = scanner.process(sweep)
            n += 1
            if not quiet and n % 10 == 0:
                print(
                    f"  sweep {res['sweep']}: {res['detections']} detections, floor {res['floor_db']:.1f} dB, "
                    f"{sum(1 for t in scanner.tracks if t.id is not None)} signals stored",
                    file=sys.stderr,
                )
            if (t_end and time.monotonic() >= t_end) or (sweeps and n >= sweeps):
                break
    except KeyboardInterrupt:
        print("\nStopping ...", file=sys.stderr)
    finally:
        rc = 0
        close_result = close()
        if proc is not None:
            rc = close_result if isinstance(close_result, int) else 0
        store.commit()
    if proc is not None and n == 0:
        print("No sweep data received from hackrf_sweep.", file=sys.stderr)
        tail = proc.error_text()
        if tail:
            print(tail, file=sys.stderr)
        print("Check: `hackrf_info` shows your HackRF, USB permissions (udev rules / sudo), no other app using it.", file=sys.stderr)
        return -1
    return n


# ---------------------------------------------------------------- commands
def cmd_scan(args) -> int:
    store = Store(args.db)
    ident = _identifier(args)
    if not ident.has_database:
        print("Note: no Artemis database found - using the built-in bandplan only. Run `hackrf-scout update-db` once.", file=sys.stderr)
    scanner = Scanner(store, ident, args.snr, args.min_hits, args.expire, args.obs_interval, _ranges(args.ignore))
    n = do_scan(args, store, scanner, args.duration, args.sweeps, args.quiet)
    if n < 0:
        return 2
    total = len(store.load_signals())
    print(f"Done: {n} sweeps processed, {len(scanner.new_signals)} new signals, {total} stored in {args.db}")
    store.close()
    return 0


def cmd_run(args) -> int:
    store = Store(args.db)
    ident = _identifier(args)
    scanner = Scanner(store, ident, args.snr, args.min_hits, args.expire, args.obs_interval, _ranges(args.ignore))
    cycle = 0
    try:
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            print(f"[cycle {cycle}] scanning for {args.scan_seconds:.0f} s ...", file=sys.stderr)
            n = do_scan(args, store, scanner, args.scan_seconds, None, args.quiet)
            if n < 0:
                return 2
            if args.capture_seconds > 0:
                _do_captures(args, store, ident, args.capture_max, args.capture_which)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    store.close()
    return 0


def _do_captures(args, store: Store, ident: Identifier, limit: int, which: str) -> int:
    rows = store.query_signals(order="max_snr")
    rows = [r for r in rows if not r["captured"] and r["max_snr"] >= args.capture_min_snr]
    if which == "unidentified":
        rows = [r for r in rows if r["ident_source"] != "artemis"]
    done = 0
    for r in rows[:limit]:
        try:
            info = capture_signal(r, args.capture_dir, args.capture_seconds, args.lna, args.vga, args.amp, args.hackrf_transfer)
        except RuntimeError as exc:
            print(f"  capture of #{r['id']} failed: {exc}", file=sys.stderr)
            continue
        store.add_capture(r["id"], datetime.now().isoformat(timespec="seconds"), info["path"], info["center_hz"], info["rate"], info["seconds"])
        store.mark_captured(r["id"])
        store.commit()
        print(f"  captured #{r['id']} {r['center_hz'] / 1e6:.3f} MHz -> {info['path']}")
        done += 1
    return done


def cmd_capture(args) -> int:
    store = Store(args.db)
    n = _do_captures(args, store, _identifier(args), args.top, args.which)
    print(f"{n} capture(s) written to {args.capture_dir}")
    store.close()
    return 0


def cmd_identify(args) -> int:
    store = Store(args.db)
    ident = _identifier(args)
    if not ident.has_database:
        print("No Artemis database found; only the bandplan will be used. Run `hackrf-scout update-db`.", file=sys.stderr)
    changed = 0
    now = datetime.now().isoformat(timespec="seconds")
    for r in store.query_signals():
        if not args.all and r["ident_source"] == "artemis":
            continue
        res = ident.identify(r["center_hz"], r["bandwidth_hz"], args.bin_width)
        store.set_ident(r["id"], res, now)
        changed += 1
    store.close()
    print(f"Re-identified {changed} signals")
    return 0


def cmd_report(args) -> int:
    store = Store(args.db)
    band = None
    if args.band:
        a, b = args.band.split(":")
        band = (float(a) * 1e6, float(b) * 1e6)
    rows = store.query_signals(args.unidentified, args.min_hits, args.sort, args.limit, band)
    sweeps = store.sweep_count()
    if not rows:
        print("No signals stored yet.")
        return 0
    print(f"{'ID':>4} {'MHz':>11} {'BW kHz':>9} {'peak dB':>8} {'SNR':>5} {'hits':>5} {'duty%':>6}  {'last seen':<19}  identification / service")
    for r in rows:
        duty = min(100.0, 100.0 * r["hits"] / max(1, sweeps - r["first_sweep"] + 1))
        what = _fmt_ident(r)
        if r["service"] and r["ident_source"] == "artemis":
            what += f"  | {r['service'].split(';')[0]}"
        print(
            f"{r['id']:>4} {r['center_hz'] / 1e6:>11.3f} {r['bandwidth_hz'] / 1e3:>9.1f} {r['max_db']:>8.1f} "
            f"{r['max_snr']:>5.1f} {r['hits']:>5d} {duty:>6.1f}  {r['last_seen'][:19]:<19}  {what}"
        )
        if args.verbose and r["ident_json"]:
            for c in json.loads(r["ident_json"])[:3]:
                mods = ",".join(c.get("modulations") or []) or "?"
                print(f"        - {c['name']} ({c['score']:.0f}%, mod: {mods}) {c.get('url') or ''}")
    print(f"\n{len(rows)} signals shown; {sweeps} sweeps recorded in {args.db}")
    store.close()
    return 0


def cmd_export(args) -> int:
    store = Store(args.db)
    rows = store.query_signals(order="center_hz")
    out = open(args.out, "w", newline="", encoding="utf-8") if args.out != "-" else sys.stdout
    cols = ["id", "center_hz", "bandwidth_hz", "first_seen", "last_seen", "hits", "max_db", "avg_db", "max_snr",
            "ident_name", "ident_score", "ident_source", "ident_url", "service", "label", "notes"]
    if args.format == "csv":
        w = csv.writer(out)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    else:
        data = []
        for r in rows:
            d = {c: r[c] for c in cols}
            d["candidates"] = json.loads(r["ident_json"]) if r["ident_json"] else []
            if args.observations:
                d["observations"] = [dict(o) for o in store.observations(r["id"])]
            data.append(d)
        json.dump(data, out, indent=2)
    if out is not sys.stdout:
        out.close()
        print(f"Wrote {len(rows)} signals to {args.out}")
    store.close()
    return 0


def cmd_update_db(args) -> int:
    from . import artemisdb

    artemisdb.update(args.dest)
    return 0


def cmd_simulate(args) -> int:
    from . import simulate

    return simulate.main(args.rest)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hackrf-scout", description="Sweep the HackRF range, find signals, identify and store them.")
    ap.add_argument("--version", action="version", version=f"hackrf-scout {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--db", default="scout.db", help="SQLite database file (default scout.db)")

    p = sub.add_parser("scan", help="sweep, detect, identify and store")
    common(p)
    _add_scan_args(p)
    p.add_argument("--sweeps", type=int, help="stop after N sweeps")
    p.add_argument("--duration", type=float, help="stop after N seconds")
    p.add_argument("--source", help="replay a saved hackrf_sweep CSV file ('-' = stdin) instead of using hardware")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("run", help="loop: scan, then record IQ of new signals, repeat")
    common(p)
    _add_scan_args(p)
    p.add_argument("--scan-seconds", type=float, default=600, help="scan time per cycle (default 600)")
    p.add_argument("--cycles", type=int, default=0, help="number of cycles, 0 = until Ctrl-C")
    p.add_argument("--capture-seconds", type=float, default=5, help="IQ seconds per signal, 0 disables captures")
    p.add_argument("--capture-max", type=int, default=3, help="max captures per cycle")
    p.add_argument("--capture-which", choices=["new", "unidentified"], default="unidentified")
    p.add_argument("--capture-min-snr", type=float, default=15.0)
    p.add_argument("--capture-dir", default="captures")
    p.add_argument("--hackrf-transfer", default="hackrf_transfer")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("capture", help="record IQ of stored signals that have not been captured yet")
    common(p)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--which", choices=["new", "unidentified"], default="new")
    p.add_argument("--capture-seconds", type=float, default=5)
    p.add_argument("--capture-min-snr", type=float, default=10.0)
    p.add_argument("--capture-dir", default="captures")
    p.add_argument("--hackrf-transfer", default="hackrf_transfer")
    p.add_argument("-l", "--lna", type=int, default=24)
    p.add_argument("-g", "--vga", type=int, default=20)
    p.add_argument("-a", "--amp", action="store_true")
    p.add_argument("--signals")
    p.add_argument("--region-keywords", default="")
    p.add_argument("--min-score", type=float, default=55.0)
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("identify", help="(re-)identify stored signals, e.g. after update-db")
    common(p)
    p.add_argument("--all", action="store_true", help="redo signals that already have an Artemis match")
    p.add_argument("--bin-width", type=float, default=100e3)
    p.add_argument("--signals")
    p.add_argument("--region-keywords", default="")
    p.add_argument("--min-score", type=float, default=55.0)
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("report", help="list stored signals")
    common(p)
    p.add_argument("--unidentified", action="store_true", help="only signals with no identification at all")
    p.add_argument("--min-hits", type=int, default=0)
    p.add_argument("--sort", default="center_hz", help="center_hz, hits, max_snr, last_seen, first_seen, bandwidth_hz, max_db")
    p.add_argument("--band", metavar="START:STOP", help="only this MHz range")
    p.add_argument("--limit", type=int)
    p.add_argument("-v", "--verbose", action="store_true", help="show top candidate matches")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("export", help="export signals as CSV or JSON")
    common(p)
    p.add_argument("--format", choices=["csv", "json"], default="csv")
    p.add_argument("--out", default="-")
    p.add_argument("--observations", action="store_true", help="include observation history (JSON only)")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("update-db", help="download the Artemis/SigID signal database (once, ~300 MB)")
    p.add_argument("--dest", default="data")
    p.set_defaults(fn=cmd_update_db)

    p = sub.add_parser("simulate", help="print simulated hackrf_sweep output (for testing without hardware)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_simulate)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "simulate":  # pass options straight through to the simulator
        from . import simulate

        return simulate.main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
