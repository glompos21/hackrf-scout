#!/usr/bin/env python3
"""Download the Artemis signal database (built from the SigID Wiki) and convert it to JSON.

This module only uses the standard library so it can be run in isolated mode:

    python3 -I hackrf_scout/artemisdb.py --dest data

The tar archive (a few hundred MB) is downloaded into a fresh temporary
directory, extracted with path-traversal protection, opened *read-only*, and
only the fields we need are copied to `artemis_signals.json`. The database is
produced by the Artemis project (https://github.com/AresValley/Artemis) from
https://www.sigidwiki.com - it is not redistributed with hackrf-scout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from contextlib import closing
from typing import Any, Dict, List

RELEASE_INFO_URL = "https://raw.githubusercontent.com/AresValley/Artemis/master/config/release-info.json"


def _get_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: str, expected_sha256: str | None = None) -> None:
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
    if expected_sha256:
        if h.hexdigest().lower() != expected_sha256.lower():
            raise RuntimeError("SHA-256 of the downloaded database does not match the published hash - refusing to use it")
        print("  SHA-256 verified", file=sys.stderr)


def _safe_extract(tar_path: str, dest: str) -> None:
    root = os.path.realpath(dest)
    with tarfile.open(tar_path) as tf:
        members = []
        for m in tf.getmembers():
            target = os.path.realpath(os.path.join(dest, m.name))
            if not (target == root or target.startswith(root + os.sep)):
                continue  # path traversal
            if m.issym() or m.islnk() or m.isdev():
                continue
            members.append(m)
        tf.extractall(dest, members=members)


def _find(root: str, filename: str) -> str:
    for base, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(base, filename)
    raise FileNotFoundError(f"{filename} not found in the extracted archive (database layout may have changed)")


def convert(sqlite_path: str) -> List[Dict[str, Any]]:
    uri = "file:" + sqlite_path.replace("?", "%3f") + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        cur = conn.cursor()
        try:
            sigs = cur.execute("SELECT SIG_ID, NAME, DESCRIPTION, URL FROM signals ORDER BY NAME").fetchall()
            freq = cur.execute("SELECT SIG_ID, VALUE FROM frequency").fetchall()
            bw = cur.execute("SELECT SIG_ID, VALUE FROM bandwidth").fetchall()
            mod = cur.execute("SELECT SIG_ID, VALUE FROM modulation").fetchall()
            loc = cur.execute("SELECT SIG_ID, VALUE FROM location").fetchall()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"Unexpected Artemis database layout: {exc}") from exc
        # categories are nice-to-have: the label table is called `categorylabel` in current
        # releases (older ones used `category_label`), and a missing table must not break us
        cat = []
        for label_table in ("categorylabel", "category_label"):
            try:
                cat = cur.execute(
                    f"SELECT category.SIG_ID, {label_table}.VALUE FROM category "
                    f"JOIN {label_table} ON category.CLB_ID = {label_table}.CLB_ID"
                ).fetchall()
                break
            except sqlite3.DatabaseError:
                continue

    def collect(rows, conv=lambda v: v):
        d: Dict[Any, list] = defaultdict(list)
        for sid, v in rows:
            if v not in (None, ""):
                try:
                    cv = conv(v)
                except (ValueError, TypeError):
                    continue
                if cv not in d[sid]:
                    d[sid].append(cv)
        return d

    freqs = collect(freq, lambda v: int(float(v)))
    bws = collect(bw, lambda v: int(float(v)))
    mods = collect(mod, lambda v: str(v).strip().upper())
    locs = collect(loc, lambda v: str(v).strip())
    cats = collect(cat, lambda v: str(v).strip().lower())

    out: List[Dict[str, Any]] = []
    for sid, name, desc, url in sigs:
        if not name or not freqs.get(sid):
            continue
        out.append(
            {
                "name": str(name).strip(),
                "description": (desc or "").strip(),
                "url": url.strip() if url and "sigidwiki" in str(url).lower() else None,
                "freqs_hz": freqs[sid],
                "bandwidths_hz": bws.get(sid, []),
                "modulations": mods.get(sid, []),
                "locations": locs.get(sid, []),
                "categories": cats.get(sid, []),
            }
        )
    return out


def update(dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    print("Fetching Artemis release info ...", file=sys.stderr)
    info = _get_json(RELEASE_INFO_URL)["sigID_DB"]
    print(f"Artemis-DB version {info.get('version')}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="artemis-db-") as tmp:
        tar_path = os.path.join(tmp, "db.tar")
        extract_dir = os.path.join(tmp, "extracted")
        os.mkdir(extract_dir)
        print(f"Downloading {info['url']}", file=sys.stderr)
        _download(info["url"], tar_path, info.get("sha256_hash"))
        print("Extracting ...", file=sys.stderr)
        _safe_extract(tar_path, extract_dir)
        signals = convert(_find(extract_dir, "data.sqlite"))
    out_path = os.path.join(dest_dir, "artemis_signals.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"source": "Artemis-DB / sigidwiki.com", "version": info.get("version"), "signals": signals}, fh)
    print(f"Wrote {len(signals)} signals to {out_path}", file=sys.stderr)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data", help="directory for artemis_signals.json (default: ./data)")
    ap.add_argument("--from-sqlite", help="convert an already extracted data.sqlite instead of downloading")
    args = ap.parse_args(argv)
    if args.from_sqlite:
        signals = convert(args.from_sqlite)
        os.makedirs(args.dest, exist_ok=True)
        out = os.path.join(args.dest, "artemis_signals.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"source": "Artemis-DB / sigidwiki.com", "signals": signals}, fh)
        print(f"Wrote {len(signals)} signals to {out}", file=sys.stderr)
    else:
        update(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
