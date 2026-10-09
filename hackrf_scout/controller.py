"""Single-owner lock for the HackRF, and start/stop control of a scanner process.

The HackRF can only be used by one program at a time, so every command that touches
the hardware (`scan`, `run`, `capture`) holds `HardwareLock` for its whole lifetime.
The lock is an `flock` on a file, so the kernel drops it when the process dies for
any reason and there are no stale lock files. Next to it, `scanner.json` says who
holds it (pid, mode, arguments), which is what the web UI shows.

`ScannerController` is what the web server uses: it starts `hackrf-scout scan|run` as a
detached child (it keeps running if the web server restarts) and stops it gracefully.
State lives in plain files under `~/.hackrf-scout/` (or `$HACKRF_SCOUT_STATE_DIR`), never
in the SQLite database, so the web process can keep the database open read-only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import bands as bandlib
from . import watch

try:  # POSIX only; on other platforms the lock is a no-op and control is unavailable
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


class HardwareBusy(RuntimeError):
    """Another hackrf-scout process already owns the HackRF."""


class ControlError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def state_dir(directory: Optional[str] = None) -> Path:
    d = directory or os.environ.get("HACKRF_SCOUT_STATE_DIR")
    return Path(d) if d else Path.home() / ".hackrf-scout"


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_status(directory: Optional[str] = None) -> Dict[str, Any]:
    """Who holds the HackRF right now? Safe to call from any process."""
    d = state_dir(directory)
    if fcntl is None:
        return {"supported": False, "running": False}
    try:
        fd = os.open(d / "scanner.lock", os.O_RDWR)
    except OSError:
        return {"supported": True, "running": False}
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return {"supported": True, "running": False}
        except OSError:
            pass  # held by someone
    finally:
        os.close(fd)
    info: Dict[str, Any] = {}
    try:
        with open(d / "scanner.json", "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            info = loaded
    except (OSError, ValueError):
        pass
    out = {"supported": True, "running": True, "starting": not _pid_alive(info.get("pid"))}
    if not out["starting"]:
        out.update(info)
    return out


class HardwareLock:
    def __init__(self, directory: Optional[str] = None):
        self.dir = state_dir(directory)
        self.info: Dict[str, Any] = {}
        self._fd: Optional[int] = None

    def acquire(self, **info: Any) -> "HardwareLock":
        if fcntl is None:
            return self
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.dir / "scanner.lock", os.O_RDWR | os.O_CREAT, 0o600)
        # a status check holds the lock for microseconds; retry briefly before calling it busy
        for attempt in range(6):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if attempt == 5:
                    os.close(fd)
                    cur = read_status(str(self.dir))
                    who = f" (pid {cur['pid']}, {cur.get('mode', 'scan')})" if cur.get("pid") else ""
                    raise HardwareBusy(f"the HackRF is already in use by another hackrf-scout process{who}")
                time.sleep(0.05)
        self._fd = fd
        self.info = {"pid": os.getpid(), "started_at": datetime.now().isoformat(timespec="seconds"), **info}
        self._write()
        return self

    def update(self, **kw: Any) -> None:
        self.info.update(kw)
        if self._fd is not None:
            self._write()

    def _write(self) -> None:
        tmp = self.dir / "scanner.json.tmp"
        tmp.write_text(json.dumps(self.info), encoding="utf-8")
        os.replace(tmp, self.dir / "scanner.json")

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            (self.dir / "scanner.json").unlink()
        except OSError:
            pass
        os.close(self._fd)  # drops the flock
        self._fd = None

    def __enter__(self) -> "HardwareLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# --------------------------------------------------------------------------- argument building
_REGION = re.compile(r"^[A-Za-z0-9 ,._-]{0,100}$")
_SCAN_KEYS = {"mode", "bands", "lna", "vga", "amp", "bin_width", "snr", "min_hits", "expire", "obs_interval",
              "region_keywords", "duration", "floor_alpha", "hysteresis", "overload_db"}
_RUN_KEYS = {"scan_seconds", "cycles", "capture_seconds", "capture_max", "capture_which", "capture_min_snr"}
_WATCH_KEYS = {"mode", "bands", "lna", "vga", "amp", "snr", "region_keywords", "duration", "slice_ms", "min_bursts"}


def _num(params: Dict[str, Any], key: str, default: Any, lo: float, hi: float, integer: bool = False, step: int = 0) -> Any:
    v = params.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        if isinstance(v, str) and v.strip():
            try:
                v = float(v)
            except ValueError:
                raise ValueError(f"{key} must be a number")
        else:
            raise ValueError(f"{key} must be a number")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{key} must be a finite number")
    if integer:
        if int(v) != v:
            raise ValueError(f"{key} must be a whole number")
        v = int(v)
    if not (lo <= v <= hi):
        raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
    if step and v % step:
        raise ValueError(f"{key} must be a multiple of {step}")
    return v


def build_scan_args(
    params: Dict[str, Any],
    *,
    db: str,
    registry: Optional[bandlib.Registry] = None,
    capture_dir: str = "captures",
    hackrf_sweep: Optional[str] = None,
    hackrf_transfer: Optional[str] = None,
) -> List[str]:
    """Turn the web form into a `hackrf-scout` argument list (never a shell string).

    Everything is validated against fixed ranges, unknown keys are rejected, and the
    database, executables and capture directory come from the server, not the request.
    """
    if not isinstance(params, dict):
        raise ValueError("parameters must be an object")
    mode = params.get("mode", "scan")
    if mode not in ("scan", "run", "watch"):
        raise ValueError("mode must be 'scan', 'run' or 'watch'")
    allowed = _WATCH_KEYS if mode == "watch" else set(_SCAN_KEYS) | (_RUN_KEYS if mode == "run" else set())
    extra = sorted(set(params) - allowed)
    if extra:
        raise ValueError(f"unknown or unsupported parameter(s) for mode '{mode}': {', '.join(extra)}")

    registry = registry or bandlib.Registry()
    if mode == "watch":
        return _build_watch_args(params, db, registry, hackrf_transfer)
    specs = params.get("bands") or []
    if not isinstance(specs, list) or len(specs) > 24 or not all(isinstance(s, str) for s in specs):
        raise ValueError("bands must be a list of up to 24 band names or START:STOP ranges")
    ranges = bandlib.merge_sweep_ranges(registry.resolve_many(specs)) if specs else []

    argv = [sys.executable, "-m", "hackrf_scout", mode, "--db", os.path.abspath(db), "--quiet"]
    for r in ranges:
        argv += ["-f", r]
    argv += ["-w", str(_num(params, "bin_width", 100000, 2445, 5_000_000, integer=True))]
    argv += ["-l", str(_num(params, "lna", 24, 0, 40, integer=True, step=8))]
    argv += ["-g", str(_num(params, "vga", 20, 0, 62, integer=True, step=2))]
    amp = params.get("amp", False)
    if not isinstance(amp, bool):
        raise ValueError("amp must be true or false")
    if amp:
        argv.append("-a")
    argv += ["--snr", f"{_num(params, 'snr', 10.0, 1, 60):g}"]
    argv += ["--min-hits", str(_num(params, "min_hits", 3, 1, 50, integer=True))]
    argv += ["--expire", str(_num(params, "expire", 20, 1, 1000, integer=True))]
    argv += ["--obs-interval", f"{_num(params, 'obs_interval', 30.0, 1, 3600):g}"]
    argv += ["--floor-alpha", f"{_num(params, 'floor_alpha', 0.2, 0.001, 1):g}"]
    argv += ["--hysteresis", f"{_num(params, 'hysteresis', 3.0, 0, 30):g}"]
    argv += ["--overload-db", f"{_num(params, 'overload_db', 6.0, 0, 60):g}"]
    region = params.get("region_keywords", "")
    if not isinstance(region, str) or not _REGION.match(region):
        raise ValueError("region_keywords may only contain letters, digits, spaces and , . _ -")
    if region.strip():
        argv += ["--region-keywords", region.strip()]
    if hackrf_sweep:
        argv += ["--hackrf-sweep", hackrf_sweep]

    if mode == "scan":
        if params.get("duration") not in (None, "", 0):
            argv += ["--duration", f"{_num(params, 'duration', None, 1, 86400):g}"]
    else:
        argv += ["--scan-seconds", f"{_num(params, 'scan_seconds', 600, 5, 86400):g}"]
        argv += ["--cycles", str(_num(params, "cycles", 0, 0, 10000, integer=True))]
        argv += ["--capture-seconds", f"{_num(params, 'capture_seconds', 5, 0, 60):g}"]
        argv += ["--capture-max", str(_num(params, "capture_max", 3, 0, 50, integer=True))]
        which = params.get("capture_which", "unidentified")
        if which not in ("new", "unidentified"):
            raise ValueError("capture_which must be 'new' or 'unidentified'")
        argv += ["--capture-which", which]
        argv += ["--capture-min-snr", f"{_num(params, 'capture_min_snr', 15.0, 0, 60):g}"]
        argv += ["--capture-dir", os.path.abspath(capture_dir)]
        if hackrf_transfer:
            argv += ["--hackrf-transfer", hackrf_transfer]
    return argv


def _build_watch_args(params: Dict[str, Any], db: str, registry: bandlib.Registry, hackrf_transfer: Optional[str]) -> List[str]:
    specs = params.get("bands") or []
    if not (isinstance(specs, list) and len(specs) == 1 and isinstance(specs[0], str)):
        raise ValueError("watch mode needs exactly one band (a preset name or START:STOP in MHz)")
    band = registry.resolve_many(specs)[0]
    watch.plan_band(band.lo_hz, band.hi_hz)  # raises a readable error if the band is too wide for one HackRF window
    argv = [sys.executable, "-m", "hackrf_scout", "watch", "--db", os.path.abspath(db), "--quiet",
            "--band", f"{band.lo_hz / 1e6:.6f}:{band.hi_hz / 1e6:.6f}"]
    argv += ["-l", str(_num(params, "lna", 24, 0, 40, integer=True, step=8))]
    argv += ["-g", str(_num(params, "vga", 20, 0, 62, integer=True, step=2))]
    amp = params.get("amp", False)
    if not isinstance(amp, bool):
        raise ValueError("amp must be true or false")
    if amp:
        argv.append("-a")
    argv += ["--snr", f"{_num(params, 'snr', 10.0, 1, 60):g}"]
    argv += ["--slice-ms", f"{_num(params, 'slice_ms', 20.0, 5, 500):g}"]
    argv += ["--min-bursts", str(_num(params, "min_bursts", 2, 1, 100, integer=True))]
    region = params.get("region_keywords", "")
    if not isinstance(region, str) or not _REGION.match(region):
        raise ValueError("region_keywords may only contain letters, digits, spaces and , . _ -")
    if region.strip():
        argv += ["--region-keywords", region.strip()]
    if params.get("duration") not in (None, "", 0):
        argv += ["--duration", f"{_num(params, 'duration', None, 1, 86400):g}"]
    if hackrf_transfer:
        argv += ["--hackrf-transfer", hackrf_transfer]
    return argv


# --------------------------------------------------------------------------- the controller
class ScannerController:
    def __init__(
        self,
        db_path: str,
        directory: Optional[str] = None,
        registry: Optional[bandlib.Registry] = None,
        capture_dir: str = "captures",
        hackrf_sweep: Optional[str] = None,
        hackrf_transfer: Optional[str] = None,
        cwd: Optional[str] = None,
        startup_wait: float = 6.0,
        stop_wait: float = 20.0,
        term_wait: float = 5.0,
    ):
        self.db_path = db_path
        self.dir = state_dir(directory)
        self.registry = registry or bandlib.Registry()
        self.capture_dir = capture_dir
        self.hackrf_sweep = hackrf_sweep
        self.hackrf_transfer = hackrf_transfer
        self.cwd = cwd
        self.startup_wait = startup_wait
        self.stop_wait = stop_wait
        self.term_wait = term_wait
        self._mutex = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stopping = False
        self._last: Optional[Dict[str, Any]] = None

    # ---- status ----------------------------------------------------------
    def _reap(self) -> None:
        if self._proc is not None:
            rc = self._proc.poll()
            if rc is not None:
                self._last = {"code": rc, "at": datetime.now().isoformat(timespec="seconds"), "output_tail": self._output_tail()}
                self._proc = None

    def _output_tail(self, lines: int = 15) -> str:
        try:
            text = (self.dir / "scanner.out").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])

    def status(self) -> Dict[str, Any]:
        self._reap()
        st = read_status(str(self.dir))
        if not st.get("supported", True):
            return {"supported": False, "state": "unsupported", "message": "start/stop needs Linux or macOS"}
        if not st["running"]:
            state = "stopped"
        elif self._stopping:
            state = "stopping"
        elif st.get("starting"):
            state = "starting"
        else:
            state = "running"
        out: Dict[str, Any] = {"supported": True, "state": state}
        for k in ("pid", "mode", "phase", "started_at", "db", "argv"):
            if k in st:
                out[k] = st[k]
        out["managed"] = bool(self._proc is not None and st.get("pid") == self._proc.pid)
        if state == "stopped" and self._last:
            out["last_exit"] = self._last
        return out

    # ---- start / stop ----------------------------------------------------
    def start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._mutex:
            if fcntl is None:
                raise ControlError("start/stop needs Linux or macOS", 501)
            self._reap()
            cur = read_status(str(self.dir))
            if cur["running"]:
                raise ControlError(f"a scanner is already running (pid {cur.get('pid', '?')}); stop it first", 409)
            try:
                argv = build_scan_args(
                    params or {}, db=self.db_path, registry=self.registry, capture_dir=self.capture_dir,
                    hackrf_sweep=self.hackrf_sweep, hackrf_transfer=self.hackrf_transfer,
                )
            except ValueError as exc:
                raise ControlError(str(exc), 400)
            self.dir.mkdir(parents=True, exist_ok=True)
            # the child must look at the same lock/state files we do, wherever they live
            env = dict(os.environ, PYTHONUNBUFFERED="1", HACKRF_SCOUT_STATE_DIR=str(self.dir))
            pkg_parent = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            with open(self.dir / "scanner.out", "wb") as out:
                try:
                    proc = subprocess.Popen(
                        argv, cwd=self.cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                        start_new_session=True, env=env, close_fds=True,
                    )
                except OSError as exc:
                    raise ControlError(f"could not start the scanner: {exc}", 500)
            self._proc, self._last, self._stopping = proc, None, False
            deadline = time.monotonic() + self.startup_wait
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    self._reap()
                    tail = (self._last or {}).get("output_tail") or "no output"
                    raise ControlError(f"the scanner exited right away (code {proc.returncode}): {tail[-400:]}", 500)
                st = read_status(str(self.dir))
                if st["running"] and st.get("pid") == proc.pid:
                    break
                time.sleep(0.1)
        return self.status()

    def _wait_free(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._reap()
            if not read_status(str(self.dir))["running"]:
                self._join_child()
                return True
            time.sleep(0.1)
        free = not read_status(str(self.dir))["running"]
        if free:
            self._join_child()
        return free

    def _join_child(self) -> None:
        """The lock is released a moment before the process finishes exiting; wait so the exit code is known."""
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            self._reap()

    def stop(self) -> Dict[str, Any]:
        with self._mutex:
            st = read_status(str(self.dir))
            if not st["running"]:
                return self.status()
            pid = st.get("pid")
            if not _pid_alive(pid):  # lock held but the owner has not written its pid yet
                end = time.monotonic() + 3
                while time.monotonic() < end and not _pid_alive(pid):
                    time.sleep(0.1)
                    pid = read_status(str(self.dir)).get("pid")
                if not _pid_alive(pid):
                    raise ControlError("the scanner is starting up; try again in a moment", 409)
            self._stopping = True
            try:
                # SIGINT/SIGTERM both make the scanner finish the current sweep, commit and exit
                for sig, wait in ((signal.SIGINT, self.stop_wait), (signal.SIGTERM, self.term_wait)):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        break
                    if self._wait_free(wait):
                        break
                else:
                    try:
                        # only take the whole process group if the scanner leads its own (web-started)
                        if os.getpgid(pid) == pid:
                            os.killpg(pid, signal.SIGKILL)
                        else:
                            os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    self._wait_free(3)
            except PermissionError:
                raise ControlError(f"not allowed to stop process {pid} (owned by another user)", 403)
            finally:
                self._stopping = False
        return self.status()

    def check_device(self) -> Dict[str, Any]:
        if read_status(str(self.dir))["running"]:
            return {"ok": True, "busy": True, "output": "A scanner is running, so the HackRF is in use and cannot be probed."}
        exe = shutil.which("hackrf_info")
        if exe is None:
            raise ControlError("'hackrf_info' not found. Install the HackRF tools (sudo apt install hackrf).", 404)
        try:
            res = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return {"ok": False, "busy": False, "output": "hackrf_info timed out"}
        text = (res.stdout + res.stderr).strip()
        return {"ok": res.returncode == 0, "busy": False, "output": text[-2000:]}
