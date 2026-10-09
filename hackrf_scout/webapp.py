"""Browser interface: live log, database browser, band filters, optional start/stop.

Needs the `web` extra (`pip install 'hackrf-scout[web]'`). The server only ever opens the
database read-only. The one thing it can change is the scanner process, and only when started
with --allow-control, always behind a bearer token.
"""

import asyncio
import hmac
import json
import os
import secrets
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders

from . import __version__
from . import baseline
from . import bands as bandlib
from . import query
from .controller import ControlError, ScannerController

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}
WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_NAMES or host.startswith("127.")


@dataclass
class WebConfig:
    db_path: str
    controller: ScannerController
    registry: bandlib.Registry = field(default_factory=bandlib.Registry)
    token: Optional[str] = None
    allow_control: bool = False
    host: str = "127.0.0.1"
    allowed_hosts: Set[str] = field(default_factory=set)
    band_warnings: List[str] = field(default_factory=list)
    anomaly_ttl: float = 10.0  # seconds a baseline evaluation is reused between requests


def _hostname(host_header: str) -> str:
    v = host_header.strip().lower()
    if v.startswith("["):
        end = v.find("]")
        return v[1:end] if end > 0 else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


class SecurityMiddleware:
    """Pure ASGI (so it never interferes with the streaming log): Host and Origin checks, security headers.

    The Host check is what stops DNS-rebinding (a web page you visit resolving its own name to
    127.0.0.1); the Origin check rejects cross-site POSTs. The bearer token is the real gate.
    """

    def __init__(self, app, cfg: WebConfig):
        self.app = app
        self.allowed = {h.lower() for h in cfg.allowed_hosts}
        loopback = is_loopback(cfg.host)
        if loopback:
            self.allowed |= LOOPBACK_NAMES
        if cfg.host.lower() not in WILDCARD_HOSTS:
            self.allowed.add(cfg.host.lower())
        self.enforce_host = loopback or bool(cfg.allowed_hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        problem = None
        if self.enforce_host and _hostname(host) not in self.allowed:
            problem = "host not allowed (use --allowed-host to add it)"
        elif scope["method"] in ("POST", "PUT", "PATCH", "DELETE"):
            origin = headers.get("origin")
            if origin is not None and urlsplit(origin).netloc.lower() != host.lower():
                problem = "cross-origin request rejected"
        if problem:
            resp = JSONResponse({"detail": problem}, status_code=403)
            await resp(scope, receive, send)
            return

        is_api = scope["path"].startswith("/api/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["Content-Security-Policy"] = CSP
                h["X-Content-Type-Options"] = "nosniff"
                h["Referrer-Policy"] = "no-referrer"
                h["X-Frame-Options"] = "DENY"
                if is_api:
                    h["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ------------------------------------------------------------------------------------ log stream
def _poll_log(db_path: str, last: int):
    try:
        conn = query.connect_ro(db_path)
    except query.DatabaseUnavailable:
        return [], last
    try:
        res = query.read_log(conn, after_id=last, limit=500)
    except sqlite3.Error:
        return [], last
    finally:
        conn.close()
    if res.get("max_id", 0) < last:
        last = res["max_id"]  # the log was reset (new database): follow it from the start
    return res["items"], last


async def log_event_stream(db_path: str, after_id: int, poll: float = 1.0, heartbeat: float = 15.0):
    """Server-sent events for new log rows. `id:` is the row id, so a reconnecting client resumes exactly."""
    last = after_id
    quiet = 0.0
    yield "retry: 3000\n\n"
    while True:
        items, last = await asyncio.to_thread(_poll_log, db_path, last)
        if items:
            for it in items:
                yield f"id: {it['id']}\ndata: {json.dumps(it, separators=(',', ':'))}\n\n"
                last = it["id"]
            quiet = 0.0
            continue
        await asyncio.sleep(poll)
        quiet += poll
        if quiet >= heartbeat:
            quiet = 0.0
            yield ": keep-alive\n\n"


# ------------------------------------------------------------------------------------ the app
def create_app(cfg: WebConfig) -> FastAPI:
    app = FastAPI(title="hackrf-scout", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityMiddleware, cfg=cfg)

    @app.exception_handler(ValueError)
    async def _bad_value(_request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(query.DatabaseUnavailable)
    async def _no_db(_request: Request, exc: query.DatabaseUnavailable):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    def auth(request: Request) -> None:
        if not cfg.token:
            return
        scheme, _, value = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(value.strip().encode(), cfg.token.encode()):
            raise HTTPException(401, "missing or wrong token", headers={"WWW-Authenticate": "Bearer"})

    def require_control() -> None:
        if not cfg.allow_control:
            raise HTTPException(403, "start/stop is disabled; restart the web command with --allow-control")

    @contextmanager
    def ro():
        conn = query.connect_ro(cfg.db_path)
        try:
            yield conn
        finally:
            conn.close()

    anomaly_lock = threading.Lock()
    anomaly_cache: Dict[str, Any] = {"at": -1e9, "value": None}

    def current_anomalies() -> Dict[str, Any]:
        """Evaluate the baselines at most once per `anomaly_ttl` seconds, however many tabs are open."""
        with anomaly_lock:
            if anomaly_cache["value"] is None or time.monotonic() - anomaly_cache["at"] >= cfg.anomaly_ttl:
                with ro() as conn:
                    anomaly_cache["value"] = query.anomalies(conn)
                anomaly_cache["at"] = time.monotonic()
            return anomaly_cache["value"]

    def signal_filters(
        band: List[str] = Query(default=[]),
        mode: str = Query("overlap"),
        unidentified: bool = False,
        min_hits: int = Query(0, ge=0),
        min_snr: Optional[float] = None,
        q: Optional[str] = Query(None, max_length=80),
        captured: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return dict(bands=cfg.registry.resolve_many(band), mode=mode, unidentified=unidentified,
                    min_hits=min_hits, min_snr=min_snr, q=q, captured=captured)

    api = APIRouter(prefix="/api", dependencies=[Depends(auth)])

    @api.get("/status")
    def get_status() -> Dict[str, Any]:
        try:
            with ro() as conn:
                data = query.status(conn, cfg.db_path)
        except query.DatabaseUnavailable:
            data = query.status(None, cfg.db_path)
        data.update(
            version=__version__, control_enabled=cfg.allow_control, token_required=bool(cfg.token),
            server_time=datetime.now().isoformat(timespec="seconds"), band_warnings=cfg.band_warnings,
            scanner=cfg.controller.status(),
        )
        return data

    # ---- bands
    @api.get("/bands")
    def get_bands() -> Dict[str, Any]:
        return {"bands": [b.to_dict() for b in cfg.registry.all()], "warnings": cfg.band_warnings}

    @api.get("/bands/summary")
    def get_band_summary(
        band: List[str] = Query(default=[]), mode: str = "overlap", min_hits: int = Query(0, ge=0)
    ) -> Dict[str, Any]:
        bands = cfg.registry.resolve_many(band) if band else cfg.registry.all()
        with ro() as conn:
            return {"bands": query.band_summary(conn, bands, mode, min_hits)}

    @api.get("/bands/activity")
    def get_band_activity(band: str, bucket: str = "hour", limit: int = Query(72, ge=1, le=1000), mode: str = "overlap") -> Dict[str, Any]:
        b = cfg.registry.resolve(band)
        with ro() as conn:
            return {"band": b.to_dict(), "bucket": bucket, "items": query.band_activity(conn, b, bucket, limit, mode)}

    # ---- signals
    @api.get("/anomalies")
    def get_anomalies(kind: List[str] = Query(default=[])) -> Dict[str, Any]:
        bad = [k for k in kind if k not in baseline.KINDS]
        if bad:
            raise ValueError(f"unknown kind(s): {', '.join(bad)} (use {', '.join(baseline.KINDS)})")
        res = current_anomalies()
        if kind:
            res = {**res, "flags": [x for x in res["flags"] if x["kind"] in kind]}
        return res

    @api.get("/signals")
    def get_signals(
        f: Dict[str, Any] = Depends(signal_filters),
        sort: str = "center_hz",
        order: Optional[str] = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=query.MAX_PAGE_SIZE),
        flagged: bool = False,
    ) -> Dict[str, Any]:
        flags = query.flag_map(current_anomalies())
        ids = list(flags) if flagged else None
        with ro() as conn:
            return query.list_signals(conn, sort=sort, order=order, page=page, page_size=page_size, ids=ids, flags=flags, **f)

    @api.get("/signals/export")
    def export_signals(
        f: Dict[str, Any] = Depends(signal_filters), format: str = "csv", observations: bool = False
    ) -> Response:
        with ro() as conn:
            text = "".join(query.export_signals(conn, format, with_observations=observations, **f))
        media = "text/csv" if format == "csv" else "application/json"
        return Response(text, media_type=media, headers={"Content-Disposition": f'attachment; filename="signals.{format}"'})

    @api.get("/signals/{signal_id}")
    def get_signal(signal_id: int) -> Dict[str, Any]:
        with ro() as conn:
            d = query.signal_detail(conn, signal_id, cfg.registry)
        if d is None:
            raise HTTPException(404, f"no signal #{signal_id}")
        return d

    # ---- other tables
    @api.get("/observations")
    def get_observations(
        signal_id: Optional[int] = None,
        band: List[str] = Query(default=[]),
        mode: str = "overlap",
        since: Optional[str] = Query(None, max_length=32),
        order: str = "desc",
        page: int = Query(1, ge=1),
        page_size: int = Query(100, ge=1, le=query.MAX_PAGE_SIZE),
    ) -> Dict[str, Any]:
        with ro() as conn:
            return query.list_observations(conn, signal_id, cfg.registry.resolve_many(band), mode, since, page, page_size, order)

    @api.get("/sweeps")
    def get_sweeps(page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=query.MAX_PAGE_SIZE)) -> Dict[str, Any]:
        with ro() as conn:
            return query.list_sweeps(conn, page, page_size)

    @api.get("/captures")
    def get_captures(page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=query.MAX_PAGE_SIZE)) -> Dict[str, Any]:
        with ro() as conn:
            return query.list_captures(conn, page, page_size)

    @api.get("/tables")
    def get_tables() -> Dict[str, Any]:
        with ro() as conn:
            return {"tables": query.table_counts(conn)}

    @api.get("/tables/{name}")
    def get_table(
        name: str, sort: Optional[str] = None, order: str = "desc",
        page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=query.MAX_PAGE_SIZE),
    ) -> Dict[str, Any]:
        with ro() as conn:
            return query.browse_table(conn, name, sort, order, page, page_size)

    # ---- log
    @api.get("/log")
    def get_log(after: Optional[int] = Query(None, ge=0), tail: Optional[int] = Query(None, ge=1, le=2000),
                limit: int = Query(500, ge=1, le=2000)) -> Dict[str, Any]:
        try:
            with ro() as conn:
                return query.read_log(conn, after, tail, limit)
        except query.DatabaseUnavailable:
            return {"items": [], "last_id": 0, "max_id": 0}

    @api.get("/log/stream")
    def stream_log(request: Request, after: Optional[int] = Query(None, ge=0)) -> StreamingResponse:
        header = request.headers.get("last-event-id", "")
        start = int(header) if header.isdigit() else after
        if start is None:  # no position given: follow from now
            try:
                with ro() as conn:
                    start = query.read_log(conn, tail=1)["max_id"]
            except query.DatabaseUnavailable:
                start = 0
        return StreamingResponse(
            log_event_stream(cfg.db_path, start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # ---- scanner control
    @api.get("/scanner")
    def get_scanner() -> Dict[str, Any]:
        return {**cfg.controller.status(), "control_enabled": cfg.allow_control}

    def _control(fn, *args: Any) -> Dict[str, Any]:
        try:
            return {**fn(*args), "control_enabled": True}
        except ControlError as exc:
            raise HTTPException(exc.status, str(exc))

    @api.post("/scanner/start", dependencies=[Depends(require_control)])
    def start_scanner(params: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return _control(cfg.controller.start, params)

    @api.post("/scanner/stop", dependencies=[Depends(require_control)])
    def stop_scanner() -> Dict[str, Any]:
        return _control(cfg.controller.stop)

    @api.post("/scanner/check", dependencies=[Depends(require_control)])
    def check_device() -> Dict[str, Any]:
        return _control(cfg.controller.check_device)

    app.include_router(api)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


# ------------------------------------------------------------------------------------ entry point
def serve(args) -> int:
    import uvicorn

    token = args.token or os.environ.get("HACKRF_SCOUT_TOKEN")
    loopback = is_loopback(args.host)
    if not loopback and not token:
        raise RuntimeError(f"--host {args.host} makes the data reachable from the network: set --token (or HACKRF_SCOUT_TOKEN)")
    generated = False
    if args.allow_control and not token:
        token, generated = secrets.token_urlsafe(24), True

    extra, warnings = bandlib.load_custom(args.bands_file or bandlib.default_bands_path())
    registry = bandlib.Registry(extra)
    controller = ScannerController(
        args.db, registry=registry, capture_dir=args.capture_dir,
        hackrf_sweep=args.hackrf_sweep, hackrf_transfer=args.hackrf_transfer,
    )
    cfg = WebConfig(
        db_path=args.db, controller=controller, registry=registry, token=token, allow_control=args.allow_control,
        host=args.host, allowed_hosts=set(args.allowed_host or []), band_warnings=warnings,
    )
    shown = "localhost" if args.host in WILDCARD_HOSTS else args.host
    url = f"http://{shown}:{args.port}/"
    print(f"hackrf-scout web on {url}  (database: {os.path.abspath(args.db)})", file=sys.stderr)
    if generated:
        print(f"  open this link once (it carries the access token): {url}#token={token}", file=sys.stderr)
    elif token:
        print("  an access token is required; the page will ask for it", file=sys.stderr)
    print(f"  start/stop from the browser: {'ENABLED' if args.allow_control else 'disabled (add --allow-control)'}", file=sys.stderr)
    for w in warnings:
        print(f"  warning: {w}", file=sys.stderr)
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    return 0
