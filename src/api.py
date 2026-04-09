"""
FastAPI application — OpenEnv API endpoints.

Episodes are identified by a UUID `episode_id` returned from POST /reset.
All subsequent calls (POST /step, GET /state, GET /grader) pass that ID as a
query parameter so the server can look up the correct MedicalOpenEnv instance.
Episodes are automatically expired after EPISODE_TTL_SECONDS of inactivity.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from ipaddress import ip_address, ip_network
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from .environment import MedicalOpenEnv
from .persistence import get_store, SQLiteEpisodeStore
from .tasks.task3_anonymization import BASELINE_ML_SCORE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


OPENENV_API_KEY = os.getenv("OPENENV_API_KEY", "").strip()
OPENENV_ENV = os.getenv("OPENENV_ENV", os.getenv("ENV", "development")).strip().lower()
OPENENV_REQUIRE_API_KEY = _env_flag("OPENENV_REQUIRE_API_KEY", default=False)
_PUBLIC_PATHS = {
    "/",
    "/config",
    "/info",
    "/favicon.ico",
    "/manifest.json",
    "/theme.css",
    "/mcp",
    "/health",
    "/health/detailed",
}
_PUBLIC_PATH_PREFIXES = (
    "/gradio_api",
    "/assets",
    "/static",
    "/file=",
)
_AUTH_DISABLED_WARNING_EMITTED = False

TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
_TRUSTED_PROXY_CONFIG = [
    part.strip() for part in os.getenv("TRUSTED_PROXY_IPS", "").split(",") if part.strip()
]
_TRUSTED_PROXY_NETWORKS = []
_TRUSTED_PROXY_HOSTS = set()
for _proxy in _TRUSTED_PROXY_CONFIG:
    try:
        _TRUSTED_PROXY_NETWORKS.append(ip_network(_proxy, strict=False))
    except ValueError:
        _TRUSTED_PROXY_HOSTS.add(_proxy)


@dataclass
class RuntimeStores:
    """Container for process-local mutable runtime state."""

    rate_limit_lock: threading.Lock = field(default_factory=threading.Lock)
    rate_limit_store: dict[str, deque[float]] = field(default_factory=dict)
    read_rate_limit_lock: threading.Lock = field(default_factory=threading.Lock)
    read_rate_limit_store: dict[str, deque[float]] = field(default_factory=dict)
    metrics_lock: threading.Lock = field(default_factory=threading.Lock)
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "episodes_created": 0,
            "steps_taken": 0,
            "grades_issued": 0,
            "grader_errors": 0,
            "rate_limit_hits": 0,
            "read_rate_limit_hits": 0,
        }
    )
    episodes_lock: threading.RLock = field(default_factory=threading.RLock)
    episodes: dict[str, Any] = field(default_factory=dict)
    persistence_store: SQLiteEpisodeStore | None = None


_runtime = RuntimeStores()


# ---------------------------------------------------------------------------
# Middlewares
# ---------------------------------------------------------------------------

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce a timeout on all requests."""
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=30.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout after 30 seconds"}
            )
        except Exception as e:
            # Fallback for other dispatch errors
            logger.exception("Middleware dispatch failed")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {type(e).__name__}"}
            )


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Assign a correlation ID per request for tracing."""
    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Require API key auth for non-public endpoints."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_public_path(path):
            return await call_next(request)

        if not OPENENV_API_KEY:
            if OPENENV_REQUIRE_API_KEY:
                logger.error("OPENENV_API_KEY is not configured; rejecting protected request")
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": "Server authentication key is not configured",
                        "error_type": "auth_configuration_error",
                    },
                )

            global _AUTH_DISABLED_WARNING_EMITTED
            if not _AUTH_DISABLED_WARNING_EMITTED:
                logger.warning(
                    "OPENENV_API_KEY is not configured; authentication is disabled because OPENENV_REQUIRE_API_KEY=0"
                )
                _AUTH_DISABLED_WARNING_EMITTED = True
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:].strip()

        if api_key != OPENENV_API_KEY:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={
                    "detail": "Invalid or missing API key",
                    "error_type": "auth_error",
                },
            )

        return await call_next(request)


def _is_public_path(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class GraderError(Exception):
    """Exception raised when the grader encounters an internal error."""

    def __init__(
        self,
        message: str,
        task_id: int | None = None,
        episode_id: str | None = None,
        input_summary: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.task_id = task_id
        self.episode_id = episode_id
        self.input_summary = input_summary
        self.original_error = original_error

    def to_dict(self) -> dict[str, Any]:
        """Return structured error info for API responses."""
        return {
            "error_type": "grader_error",
            "message": self.message,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "details": str(self.original_error) if self.original_error else None,
        }


# ---------------------------------------------------------------------------
# Per-IP Rate Limiter (sliding window)
# ---------------------------------------------------------------------------

# Rate limit configuration — tunable via env vars for different deployment profiles
# (e.g. lower RATE_LIMIT_REQUESTS=5 on a tiny Space, raise for a load-testing setup)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # max requests per window
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))  # window size in seconds
RATE_LIMIT_ENTRY_TTL_SECONDS = int(
    os.getenv("RATE_LIMIT_ENTRY_TTL_SECONDS", "86400")
)  # prune inactive IP entries after 24h
_rate_limit_lock = _runtime.rate_limit_lock
_rate_limit_store: dict[str, deque[float]] = _runtime.rate_limit_store  # IP -> deque of timestamps

# Separate read-endpoint limit to protect expensive GET handlers such as /baseline.
# Defaults are intentionally higher than write limits to avoid penalizing normal dashboards.
READ_RATE_LIMIT_REQUESTS = int(os.getenv("READ_RATE_LIMIT_REQUESTS", "60"))
READ_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("READ_RATE_LIMIT_WINDOW_SECONDS", str(RATE_LIMIT_WINDOW_SECONDS))
)
_read_rate_limit_lock = _runtime.read_rate_limit_lock
_read_rate_limit_store: dict[str, deque[float]] = _runtime.read_rate_limit_store  # IP -> deque of timestamps


def _get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, handling X-Forwarded-For behind proxies.
    
    Behind a reverse proxy (like Hugging Face Spaces), request.client.host
    will be the proxy's internal IP. Use X-Forwarded-For header instead.
    """
    proxy_ip = request.client.host if request.client else ""

    if TRUST_PROXY_HEADERS and proxy_ip:
        trusted = False
        if proxy_ip in _TRUSTED_PROXY_HOSTS:
            trusted = True
        else:
            try:
                parsed_proxy_ip = ip_address(proxy_ip)
                trusted = any(parsed_proxy_ip in net for net in _TRUSTED_PROXY_NETWORKS)
            except ValueError:
                trusted = False

        if trusted:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2...
                # Take the first (original client) IP if it parses as an IP.
                candidate = forwarded.split(",")[0].strip()
                try:
                    ip_address(candidate)
                    return candidate
                except ValueError:
                    logger.warning("Ignoring malformed X-Forwarded-For value: %s", candidate)

    return proxy_ip or "unknown"


def _check_rate_limit(client_ip: str) -> bool:
    """
    Check if client IP is within rate limit using sliding window.
    Returns True if request is allowed, False if rate limited.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        if client_ip not in _rate_limit_store:
            _rate_limit_store[client_ip] = deque()

        timestamps = _rate_limit_store[client_ip]

        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return False

        timestamps.append(now)
        return True


def _get_rate_limit_retry_after(client_ip: str) -> int:
    """Get seconds until the client can retry."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(client_ip, deque())
        if not timestamps:
            return 0

        oldest_in_window = next((t for t in timestamps if t >= window_start), None)
        if oldest_in_window:
            return max(1, int(oldest_in_window + RATE_LIMIT_WINDOW_SECONDS - now) + 1)
        return 0


def _check_read_rate_limit(client_ip: str) -> bool:
    """Check if client IP is within read-endpoint rate limit."""
    now = time.time()
    window_start = now - READ_RATE_LIMIT_WINDOW_SECONDS

    with _read_rate_limit_lock:
        if client_ip not in _read_rate_limit_store:
            _read_rate_limit_store[client_ip] = deque()

        timestamps = _read_rate_limit_store[client_ip]

        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= READ_RATE_LIMIT_REQUESTS:
            return False

        timestamps.append(now)
        return True


def _get_read_rate_limit_retry_after(client_ip: str) -> int:
    """Get seconds until the client can retry read endpoints."""
    now = time.time()
    window_start = now - READ_RATE_LIMIT_WINDOW_SECONDS

    with _read_rate_limit_lock:
        timestamps = _read_rate_limit_store.get(client_ip, deque())
        if not timestamps:
            return 0

        oldest_in_window = next((t for t in timestamps if t >= window_start), None)
        if oldest_in_window:
            return max(1, int(oldest_in_window + READ_RATE_LIMIT_WINDOW_SECONDS - now) + 1)
        return 0


def _enforce_read_rate_limit(request: Request, endpoint_name: str) -> None:
    """Apply read-endpoint rate limit and raise HTTP 429 when exceeded."""
    client_ip = _get_client_ip(request)
    if _check_read_rate_limit(client_ip):
        return

    _increment_metric("read_rate_limit_hits")
    retry_after = _get_read_rate_limit_retry_after(client_ip)
    request.state.retry_after = retry_after
    raise HTTPException(
        status_code=429,
        detail=(
            f"Rate limit exceeded for {endpoint_name}: maximum {READ_RATE_LIMIT_REQUESTS} requests "
            f"per {READ_RATE_LIMIT_WINDOW_SECONDS} seconds. "
            f"Please retry after {retry_after} seconds."
        ),
        headers={"Retry-After": str(retry_after)},
    )


def _purge_stale_rate_limits() -> None:
    """Remove rate limit entries with no recent requests. Thread-safe."""
    now = time.time()
    write_window_start = now - RATE_LIMIT_WINDOW_SECONDS
    read_window_start = now - READ_RATE_LIMIT_WINDOW_SECONDS
    inactive_cutoff = now - RATE_LIMIT_ENTRY_TTL_SECONDS

    with _rate_limit_lock:
        stale_ips = []
        for ip, timestamps in _rate_limit_store.items():
            if timestamps and timestamps[-1] < inactive_cutoff:
                stale_ips.append(ip)
                continue
            while timestamps and timestamps[0] < write_window_start:
                timestamps.popleft()
            if not timestamps:
                stale_ips.append(ip)
        for ip in stale_ips:
            del _rate_limit_store[ip]

    with _read_rate_limit_lock:
        stale_ips = []
        for ip, timestamps in _read_rate_limit_store.items():
            if timestamps and timestamps[-1] < inactive_cutoff:
                stale_ips.append(ip)
                continue
            while timestamps and timestamps[0] < read_window_start:
                timestamps.popleft()
            if not timestamps:
                stale_ips.append(ip)
        for ip in stale_ips:
            del _read_rate_limit_store[ip]

# ---------------------------------------------------------------------------
# Metrics counters (thread-safe, in-process)
# Tracks key operational events for the /metrics endpoint and health checks.
# ---------------------------------------------------------------------------

_metrics_lock = _runtime.metrics_lock
_metrics: dict[str, int] = _runtime.metrics


def _increment_metric(key: str) -> None:
    """Thread-safe increment of a named metric counter."""
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Episode store
# ---------------------------------------------------------------------------

EPISODE_TTL_SECONDS = 3600  # 1 hour of inactivity → auto-expire
MAX_EPISODES = 100  # Prevent unbounded memory consumption from /reset spam
MAX_NOTES_LENGTH = 10_000  # ~2× a realistic clinical note; guards against DoS via huge payloads
_episodes_lock = _runtime.episodes_lock  # Protects concurrent access to _episodes


@dataclass
class EpisodeEntry:
    env: MedicalOpenEnv
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touch(self) -> None:
        self.last_used = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_used) > EPISODE_TTL_SECONDS


# Shared, module-level store  {episode_id: EpisodeEntry}
# Access to this dict must be protected by _episodes_lock
_episodes: dict[str, EpisodeEntry] = _runtime.episodes

# SQLite persistence layer for durable storage across restarts
_persistence_store: SQLiteEpisodeStore | None = _runtime.persistence_store


def configure_runtime(runtime: RuntimeStores | None = None) -> RuntimeStores:
    """Allow callers to inject runtime stores instead of relying on fixed globals."""
    global _runtime
    global _rate_limit_lock, _rate_limit_store
    global _read_rate_limit_lock, _read_rate_limit_store
    global _metrics_lock, _metrics
    global _episodes_lock, _episodes
    global _persistence_store

    _runtime = runtime or RuntimeStores()
    _rate_limit_lock = _runtime.rate_limit_lock
    _rate_limit_store = _runtime.rate_limit_store
    _read_rate_limit_lock = _runtime.read_rate_limit_lock
    _read_rate_limit_store = _runtime.read_rate_limit_store
    _metrics_lock = _runtime.metrics_lock
    _metrics = _runtime.metrics
    _episodes_lock = _runtime.episodes_lock
    _episodes = _runtime.episodes
    _persistence_store = _runtime.persistence_store
    return _runtime


def _init_persistence() -> None:
    """Initialize SQLite persistence if enabled via env var."""
    global _persistence_store
    
    db_path = os.getenv("EPISODE_DB_PATH")
    if db_path:
        try:
            # Ensure directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            _persistence_store = get_store(db_path)
            _runtime.persistence_store = _persistence_store
            logger.info(f"SQLite persistence enabled: {db_path}")
            
            # Restore episodes from previous session
            restored_count = 0
            # Note: We can't auto-restore without knowing episode_ids
            # Episodes are restored on-demand when accessed
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite persistence: {e}. Using in-memory only.")
            _persistence_store = None
            _runtime.persistence_store = _persistence_store
    else:
        logger.info("SQLite persistence disabled (no EPISODE_DB_PATH). Using in-memory storage.")
        _persistence_store = None
        _runtime.persistence_store = _persistence_store


def _get_correlation_id(request: Request | None) -> str:
    """Safely fetch correlation ID from request state."""
    if request is None:
        return "unknown"
    return getattr(request.state, "correlation_id", "unknown")


def _validate_episode_id(episode_id: str) -> str:
    """Validate that episode_id is a canonical UUID string."""
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise HTTPException(status_code=400, detail="Invalid episode_id format")

    candidate = episode_id.strip()
    if len(candidate) > 64:
        raise HTTPException(status_code=400, detail="Invalid episode_id format")

    try:
        parsed = uuid.UUID(candidate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid episode_id format") from e

    return str(parsed)


def _get_episode(episode_id: str) -> EpisodeEntry:
    """Resolve episode_id to its EpisodeEntry, raising 404 if missing/expired. Thread-safe."""
    episode_id = _validate_episode_id(episode_id)
    with _episodes_lock:
        entry: EpisodeEntry | None = _episodes.get(episode_id)
        
        # If not in memory, try to load from persistence
        if entry is None and _persistence_store:
            try:
                result = _persistence_store.load_episode(episode_id)
                if result:
                    env, task_id, seed = result
                    # Restore to memory
                    entry = EpisodeEntry(env=env)
                    _episodes[episode_id] = entry
                    logger.info(f"Restored episode {episode_id} from SQLite")
            except Exception as e:
                logger.warning(f"Failed to load episode {episode_id} from SQLite: {e}")
        
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Episode '{episode_id}' not found. Call POST /reset to start a new episode.",
            )
        if entry.is_expired():
            _episodes.pop(episode_id, None)
            # Also delete from persistence
            if _persistence_store:
                try:
                    _persistence_store.delete_episode(episode_id)
                except Exception:
                    pass
            raise HTTPException(
                status_code=404,
                detail=f"Episode '{episode_id}' has expired. Call POST /reset to start a new episode.",
            )
        entry.touch()
        return entry


def _purge_expired() -> None:
    """Remove all expired sessions and stale rate limits (called periodically). Thread-safe."""
    with _episodes_lock:
        expired = [eid for eid, e in _episodes.items() if e.is_expired()]
        for eid in expired:
            _episodes.pop(eid, None)
            # Also delete from persistence
            if _persistence_store:
                try:
                    _persistence_store.delete_episode(eid)
                except Exception:
                    pass
    # Also clean up stale rate limit entries
    _purge_stale_rate_limits()
    
    # Purge from SQLite directly
    if _persistence_store:
        try:
            _persistence_store.purge_expired(EPISODE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Failed to purge expired episodes from SQLite: {e}")


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------

async def _expiry_task() -> None:
    """Background task: purge expired episodes every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        _purge_expired()


async def _rate_limit_cleanup_task() -> None:
    """Background task: purge stale rate limits more frequently (every 2 minutes)."""
    while True:
        await asyncio.sleep(120)
        _purge_stale_rate_limits()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for background tasks."""
    # Initialize persistence layer
    _init_persistence()
    
    expiry_task = asyncio.create_task(_expiry_task())
    cleanup_task = asyncio.create_task(_rate_limit_cleanup_task())
    yield
    
    # Cleanup persistence
    if _persistence_store:
        try:
            _persistence_store.close()
        except Exception:
            pass
    
    expiry_task.cancel()
    cleanup_task.cancel()
    try:
        await asyncio.gather(expiry_task, cleanup_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Medical Records Data Cleaner & PII Redactor v3",
    description=(
        "A healthcare AI training environment for EHR cleaning, PHI redaction, "
        "and clinical knowledge extraction. Follows OpenEnv standardization (RFC 001-003)."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# Add custom timeout middleware first
app.add_middleware(TimeoutMiddleware)
# Correlation ID middleware for request tracing
app.add_middleware(CorrelationIDMiddleware)
# API key auth middleware for all protected endpoints
app.add_middleware(APIKeyAuthMiddleware)


# ---------------------------------------------------------------------------
# Error schema + handlers
# ---------------------------------------------------------------------------

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "detail": {"type": ["string", "object"]},
        "error_type": {"type": "string"},
        "task_id": {"type": ["integer", "null"]},
        "episode_id": {"type": ["string", "null"]},
        "correlation_id": {"type": "string"},
        "retry_after": {"type": ["integer", "null"]},
    },
}


def _enrich_error_payload(request: Request, detail: Any, error_type: str = "http_error") -> dict:
    return {
        "detail": detail,
        "error_type": error_type,
        "task_id": getattr(request.state, "task_id", None),
        "episode_id": getattr(request.state, "episode_id", None),
        "correlation_id": getattr(request.state, "correlation_id", "unknown"),
        "retry_after": getattr(request.state, "retry_after", None),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Capture retry-after from headers if present
    retry_after = None
    if exc.headers and "Retry-After" in exc.headers:
        try:
            retry_after = int(exc.headers["Retry-After"])
        except Exception:
            retry_after = exc.headers["Retry-After"]
    request.state.retry_after = retry_after
    payload = _enrich_error_payload(request, exc.detail)
    return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload = _enrich_error_payload(request, exc.errors())
    return JSONResponse(status_code=422, content=payload)


# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------

def _get_cors_origins() -> list[str]:
    """
    Get allowed CORS origins from environment variable.
    
    Reads from CORS_ORIGINS env var (comma-separated list of origins).
    Defaults to localhost origins for development.
    """
    default_origins = ["http://localhost:7860", "http://127.0.0.1:7860"]
    cors_env = os.environ.get("CORS_ORIGINS", "").strip()
    
    if not cors_env:
        logger.warning(
            "CORS_ORIGINS env var not set — using localhost defaults. "
            "Set CORS_ORIGINS in production to restrict cross-origin access "
            "(e.g. CORS_ORIGINS=https://your-hf-space.hf.space)."
        )
        return default_origins
    
    # Parse comma-separated origins, strip whitespace
    origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    if not origins:
        logger.warning("CORS_ORIGINS resolved to an empty list — using localhost defaults")
        return default_origins

    if any("*" in origin for origin in origins):
        raise ValueError("CORS_ORIGINS cannot contain wildcard (*)")

    return origins


app.add_middleware(
    CORSMiddleware,
    # Configurable origins via CORS_ORIGINS env var (defaults to localhost:7860)
    allow_origins=_get_cors_origins(),
    # Only allow safe read/write methods
    allow_methods=["GET", "POST", "OPTIONS"],
    # Restrict headers to commonly needed ones
    allow_headers=["Content-Type", "Accept"],
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42

    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    # Tasks 1-3 use 'records', Task 4 uses 'knowledge'
    records: list[dict[str, Any]] | None = None
    knowledge: list[dict[str, Any]] | None = None
    is_final: bool = False


# ---------------------------------------------------------------------------
# Action schema (returned by /tasks)
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = {
    "type": "object",
    "description": "Agent's processed output to submit at each step.",
    "properties": {
        "records": {
            "type": "array",
            "description": "Tasks 1-3: List of processed PatientRecord objects (matching input schema).",
            "items": {"type": "object"},
        },
        "knowledge": {
            "type": "array",
            "description": "Task 4 only: List of knowledge objects with 'entities' and 'summary'.",
            "items": {"type": "object"},
        },
        "is_final": {
            "type": "boolean",
            "default": False,
            "description": "Set true to signal final submission and end the episode early.",
        },
    },
}

_MCP_COMPAT_TOOLS = [
    {
        "name": "reset",
        "description": "Start a new episode in the environment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "seed": {"type": "integer"},
            },
        },
    },
    {
        "name": "step",
        "description": "Submit processed records or knowledge and advance the episode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "records": {"type": "array", "items": {"type": "object"}},
                "knowledge": {"type": "array", "items": {"type": "object"}},
                "is_final": {"type": "boolean"},
            },
            "required": ["episode_id"],
        },
    },
    {
        "name": "state",
        "description": "Get current state and audit trail of an episode.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "export",
        "description": "Export full episode snapshot with reward trend.",
        "inputSchema": {
            "type": "object",
            "properties": {"episode_id": {"type": "string"}},
            "required": ["episode_id"],
        },
    },
    {
        "name": "tasks",
        "description": "List available tasks and schemas.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "schema",
        "description": "Return schema examples for the MCP tool surface.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _jsonrpc_result(rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": result,
    }


def _jsonrpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _build_task_catalog() -> list[dict[str, Any]]:
    """Single source of truth for task metadata exposed by API endpoints."""
    return [
        {
            "id": 1,
            "name": "Data Hygiene & Standardisation",
            "difficulty": "easy",
            "description": (
                "Fix messy EHR records: normalise dates to ISO 8601, "
                "standardise medication units, replace invalid ICD-10 codes, "
                "and populate missing required fields."
            ),
            "grader": {
                "type": "formula",
                "expression": "correct_fields / total_fields",
                "pass_bar": "score >= 0.85",
            },
            "grader_formula": "correct_fields / total_fields",
            "pass_bar": "score >= 0.85",
            "max_steps": 10,
            "action_payload_key": "records",
        },
        {
            "id": 2,
            "name": "PHI Detection & Redaction",
            "difficulty": "medium",
            "description": (
                "Replace all Protected Health Information with typed redaction "
                "tokens ([REDACTED_NAME], [REDACTED_MRN], ...) while preserving "
                "all clinical content."
            ),
            "grader": {
                "type": "formula",
                "expression": "phi_score * 0.6 + utility_score * 0.4",
                "pass_bar": "phi_score == 1.0 AND utility_score >= 0.8",
            },
            "grader_formula": "phi_score * 0.6 + utility_score * 0.4",
            "pass_bar": "phi_score == 1.0 AND utility_score >= 0.8",
            "max_steps": 10,
            "action_payload_key": "records",
        },
        {
            "id": 3,
            "name": "Full Anonymisation + Downstream Utility",
            "difficulty": "hard",
            "description": (
                "Fully de-identify records while preserving enough clinical signal "
                "for a deterministic disease-risk model to remain accurate. "
                "Use age-group buckets, preserve ICD-10 chapter prefixes, etc."
            ),
            "grader": {
                "type": "formula",
                "expression": "(avg_phi * 0.4) + (avg_ml_utility * 0.3) + (avg_fid * 0.2) + (k_score * 0.1)",
                "pass_bar": f"phi_score == 1.0 AND ml_utility_score >= {BASELINE_ML_SCORE}",
            },
            "grader_formula": "(avg_phi * 0.4) + (avg_ml_utility * 0.3) + (avg_fid * 0.2) + (k_score * 0.1)",
            "pass_bar": f"phi_score == 1.0 AND ml_utility_score >= {BASELINE_ML_SCORE}",
            "max_steps": 10,
            "action_payload_key": "records",
        },
        {
            "id": 4,
            "name": "Clinical Knowledge Extraction",
            "difficulty": "hard",
            "description": (
                "Extract clinical entities (diseases, treatments, labs), "
                "standardize to ICD-10-CM codes, and generate a concise "
                "narrative summary of the patient's clinical state. "
                "Submit one knowledge object per input record, in the same order as "
                "observation.records (knowledge items have no record_id; alignment is by index)."
            ),
            "grader": {
                "type": "formula",
                "expression": (
                    "avg_entity_extraction * 0.4 + avg_code_precision * 0.3 + "
                    "avg_summary_fidelity * 0.3"
                ),
                "pass_bar": "avg_entity_extraction >= 0.75 AND avg_summary_fidelity >= 0.50",
            },
            "grader_formula": (
                "avg_entity_extraction * 0.4 + avg_code_precision * 0.3 + "
                "avg_summary_fidelity * 0.3"
            ),
            "pass_bar": "avg_entity_extraction >= 0.75 AND avg_summary_fidelity >= 0.50",
            "max_steps": 10,
            "action_payload_key": "knowledge",
        },
        {
            "id": 5,
            "name": "Contextual PII Disambiguation",
            "difficulty": "expert",
            "description": (
                "Decide which mentions in clinical notes are patient identifiers vs. "
                "providers/facilities/family. Redact only patient PII, keep providers."
            ),
            "grader": {
                "type": "formula",
                "expression": "patient_phi_score * 0.5 + provider_phi_score * 0.3 + contextual_accuracy * 0.2",
                "pass_bar": "overall_score >= 0.70 AND patient_phi_score >= 0.80",
            },
            "grader_formula": "patient_phi_score * 0.5 + provider_phi_score * 0.3 + contextual_accuracy * 0.2",
            "pass_bar": "overall_score >= 0.70 AND patient_phi_score >= 0.80",
            "max_steps": 10,
            "action_payload_key": "records",
        },
    ]


@app.post("/mcp", tags=["open_env"])
async def mcp_jsonrpc_compat(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility endpoint for validators expecting JSON-RPC at POST /mcp.

    Full FastMCP transport is mounted under /mcp/* in src.main.
    """
    rpc_id = payload.get("id")
    method = payload.get("method")

    if method == "tools/list":
        return _jsonrpc_result(rpc_id, {"tools": _MCP_COMPAT_TOOLS})

    return _jsonrpc_error(
        rpc_id,
        -32601,
        "Method not found on /mcp compatibility endpoint; use mounted FastMCP transport under /mcp/*",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks() -> dict:
    """List all available tasks with descriptions, pass bars, and action schema."""
    return {
        "action_schema": _ACTION_SCHEMA,
        "episode_contract": {
            "deterministic_seeded_reset": True,
            "max_steps_per_episode": 10,
            "step_endpoint": "POST /step?episode_id=<uuid>",
            "state_endpoint": "GET /state?episode_id=<uuid>",
            "grader_endpoint": "GET /grader?episode_id=<uuid>",
        },
        "tasks": _build_task_catalog(),
    }


@app.post("/reset", status_code=200)
def reset_endpoint(request: Request, req: ResetRequest = Body(default=None)) -> dict:
    """
    Start a new episode.

    Returns an `episode_id` (UUID string) along with the first Observation.
    Pass this `episode_id` to all subsequent /step, /state, and /grader calls.
    """
    # Handle case where no body is provided (use defaults)
    if req is None:
        req = ResetRequest()
    
    return reset_episode(req, request)


def reset_episode(req: ResetRequest, request: Request) -> dict:
    """
    Internal implementation for reset logic.
    """
    # Per-IP rate limiting to prevent DoS
    client_ip = _get_client_ip(request)
    if not _check_rate_limit(client_ip):
        _increment_metric("rate_limit_hits")
        retry_after = _get_rate_limit_retry_after(client_ip)
        request.state.retry_after = retry_after
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: maximum {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS} seconds. "
                f"Please retry after {retry_after} seconds."
            ),
            headers={"Retry-After": str(retry_after)},
        )

    with _episodes_lock:
        # Purge expired sessions before checking capacity to avoid false 429s
        _purge_expired()
        
        # [P1] Prevent unbounded memory growth from /reset spam
        if len(_episodes) >= MAX_EPISODES:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many active episodes ({len(_episodes)}/{MAX_EPISODES}). "
                    "Please wait for older episodes to expire or finish and try again."
                ),
            )

    env = MedicalOpenEnv()
    corr_id = _get_correlation_id(request)
    env._correlation_id = corr_id
    request.state.task_id = req.task_id
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    episode_id = str(uuid.uuid4())
    with _episodes_lock:
        _episodes[episode_id] = EpisodeEntry(env=env)
    
    # Persist to SQLite if enabled
    if _persistence_store:
        try:
            _persistence_store.save_episode(episode_id, env, req.task_id, req.seed)
        except Exception as e:
            logger.warning(f"Failed to persist episode {episode_id}: {e}")

    _increment_metric("episodes_created")
    logger.info(
        "api.reset",
        extra={
            "episode_id": episode_id,
            "task_id": req.task_id,
            "seed": req.seed,
            "correlation_id": corr_id,
        },
    )
    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
        "correlation_id": corr_id,
    }


def _validate_record_structure(record: dict, task_id: int) -> tuple[bool, str]:
    """
    Validate that a record has required fields for the task.
    Only validates critical fields that would cause grader crashes.

    Returns:
        (is_valid, error_message)
    """
    # Only validate fields that are absolutely required to prevent crashes
    # Most fields are optional or have defaults in the grader
    # Relaxed validation: only check for crash-causing issues
    if task_id == 1:
        if "icd10_codes" in record and not isinstance(record.get("icd10_codes"), list):
            return False, "Field 'icd10_codes' must be a list"
        if "medications" in record and not isinstance(record.get("medications"), list):
            return False, "Field 'medications' must be a list"

    if task_id in (2, 3):
        if "clinical_notes" in record and not isinstance(record.get("clinical_notes"), str):
            return False, "Field 'clinical_notes' must be a string"
        notes = record.get("clinical_notes", "")
        if isinstance(notes, str) and len(notes) > MAX_NOTES_LENGTH:
            return False, (
                f"clinical_notes exceeds {MAX_NOTES_LENGTH} chars "
                f"({len(notes)} submitted). Truncate before submitting."
            )

    if task_id == 5:
        # Relaxed: allow records without clinical_notes (grader handles empty gracefully)
        notes = record.get("clinical_notes", "")
        if notes is not None and not isinstance(notes, str):
            return False, "Field 'clinical_notes' must be a string"
        if isinstance(notes, str) and len(notes) > MAX_NOTES_LENGTH:
            return False, (
                f"clinical_notes exceeds {MAX_NOTES_LENGTH} chars "
                f"({len(notes)} submitted). Truncate before submitting."
            )

    return True, ""


def _validate_knowledge_structure(knowledge: dict) -> tuple[bool, str]:
    """
    Validate that a knowledge object has required fields for Task 4.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(knowledge, dict):
        return False, "Knowledge object must be a dictionary"
    
    if "entities" not in knowledge:
        return False, "Missing required field: 'entities'"
    
    if "summary" not in knowledge:
        return False, "Missing required field: 'summary'"
    
    if not isinstance(knowledge.get("entities"), list):
        return False, "Field 'entities' must be a list"
    
    if not isinstance(knowledge.get("summary"), str):
        return False, "Field 'summary' must be a string"
    
    return True, ""


def _summarize_input(req: StepRequest, max_len: int = 200) -> str:
    """Create a summary of the request for logging purposes."""
    if req.records:
        summary = f"records={len(req.records)} items"
    elif req.knowledge:
        summary = f"knowledge={len(req.knowledge)} items"
    else:
        summary = "empty payload"
    return summary[:max_len]


def _validate_task_records_payload(req: StepRequest, task_id: int) -> None:
    """Validate records payload. Relaxed: accepts empty records for validator compatibility."""
    # Accept None/empty records — the grader will return a low score
    if req.records is None:
        # Coerce to empty list so step() doesn't crash
        req.records = []
        return

    for i, record in enumerate(req.records):
        valid, error_msg = _validate_record_structure(record, task_id)
        if not valid:
            raise HTTPException(status_code=422, detail=f"Record {i}: {error_msg}")
        # Enforce MAX_NOTES_LENGTH for clinical_notes
        notes = record.get("clinical_notes", "")
        if isinstance(notes, str) and len(notes) > MAX_NOTES_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Record {i} clinical_notes exceeds {MAX_NOTES_LENGTH} chars ({len(notes)} submitted). Truncate before submitting.",
            )


def _validate_task_knowledge_payload(req: StepRequest, task_id: int) -> None:
    """Validate knowledge payload. Relaxed: accepts empty knowledge for validator compatibility."""
    # Accept None/empty knowledge — the grader will return a low score
    if req.knowledge is None:
        # Coerce to empty list so step() doesn't crash
        req.knowledge = []
        return

    for i, knowledge_obj in enumerate(req.knowledge):
        valid, error_msg = _validate_knowledge_structure(knowledge_obj)
        if not valid:
            raise HTTPException(status_code=422, detail=f"Knowledge object {i}: {error_msg}")
        # Enforce MAX_NOTES_LENGTH for summary
        summary = knowledge_obj.get("summary", "")
        if isinstance(summary, str) and len(summary) > MAX_NOTES_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"Knowledge {i} summary exceeds {MAX_NOTES_LENGTH} chars ({len(summary)} submitted). Truncate before submitting.",
            )


_TASK_PAYLOAD_VALIDATORS: dict[int, Callable[[StepRequest, int], None]] = {
    1: _validate_task_records_payload,
    2: _validate_task_records_payload,
    3: _validate_task_records_payload,
    4: _validate_task_knowledge_payload,
    5: _validate_task_records_payload,
}


@app.post("/step")
def step_episode(
    req: StepRequest,
    episode_id: uuid.UUID = Query(..., description="episode_id returned by POST /reset"),
    request: Request = None,
) -> dict:
    """Submit processed records and receive a reward + next observation. [P2] Thread-safe."""
    from .models import Action

    episode_id_str = str(episode_id)
    entry = _get_episode(episode_id_str)
    corr_id = _get_correlation_id(request)
    request.state.episode_id = episode_id_str
    
    # [P2] Protect against concurrent step() calls on same episode
    with entry.lock:
        # Validate task-specific payload requirements
        task_id = entry.env._task_id
        request.state.task_id = task_id
        entry.env._correlation_id = corr_id
        payload_validator = _TASK_PAYLOAD_VALIDATORS.get(task_id)
        if payload_validator is None:
            raise HTTPException(status_code=400, detail=f"Unsupported task_id: {task_id}")
        payload_validator(req, task_id)

        # Capture current state before grading for rollback if needed
        current_step = entry.env._step
        
        try:
            action = Action(
                records=req.records, 
                knowledge=req.knowledge,
                is_final=req.is_final
            )
        except ValueError as e:
            # Action construction/validation errors
            raise HTTPException(status_code=422, detail=str(e))

        try:
            next_obs, reward, done, info = entry.env.step(action)
            _increment_metric("steps_taken")
            _increment_metric("grades_issued")
            logger.info(
                "api.step",
                extra={
                    "episode_id": episode_id_str,
                    "task_id": task_id,
                    "step": entry.env._step,
                    "done": done,
                    "score": reward.score,
                    "passed": info.get("passed", False),
                    "correlation_id": corr_id,
                },
            )
        except RuntimeError as e:
            # Expected runtime errors from environment state (e.g., episode already done)
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            # Validation errors from step logic
            raise HTTPException(status_code=422, detail=str(e))
        except GraderError as e:
            _increment_metric("grader_errors")
            # Grader-specific errors - already logged with context
            logger.error(
                "Grader error in step: task_id=%s, episode_id=%s, input=%s, error=%s",
                e.task_id, e.episode_id, e.input_summary, str(e.original_error)
            )
            raise HTTPException(
                status_code=500,
                detail=e.to_dict(),
            )
        except Exception as e:
            _increment_metric("grader_errors")
            # Unexpected grader/serialization errors - wrap in GraderError for context
            input_summary = _summarize_input(req)
            grader_err = GraderError(
                message="An unexpected error occurred during grading",
                task_id=task_id,
                episode_id=episode_id_str,
                input_summary=input_summary,
                original_error=e,
            )
            logger.exception(
                "Unexpected grader error: task_id=%s, episode_id=%s, input=%s",
                task_id, episode_id_str, input_summary
            )
            # Attempt to restore episode state consistency
            try:
                entry.env._step = current_step
            except Exception:
                logger.warning("Failed to restore episode state after grader error")

            raise HTTPException(
                status_code=500,
                detail=grader_err.to_dict(),
            )

    # Include last history entry as a lightweight trace snapshot
    trace_entry = entry.env._history[-1] if entry.env._history else None

    return {
        "episode_id": episode_id_str,
        "observation": next_obs.model_dump(),
        "reward": float(reward.score),
        "done": done,
        "correlation_id": corr_id,
        "info": {
            **info,
            "breakdown": reward.breakdown,
            "grader_info": reward.info,
            "correlation_id": corr_id,
            "trace": trace_entry,
        },
    }


@app.get("/state")
def get_state(
    request: Request,
    episode_id: uuid.UUID = Query(..., description="episode_id returned by POST /reset"),
) -> dict:
    """Return current episode state snapshot."""
    _enforce_read_rate_limit(request, "/state")

    episode_id_str = str(episode_id)
    entry = _get_episode(episode_id_str)
    with entry.lock:
        return {"episode_id": episode_id_str, **entry.env.state().model_dump()}


@app.get("/grader")
def regrade(
    request: Request,
    episode_id: uuid.UUID = Query(..., description="episode_id returned by POST /reset"),
) -> dict:
    """Re-grade the most recent submission (idempotent)."""
    _enforce_read_rate_limit(request, "/grader")

    episode_id_str = str(episode_id)
    entry = _get_episode(episode_id_str)
    with entry.lock:
        return {"episode_id": episode_id_str, **entry.env.regrade()}


@app.get("/baseline")
def run_baseline(request: Request) -> dict:
    """
    Run the standardized Hybrid (Rules + Local ML) baseline on ALL tasks (seed=42).
    Returns per-task scores without requiring OPENAI_API_KEY.
    """
    _enforce_read_rate_limit(request, "/baseline")

    from .baseline_agent import hybrid_baseline

    results = []
    for task_id in (1, 2, 3, 4, 5):
        task_env = MedicalOpenEnv()
        task_env.reset(task_id=task_id, seed=42)
        try:
            result = hybrid_baseline(task_env)
            results.append(result)
        except Exception as e:
            logger.exception(f"Baseline task {task_id} failed")
            results.append({"task_id": task_id, "error": str(e)})

    avg_score: float = sum(float(r.get("score", 0.0)) for r in results) / len(results)
    return {
        "agent": "hybrid (rules + Local ML)",
        "seed": 42,
        "results": results,
        "average_score": float(round(avg_score, 4)),
    }


@app.get("/contract", tags=["open_env"])
def get_contract(request: Request) -> dict:
    """OpenEnv-style environment contract for clients and evaluators."""
    _enforce_read_rate_limit(request, "/contract")

    from .models import Observation, Reward, State

    reset_schema = (
        ResetRequest.model_json_schema()
        if hasattr(ResetRequest, "model_json_schema")
        else ResetRequest.schema()
    )
    step_schema = (
        StepRequest.model_json_schema()
        if hasattr(StepRequest, "model_json_schema")
        else StepRequest.schema()
    )
    obs_schema = (
        Observation.model_json_schema()
        if hasattr(Observation, "model_json_schema")
        else Observation.schema()
    )
    reward_schema = (
        Reward.model_json_schema()
        if hasattr(Reward, "model_json_schema")
        else Reward.schema()
    )
    state_schema = (
        State.model_json_schema()
        if hasattr(State, "model_json_schema")
        else State.schema()
    )

    return {
        "name": "Medical Records Data Cleaner & PII Redactor",
        "version": app.version,
        "mode": "agentic",
        "determinism": {
            "seeded_reset": True,
            "same_seed_same_records": True,
            "max_steps_per_episode": 10,
        },
        "api_surface": {
            "tasks": "GET /tasks",
            "reset": "POST /reset",
            "step": "POST /step?episode_id=<uuid>",
            "state": "GET /state?episode_id=<uuid>",
            "grader": "GET /grader?episode_id=<uuid>",
            "export": "GET /export?episode_id=<uuid>",
            "metrics": "GET /metrics",
            "health": "GET /health",
            "health_detailed": "GET /health/detailed",
            "schema": "GET /schema",
            "metadata": "GET /metadata",
            "mode": "GET /mode",
            "openapi": "GET /openapi.json",
        },
        "episode_structure": {
            "create": "POST /reset returns a new episode_id and initial observation",
            "iterate": "POST /step with same episode_id until done=true or is_final=true",
            "inspect": "GET /state to read server-side state and audit trail",
            "regrade": "GET /grader for idempotent re-grading of last submission",
            "export": "GET /export for replay/debug reward trend",
            "expiry": f"episodes expire after {EPISODE_TTL_SECONDS} seconds of inactivity",
        },
        "tasks": _build_task_catalog(),
        "schemas": {
            "reset_request": reset_schema,
            "step_request": step_schema,
            "action": _ACTION_SCHEMA,
            "observation": obs_schema,
            "reward": reward_schema,
            "state": state_schema,
            "error": ERROR_SCHEMA,
        },
        "examples": {
            "reset_request": {"task_id": 2, "seed": 42},
            "step_request_task2": {
                "records": [{"record_id": "rec-001", "clinical_notes": "[REDACTED]"}],
                "is_final": True,
            },
            "step_request_task4": {
                "knowledge": [
                    {
                        "entities": [{"text": "hypertension", "type": "Condition", "code": "I10"}],
                        "summary": "Patient with essential hypertension.",
                    }
                ],
                "is_final": True,
            },
        },
    }


@app.get("/openapi.json", tags=["open_env"])
def get_openapi_explicit() -> dict:
    """Explicitly serve OpenAPI schema at root for OpenEnv validator, in case Gradio swallows it."""
    from fastapi.openapi.utils import get_openapi
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

@app.get("/mode", tags=["open_env"])
def get_mode() -> dict:
    """Return the operation mode of this environment (required by OpenEnv spec)."""
    return {"mode": "agentic"}


@app.get("/health", tags=["open_env"])
def get_health() -> dict:
    """Standard health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/detailed", tags=["open_env"])
def detailed_health(request: Request) -> dict:
    """Detailed health check including dependency status."""
    _enforce_read_rate_limit(request, "/health/detailed")

    health = {
        "status": "healthy",
        "episodes_active": len(_episodes),
        "ner_available": False,
        "bertscore_available": False,
        "persistence": {
            "enabled": False,
            "db_path": None,
            "initialized": False,
            "last_error": None,
            "schema_version": None,
        },
    }

    try:
        from .baseline_agent import get_ner_agent
        agent = get_ner_agent()
        health["ner_available"] = agent.nlp is not None
    except Exception:
        pass

    try:
        import evaluate
        health["bertscore_available"] = True
    except Exception:
        pass

    if _persistence_store:
        try:
            health["persistence"] = _persistence_store.status()
        except Exception as e:
            health["persistence"] = {
                "enabled": True,
                "db_path": os.getenv("EPISODE_DB_PATH"),
                "initialized": False,
                "last_error": str(e),
                "schema_version": None,
            }

    # Capacity and metrics snapshot
    with _metrics_lock:
        health["metrics"] = dict(_metrics)
    health["max_episodes"] = MAX_EPISODES
    health["rate_limit_config"] = {
        "requests_per_window": RATE_LIMIT_REQUESTS,
        "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
        "entry_ttl_seconds": RATE_LIMIT_ENTRY_TTL_SECONDS,
        "read_requests_per_window": READ_RATE_LIMIT_REQUESTS,
        "read_window_seconds": READ_RATE_LIMIT_WINDOW_SECONDS,
    }

    return health


@app.get("/metrics", tags=["open_env"])
def get_metrics(request: Request) -> dict:
    """
    Prometheus-style counters for operational monitoring.

    Tracks episodes created, steps taken, grades issued, grader errors,
    and rate-limit hits since the last server restart.
    """
    _enforce_read_rate_limit(request, "/metrics")

    with _metrics_lock:
        counters = dict(_metrics)
    return {
        "counters": counters,
        "capacity": {
            "active_episodes": len(_episodes),
            "max_episodes": MAX_EPISODES,
        },
        "rate_limit": {
            "requests_per_window": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "entry_ttl_seconds": RATE_LIMIT_ENTRY_TTL_SECONDS,
            "read_requests_per_window": READ_RATE_LIMIT_REQUESTS,
            "read_window_seconds": READ_RATE_LIMIT_WINDOW_SECONDS,
        },
        "error_schema": ERROR_SCHEMA,
    }


@app.get("/export")
def export_episode(
    request: Request,
    episode_id: uuid.UUID = Query(..., description="episode_id returned by POST /reset"),
) -> dict:
    """
    Export a full episode snapshot for debugging and offline replay.

    Returns the complete episode state including the per-step audit trail
    (score, breakdown, passed) and a chart-friendly reward_trend list.
    Useful for visualising reward curves, diagnosing score regressions, or
    saving episode state before it expires.
    """
    _enforce_read_rate_limit(request, "/export")

    episode_id_str = str(episode_id)
    entry = _get_episode(episode_id_str)
    with entry.lock:
        state_model = entry.env.state()
        state = state_model.model_dump()

    # Build a chart-friendly reward trend (one point per step taken)
    reward_trend = [
        {
            "step": h["step"],
            "score": h["score"],
            "passed": h["passed"],
        }
        for h in state.get("audit_trail", [])
    ]

    return {
        "episode_id": episode_id_str,
        **state,
        "reward_trend": reward_trend,
    }


@app.get("/metadata", tags=["open_env"])
def get_metadata() -> dict:
    """Standard metadata endpoint returning name and description."""
    return {
        "name": "Medical Records Data Cleaner & PII Redactor",
        "version": app.version,
        "description": "Synthetic healthcare OpenEnv for medical record cleaning and PHI redaction.",
        "tasks_count": len(_build_task_catalog()),
        "supports_episode_ids": True,
        "supports_deterministic_seed": True,
        "supports_mcp": True,
    }


@app.get("/schema", tags=["open_env"])
def get_schema() -> dict:
    """Standard schemas endpoint for RL clients."""
    # Build complete JSON schemas for internal models
    from .models import Observation, Reward, State

    observation_schema = (
        Observation.model_json_schema()
        if hasattr(Observation, "model_json_schema")
        else Observation.schema()
    )
    reward_schema = (
        Reward.model_json_schema()
        if hasattr(Reward, "model_json_schema")
        else Reward.schema()
    )
    state_schema = (
        State.model_json_schema()
        if hasattr(State, "model_json_schema")
        else State.schema()
    )

    return {
        "action": _ACTION_SCHEMA,
        "observation": observation_schema,
        "reward": reward_schema,
        "error": ERROR_SCHEMA,
        "state": state_schema,
    }
