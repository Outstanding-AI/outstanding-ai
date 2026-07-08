"""Low-cardinality S3 request attribution for AI runtime lake reads."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from src.common.sql_attribution import sanitize_sql_label

logger = logging.getLogger("s3.attribution")

S3_ATTRIBUTION_VERSION = "solvix_s3:v1"
_REQUEST_CONTEXT_KEY = "solvix_s3_request_attribution"


@dataclass
class _RequestInfo:
    operation: str
    source: str
    request_tier: str
    layer_table: str


@dataclass
class S3RequestAccumulator:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_requests: int = 0
    failure_count: int = 0
    by_operation: Counter[str] = field(default_factory=Counter)
    by_request_tier: Counter[str] = field(default_factory=Counter)
    by_source: Counter[str] = field(default_factory=Counter)
    by_layer_table: Counter[str] = field(default_factory=Counter)

    def record(self, info: _RequestInfo, *, status_code: int | None = None) -> None:
        with self._lock:
            self.total_requests += 1
            self.failure_count += int(bool(status_code and status_code >= 400))
            self.by_operation[info.operation] += 1
            self.by_request_tier[info.request_tier] += 1
            self.by_source[info.source] += 1
            self.by_layer_table[info.layer_table] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": S3_ATTRIBUTION_VERSION,
                "total_requests": self.total_requests,
                "failure_count": self.failure_count,
                "by_operation": dict(self.by_operation),
                "by_request_tier": dict(self.by_request_tier),
                "by_source": dict(self.by_source),
                "by_layer_table": dict(self.by_layer_table),
            }


_accumulator_ctx: ContextVar[S3RequestAccumulator | None] = ContextVar(
    "ai_s3_request_accumulator",
    default=None,
)


@contextmanager
def s3_request_attribution_context() -> Iterator[S3RequestAccumulator]:
    acc = S3RequestAccumulator()
    token = _accumulator_ctx.set(acc)
    try:
        yield acc
    finally:
        _accumulator_ctx.reset(token)


def create_instrumented_s3_client(*, source: str = "ai.unknown", **client_kwargs: Any) -> Any:
    import boto3

    client = boto3.client("s3", **client_kwargs)
    instrument_s3_client(client, source=source)
    return client


def instrument_s3_client(client: Any, *, source: str = "ai.unknown") -> Any:
    events = getattr(getattr(client, "meta", None), "events", None)
    if events is None:
        return client
    marker = "_solvix_s3_attribution_registered"
    if getattr(client, marker, False):
        return client
    safe_source = sanitize_sql_label(source, max_length=120) or "ai.unknown"
    events.register(
        "before-parameter-build.s3.*",
        lambda **kwargs: _before_parameter_build(default_source=safe_source, **kwargs),
    )
    events.register("after-call.s3.*", _after_call)
    setattr(client, marker, True)
    return client


def log_s3_request_summary(
    accumulator: S3RequestAccumulator, *, context: dict[str, Any] | None = None
) -> None:
    summary = accumulator.snapshot()
    if summary["total_requests"]:
        logger.info("s3_request_summary", extra={**(context or {}), "s3_request_summary": summary})


def _before_parameter_build(
    params: dict[str, Any] | None = None,
    model: Any = None,
    context: dict | None = None,
    default_source: str = "ai.unknown",
    **_: Any,
) -> None:
    if context is None:
        return
    operation = str(getattr(model, "name", "") or "unknown")
    key = str((params or {}).get("Key") or (params or {}).get("Prefix") or "")
    context[_REQUEST_CONTEXT_KEY] = _RequestInfo(
        operation=operation,
        source=sanitize_sql_label(default_source, max_length=120) or "ai.unknown",
        request_tier=_request_tier(operation),
        layer_table=_classify_layer_table(key),
    )


def _after_call(http_response: Any = None, context: dict | None = None, **_: Any) -> None:
    acc = _accumulator_ctx.get()
    if acc is None or context is None:
        return
    info = context.get(_REQUEST_CONTEXT_KEY)
    if isinstance(info, _RequestInfo):
        acc.record(info, status_code=getattr(http_response, "status_code", None))


def _request_tier(operation: str) -> str:
    if operation in {"PutObject", "CopyObject"}:
        return "tier1_write_or_copy"
    if operation in {"ListObjects", "ListObjectsV2"}:
        return "tier1_list"
    if operation in {"GetObject", "HeadObject"}:
        return "tier2_read_head_get"
    if operation in {"DeleteObject", "DeleteObjects"}:
        return "delete"
    return "unknown"


def _classify_layer_table(key: str) -> str:
    parts = [part for part in str(key or "").lstrip("/").split("/") if part]
    if not parts:
        return "unknown"
    layer = sanitize_sql_label(parts[0], max_length=80) or "unknown"
    table = (
        parts[1]
        if layer in {"silver_core", "silver_application", "gold", "gold_staging", "gold_backup"}
        and len(parts) > 1
        else None
    )
    safe_table = sanitize_sql_label(table, max_length=120)
    return f"{layer}.{safe_table}" if safe_table else layer
