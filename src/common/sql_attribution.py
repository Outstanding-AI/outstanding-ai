"""SQL attribution helpers for AI runtime lake reads."""

from __future__ import annotations

import re
from typing import Any

SQL_ATTRIBUTION_VERSION = "solvix_sql:v1"
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def sanitize_sql_label(value: Any, *, max_length: int = 120) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    cleaned = _SAFE_LABEL_RE.sub("_", raw).strip("_.:-")
    return cleaned[:max_length] or None


def athena_attribution_comment(
    *,
    source: str | None,
    tenant_id: str | None = None,
    sync_run_id: str | None = None,
    runtime: str = "ai",
    component: str = "lake_reader",
    query_class: str = "select",
) -> str:
    safe_runtime = sanitize_sql_label(runtime, max_length=40) or "ai"
    safe_component = sanitize_sql_label(component, max_length=80)
    safe_source = (
        sanitize_sql_label(source, max_length=120) or f"{safe_runtime}.lake_reader.unknown"
    )
    parts = [
        SQL_ATTRIBUTION_VERSION,
        f"runtime={safe_runtime}",
    ]
    if safe_component:
        parts.append(f"component={safe_component}")
    parts.append(f"source={safe_source}")
    if tenant_id:
        parts.append(f"tenant={tenant_id}")
    sync = sanitize_sql_label(sync_run_id, max_length=120)
    if sync:
        parts.append(f"sync_run_id={sync}")
    query = sanitize_sql_label(query_class, max_length=40)
    if query:
        parts.append(f"query_class={query}")
    return "/* " + ";".join(parts) + " */\n"
