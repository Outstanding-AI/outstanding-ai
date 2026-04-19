"""Metrics utilities for structured logging and performance tracking."""

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def timed_operation(operation_name: str, **extra: Any):
    """Context manager for timing operations with structured logging.

    Usage:
        with timed_operation("draft_generation", party_id=party_id):
            result = await generate_draft(...)

    Logs:
        - On success: {operation_name} completed with latency_ms and success=True
        - On failure: {operation_name} failed with latency_ms, success=False, and error

    Args:
        operation_name: Name of the operation for logging (e.g., "draft_generation")
        **extra: Additional key-value pairs to include in log output
    """
    start = time.perf_counter()
    try:
        yield
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{operation_name} completed",
            extra={
                "metric_type": operation_name,
                "latency_ms": round(latency_ms, 2),
                "success": True,
                **extra,
            },
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error(
            f"{operation_name} failed",
            extra={
                "metric_type": operation_name,
                "latency_ms": round(latency_ms, 2),
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                **extra,
            },
        )
        raise


def log_metric(metric_type: str, **data: Any) -> None:
    """Log a metric event with structured data.

    Usage:
        log_metric("rate_limit_hit", provider="vertex", model="gemini-2.5-flash")

    Args:
        metric_type: Type of metric event
        **data: Metric data to log
    """
    logger.info(
        f"{metric_type}",
        extra={
            "metric_type": metric_type,
            **data,
        },
    )
