"""
Guardrail execution strategies -- parallel and sequential.

Provides the two execution modes used by ``GuardrailPipeline``:
- ``validate_parallel()``: ThreadPoolExecutor with ``as_completed``
- ``validate_sequential()``: loop with optional fail-fast on CRITICAL
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from .base import BaseGuardrail, GuardrailPipelineResult, GuardrailResult, GuardrailSeverity

logger = logging.getLogger(__name__)

# Thread pool for parallel guardrail execution (7 guardrails share 6 worker threads)
_guardrail_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="guardrail")


def validate_parallel(
    guardrails: List[BaseGuardrail],
    run_single: Callable,
    output: str,
    context,
    **kwargs,
) -> GuardrailPipelineResult:
    """Run all guardrails in parallel using thread pool.

    Note: fail_fast is not supported in parallel mode since all guardrails
    run concurrently. All results are collected and evaluated together.

    Args:
        guardrails: List of guardrail instances to execute.
        run_single: Callback to execute a single guardrail (from pipeline).
        output: The AI-generated output to validate.
        context: The input context (CaseContext).
        **kwargs: Additional context (extracted_data, etc.)

    Returns:
        GuardrailPipelineResult with all validation results.
    """
    pipeline_start = time.perf_counter()
    all_results: list[GuardrailResult] = []
    blocking_guardrails: list[str] = []
    should_block = False
    guardrail_latencies = {}

    # Submit all guardrails concurrently.  Each guardrail runs in
    # its own thread, allowing I/O-bound checks (e.g., entity
    # verification LLM call) to overlap with CPU-bound regex checks.
    futures = {
        _guardrail_executor.submit(run_single, guardrail, output, context, **kwargs): guardrail
        for guardrail in guardrails
    }

    # Collect results as they complete (order is non-deterministic).
    # Blocking failures and exceptions are tracked for the aggregate
    # pipeline result.
    for future in as_completed(futures):
        guardrail_name, results, exception, latency_ms = future.result()
        guardrail_latencies[guardrail_name] = latency_ms

        if exception:
            logger.error(f"Guardrail {guardrail_name} raised exception: {exception}")
            all_results.append(
                GuardrailResult(
                    passed=False,
                    guardrail_name=guardrail_name,
                    severity=GuardrailSeverity.HIGH,
                    message=f"Guardrail execution error: {str(exception)}",
                    details={"exception": str(exception)},
                )
            )
            should_block = True
            blocking_guardrails.append(guardrail_name)
        else:
            all_results.extend(results)
            for result in results:
                if result.should_block:
                    should_block = True
                    if guardrail_name not in blocking_guardrails:
                        blocking_guardrails.append(guardrail_name)
                    logger.warning(f"Guardrail {guardrail_name} BLOCKED output: {result.message}")

    all_passed = all(r.passed for r in all_results)
    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000

    # Log pipeline summary
    logger.info(
        "Guardrail pipeline completed",
        extra={
            "metric_type": "guardrail_pipeline",
            "latency_ms": round(pipeline_latency_ms, 2),
            "all_passed": all_passed,
            "should_block": should_block,
            "guardrails_run": len(guardrails),
            "blocking_guardrails": blocking_guardrails,
            "guardrail_latencies": {k: round(v, 2) for k, v in guardrail_latencies.items()},
        },
    )

    # retry_suggested is True when there are blocking failures but
    # few enough (<= 2) that targeted LLM feedback is likely to fix
    # them.  The draft generator uses this to decide whether to retry.
    return GuardrailPipelineResult(
        all_passed=all_passed,
        should_block=should_block,
        results=all_results,
        retry_suggested=should_block and len(blocking_guardrails) <= 2,
        blocking_guardrails=blocking_guardrails,
    )


def validate_sequential(
    guardrails: List[BaseGuardrail],
    output: str,
    context,
    fail_fast: bool = True,
    **kwargs,
) -> GuardrailPipelineResult:
    """Run all guardrails sequentially (original implementation).

    Args:
        guardrails: List of guardrail instances to execute.
        output: The AI-generated output to validate.
        context: The input context (CaseContext).
        fail_fast: If True, stop on first critical failure.
        **kwargs: Additional context (extracted_data, etc.)

    Returns:
        GuardrailPipelineResult with all validation results.
    """
    pipeline_start = time.perf_counter()
    all_results: list[GuardrailResult] = []
    blocking_guardrails: list[str] = []
    should_block = False
    guardrail_latencies = {}

    for guardrail in guardrails:
        start_time = time.perf_counter()
        try:
            results = guardrail.validate(output, context, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
            guardrail_latencies[guardrail.name] = latency_ms
            passed = all(r.passed for r in results)

            logger.info(
                "Guardrail completed",
                extra={
                    "metric_type": "guardrail_completed",
                    "guardrail": guardrail.name,
                    "severity": guardrail.severity.value,
                    "latency_ms": round(latency_ms, 2),
                    "passed": passed,
                    "checks_count": len(results),
                },
            )

            all_results.extend(results)

            # Check for blocking failures
            for result in results:
                if result.should_block:
                    should_block = True
                    blocking_guardrails.append(guardrail.name)

                    logger.warning(f"Guardrail {guardrail.name} BLOCKED output: {result.message}")

                    if fail_fast and result.severity == GuardrailSeverity.CRITICAL:
                        pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000
                        logger.info(
                            "Guardrail pipeline completed (fail-fast)",
                            extra={
                                "metric_type": "guardrail_pipeline",
                                "latency_ms": round(pipeline_latency_ms, 2),
                                "all_passed": False,
                                "should_block": True,
                                "guardrails_run": len(guardrail_latencies),
                                "blocking_guardrails": blocking_guardrails,
                                "fail_fast": True,
                            },
                        )
                        # Stop immediately on critical failure
                        return GuardrailPipelineResult(
                            all_passed=False,
                            should_block=True,
                            results=all_results,
                            retry_suggested=True,
                            blocking_guardrails=blocking_guardrails,
                        )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            guardrail_latencies[guardrail.name] = latency_ms
            logger.error(
                "Guardrail failed with exception",
                extra={
                    "metric_type": "guardrail_error",
                    "guardrail": guardrail.name,
                    "severity": guardrail.severity.value,
                    "latency_ms": round(latency_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            # Create a failure result for the exception
            all_results.append(
                GuardrailResult(
                    passed=False,
                    guardrail_name=guardrail.name,
                    severity=GuardrailSeverity.HIGH,
                    message=f"Guardrail execution error: {str(e)}",
                    details={"exception": str(e)},
                )
            )
            should_block = True
            blocking_guardrails.append(guardrail.name)

    all_passed = all(r.passed for r in all_results)
    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000

    logger.info(
        "Guardrail pipeline completed",
        extra={
            "metric_type": "guardrail_pipeline",
            "latency_ms": round(pipeline_latency_ms, 2),
            "all_passed": all_passed,
            "should_block": should_block,
            "guardrails_run": len(guardrails),
            "blocking_guardrails": blocking_guardrails,
            "guardrail_latencies": {k: round(v, 2) for k, v in guardrail_latencies.items()},
        },
    )

    return GuardrailPipelineResult(
        all_passed=all_passed,
        should_block=should_block,
        results=all_results,
        retry_suggested=should_block and len(blocking_guardrails) <= 2,
        blocking_guardrails=blocking_guardrails,
    )
