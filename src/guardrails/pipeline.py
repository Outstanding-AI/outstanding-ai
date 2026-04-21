"""Guardrail Pipeline -- orchestrate all 7 guardrails.

Run guardrails either in parallel (default, via ``ThreadPoolExecutor``)
or sequentially (with optional fail-fast on CRITICAL failures).  The
pipeline collects individual ``GuardrailResult`` objects, determines
whether any blocking failures exist, and returns a single
``GuardrailPipelineResult`` consumed by the draft generator and
classifier.

Execution order is by severity (CRITICAL first, LOW last) so that the
most important checks are prioritised in sequential mode and their
results appear first in logs.

Thread pool size is fixed at 6 worker threads for the default
7-guardrail set. The pool is module-level to avoid per-request
thread creation overhead.
"""

import logging
import time
from typing import Tuple

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailPipelineResult, GuardrailResult, GuardrailSeverity
from .contextual import ContextualCoherenceGuardrail
from .entity import EntityVerificationGuardrail
from .executor import validate_parallel, validate_sequential
from .factual_grounding import FactualGroundingGuardrail
from .feedback import get_retry_prompt_addition
from .numerical import NumericalConsistencyGuardrail
from .placeholder import PlaceholderValidationGuardrail
from .temporal import TemporalConsistencyGuardrail

logger = logging.getLogger(__name__)

# Default max retries for failed guardrails
DEFAULT_MAX_RETRIES = 2


class GuardrailPipeline:
    """Orchestrate all 7 guardrails via parallel or sequential execution.

    Default mode is **parallel** using a module-level ``ThreadPoolExecutor``
    (6 shared workers across 7 guardrails). This allows I/O-bound checks (entity
    verification LLM call) to overlap with CPU-bound regex checks.

    Guardrails are sorted by severity (CRITICAL first) so that in
    sequential/fail-fast mode the most important checks run first.

    Severity determines blocking behaviour:
    - CRITICAL / HIGH: block output (``should_block=True``)
    - MEDIUM: warn, allow with flag
    - LOW: log only, allow

    A singleton instance (``guardrail_pipeline``) is exported at module
    level and shared by the draft generator and email classifier.
    """

    def __init__(self, guardrails: list[BaseGuardrail] = None):
        """
        Initialize the pipeline with guardrails.

        Args:
            guardrails: List of guardrails to run. If None, uses default set.
        """
        if guardrails is None:
            self.guardrails = self._get_default_guardrails()
        else:
            self.guardrails = guardrails

        # Sort by severity (critical first)
        severity_order = {
            GuardrailSeverity.CRITICAL: 0,
            GuardrailSeverity.HIGH: 1,
            GuardrailSeverity.MEDIUM: 2,
            GuardrailSeverity.LOW: 3,
        }
        self.guardrails.sort(key=lambda g: severity_order[g.severity])

        logger.info(
            f"Initialized guardrail pipeline with {len(self.guardrails)} guardrails: "
            f"{[g.name for g in self.guardrails]}"
        )

    def _get_default_guardrails(self) -> list[BaseGuardrail]:
        """Return the default set of 7 guardrails.

        Order here does not matter -- the ``__init__`` sorts by severity.
        PlaceholderValidation is listed first as a hint that it is the
        cheapest (deterministic, zero LLM calls).
        """
        from .tone_clamping import ToneClampingGuardrail

        return [
            PlaceholderValidationGuardrail(),  # Deterministic, zero-cost — runs first
            FactualGroundingGuardrail(),
            NumericalConsistencyGuardrail(),
            ToneClampingGuardrail(),  # Confirms a runtime-selected tone was provided
            EntityVerificationGuardrail(),
            TemporalConsistencyGuardrail(),
            ContextualCoherenceGuardrail(),
        ]

    def _run_single_guardrail(
        self,
        guardrail: BaseGuardrail,
        output: str,
        context: CaseContext,
        **kwargs,
    ) -> Tuple[str, list[GuardrailResult], Exception | None, float]:
        """
        Run a single guardrail and return results with timing.

        Returns:
            Tuple of (guardrail_name, results, exception_if_any, latency_ms)
        """
        start_time = time.perf_counter()
        try:
            results = guardrail.validate(output, context, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000
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
            return (guardrail.name, results, None, latency_ms)
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
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
            return (guardrail.name, [], e, latency_ms)

    def validate(
        self,
        output: str,
        context: CaseContext,
        fail_fast: bool = True,
        parallel: bool = True,
        **kwargs,
    ) -> GuardrailPipelineResult:
        """
        Run all guardrails on the output.

        Args:
            output: The AI-generated output to validate
            context: The input context
            fail_fast: If True, stop on first critical failure
            parallel: If True, run guardrails in parallel (default)
            **kwargs: Additional context (extracted_data, etc.)

        Returns:
            GuardrailPipelineResult with all validation results
        """
        if parallel:
            return validate_parallel(
                self.guardrails, self._run_single_guardrail, output, context, **kwargs
            )
        return validate_sequential(self.guardrails, output, context, fail_fast, **kwargs)

    def get_retry_prompt_addition(self, pipeline_result: GuardrailPipelineResult, **kwargs) -> str:
        """Generate context-aware retry prompt based on guardrail failures.

        Delegates to ``feedback.get_retry_prompt_addition()``.

        Args:
            pipeline_result: Results from the guardrail pipeline.
            **kwargs: Draft context flags -- ``skip_invoice_table`` and
                ``closure_mode``.

        Returns:
            Prompt addition string, or empty string if all passed.
        """
        return get_retry_prompt_addition(pipeline_result, **kwargs)


# Singleton instance shared by generator.py and classifier.py.
guardrail_pipeline = GuardrailPipeline()
