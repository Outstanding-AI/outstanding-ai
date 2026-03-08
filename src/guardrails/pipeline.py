"""Guardrail Pipeline - orchestrates all guardrails."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

from src.api.models.requests import CaseContext

from .base import BaseGuardrail, GuardrailPipelineResult, GuardrailResult, GuardrailSeverity
from .contextual import ContextualCoherenceGuardrail
from .entity import EntityVerificationGuardrail
from .factual_grounding import FactualGroundingGuardrail
from .numerical import NumericalConsistencyGuardrail
from .placeholder import PlaceholderValidationGuardrail
from .temporal import TemporalConsistencyGuardrail

logger = logging.getLogger(__name__)

# Default max retries for failed guardrails
DEFAULT_MAX_RETRIES = 2

# Thread pool for parallel guardrail execution (6 guardrails = 6 workers)
_guardrail_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="guardrail")


class GuardrailPipeline:
    """
    Orchestrates all guardrails in a pipeline.

    Guardrails are run in order of severity:
    1. Critical guardrails (block on failure)
    2. High guardrails (block on failure)
    3. Medium guardrails (warn, allow with flag)
    4. Low guardrails (log only)
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
        """Get the default set of guardrails."""
        return [
            PlaceholderValidationGuardrail(),  # Deterministic, zero-cost — runs first
            FactualGroundingGuardrail(),
            NumericalConsistencyGuardrail(),
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
            return self._validate_parallel(output, context, **kwargs)
        return self._validate_sequential(output, context, fail_fast, **kwargs)

    def _validate_parallel(
        self,
        output: str,
        context: CaseContext,
        **kwargs,
    ) -> GuardrailPipelineResult:
        """
        Run all guardrails in parallel using thread pool.

        Note: fail_fast is not supported in parallel mode since all guardrails
        run concurrently. All results are collected and evaluated together.
        """
        pipeline_start = time.perf_counter()
        all_results: list[GuardrailResult] = []
        blocking_guardrails: list[str] = []
        should_block = False
        guardrail_latencies = {}

        # Submit all guardrails to thread pool
        futures = {
            _guardrail_executor.submit(
                self._run_single_guardrail, guardrail, output, context, **kwargs
            ): guardrail
            for guardrail in self.guardrails
        }

        # Collect results as they complete
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
                        logger.warning(
                            f"Guardrail {guardrail_name} BLOCKED output: {result.message}"
                        )

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
                "guardrails_run": len(self.guardrails),
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

    def _validate_sequential(
        self,
        output: str,
        context: CaseContext,
        fail_fast: bool = True,
        **kwargs,
    ) -> GuardrailPipelineResult:
        """
        Run all guardrails sequentially (original implementation).

        Args:
            output: The AI-generated output to validate
            context: The input context
            fail_fast: If True, stop on first critical failure
            **kwargs: Additional context (extracted_data, etc.)

        Returns:
            GuardrailPipelineResult with all validation results
        """
        pipeline_start = time.perf_counter()
        all_results: list[GuardrailResult] = []
        blocking_guardrails: list[str] = []
        should_block = False
        guardrail_latencies = {}

        for guardrail in self.guardrails:
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

                        logger.warning(
                            f"Guardrail {guardrail.name} BLOCKED output: {result.message}"
                        )

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
                "guardrails_run": len(self.guardrails),
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

    def get_retry_prompt_addition(self, pipeline_result: GuardrailPipelineResult) -> str:
        """
        Generate additional prompt instructions based on guardrail failures.

        Used when retrying after a guardrail failure.
        """
        if pipeline_result.all_passed:
            return ""

        additions = [
            "\n\n**IMPORTANT VALIDATION REQUIREMENTS:**",
            "The previous response had validation errors. Please ensure:",
        ]

        for result in pipeline_result.results:
            if not result.passed:
                if result.guardrail_name == "placeholder_validation":
                    additions.append(
                        "- Do NOT invent placeholders like [SOMETHING] or {SOMETHING}. "
                        "The ONLY allowed placeholder is {INVOICE_TABLE}. "
                        "Use actual values from the context provided."
                    )
                    if result.found:
                        additions.append(
                            f"- Remove these hallucinated placeholders: {result.found}"
                        )
                elif result.guardrail_name == "factual_grounding":
                    additions.append(
                        "- ONLY use invoice numbers and amounts from the provided context"
                    )
                    if result.expected:
                        additions.append(f"- Valid invoices: {result.expected}")
                elif result.guardrail_name == "numerical_consistency":
                    additions.append("- Verify all calculations match (totals = sum of parts)")
                    if result.details.get("calculated_total"):
                        additions.append(f"- Correct total: {result.details['calculated_total']}")
                elif result.guardrail_name == "entity_verification":
                    additions.append("- Use exact customer code and company name from context")
                    if result.expected:
                        additions.append(f"- Expected: {result.expected}")

        return "\n".join(additions)


# Singleton instance for easy import
guardrail_pipeline = GuardrailPipeline()
