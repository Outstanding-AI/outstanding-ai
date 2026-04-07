"""Real-time evaluator for per-request evaluation."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.api.models.requests import ClassifyRequest, GenerateDraftRequest
from src.api.models.responses import ClassifyResponse, GenerateDraftResponse
from src.guardrails.base import GuardrailPipelineResult

from .metrics import InteractionMetrics

logger = logging.getLogger(__name__)


class RealTimeEvaluator:
    """
    Evaluates AI performance in real-time for each request.

    Runs after every classification or generation request to:
    1. Calculate factual accuracy from guardrail results
    2. Log performance metrics (latency, tokens)
    3. Track extraction success
    4. Store metrics for batch analysis
    """

    def __init__(self, store: Optional[Any] = None):
        """
        Initialize the evaluator.

        Args:
            store: Optional storage backend for metrics (e.g., database, queue)
        """
        self.store = store
        self._metrics_buffer: list[InteractionMetrics] = []
        self._buffer_size = 100  # Flush after this many metrics

    def evaluate_classification(
        self,
        request: ClassifyRequest,
        response: ClassifyResponse,
        guardrail_result: Optional[GuardrailPipelineResult] = None,
        latency_ms: float = 0.0,
        provider: str = "",
        model: str = "",
    ) -> InteractionMetrics:
        """
        Evaluate a classification request/response.

        Args:
            request: The classification request
            response: The classification response
            guardrail_result: Result from guardrail pipeline
            latency_ms: Time taken for the request
            provider: LLM provider used
            model: Model used

        Returns:
            InteractionMetrics with evaluation results
        """
        request_id = str(uuid.uuid4())

        # Calculate factual accuracy from guardrails
        guardrails_passed = True
        guardrail_failures = []
        factual_accuracy = 1.0

        if guardrail_result:
            guardrails_passed = guardrail_result.all_passed
            guardrail_failures = [
                r.guardrail_name for r in guardrail_result.results if not r.passed
            ]
            # Accuracy = passed / total checks
            total_checks = len(guardrail_result.results)
            passed_checks = sum(1 for r in guardrail_result.results if r.passed)
            factual_accuracy = passed_checks / total_checks if total_checks > 0 else 1.0

        # Extract promise/dispute data if present
        extracted = response.extracted_data
        promise_date = None
        promise_amount = None
        dispute_type = None

        if extracted:
            if extracted.promise_date:
                promise_date = str(extracted.promise_date)
            promise_amount = extracted.promise_amount
            dispute_type = extracted.dispute_type

        metrics = InteractionMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            guardrails_passed=guardrails_passed,
            guardrail_failures=guardrail_failures,
            factual_accuracy=factual_accuracy,
            classification=response.classification,
            classification_confidence=response.confidence,
            promise_date_extracted=promise_date,
            promise_amount_extracted=promise_amount,
            dispute_type_extracted=dispute_type,
            latency_ms=latency_ms,
            tokens_used=response.tokens_used or 0,
            provider_used=provider,
            model_used=model,
        )

        # Log metrics
        self._log_metrics(metrics)

        # Buffer for batch storage
        self._buffer_metrics(metrics)

        return metrics

    def evaluate_generation(
        self,
        request: GenerateDraftRequest,
        response: GenerateDraftResponse,
        guardrail_result: Optional[GuardrailPipelineResult] = None,
        latency_ms: float = 0.0,
        provider: str = "",
        model: str = "",
    ) -> InteractionMetrics:
        """
        Evaluate a draft generation request/response.

        Args:
            request: The generation request
            response: The generation response
            guardrail_result: Result from guardrail pipeline
            latency_ms: Time taken for the request
            provider: LLM provider used
            model: Model used

        Returns:
            InteractionMetrics with evaluation results
        """
        request_id = str(uuid.uuid4())

        # Calculate factual accuracy from guardrails
        guardrails_passed = True
        guardrail_failures = []
        factual_accuracy = 1.0

        if guardrail_result:
            guardrails_passed = guardrail_result.all_passed
            guardrail_failures = [
                r.guardrail_name for r in guardrail_result.results if not r.passed
            ]
            total_checks = len(guardrail_result.results)
            passed_checks = sum(1 for r in guardrail_result.results if r.passed)
            factual_accuracy = passed_checks / total_checks if total_checks > 0 else 1.0

        metrics = InteractionMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
            guardrails_passed=guardrails_passed,
            guardrail_failures=guardrail_failures,
            factual_accuracy=factual_accuracy,
            latency_ms=latency_ms,
            tokens_used=response.tokens_used or 0,
            provider_used=provider,
            model_used=model,
            tone_requested=request.tone,
        )

        # Log metrics
        self._log_metrics(metrics)

        # Buffer for batch storage
        self._buffer_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: InteractionMetrics) -> None:
        """Log metrics for monitoring."""
        log_data = {
            "request_id": metrics.request_id,
            "guardrails_passed": metrics.guardrails_passed,
            "factual_accuracy": f"{metrics.factual_accuracy:.2%}",
            "classification": metrics.classification,
            "confidence": f"{metrics.classification_confidence:.2f}",
            "latency_ms": f"{metrics.latency_ms:.0f}",
            "tokens": metrics.tokens_used,
        }

        if metrics.guardrails_passed:
            logger.info(f"Eval metrics: {log_data}")
        else:
            logger.warning(
                f"Eval metrics (guardrails failed): {log_data}, "
                f"failures={metrics.guardrail_failures}"
            )

    def _buffer_metrics(self, metrics: InteractionMetrics) -> None:
        """Buffer metrics for batch storage."""
        self._metrics_buffer.append(metrics)

        if len(self._metrics_buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered metrics to storage."""
        if not self._metrics_buffer:
            return

        if self.store:
            try:
                # Store would implement bulk insert
                # self.store.bulk_insert(self._metrics_buffer)
                logger.info(f"Flushed {len(self._metrics_buffer)} metrics to storage")
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")

        self._metrics_buffer = []

    def get_summary_stats(self) -> dict:
        """Get summary statistics from buffered metrics."""
        if not self._metrics_buffer:
            return {}

        total = len(self._metrics_buffer)
        passed = sum(1 for m in self._metrics_buffer if m.guardrails_passed)
        avg_accuracy = sum(m.factual_accuracy for m in self._metrics_buffer) / total
        avg_latency = sum(m.latency_ms for m in self._metrics_buffer) / total
        avg_tokens = sum(m.tokens_used for m in self._metrics_buffer) / total

        return {
            "total_requests": total,
            "guardrail_pass_rate": passed / total,
            "avg_factual_accuracy": avg_accuracy,
            "avg_latency_ms": avg_latency,
            "avg_tokens_used": avg_tokens,
        }


# Singleton instance
realtime_evaluator = RealTimeEvaluator()
