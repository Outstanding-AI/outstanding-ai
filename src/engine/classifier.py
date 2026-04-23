"""
Email classification engine.

Classify inbound debtor emails into one of 23 categories using a structured
LLM call.  The classifier also extracts structured intent data (promise
dates, disputed amounts, insolvency details, redirect contacts, etc.) that
the Django backend uses to drive obligation-level side-effects.

Categories (grouped by post-classification draft action):

**Follow-up** (discard old draft, generate new):
    COOPERATIVE, REQUEST_INFO, PLAN_REQUEST, ALREADY_PAID, REDIRECT,
    QUERY_QUESTION, ESCALATION_REQUEST

**Discard only** (no new draft -- case paused or needs review):
    PROMISE_TO_PAY, DISPUTE, INSOLVENCY, HARDSHIP, UNSUBSCRIBE, HOSTILE,
    OUT_OF_OFFICE, AMOUNT_DISAGREEMENT, RETENTION_CLAIM, LEGAL_RESPONSE

**No action** (keep existing draft):
    UNCLEAR, GENERIC_ACKNOWLEDGEMENT, PAYMENT_CONFIRMATION,
    REMITTANCE_ADVICE, EMAIL_BOUNCE, PARTIAL_PAYMENT_NOTIFICATION

The classifier runs guardrails on its own *reasoning* output to validate
any facts the LLM mentions (invoice numbers, amounts).  Guardrail results
are returned alongside the classification but do NOT block classification
delivery -- they serve as a quality signal for downstream consumers.
"""

import json
import logging
from datetime import date

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import ClassifyRequest
from src.api.models.responses import (
    ClassifyResponse,
    ExtractedData,
    GuardrailValidation,
    IntentDetail,
)
from src.config.settings import settings
from src.guardrails.base import GuardrailSeverity
from src.guardrails.pipeline import guardrail_pipeline
from src.llm.factory import llm_client
from src.llm.schemas import ClassificationLLMResponse
from src.prompts import CLASSIFY_EMAIL_SYSTEM, CLASSIFY_EMAIL_USER
from src.prompts._sanitize import sanitize_delimiter_tags

from .formatters import format_industry_context_for_classification, format_invoice_table

logger = logging.getLogger(__name__)


class EmailClassifier:
    """Classify inbound debtor emails into categories with intent extraction.

    The classifier is stateless; all context arrives via the request object.
    A singleton instance (``classifier``) is exported at module level for
    use by the FastAPI route handler.

    Pipeline:
    1. Build a rich user prompt with party context, per-invoice table,
       industry context, and the raw email (subject + body).
    2. Call the LLM with ``ClassificationLLMResponse`` structured output.
    3. Parse extracted data (promise dates, amounts, redirect contacts).
    4. Run guardrails on the LLM reasoning text.
    5. Return classification, confidence, extracted data, and guardrail
       validation metadata.
    """

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """
        Classify an inbound email.

        Args:
            request: Classification request with email and context

        Returns:
            Classification result with confidence, extracted data, and guardrail validation
        """
        # Compute derived aggregates from obligations for the prompt.
        # These give the LLM a summary view alongside the per-invoice table.
        total_outstanding = (
            request.context.total_outstanding_base
            if request.context.total_outstanding_base is not None
            else sum(
                o.amount_due_base if o.amount_due_base is not None else o.amount_due
                for o in request.context.obligations
            )
        )
        days_overdue_max = max((o.days_past_due for o in request.context.obligations), default=0)

        # Industry context helps the LLM interpret domain-specific
        # language (e.g., retention claims in construction, seasonal
        # patterns in agriculture).
        industry_context = format_industry_context_for_classification(request.context.industry)

        # Per-invoice table gives the LLM granular visibility into each
        # obligation so it can match invoice references in the email body
        # and detect per-invoice intents (e.g., "we paid INV-1234").
        invoice_table = format_invoice_table(request.context)

        # Build user prompt with context
        user_prompt = CLASSIFY_EMAIL_USER.format(
            party_name=request.context.party.name,
            customer_code=request.context.party.customer_code,
            currency=request.context.base_currency,
            total_outstanding=total_outstanding,
            days_overdue_max=days_overdue_max,
            broken_promises_count=request.context.broken_promises_count,
            segment=(
                request.context.behavior.behaviour_segment
                if request.context.behavior
                else "unknown"
            ),
            active_dispute=request.context.active_dispute,
            hardship_indicated=request.context.hardship_indicated,
            invoice_table=invoice_table,
            is_verified=request.context.party.is_verified,
            party_source=request.context.party.source,
            industry_context=industry_context,
            from_name=request.email.from_name or "Unknown",
            from_address=request.email.from_address,
            subject=sanitize_delimiter_tags(request.email.subject),
            body=sanitize_delimiter_tags(request.email.body),
        )

        # Use response_schema for guaranteed valid JSON (no markdown wrapping)
        response = await llm_client.complete(
            system_prompt=CLASSIFY_EMAIL_SYSTEM,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            response_schema=ClassificationLLMResponse,
            caller="classification",
        )

        # Parse JSON response - structured output guarantees valid JSON
        tokens_used = response.usage.get("total_tokens", 0)
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)
        raw_result = json.loads(response.content)

        # Validate LLM response using Pydantic schema
        try:
            result = ClassificationLLMResponse(**raw_result)
        except ValidationError as e:
            logger.error(f"LLM response validation failed: {e}")
            raise LLMResponseInvalidError(
                message="LLM returned invalid classification response",
                details={"validation_errors": e.errors()},
            )

        # Parse extracted data — flat (legacy) + per-intent (PR4)
        extracted = _build_extracted_data(result.extracted_data)

        intent_details: list[IntentDetail] | None = None
        if result.intent_details:
            parsed_details: list[IntentDetail] = []
            for detail in result.intent_details:
                detail_extracted = _build_extracted_data(detail.extracted_data)
                parsed_details.append(
                    IntentDetail(intent=detail.intent, extracted_data=detail_extracted)
                )
            intent_details = parsed_details or None

            # Back-fill the flat extracted_data with the primary intent's
            # extraction when the LLM omitted the legacy field but provided
            # intent_details. Pre-PR4 consumers still see the right data.
            if extracted is None and intent_details:
                primary_detail = next(
                    (d for d in intent_details if d.intent == result.classification),
                    intent_details[0],
                )
                extracted = primary_detail.extracted_data

        # Run guardrails on LLM reasoning (validate any facts mentioned)
        guardrail_validation = None
        if result.reasoning:
            guardrail_result = guardrail_pipeline.validate(
                output=result.reasoning,
                context=request.context,
                extracted_data=extracted,
            )

            # Calculate factual accuracy
            total_checks = len(guardrail_result.results)
            passed_checks = sum(1 for r in guardrail_result.results if r.passed)
            factual_accuracy = passed_checks / total_checks if total_checks > 0 else 1.0

            # Separate warnings from blocking failures
            warnings = [
                r.guardrail_name
                for r in guardrail_result.results
                if not r.passed and r.severity in (GuardrailSeverity.MEDIUM, GuardrailSeverity.LOW)
            ]

            guardrail_validation = GuardrailValidation(
                all_passed=guardrail_result.all_passed,
                guardrails_run=total_checks,
                guardrails_passed=passed_checks,
                blocking_failures=guardrail_result.blocking_guardrails,
                warnings=warnings,
                review_findings=guardrail_result.review_findings,
                factual_accuracy=factual_accuracy,
            )

            if not guardrail_result.all_passed:
                logger.warning(
                    f"Guardrails failed for {request.context.party.customer_code}: "
                    f"blocking={guardrail_result.blocking_guardrails}, warnings={warnings}"
                )

        logger.info(
            f"Classified email for {request.context.party.customer_code}: "
            f"{result.classification} (confidence: {result.confidence:.2f})"
        )

        return ClassifyResponse(
            classification=result.classification,
            confidence=result.confidence,
            reasoning=result.reasoning,
            secondary_intents=result.secondary_intents,
            extracted_data=extracted,
            intent_details=intent_details,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            forbidden_content_detected=result.forbidden_content_detected or [],
            guardrail_validation=guardrail_validation,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != llm_client.primary_provider_name),
        )


def _build_extracted_data(raw) -> ExtractedData | None:
    """Convert an ``LLMExtractedData`` (string dates) to ``ExtractedData`` (date objects).

    Returns ``None`` if ``raw`` is ``None`` or every field is ``None`` — keeps
    the response payload light when the LLM had nothing to extract for an
    intent. Date parse failures are logged (not raised) so a single malformed
    date does not drop the entire extraction.

    Used by both the legacy flat ``extracted_data`` path and the PR4
    ``intent_details[*].extracted_data`` per-intent extractions.
    """
    if raw is None:
        return None
    if not any(v is not None for v in raw.model_dump().values()):
        return None

    def _parse_date(value, field_name: str):
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            logger.warning("Could not parse %s: %s", field_name, value)
            return None

    return ExtractedData(
        promise_date=_parse_date(raw.promise_date, "promise_date"),
        promise_amount=raw.promise_amount,
        dispute_type=raw.dispute_type,
        dispute_reason=raw.dispute_reason,
        invoice_refs=raw.invoice_refs,
        disputed_amount=raw.disputed_amount,
        claimed_amount=raw.claimed_amount,
        claimed_date=_parse_date(raw.claimed_date, "claimed_date"),
        claimed_reference=raw.claimed_reference,
        claimed_details=raw.claimed_details,
        insolvency_type=raw.insolvency_type,
        insolvency_details=raw.insolvency_details,
        administrator_name=raw.administrator_name,
        administrator_email=raw.administrator_email,
        reference_number=raw.reference_number,
        return_date=_parse_date(raw.return_date, "return_date"),
        redirect_name=raw.redirect_name,
        redirect_contact=raw.redirect_contact,
        redirect_email=raw.redirect_email,
        bounced_email=raw.bounced_email,
    )


# Singleton instance used by the /classify route handler.
classifier = EmailClassifier()
