"""
Email classification engine.

Classify inbound debtor emails into a controlled set of categories using a structured
LLM call.  The classifier also extracts structured intent data (promise
dates, disputed amounts, insolvency details, redirect contacts, etc.) that
the Django backend uses to drive obligation-level side-effects.

Categories (grouped by post-classification draft action):

**Follow-up** (discard old draft, generate new):
    COOPERATIVE, REQUEST_INFO, PLAN_REQUEST, ALREADY_PAID, REDIRECT,
    QUERY_QUESTION, ESCALATION_REQUEST

**Discard only** (no new draft -- case paused or needs review):
    PROMISE_TO_PAY, DISPUTE, DEBTOR_INTERNAL_PROCESSING_BLOCKER, INSOLVENCY,
    HARDSHIP, UNSUBSCRIBE, HOSTILE, OUT_OF_OFFICE, AMOUNT_DISAGREEMENT,
    RETENTION_CLAIM, LEGAL_RESPONSE

**No action** (keep existing draft):
    UNCLEAR, GENERIC_ACKNOWLEDGEMENT, PAYMENT_CONFIRMATION,
    REMITTANCE_ADVICE, EMAIL_BOUNCE, PARTIAL_PAYMENT_NOTIFICATION

The classifier returns evidence only.  Its structured extraction is later
reconciled by the backend against the current, scoped accounting records
before it can affect a collection control.  Do not run debtor-facing draft
guardrails over the classifier's internal reasoning: those rules validate
email salutation, identity, tone, policy language, and collection wording,
none of which exists in a classification rationale.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import ClassifyRequest
from src.api.models.responses import ClassifyResponse, ExtractedData, IntentDetail
from src.config.settings import settings
from src.llm.factory import llm_client
from src.llm.schemas import ClassificationLLMResponse
from src.prompts import CLASSIFY_EMAIL_SYSTEM, CLASSIFY_EMAIL_USER
from src.prompts._sanitize import sanitize_delimiter_tags

from .audit import build_ai_audit
from .formatters import format_industry_context_for_classification, format_invoice_table

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT_TEMPLATE_ID = "classification"
CLASSIFICATION_PROMPT_TEMPLATE_VERSION = "partial_settlement_atomic_v6"
# Classifier responses are structured evidence, not debtor-facing drafts.
# The outbound guardrail pipeline must remain absent here.  In particular,
# IdentityScopeGuardrail's required ``Hello`` salutation is valid only for a
# rendered collection email and was producing false HIGH-severity alerts when
# applied to a one-line internal classification rationale.
CLASSIFICATION_GUARDRAIL_PIPELINE_VERSION: str | None = None

_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_WEEKDAY_ABBREVIATION_INDEX = {name[:3]: value for name, value in _WEEKDAY_INDEX.items()}
_WEEKDAY_PATTERN = "|".join(_WEEKDAY_INDEX)
_NEXT_WEEKDAY_RE = re.compile(rf"\bnext\s+({_WEEKDAY_PATTERN})\b", re.IGNORECASE)
_THIS_WEEKDAY_RE = re.compile(rf"\bthis\s+({_WEEKDAY_PATTERN})\b", re.IGNORECASE)
_ANCHORED_WEEKDAY_RE = re.compile(
    rf"\b(?:by|on|before|until)\s+({_WEEKDAY_PATTERN})\b",
    re.IGNORECASE,
)
_VAGUE_PROMISE_TIMING_RE = re.compile(
    r"\b(?:end\s+of\s+(?:the\s+)?(?:month|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"|next\s+payment\s+run|upcoming\s+payment\s+run|later\s+this\s+month|soon)\b",
    re.IGNORECASE,
)
_NEXT_PAYMENT_RUN_RE = re.compile(
    r"\b(?:the\s+)?next\s+payment\s+run\b",
    re.IGNORECASE,
)
_BOUNDED_PROMISE_PERIOD_RE = re.compile(
    r"\b(?:next\s+week|next\s+month|"
    r"(?:in|within)\s+(?:one|two|three|four|\d+)\s+(?:week|weeks|month|months)|"
    r"(?:one|two|three|four|\d+)\s+(?:week|weeks|month|months)(?:'|’)?\s+time)\b",
    re.IGNORECASE,
)
_EXACT_DATE_CUE_RE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b"
    rf"|\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|"
    rf"september|october|november|december)(?:\s+\d{{4}})?\b"
    rf"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    rf"\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"
    rf"|\b(?:today|tomorrow|next\s+(?:{_WEEKDAY_PATTERN})|this\s+(?:{_WEEKDAY_PATTERN})|"
    rf"(?:by|on|before|until)\s+(?:{_WEEKDAY_PATTERN}))\b",
    re.IGNORECASE,
)


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
    4. Return classification evidence.  The backend reconciles structured
       invoice scope and financial controls against its current, sync-scoped
       accounting context before durable side effects are applied.
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
        reference_date = _classification_reference_date(request)
        commitment_chase_dates = _configured_commitment_chase_dates(
            request,
            reference_date=reference_date,
        )
        received_at_display = (
            request.email.received_at.isoformat() if request.email.received_at else "not supplied"
        )

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
            forwarded_context=_format_forwarded_context_for_prompt(request.email.forwarded_context),
            historical_backfill_context=_format_historical_backfill_context(
                request.context.historical_backfill
            ),
            received_at=received_at_display,
            reference_date=reference_date.isoformat(),
            commitment_chase_dates=_format_commitment_chase_dates(commitment_chase_dates),
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

        relative_date_source = _relative_date_source_text(request)

        # Parse extracted data — flat (legacy) + per-intent (PR4)
        extracted = _build_extracted_data(
            result.extracted_data,
            reference_date=reference_date,
            source_text=relative_date_source,
            intent=result.classification,
            allowed_commitment_dates=commitment_chase_dates,
        )

        intent_details: list[IntentDetail] | None = None
        if result.intent_details:
            parsed_details: list[IntentDetail] = []
            for detail in result.intent_details:
                detail_extracted = _build_extracted_data(
                    detail.extracted_data,
                    reference_date=reference_date,
                    source_text=relative_date_source,
                    intent=detail.intent,
                    allowed_commitment_dates=commitment_chase_dates,
                )
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

        # ``reasoning`` is an internal explanation of the structured evidence,
        # never an email body.  It must not enter the outbound draft guardrail
        # pipeline: that pipeline enforces customer-email shape (including an
        # exact ``Hello`` greeting) and interprets debtor claims as our own
        # asserted accounting figures.  The authoritative enforcement boundary
        # for classifier output is the backend's scoped, current-accounting
        # reconciliation before it writes collection controls.
        guardrail_validation = None

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
            forbidden_content_detected=[
                item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                for item in (result.forbidden_content_detected or [])
            ],
            guardrail_validation=guardrail_validation,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != llm_client.primary_provider_name),
            classification_evidence_only=True,
            ai_audit=build_ai_audit(
                response=response,
                context=request.context,
                prompt_template_id=CLASSIFICATION_PROMPT_TEMPLATE_ID,
                prompt_template_version=CLASSIFICATION_PROMPT_TEMPLATE_VERSION,
                system_prompt=CLASSIFY_EMAIL_SYSTEM,
                user_prompt=user_prompt,
                prompt_input={
                    "context": request.context.model_dump(mode="json", exclude_none=True),
                    "email": request.email.model_dump(mode="json", exclude_none=True),
                },
                guardrail_pipeline_version=CLASSIFICATION_GUARDRAIL_PIPELINE_VERSION,
                token_count=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                inference_profile="classification",
            ),
        )


def _build_extracted_data(
    raw,
    *,
    reference_date: date | None = None,
    source_text: str | None = None,
    intent: str | None = None,
    allowed_commitment_dates: tuple[date, ...] = (),
) -> ExtractedData | None:
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

    normalized_intent = str(intent or "").upper()

    def _parse_date(value, field_name: str):
        if not value:
            return None
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            resolved = _resolve_relative_date(str(value), reference_date)
            if resolved:
                return resolved
            logger.warning("Could not parse %s: %s", field_name, value)
            return None

    def _parse_date_with_source_fallback(value, field_name: str, allowed_intents: set[str]):
        parsed = _parse_date(value, field_name)
        if parsed or normalized_intent not in allowed_intents:
            return parsed
        return _resolve_relative_date(source_text, reference_date)

    promise_date = _parse_date_with_source_fallback(
        raw.promise_date,
        "promise_date",
        {"PROMISE_TO_PAY"},
    )
    promise_date_evidence_text = raw.promise_date_evidence_text
    if normalized_intent == "PROMISE_TO_PAY" and _requires_schedule_date_fallback(source_text):
        # Vague but bounded debtor timing is normalized only to one of the
        # tenant's configured collection dates supplied to the model.  This
        # keeps the LLM useful for semantic interpretation without allowing
        # it to invent an arbitrary calendar date.
        payment_run_date = _resolve_next_payment_run_date(
            source_text,
            reference_date=reference_date,
            allowed_commitment_dates=allowed_commitment_dates,
        )
        if payment_run_date is not None:
            # "Next payment run" is a debtor-authored commitment, but not an
            # exact calendar date.  Its operational date is therefore derived
            # deterministically from the tenant's configured collection
            # calendar, never accepted from the model.  Preserve the original
            # debtor wording as evidence so the resolved date is auditable.
            promise_date = payment_run_date
            promise_date_evidence_text = _vague_promise_period_evidence(source_text)
        elif promise_date not in set(allowed_commitment_dates):
            promise_date = None
        elif not promise_date_evidence_text:
            promise_date_evidence_text = _vague_promise_period_evidence(source_text)

    proposed_payment_date = _parse_date_with_source_fallback(
        raw.proposed_payment_date,
        "proposed_payment_date",
        {"AMOUNT_DISAGREEMENT"},
    )
    proposed_payment_date_evidence_text = raw.proposed_payment_date_evidence_text
    if (
        normalized_intent == "AMOUNT_DISAGREEMENT"
        and raw.partial_settlement
        and _requires_schedule_date_fallback(source_text)
    ):
        # The same bounded calendar policy applies when a debtor commits to a
        # partial payment.  The original payment-run words remain the source
        # evidence; the service, not the model, resolves the operational date.
        payment_run_date = _resolve_next_payment_run_date(
            source_text,
            reference_date=reference_date,
            allowed_commitment_dates=allowed_commitment_dates,
        )
        if payment_run_date is not None:
            proposed_payment_date = payment_run_date
            proposed_payment_date_evidence_text = _vague_promise_period_evidence(source_text)
        elif proposed_payment_date not in set(allowed_commitment_dates):
            proposed_payment_date = None
        elif not proposed_payment_date_evidence_text:
            proposed_payment_date_evidence_text = _vague_promise_period_evidence(source_text)

    return ExtractedData(
        promise_date=promise_date,
        promise_amount=raw.promise_amount,
        # An amount-less debtor commitment means the full current balance for
        # its resolved invoice scope.  Keep the numeric field empty so a Sage
        # balance is never represented as debtor-authored evidence.  If the
        # model proposed any amount (even one later rejected by another
        # boundary), it must not simultaneously claim this default.
        full_current_balance=(normalized_intent == "PROMISE_TO_PAY" and raw.promise_amount is None),
        promise_date_evidence_text=promise_date_evidence_text,
        promise_amount_evidence_text=raw.promise_amount_evidence_text,
        promise_strength=raw.promise_strength,
        dispute_type=raw.dispute_type,
        dispute_reason=raw.dispute_reason,
        invoice_refs=raw.invoice_refs,
        disputed_amount=raw.disputed_amount,
        disputed_amount_evidence_text=raw.disputed_amount_evidence_text,
        partial_settlement=raw.partial_settlement,
        proposed_payment_amount=raw.proposed_payment_amount,
        proposed_payment_amount_evidence_text=raw.proposed_payment_amount_evidence_text,
        proposed_payment_date=proposed_payment_date,
        proposed_payment_date_evidence_text=proposed_payment_date_evidence_text,
        claimed_due_date=_parse_date(raw.claimed_due_date, "claimed_due_date"),
        claimed_payment_date=_parse_date_with_source_fallback(
            raw.claimed_payment_date,
            "claimed_payment_date",
            {"PAYMENT_TIMING_DISPUTE", "DEBTOR_INTERNAL_PROCESSING_BLOCKER"},
        ),
        claimed_due_date_evidence_text=raw.claimed_due_date_evidence_text,
        claimed_payment_date_evidence_text=raw.claimed_payment_date_evidence_text,
        payment_timing_reason=raw.payment_timing_reason,
        internal_blocker_type=raw.internal_blocker_type,
        internal_blocker_reason=raw.internal_blocker_reason,
        internal_blocker_owner_hint=raw.internal_blocker_owner_hint,
        claimed_amount=raw.claimed_amount,
        claimed_date=_parse_date_with_source_fallback(
            raw.claimed_date,
            "claimed_date",
            {
                "ALREADY_PAID",
                "PAYMENT_CONFIRMATION",
                "REMITTANCE_ADVICE",
                "PARTIAL_PAYMENT_NOTIFICATION",
            },
        ),
        claimed_amount_evidence_text=raw.claimed_amount_evidence_text,
        claimed_date_evidence_text=raw.claimed_date_evidence_text,
        claimed_reference=raw.claimed_reference,
        claimed_details=raw.claimed_details,
        insolvency_type=raw.insolvency_type,
        insolvency_details=raw.insolvency_details,
        administrator_name=raw.administrator_name,
        administrator_email=raw.administrator_email,
        reference_number=raw.reference_number,
        return_date=_parse_date_with_source_fallback(
            raw.return_date,
            "return_date",
            {"OUT_OF_OFFICE"},
        ),
        redirect_name=raw.redirect_name,
        redirect_contact=raw.redirect_contact,
        redirect_email=raw.redirect_email,
        bounced_email=raw.bounced_email,
        account_wide=raw.account_wide,
    )


def _classification_policy(request: ClassifyRequest) -> dict[str, Any]:
    """Return the effective scheduled-prep policy, if the tenant has one."""

    tenant_settings = request.context.tenant_settings or {}
    protocol = tenant_settings.get("escalation_protocol") or {}
    global_config = protocol.get("global") if isinstance(protocol, dict) else None
    policy = (
        global_config.get("draft_generation_policy") if isinstance(global_config, dict) else None
    )
    return policy if isinstance(policy, dict) else {}


def _classification_business_timezone(request: ClassifyRequest) -> ZoneInfo:
    """Use the collection policy's IANA timezone for debtor-relative dates.

    The generation policy is evaluated in tenant business time, so a payment
    run must use the same local calendar.  Fall back to UTC only for legacy or
    invalid policies, preserving the prior safe behaviour.
    """

    timezone_name = str(_classification_policy(request).get("timezone") or "UTC")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Invalid collection policy timezone; using UTC", timezone_name=timezone_name)
        return ZoneInfo("UTC")


def _classification_reference_date(request: ClassifyRequest) -> date:
    business_timezone = _classification_business_timezone(request)
    if request.email.received_at:
        return request.email.received_at.astimezone(business_timezone).date()

    cutoff = getattr(request.context, "application_decision_cutoff", None)
    if isinstance(cutoff, datetime):
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return cutoff.astimezone(business_timezone).date()
    if isinstance(cutoff, date):
        return cutoff
    if isinstance(cutoff, str):
        try:
            parsed = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.date()
            return parsed.astimezone(business_timezone).date()
        except ValueError:
            pass
    return datetime.now(business_timezone).date()


def _relative_date_source_text(request: ClassifyRequest) -> str:
    context = request.email.forwarded_context or {}
    current_reply = context.get("current_reply") if isinstance(context, dict) else None
    if isinstance(current_reply, dict) and current_reply.get("body_excerpt"):
        return str(current_reply["body_excerpt"])
    return request.email.body


def _resolve_relative_date(text: str | None, reference_date: date | None) -> date | None:
    if not text or not reference_date:
        return None
    lowered = str(text).lower()
    if re.search(r"\btoday\b", lowered):
        return reference_date
    if re.search(r"\btomorrow\b", lowered):
        return reference_date + timedelta(days=1)
    if re.search(r"\byesterday\b", lowered):
        return reference_date - timedelta(days=1)
    match = _NEXT_WEEKDAY_RE.search(lowered)
    if match:
        return _next_weekday(reference_date, _WEEKDAY_INDEX[match.group(1).lower()])

    match = _THIS_WEEKDAY_RE.search(lowered) or _ANCHORED_WEEKDAY_RE.search(lowered)
    if match:
        return _next_weekday(
            reference_date,
            _WEEKDAY_INDEX[match.group(1).lower()],
            include_today=True,
        )
    return None


def _requires_schedule_date_fallback(text: str | None) -> bool:
    """Return True when a commitment has timing but no exact calendar date.

    Month-end language and payment-run references are useful commitment
    evidence, but converting them into a fabricated calendar day changes the
    debtor's claim.  The backend owns the deterministic fallback to the next
    configured draft-generation date.
    """

    if not text:
        return False
    value = str(text)
    vague_timing = bool(
        _VAGUE_PROMISE_TIMING_RE.search(value) or _BOUNDED_PROMISE_PERIOD_RE.search(value)
    )
    return vague_timing and _EXACT_DATE_CUE_RE.search(value) is None


def _resolve_next_payment_run_date(
    text: str | None,
    *,
    reference_date: date | None,
    allowed_commitment_dates: tuple[date, ...],
) -> date | None:
    """Resolve an unqualified next-payment-run promise from tenant policy.

    A debtor can firmly commit to their next payment run without stating a
    calendar date.  The collection policy already supplies an ordered, closed
    list of tenant-local payment-run dates (the day before a draft window).
    Select the earliest permitted date on or after the debtor's local received
    date.  When the debtor says ``next week`` or ``next month``, constrain the
    selection to that period before falling forward.  This is a deterministic
    operational interpretation, not an LLM-generated date.
    """

    if not text or not reference_date or not _NEXT_PAYMENT_RUN_RE.search(str(text)):
        return None
    future_dates = sorted(value for value in allowed_commitment_dates if value >= reference_date)
    if not future_dates:
        return None

    lowered = str(text).lower()
    if re.search(r"\bnext\s+week\b", lowered):
        period_start = reference_date + timedelta(days=7 - reference_date.weekday())
        period_end = period_start + timedelta(days=6)
        in_period = [value for value in future_dates if period_start <= value <= period_end]
        if in_period:
            return in_period[0]
        after_period = [value for value in future_dates if value > period_end]
        return after_period[0] if after_period else None
    if re.search(r"\bnext\s+month\b", lowered):
        first_of_this_month = reference_date.replace(day=1)
        period_start = (first_of_this_month + timedelta(days=32)).replace(day=1)
        following_month = (period_start + timedelta(days=32)).replace(day=1)
        period_end = following_month - timedelta(days=1)
        in_period = [value for value in future_dates if period_start <= value <= period_end]
        if in_period:
            return in_period[0]
        after_period = [value for value in future_dates if value > period_end]
        return after_period[0] if after_period else None
    return future_dates[0]


def _vague_promise_period_evidence(text: str | None) -> str | None:
    if not text:
        return None
    match = _BOUNDED_PROMISE_PERIOD_RE.search(str(text))
    if match:
        return match.group(0)
    match = _NEXT_PAYMENT_RUN_RE.search(str(text))
    if match:
        return match.group(0)
    match = _VAGUE_PROMISE_TIMING_RE.search(str(text))
    return match.group(0) if match else None


def _configured_commitment_chase_dates(
    request: ClassifyRequest,
    *,
    reference_date: date,
    horizon_days: int = 124,
) -> tuple[date, ...]:
    """Return future payment-run dates allowed for vague commitments.

    An unqualified debtor statement that payment is in its "next payment
    run" means payment is due on the calendar day immediately before the next
    scheduled draft-generation day.  We derive that operational payment date
    from the tenant's configured draft-generation windows; the LLM receives
    the resulting closed list and cannot invent an arbitrary calendar date.
    """

    policy = _classification_policy(request)
    if not isinstance(policy, dict) or policy.get("mode") != "scheduled_prep":
        return ()

    weekdays: set[int] = set()
    for window in policy.get("windows") or []:
        if not isinstance(window, dict):
            continue
        weekday_name = str(window.get("weekday") or "").strip().lower()
        weekday = _WEEKDAY_INDEX.get(weekday_name)
        if weekday is None:
            weekday = _WEEKDAY_ABBREVIATION_INDEX.get(weekday_name[:3])
        if weekday is not None:
            weekdays.add(weekday)
    if not weekdays:
        return ()

    dates: list[date] = []
    cursor = reference_date + timedelta(days=1)
    end = reference_date + timedelta(days=max(1, horizon_days))
    while cursor <= end:
        if cursor.weekday() in weekdays:
            payment_run_date = cursor - timedelta(days=1)
            # Do not backdate a newly received promise.  A payment run whose
            # operational date is already past is not a future commitment.
            if payment_run_date >= reference_date:
                dates.append(payment_run_date)
        cursor += timedelta(days=1)
    return tuple(dates)


def _format_commitment_chase_dates(values: tuple[date, ...]) -> str:
    if not values:
        return "None configured; leave vague commitment dates null."
    return "\n".join(f"- {value.isoformat()} ({value.strftime('%A')})" for value in values)


def _next_weekday(
    reference_date: date, target_weekday: int, *, include_today: bool = False
) -> date:
    days_ahead = (target_weekday - reference_date.weekday()) % 7
    if days_ahead == 0 and not include_today:
        days_ahead = 7
    return reference_date + timedelta(days=days_ahead)


# Singleton instance used by the /classify route handler.
classifier = EmailClassifier()


def _format_forwarded_context_for_prompt(context: dict[str, Any] | None) -> str:
    """Render trusted ingestion-derived forward metadata without raw body text."""

    if not context:
        return "None detected."
    safe_context = {
        key: value
        for key, value in context.items()
        if key
        in {
            "source_type",
            "detection_methods",
            "body_forward_markers",
            "forward_header_fields",
            "internal_routing_cues",
            "matched_oai_draft_id",
            "validated_invoice_refs",
            "unresolved_invoice_refs",
            "same_thread_oai_draft_ids",
            "forwarded_lineage",
            "prompt_budget",
            "evidence_snippets",
            "instruction",
        }
    }
    return sanitize_delimiter_tags(json.dumps(safe_context, sort_keys=True, default=str))


def _format_historical_backfill_context(context: dict[str, Any] | None) -> str:
    """Render body-free historical audit metadata for collection-thread backfill."""

    if not context:
        return "Not a historical backfill classification."
    safe_context = {
        key: value
        for key, value in context.items()
        if key
        in {
            "mode",
            "message_role",
            "direction",
            "folder_name",
            "mail_message_id",
            "conversation_id",
            "chain_message_ordinal",
            "invoice_refs",
            "current_sage_scope",
            "protocol_touch_type",
            "protocol_level",
            "protocol_touch_index",
            "protocol_cadence_status",
            "classification_instruction",
        }
    }
    return sanitize_delimiter_tags(json.dumps(safe_context, sort_keys=True, default=str))
