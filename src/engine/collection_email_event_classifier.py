"""Email-native collection-chain event classifier.

This deliberately does not receive Sage, policy, or routing context.  It
classifies one chronological event and a bounded set of prior email evidence.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailEventRequest
from src.api.models.responses import CollectionEmailEventResponse
from src.config.constants import CLASSIFICATION_CATEGORIES
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import (
    CollectionEmailAmountAssertion,
    CollectionEmailDateAssertion,
    CollectionEmailEventLLMResponse,
)

from ._evidence_grounding import (
    amount_is_explicit_in_span,
    date_is_explicit_in_span,
    locate_evidence,
)
from .audit import build_ai_audit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "collection_email_event"
PROMPT_TEMPLATE_VERSION = "v12"
_MAX_GROUNDING_ATTEMPTS = 3

# A manual outbound message is authored by our organisation. It can record a
# collection touch, escalation, or acknowledgement, but cannot truthfully
# report a debtor-authored promise, dispute, remittance, or payment claim.
_MANUAL_OUTBOUND_SEMANTIC_CLASSES = frozenset(
    {
        "OUTBOUND_COLLECTION_ACTION",
        "OUTBOUND_ESCALATION_ACTION",
        "OUTBOUND_PROMISE_ACKNOWLEDGEMENT",
    }
)
_MANUAL_OUTBOUND_LIFECYCLE_STATUSES = frozenset(
    {"active", "awaiting_debtor_response", "not_applicable", "uncertain"}
)

_GROUNDING_VALIDATION_REMEDIATION = {
    "amount_evidence_not_verbatim": (
        "amount_evidence_text must be an exact, unique verbatim substring of the current message "
        "body. Quote the exact text containing the amount, or set amount, currency, and "
        "amount_evidence_text to null if you cannot quote it verbatim."
    ),
    "amount_evidence_not_unique": (
        "amount_evidence_text matched more than one place in the body. Extend the quoted span until "
        "it identifies the amount uniquely."
    ),
    "amount_evidence_does_not_support_value": (
        "The quoted amount_evidence_text does not contain the numeric amount you asserted. Either "
        "correct amount to match the quoted text, or set amount, currency, and amount_evidence_text "
        "to null."
    ),
    "date_evidence_not_verbatim": (
        "date_evidence_text must be an exact, unique verbatim substring of the current message body. "
        "Quote the exact text containing the date, or set date_value and date_evidence_text to null "
        "if you cannot quote it verbatim."
    ),
    "date_evidence_not_unique": (
        "date_evidence_text matched more than one place in the body. Extend the quoted span until it "
        "identifies the date uniquely."
    ),
    "date_evidence_does_not_support_value": (
        "The quoted date_evidence_text does not contain the calendar date you asserted. Either "
        "correct date_value to match the quoted text, or set date_value and date_evidence_text to "
        "null."
    ),
    "manual_outbound_semantic_classification_invalid": (
        "For mode manual_outbound, semantic_classification and every secondary intent must be null "
        "or one of OUTBOUND_COLLECTION_ACTION, OUTBOUND_ESCALATION_ACTION, and "
        "OUTBOUND_PROMISE_ACKNOWLEDGEMENT. Debtor-authored response categories are invalid."
    ),
    "manual_outbound_lifecycle_status_invalid": (
        "For mode manual_outbound, lifecycle_status must be active, awaiting_debtor_response, "
        "not_applicable, or uncertain. A manual outbound email cannot itself claim financial "
        "confirmation or close a collection conversation."
    ),
    "manual_outbound_neutral_intent_details_invalid": (
        "When manual_outbound has no semantic_classification, secondary_intents and intent_details "
        "must both be empty."
    ),
    "manual_outbound_nonsemantic_fields_invalid": (
        "For mode manual_outbound, secondary_intents, intent_details, invoice_assertions, "
        "amount_assertions, and date_assertions must all be empty. A manually authored "
        "outbound email records only one non-financial operator-action label."
    ),
}

_CONTROLLED_TAXONOMY = ", ".join(sorted(CLASSIFICATION_CATEGORIES))
_SYSTEM_PROMPT = (
    """You classify one accounts-receivable email-chain event.
Decide only collection relevance, email lifecycle, and debtor-response facts.
Use the current message and bounded prior email context; quoted or forwarded
text is not authored intent. Do not use or infer Sage balances, payment state,
debtor policy, recipients, draft routing, or a collection chain choice.
The context is causal: it contains only earlier retained events from this one
conversation. Never infer an outcome from a later message, from silence, or
from any fact not present in the current event or explicit prior evidence.
Treat a manually authored outbound message as authored email evidence. Treat a
system-generated outbound message only as the supplied deterministic draft fact;
do not invent its invoice scope. A deleted or unavailable event has no authored
content and must produce no invoice or debtor-response assertion.
When semantic_classification is present, use exactly one value from the same
controlled debtor-response taxonomy as the operational classifier:
"""
    + _CONTROLLED_TAXONOMY
    + ".\n"
    + """

For a known collection chain, preserve collection relevance unless this event
explicitly closes or reopens the email conversation. A debtor payment or
promise claim is pending_financial_confirmation, never proof of payment.

DISPUTE TRIAGE — APPLY BEFORE RULING AN INBOUND DEBTOR RESPONSE NON-DISPUTE
First inspect the current debtor-authored text for a challenge to the invoice,
amount, payment terms, delivery, billing party, tax, price, contractual basis,
or another condition that prevents or questions payment. Do not rely on a
subject line, quoted history, prior messages, or silence. The absence of the
word “dispute” is not enough to rule out a challenge; equally, a missing or
vague challenge is never enough to create a blocking dispute control.

Use these mutually exclusive decision boundaries for each invoice-scoped
statement:
* DISPUTE: the debtor challenges whether an invoice is valid or payable at all,
  or asks for correction/investigation because of an asserted invoice error,
  ownership/billing issue, delivery/service issue, duplicate/incorrect charge,
  tax issue, cancellation, credit, or contractual objection.
* AMOUNT_DISAGREEMENT: the debtor accepts that the invoice is owed but asserts
  that its balance, rate, quantity, tax, credit, or other monetary figure is
  wrong.
* PAYMENT_TIMING_DISPUTE: the debtor challenges the invoice's due date,
  payment terms, or whether it is currently payable. A future payment date is
  not this intent when it is an unqualified payment commitment; use
  PROMISE_TO_PAY instead.
* DEBTOR_INTERNAL_PROCESSING_BLOCKER: the debtor does not challenge the
  invoice's correctness, but expressly says a concrete debtor-side process
  prevents payment—for example an unresolved approval, matching, receipt,
  portal, or payment-run dependency. Do not turn a debtor-side processing
  blocker into DISPUTE.
* QUERY_QUESTION: the debtor asks for information but does not assert a
  challenge or a concrete payment-blocking process.

Statements such as an acknowledgement, “checking”, “looking into it”,
“please assist”, “not seen yet”, a bare PO/reference, a generic portal link,
or a generic request-receipt SLA do not prove a dispute or blocker. If the
current text plausibly signals an invoice concern but does not establish one
of the boundaries above, abstain from a material control: set
semantic_classification to null, use lifecycle_status="uncertain", leave
intent_details empty, and add reason_codes containing
"possible_invoice_concern_insufficient_for_control". Never use a material
dispute class just to avoid missing a concern; the system retains that
uncertainty for later evidence without inventing a query state.

``prior_evidence`` can contain one ``chain_invoice_context`` object. Its
invoice_candidates are body-free identifiers explicitly established in earlier
messages from this chain; they are not Sage results. For each response intent:
use an invoice named in the current authored text first. If the current text is
deictic (for example, "we will pay it Friday" or "we dispute this") you may
link it to exactly one candidate only when candidate_count is 1 and
is_truncated is false; include that invoice in the intent's invoice_refs and
add ``contextual_single_invoice_link`` to reason_codes. When the candidate set
is empty, multiple, or truncated, do not guess and leave invoice_refs empty,
with ``ambiguous_contextual_invoice_scope`` or
``missing_contextual_invoice_scope``. Never assign one promise, dispute, or
remittance to every invoice in a chain.
Return a JSON object only, with exactly these keys and types:
{
  "relevance_status": "collection" | "non_collection" | "uncertain",
  "lifecycle_status": "active" | "awaiting_debtor_response" |
      "pending_financial_confirmation" | "closed_by_email" | "uncertain" |
      "not_applicable",
  "semantic_classification": an existing uppercase debtor-response taxonomy
      value or null,
  "secondary_intents": [uppercase taxonomy values],
  "intent_details": [{"intent": uppercase taxonomy value,
      "extracted_data": {"invoice_refs": [strings], and only the controlled
      fields belonging to this intent. For every non-null promise_amount,
      disputed_amount, or claimed_amount include its matching
      *_amount_evidence_text field. For every non-null promise_date,
      claimed_date, claimed_due_date, or claimed_payment_date include its
      matching *_date_evidence_text field. A PROMISE_TO_PAY with no numeric
      amount stated in the current debtor-authored text must carry
      "promise_amount": null and "full_current_balance": true. An explicitly
      stated numeric promise amount must carry "full_current_balance": false.}}],
  "invoice_assertions": ["invoice reference"],
  "amount_assertions": [{"invoice_ref": string-or-null, "amount":
      number-or-null, "currency": string-or-null, "assertion_type":
      "claimed_paid" | "claimed_due" | "promised_payment" |
      "disputed_amount" | "remittance_amount" | "unknown"}],
  "date_assertions": [{"invoice_ref": string-or-null, "date_value":
      string-or-null, "assertion_type": "promise_date" | "payment_date" |
      "due_date" | "remittance_date" | "other"}],
  "reason_codes": ["controlled_snake_case_code"],
  "confidence": number from 0 through 1
}
Use [] or null when a field has no evidence. Do not add keys or prose outside
that JSON object. When more than one intent exists, keep every intent's invoice
references and amount/date facts isolated in its own intent_details entry. The
first intent_details entry must match semantic_classification. This is the same
debtor-response taxonomy and per-intent extraction contract used by the
operational debtor-response classifier.

GROUNDING
Every non-null amount and date_value must be backed by an exact verbatim
substring of the current message body, returned in that assertion's
amount_evidence_text / date_evidence_text field. Quote only the minimal span
containing the value; extend it only as needed to make the quote unique in
the body. Never paraphrase, reformat, or normalize the quoted text — copy it
exactly as written, including original date/number formatting. If you cannot
find or quote exact verbatim text supporting an amount or date, set that
amount/date_value and its evidence field to null rather than guessing.
For PROMISE_TO_PAY, an amount-less commitment means the full current balance
of the safely reconciled invoice scope: set full_current_balance=true while
keeping promise_amount, promise_amount_evidence_text, and promised-payment
amount_assertions empty. This flag is accounting-resolution provenance, not a
claim that the debtor stated a number. If the debtor states a numeric partial
or exact amount, set full_current_balance=false and ground that amount
verbatim. Never derive a debtor-stated amount from chain metadata or context.
Invoice references you name explicitly in intent_details/invoice_assertions
do not require a separate evidence field, but must still come from text
actually present in the current message or a valid single-candidate
contextual link as described above — never invent an invoice number."""
)

_USER_PROMPT = """Mode: {mode}\n\nEmail event evidence:\n{payload}"""

_MANUAL_OUTBOUND_MODE_CONTRACT = """
MANUAL-OUTBOUND MODE CONTRACT
The current message was authored by our organisation. Classify only its current
operational meaning; earlier-chain context may resolve scope after a current
meaning is established, but can never create that meaning.

For this mode, semantic_classification may be only null,
OUTBOUND_COLLECTION_ACTION, OUTBOUND_ESCALATION_ACTION, or
OUTBOUND_PROMISE_ACKNOWLEDGEMENT. Do not use debtor-authored response
categories such as PROMISE_TO_PAY, REQUEST_INFO, ALREADY_PAID, DISPUTE,
REMITTANCE_ADVICE, GENERIC_ACKNOWLEDGEMENT, or COOPERATIVE.

Choose OUTBOUND_COLLECTION_ACTION only when the current authored message
expressly asks the recipient to pay, settle, or provide a payment date, status,
or update for an invoice or account balance. Choose
OUTBOUND_ESCALATION_ACTION only when the current authored message expressly
communicates an escalation or consequence of continued non-payment. Choose
OUTBOUND_PROMISE_ACKNOWLEDGEMENT only when the current authored message
expressly acknowledges a debtor's previously stated payment promise or payment
confirmation.

Otherwise semantic_classification must be null with empty secondary_intents
and intent_details. Invoice issuance, attachment, statement distribution,
document resend, reference numbers, due dates, and invitations to ask questions
are neutral unless the current text also contains one of the explicit meanings
above. Before selecting a non-null value, decide whether that meaning remains
in the current message after ignoring prior_messages, prior_evidence, and
chain_status. If it does not, return null.

Return exactly one semantic classification or null. Always leave
secondary_intents, intent_details, invoice_assertions, amount_assertions, and
date_assertions empty. Do not extract or infer invoice scope, balance, payment
amount, payment date, promise, remittance, dispute, query, or settlement fact
from an operator-authored email.
"""


def _parse_response_object(content: str) -> dict:
    """Parse strict JSON, allowing only the common fenced-JSON transport wrapper."""
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            fence = lines[0].strip().lower()
            if fence in {"```", "```json"}:
                text = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("collection_email_event_response_must_be_object")
    return parsed


def _grounded_or_nulled_amount(
    assertion: CollectionEmailAmountAssertion, *, body: str
) -> CollectionEmailAmountAssertion:
    """Verify amount_evidence_text, or null the whole claim out (abstain rather than guess).

    A model that omits evidence entirely is treated as an abstention (soft
    null, no retry — nothing to correct). A model that supplies evidence that
    doesn't actually verify is a correctable mistake and raises ValueError so
    the retry loop in ``classify()`` can give it one more attempt.
    """
    if assertion.amount is None:
        return assertion
    if not assertion.amount_evidence_text:
        return assertion.model_copy(
            update={"amount": None, "currency": None, "amount_evidence_text": None}
        )
    _, _, span = locate_evidence(body, assertion.amount_evidence_text, field="amount")
    if not amount_is_explicit_in_span(assertion.amount, span):
        raise ValueError("amount_evidence_does_not_support_value")
    return assertion


def _grounded_or_nulled_date(
    assertion: CollectionEmailDateAssertion, *, body: str
) -> CollectionEmailDateAssertion:
    if assertion.date_value is None:
        return assertion
    if not assertion.date_evidence_text:
        return assertion.model_copy(update={"date_value": None, "date_evidence_text": None})
    _, _, span = locate_evidence(body, assertion.date_evidence_text, field="date")
    if not date_is_explicit_in_span(assertion.date_value, span):
        raise ValueError("date_evidence_does_not_support_value")
    return assertion


def _apply_grounding(
    parsed: CollectionEmailEventLLMResponse, *, body: str
) -> CollectionEmailEventLLMResponse:
    """Ground every amount/date assertion against the current message body.

    Raises ValueError (a retry-worthy, snake_case-coded failure) when the
    model supplied evidence that doesn't verify. Missing evidence is not an
    error here — it's silently downgraded to an abstention, matching the
    "never guess" grounding contract without wasting a retry on a model that
    simply chose not to extract a value.
    """
    grounded_amounts = [
        _grounded_or_nulled_amount(item, body=body) for item in parsed.amount_assertions
    ]
    grounded_dates = [_grounded_or_nulled_date(item, body=body) for item in parsed.date_assertions]
    grounded_details = []
    for detail in parsed.intent_details:
        extracted = detail.extracted_data
        if extracted is None:
            grounded_details.append(detail)
            continue
        updates: dict[str, object] = {}
        # Decide amount provenance from the model's original monetary claim,
        # before grounding can null an unsupported value. A genuinely
        # amount-less promise defaults to the full current balance; a proposed
        # explicit amount that fails grounding must remain neither a trusted
        # partial amount nor a silently widened full-balance promise.
        if str(detail.intent or "").upper() == "PROMISE_TO_PAY":
            updates["full_current_balance"] = extracted.promise_amount is None
        else:
            updates["full_current_balance"] = False
        for value_field, evidence_field in (
            ("promise_amount", "promise_amount_evidence_text"),
            ("disputed_amount", "disputed_amount_evidence_text"),
            ("claimed_amount", "claimed_amount_evidence_text"),
        ):
            value = getattr(extracted, value_field)
            evidence = getattr(extracted, evidence_field)
            if value is None:
                continue
            if not evidence:
                updates[value_field] = None
                updates[evidence_field] = None
                continue
            _, _, span = locate_evidence(body, evidence, field="amount")
            if not amount_is_explicit_in_span(value, span):
                raise ValueError("amount_evidence_does_not_support_value")
        for value_field, evidence_field in (
            ("promise_date", "promise_date_evidence_text"),
            ("claimed_date", "claimed_date_evidence_text"),
            ("claimed_due_date", "claimed_due_date_evidence_text"),
            ("claimed_payment_date", "claimed_payment_date_evidence_text"),
        ):
            value = getattr(extracted, value_field)
            evidence = getattr(extracted, evidence_field)
            if value is None:
                continue
            if not evidence:
                updates[value_field] = None
                updates[evidence_field] = None
                continue
            _, _, span = locate_evidence(body, evidence, field="date")
            if not date_is_explicit_in_span(value, span):
                raise ValueError("date_evidence_does_not_support_value")
        grounded_details.append(
            detail.model_copy(update={"extracted_data": extracted.model_copy(update=updates)})
        )
    return parsed.model_copy(
        update={
            "amount_assertions": grounded_amounts,
            "date_assertions": grounded_dates,
            "intent_details": grounded_details,
        }
    )


def _validate_manual_outbound_contract(parsed: CollectionEmailEventLLMResponse) -> None:
    """Fail closed when a sender-role-specific output contradicts manual-outbound mode."""

    if parsed.lifecycle_status not in _MANUAL_OUTBOUND_LIFECYCLE_STATUSES:
        raise ValueError("manual_outbound_lifecycle_status_invalid")
    intents = [
        value
        for value in [parsed.semantic_classification, *parsed.secondary_intents]
        if value is not None
    ]
    if any(intent not in _MANUAL_OUTBOUND_SEMANTIC_CLASSES for intent in intents):
        raise ValueError("manual_outbound_semantic_classification_invalid")
    if not parsed.semantic_classification and (parsed.secondary_intents or parsed.intent_details):
        raise ValueError("manual_outbound_neutral_intent_details_invalid")
    if any(
        (
            parsed.secondary_intents,
            parsed.intent_details,
            parsed.invoice_assertions,
            parsed.amount_assertions,
            parsed.date_assertions,
        )
    ):
        raise ValueError("manual_outbound_nonsemantic_fields_invalid")


def _grounding_validation_code(exc: Exception) -> str:
    """Map a raised validation/grounding error to a stable snake_case code.

    Our own grounding ValueErrors already raise with the code as the message
    (see locate_evidence / _grounded_or_nulled_*); anything else (a raw
    Pydantic ValidationError, a JSON decode error) falls back to a generic
    schema-invalid code.
    """
    if isinstance(exc, ValueError) and str(exc):
        code = str(exc)
        if re.fullmatch(r"[a-z0-9_]+", code):
            return code
    return "collection_email_event_response_schema_invalid"


def _invalid_response_telemetry(response) -> dict[str, object]:
    """Keep billable model telemetry when strict output parsing fails.

    No model text, prompts, addresses, or request payloads are included. The
    backend uses this safe envelope to settle the reserved budget and write a
    failed LLM audit row instead of recording paid invalid output as zero cost.
    """

    usage = response.usage if isinstance(getattr(response, "usage", None), dict) else {}
    return {
        "provider": str(getattr(response, "provider", "unknown") or "unknown"),
        "model": str(getattr(response, "model", "unknown") or "unknown"),
        "is_fallback": bool(getattr(response, "is_fallback", False)),
        "tokens_used": int(usage.get("total_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


class CollectionEmailEventClassifier:
    def __init__(self) -> None:
        # Per-caller default override (None/None until validated against the
        # labeled eval dataset — see settings.collection_email_event_*_model).
        default_override = {
            key: value
            for key, value in {
                "vertex": settings.collection_email_event_vertex_model,
                "openai": settings.collection_email_event_openai_model,
            }.items()
            if value
        }
        self._client = LLMProviderWithFallback(
            primary_provider="vertex",
            fallback_provider="openai",
            model_override=default_override,
        )
        manual_outbound_override = {
            key: value
            for key, value in {
                "vertex": settings.manual_outbound_email_vertex_model,
                "openai": settings.manual_outbound_email_openai_model,
            }.items()
            if value
        }
        self._manual_outbound_client = LLMProviderWithFallback(
            primary_provider="vertex",
            fallback_provider="openai",
            model_override=manual_outbound_override,
        )

    async def classify(self, request: CollectionEmailEventRequest) -> CollectionEmailEventResponse:
        client = self._manual_outbound_client if request.mode == "manual_outbound" else self._client
        if request.model_override:
            # Eval-harness path only (see the field docstring on
            # CollectionEmailEventRequest) — build a one-off client instead of
            # mutating the shared instance's override.
            client = LLMProviderWithFallback(
                primary_provider="vertex",
                fallback_provider="openai",
                model_override=dict(request.model_override),
            )
        prompt_input = request.model_dump(
            mode="json", exclude_none=True, exclude={"model_override"}
        )
        body = str((request.current_message or {}).get("body") or "")
        active_system_prompt = (
            _SYSTEM_PROMPT + _MANUAL_OUTBOUND_MODE_CONTRACT
            if request.mode == "manual_outbound"
            else _SYSTEM_PROMPT
        )
        last_exc: Exception | None = None
        response = None
        user_prompt = ""
        attempt_count = 0
        for _ in range(_MAX_GROUNDING_ATTEMPTS):
            attempt_count += 1
            user_prompt = _USER_PROMPT.format(
                mode=request.mode,
                payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
            )
            response = await client.complete(
                system_prompt=active_system_prompt,
                user_prompt=user_prompt,
                temperature=settings.classification_temperature,
                # Vertex rejects this otherwise-valid nested JSON Schema because
                # it exceeds its serving-state budget. JSON mode plus strict
                # Pydantic parsing preserves the contract without sending a
                # provider-native schema that either provider cannot serve.
                json_mode=True,
                caller="collection_email_event",
            )
            try:
                parsed = CollectionEmailEventLLMResponse(**_parse_response_object(response.content))
                parsed = _apply_grounding(parsed, body=body)
                if request.mode == "manual_outbound":
                    _validate_manual_outbound_contract(parsed)
            except (ValidationError, ValueError, TypeError) as exc:
                last_exc = exc
                validation_code = _grounding_validation_code(exc)
                remediation = _GROUNDING_VALIDATION_REMEDIATION.get(
                    validation_code,
                    "Correct the proposed extraction to satisfy the system contract exactly.",
                )
                active_system_prompt = (
                    f"{_SYSTEM_PROMPT}"
                    f"{_MANUAL_OUTBOUND_MODE_CONTRACT if request.mode == 'manual_outbound' else ''}"
                    "\n\nVALIDATION CORRECTION MODE\n"
                    f"The previous output failed {validation_code}. {remediation} "
                    "Return a fresh corrected extraction and do not repeat the invalid field value."
                )
                continue
            return CollectionEmailEventResponse(
                relevance_status=parsed.relevance_status,
                lifecycle_status=parsed.lifecycle_status,
                semantic_classification=parsed.semantic_classification,
                secondary_intents=parsed.secondary_intents,
                intent_details=[
                    item.model_dump(exclude_none=True) for item in parsed.intent_details
                ],
                invoice_assertions=parsed.invoice_assertions,
                amount_assertions=[
                    item.model_dump(exclude_none=True) for item in parsed.amount_assertions
                ],
                date_assertions=[
                    item.model_dump(exclude_none=True) for item in parsed.date_assertions
                ],
                reason_codes=parsed.reason_codes,
                confidence=parsed.confidence,
                tokens_used=response.usage.get("total_tokens", 0),
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                provider=response.provider,
                model=response.model,
                is_fallback=response.provider != client.primary_provider_name,
                ai_audit=build_ai_audit(
                    response=response,
                    prompt_template_id=PROMPT_TEMPLATE_ID,
                    prompt_template_version=PROMPT_TEMPLATE_VERSION,
                    system_prompt=active_system_prompt,
                    user_prompt=user_prompt,
                    prompt_input=prompt_input,
                    token_count=response.usage.get("total_tokens", 0),
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    inference_profile="classification",
                ),
            )
        assert response is not None and last_exc is not None
        exc = last_exc
        validation_errors = []
        if isinstance(exc, ValidationError):
            validation_errors = [
                {
                    "location": ".".join(str(part) for part in error.get("loc", ())),
                    "type": str(error.get("type") or "validation_error"),
                }
                for error in exc.errors()[:8]
            ]
        elif isinstance(exc, json.JSONDecodeError):
            validation_errors = [{"location": "response", "type": "json_decode_error"}]
        else:
            validation_errors = [{"location": "grounding", "type": _grounding_validation_code(exc)}]
        # Keep diagnostics useful without recording model output, prompt
        # text, or customer content in application logs or API errors.
        logger.warning(
            "Collection-email event response failed strict validation",
            extra={
                "mode": request.mode,
                "validation_errors": validation_errors,
                "error_type": type(exc).__name__,
                "attempt_count": attempt_count,
            },
        )
        raise LLMResponseInvalidError(
            message="LLM returned invalid collection-email event response",
            details={
                "mode": request.mode,
                "validation_errors": validation_errors,
                "telemetry": _invalid_response_telemetry(response),
                "attempt_count": attempt_count,
            },
        ) from exc


collection_email_event_classifier = CollectionEmailEventClassifier()
