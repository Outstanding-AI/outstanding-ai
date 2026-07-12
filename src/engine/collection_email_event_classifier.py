"""Email-native collection-chain event classifier.

This deliberately does not receive Sage, policy, or routing context.  It
classifies one chronological event and a bounded set of prior email evidence.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailEventRequest
from src.api.models.responses import CollectionEmailEventResponse
from src.config.constants import CLASSIFICATION_CATEGORIES
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionEmailEventLLMResponse

from .audit import build_ai_audit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "collection_email_event"
PROMPT_TEMPLATE_VERSION = "v7"

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
      fields belonging to this intent}}],
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
operational debtor-response classifier."""
)

_USER_PROMPT = """Mode: {mode}\n\nEmail event evidence:\n{payload}"""


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
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def classify(self, request: CollectionEmailEventRequest) -> CollectionEmailEventResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = _USER_PROMPT.format(
            mode=request.mode,
            payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
        )
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
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
        except (ValidationError, ValueError, TypeError) as exc:
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
            # Keep diagnostics useful without recording model output, prompt
            # text, or customer content in application logs or API errors.
            logger.warning(
                "Collection-email event response failed strict validation",
                extra={
                    "mode": request.mode,
                    "validation_errors": validation_errors,
                    "error_type": type(exc).__name__,
                },
            )
            raise LLMResponseInvalidError(
                message="LLM returned invalid collection-email event response",
                details={
                    "mode": request.mode,
                    "validation_errors": validation_errors,
                    "telemetry": _invalid_response_telemetry(response),
                },
            ) from exc
        return CollectionEmailEventResponse(
            relevance_status=parsed.relevance_status,
            lifecycle_status=parsed.lifecycle_status,
            semantic_classification=parsed.semantic_classification,
            secondary_intents=parsed.secondary_intents,
            intent_details=[item.model_dump(exclude_none=True) for item in parsed.intent_details],
            invoice_assertions=parsed.invoice_assertions,
            amount_assertions=[
                item.model_dump(exclude_none=True) for item in parsed.amount_assertions
            ],
            date_assertions=[item.model_dump(exclude_none=True) for item in parsed.date_assertions],
            reason_codes=parsed.reason_codes,
            confidence=parsed.confidence,
            tokens_used=response.usage.get("total_tokens", 0),
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            provider=response.provider,
            model=response.model,
            is_fallback=response.provider != self._client.primary_provider_name,
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_input=prompt_input,
                token_count=response.usage.get("total_tokens", 0),
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                inference_profile="classification",
            ),
        )


collection_email_event_classifier = CollectionEmailEventClassifier()
