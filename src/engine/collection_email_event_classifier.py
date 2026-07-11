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
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionEmailEventLLMResponse

from .audit import build_ai_audit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "collection_email_event"
PROMPT_TEMPLATE_VERSION = "v4"

_SYSTEM_PROMPT = """You classify one accounts-receivable email-chain event.
Decide only collection relevance, email lifecycle, and debtor-response facts.
Use the current message and bounded prior email context; quoted or forwarded
text is not authored intent. Do not use or infer Sage balances, payment state,
debtor policy, recipients, draft routing, or a collection chain choice.
When semantic_classification is present, use the existing controlled debtor
response taxonomy exactly (for example PROMISE_TO_PAY, DISPUTE,
REMITTANCE_ADVICE, ALREADY_PAID, or UNCLEAR).

For a known collection chain, preserve collection relevance unless this event
explicitly closes or reopens the email conversation. A debtor payment or
promise claim is pending_financial_confirmation, never proof of payment.
Return a JSON object only, with exactly these keys and types:
{
  "relevance_status": "collection" | "non_collection" | "uncertain",
  "lifecycle_status": "active" | "awaiting_debtor_response" |
      "pending_financial_confirmation" | "closed_by_email" | "uncertain" |
      "not_applicable",
  "semantic_classification": an existing uppercase debtor-response taxonomy
      value or null,
  "secondary_intents": [uppercase taxonomy values],
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
that JSON object."""

_USER_PROMPT = """Mode: {mode}\n\nEmail event evidence:\n{payload}"""


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
            parsed = CollectionEmailEventLLMResponse(**json.loads(response.content))
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
                details={"mode": request.mode, "validation_errors": validation_errors},
            ) from exc
        return CollectionEmailEventResponse(
            relevance_status=parsed.relevance_status,
            lifecycle_status=parsed.lifecycle_status,
            semantic_classification=parsed.semantic_classification,
            secondary_intents=parsed.secondary_intents,
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
