"""Email-native collection-chain event classifier.

This deliberately does not receive Sage, policy, or routing context.  It
classifies one chronological event and a bounded set of prior email evidence.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailEventRequest
from src.api.models.responses import CollectionEmailEventResponse
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionEmailEventLLMResponse

from .audit import build_ai_audit

PROMPT_TEMPLATE_ID = "collection_email_event"
PROMPT_TEMPLATE_VERSION = "v2"

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
Return strict JSON only."""

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
            response_schema=CollectionEmailEventLLMResponse,
            caller="collection_email_event",
        )
        try:
            parsed = CollectionEmailEventLLMResponse(**json.loads(response.content))
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid collection-email event response",
                details={"mode": request.mode},
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
