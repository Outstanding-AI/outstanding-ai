"""Vertex-first bounded fact extraction for one collection-mail event."""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionEmailFactExtractionRequest
from src.api.models.responses import CollectionEmailFactExtractionResponse
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionEmailFactExtractionLLMResponse

from .audit import build_ai_audit
from .collection_email_event_classifier import _parse_response_object

PROMPT_TEMPLATE_ID = "collection_email_fact_extraction"
PROMPT_TEMPLATE_VERSION = "v2"
_SYSTEM_PROMPT = """Extract only invoice references, monetary assertions, and date assertions from one email event.
Use the current message and bounded prior context only to resolve references. Ignore quoted/forwarded text as authored intent.
Do not decide whether the chain is collection-related, do not classify a debtor response, and do not infer Sage truth, policy,
recipients, a route, or a draft. Return only strict JSON with keys invoice_assertions, amount_assertions, date_assertions,
confidence, and reason_codes. Bind amount and due-date assertions to an
invoice_ref only when the authored text supports that relationship. Never
invent a pairing. Preserve unbound amount-only and due-date-only assertions so
the deterministic reconciler can abstain or use its unique combined fallback."""


class CollectionEmailFactExtractor:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def extract(
        self, request: CollectionEmailFactExtractionRequest
    ) -> CollectionEmailFactExtractionResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            json_mode=True,
            caller="collection_email_fact_extraction",
        )
        try:
            parsed = CollectionEmailFactExtractionLLMResponse(
                **_parse_response_object(response.content)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid collection-email fact response",
                details={"operation": "collection_email_fact_extraction"},
            ) from exc
        return CollectionEmailFactExtractionResponse(
            invoice_assertions=parsed.invoice_assertions,
            amount_assertions=[
                item.model_dump(exclude_none=True) for item in parsed.amount_assertions
            ],
            date_assertions=[item.model_dump(exclude_none=True) for item in parsed.date_assertions],
            confidence=parsed.confidence,
            reason_codes=parsed.reason_codes,
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


collection_email_fact_extractor = CollectionEmailFactExtractor()
