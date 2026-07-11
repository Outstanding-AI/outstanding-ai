"""Vertex-first collection-chain identifier after reconciliation."""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import CollectionChainIdentificationRequest
from src.api.models.responses import CollectionChainIdentificationResponse
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import CollectionChainIdentificationLLMResponse

from .audit import build_ai_audit
from .collection_email_event_classifier import _invalid_response_telemetry, _parse_response_object

PROMPT_TEMPLATE_ID = "collection_chain_identifier"
PROMPT_TEMPLATE_VERSION = "v2"
_SYSTEM_PROMPT = """Decide only whether a bounded email-chain event establishes, confirms, reopens, closes, or leaves uncertain a collection-related chain.
Use the current message, exact/bounded prior email context, extracted email facts, and reconciled scope outcome codes. Reconciled outcomes are facts about matching, not proof of collection relevance. Ignore quoted or forwarded text as authored intent.
Do not classify debtor promises/disputes/remittances, do not decide Sage truth, policy, recipients, routing, or drafting.
Return only strict JSON with collection_status, event_effect, confidence, reason_codes, and evidence_message_ordinals."""


class CollectionChainIdentifier:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex", fallback_provider="openai"
        )

    async def identify(
        self, request: CollectionChainIdentificationRequest
    ) -> CollectionChainIdentificationResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            # The identifier response is deliberately small and closed, so it
            # can use provider-native structured output without exposing the
            # broader event-classifier schema to Vertex.
            response_schema=CollectionChainIdentificationLLMResponse,
            caller="collection_chain_identifier",
        )
        try:
            parsed = CollectionChainIdentificationLLMResponse(
                **_parse_response_object(response.content)
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise LLMResponseInvalidError(
                message="LLM returned invalid collection-chain identifier response",
                details={
                    "operation": "collection_chain_identifier",
                    "telemetry": _invalid_response_telemetry(response),
                },
            ) from exc
        return CollectionChainIdentificationResponse(
            collection_status=parsed.collection_status,
            event_effect=parsed.event_effect,
            confidence=parsed.confidence,
            reason_codes=parsed.reason_codes,
            evidence_message_ordinals=parsed.evidence_message_ordinals,
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


collection_chain_identifier = CollectionChainIdentifier()
