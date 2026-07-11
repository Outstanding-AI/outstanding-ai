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
PROMPT_TEMPLATE_VERSION = "v3"
_SYSTEM_PROMPT = """Decide only whether a bounded email-chain event establishes, confirms, reopens, closes, or leaves uncertain a collection-related chain.
Use the current message, exact/bounded prior email context, extracted email facts, and reconciled scope outcome codes. Reconciled outcomes are facts about matching, not proof of collection relevance. Ignore quoted or forwarded text as authored intent.
Do not classify debtor promises/disputes/remittances, do not decide Sage truth, policy, recipients, routing, or drafting.
Return only strict JSON with collection_status, event_effect, confidence, reason_codes, and evidence_message_ordinals.
Always include all five keys. If insufficient email evidence exists, return:
{"collection_status":"uncertain","event_effect":"no_change","confidence":0.0,"reason_codes":["insufficient_email_evidence"],"evidence_message_ordinals":[]}
collection_status is collection, non_collection, or uncertain. event_effect is
new, confirmed, reopened, closed, or no_change. evidence_message_ordinals is a
list of the bounded-context message ordinals only.
Do not add prose or any other key."""

_CHAIN_KEYS = {
    "collection_status",
    "relevance_label",
    "event_effect",
    "confidence",
    "reason_codes",
    "evidence_message_ordinals",
}


def _canonical_chain_response_object(content: object) -> dict:
    """Fail closed while adapting the documented historical relevance aliases."""

    raw = _parse_response_object(content)
    if set(raw) - _CHAIN_KEYS:
        raise ValueError("collection_chain_identifier_unknown_fields")
    status = raw.get("collection_status", raw.get("relevance_label"))
    status = {
        "collection_related": "collection",
        "non_collection": "non_collection",
        "uncertain": "uncertain",
    }.get(status, status)
    effect = raw.get("event_effect")
    reasons = raw.get("reason_codes") or []
    if not isinstance(reasons, list):
        raise ValueError("collection_chain_identifier_reason_codes_must_be_list")
    ordinals = raw.get("evidence_message_ordinals") or []
    if not isinstance(ordinals, list):
        raise ValueError("collection_chain_identifier_ordinals_must_be_list")
    if effect is None:
        # Missing lifecycle effect must never activate or close a chain.
        status, effect = "uncertain", "no_change"
        reasons = [*reasons, "missing_event_effect_abstention"]
    return {
        "collection_status": status,
        "event_effect": effect,
        "confidence": raw.get("confidence", 0.0),
        "reason_codes": reasons,
        "evidence_message_ordinals": ordinals,
    }


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
            # Keep this aligned with fact extraction: Vertex may reject even
            # bounded nested schemas under its serving-state limit.
            json_mode=True,
            caller="collection_chain_identifier",
        )
        try:
            parsed = CollectionChainIdentificationLLMResponse(
                **_canonical_chain_response_object(response.content)
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
