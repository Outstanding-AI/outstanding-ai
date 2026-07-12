"""Vertex-first collection-chain identifier after reconciliation."""

from __future__ import annotations

import json
import re

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
PROMPT_TEMPLATE_VERSION = "v5"
_SYSTEM_PROMPT = """Decide only whether a bounded email-chain event establishes, confirms, reopens, closes, or leaves uncertain a collection-related chain.
Use the current message, exact/bounded prior email context, the body-free prior-chain invoice ledger, extracted email facts, and reconciled scope outcome codes. Reconciled outcomes are facts about matching, not proof of collection relevance. Ignore quoted or forwarded text as authored intent.
The prior-chain ledger preserves facts from preceding messages, including an
earlier invoice reference followed by a later promise or dispute response. It
is context only: do not turn a ledger entry into a Sage claim or a routing
decision. A chain can be collection-related even when its current invoice
mapping is unresolved.
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
_TRANSPORT_ONLY_CHAIN_KEYS = {"reason", "explanation", "summary"}
_COLLECTION_STATUS_ALIASES = {
    "collection": "collection",
    "collection_related": "collection",
    "collection-related": "collection",
    "non_collection": "non_collection",
    "non-collection": "non_collection",
    "not_collection": "non_collection",
    "not-related": "non_collection",
    "uncertain": "uncertain",
    "unknown": "uncertain",
    "insufficient_evidence": "uncertain",
}
_EVENT_EFFECT_ALIASES = {
    "new": "new",
    "new_chain": "new",
    "opened": "new",
    "confirmed": "confirmed",
    "ongoing": "confirmed",
    "existing": "confirmed",
    "reopened": "reopened",
    "reopen": "reopened",
    "closed": "closed",
    "close": "closed",
    "no_change": "no_change",
    "no change": "no_change",
    "unchanged": "no_change",
    "none": "no_change",
}


def _canonical_chain_response_object(content: object) -> dict:
    """Fail closed while adapting the documented historical relevance aliases."""

    raw = _parse_response_object(content)
    if set(raw) - _CHAIN_KEYS - _TRANSPORT_ONLY_CHAIN_KEYS:
        raise ValueError("collection_chain_identifier_unknown_fields")
    raw_statuses = {
        str(value).strip().lower()
        for value in (raw.get("collection_status"), raw.get("relevance_label"))
        if value not in (None, "")
    }
    status = (
        _COLLECTION_STATUS_ALIASES.get(next(iter(raw_statuses)), "uncertain")
        if len(raw_statuses) == 1
        else "uncertain"
    )
    effect = _EVENT_EFFECT_ALIASES.get(str(raw.get("event_effect") or "").strip().lower())
    reasons = raw.get("reason_codes") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)] if isinstance(reasons, str) else []
    ordinals = raw.get("evidence_message_ordinals") or []
    if not isinstance(ordinals, list):
        ordinals = []
    normalized_ordinals = []
    for ordinal in ordinals:
        try:
            parsed = int(ordinal)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            normalized_ordinals.append(parsed)
    normalized_reasons = [str(reason).strip().lower() for reason in reasons]
    reasons = [reason for reason in normalized_reasons if re.fullmatch(r"[a-z0-9_]{1,80}", reason)]
    if len(reasons) != len(normalized_reasons):
        reasons.append("uncontrolled_reason_code_discarded")
    reasons = reasons[:20]
    if not raw_statuses:
        reasons = [*reasons, "missing_collection_status_abstention"]
    elif (
        len(raw_statuses) != 1
        or status == "uncertain"
        and next(iter(raw_statuses)) not in _COLLECTION_STATUS_ALIASES
    ):
        reasons = [*reasons, "invalid_collection_status_abstention"]
    if effect is None:
        # An unknown lifecycle effect must never activate or close a chain.
        status, effect = "uncertain", "no_change"
        reasons = [*reasons, "invalid_event_effect_abstention"]
    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    if not 0.0 <= confidence <= 1.0:
        confidence = 0.0
        reasons = [*reasons, "invalid_confidence_abstention"]
    return {
        "collection_status": status,
        "event_effect": effect,
        "confidence": confidence,
        "reason_codes": reasons[:20],
        "evidence_message_ordinals": normalized_ordinals[:20],
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
