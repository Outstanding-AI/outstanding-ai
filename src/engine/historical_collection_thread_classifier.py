"""Historical collection-thread protocol and adjudication classifier."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import HistoricalCollectionThreadRequest
from src.api.models.responses import HistoricalCollectionThreadResponse
from src.config.settings import settings
from src.llm.factory import llm_client
from src.llm.schemas import HistoricalCollectionThreadLLMResponse

from .audit import build_ai_audit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "historical_collection_thread"
PROMPT_TEMPLATE_VERSION = "v1"

SYSTEM_PROMPT = """You classify historical accounts-receivable collection thread evidence.

You must separate deterministic facts from interpretation:
- Deterministic facts can show chronology, sender/recipient/contact changes, invoice mentions, rolling invoice exposure, and current Sage validation.
- You decide protocol interpretation: whether an outbound touch is a same-contact escalation, cross-contact escalation, debtor-reply response, manual follow-up, promise/remittance acknowledgement, non-collection, or unknown.
- Cross-contact changes can be facts, but they are not automatically escalation unless the conversation context supports escalation intent.
- An outbound response after a debtor reply is not escalation unless the chain context supports escalation.
- Current Sage/Silver validation controls what can be chased now. Do not make paid, closed, or currently non-chaseable invoices active.
- If multiple threads plausibly compete, recommend needs_review rather than forcing active.

Return only JSON matching the schema."""


USER_PROMPT = """Mode: {mode}

Evidence JSON:
{payload}

Decide using the rules in the system prompt. For message_protocol mode, classify only the current message using prior chronological context. For debtor_thread_adjudication mode, recommend one active thread only when current open-overdue exposure and the evidence support it; otherwise use needs_review."""


class HistoricalCollectionThreadClassifier:
    async def classify(
        self, request: HistoricalCollectionThreadRequest
    ) -> HistoricalCollectionThreadResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = USER_PROMPT.format(
            mode=request.mode,
            payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
        )
        response = await llm_client.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            response_schema=HistoricalCollectionThreadLLMResponse,
            caller="historical_collection_thread",
        )
        raw = json.loads(response.content)
        try:
            parsed = HistoricalCollectionThreadLLMResponse(**raw)
        except ValidationError as exc:
            logger.error("Historical collection-thread LLM response validation failed: %s", exc)
            raise LLMResponseInvalidError(
                message="LLM returned invalid historical collection-thread response",
                details={"validation_errors": exc.errors()},
            ) from exc

        tokens_used = response.usage.get("total_tokens", 0)
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)
        return HistoricalCollectionThreadResponse(
            classification=parsed.classification or parsed.protocol_touch_type,
            protocol_touch_type=parsed.protocol_touch_type,
            is_escalation=parsed.is_escalation,
            escalation_kind=parsed.escalation_kind,
            debtor_reply_response=parsed.debtor_reply_response,
            commitment_acknowledgement_type=parsed.commitment_acknowledgement_type,
            confidence=parsed.confidence,
            reason=parsed.reason,
            evidence_message_ids=parsed.evidence_message_ids,
            recommended_active_thread_id=parsed.recommended_active_thread_id,
            thread_actions=parsed.thread_actions,
            guardrail_warnings=parsed.guardrail_warnings,
            secondary_intents=parsed.secondary_intents,
            intent_details=parsed.intent_details,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != llm_client.primary_provider_name),
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=PROMPT_TEMPLATE_ID,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_input=prompt_input,
                token_count=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                inference_profile="classification",
            ),
        )


historical_collection_thread_classifier = HistoricalCollectionThreadClassifier()
