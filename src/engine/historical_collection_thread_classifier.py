"""Historical collection-thread protocol, adjudication, and relevance classifier."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import HistoricalCollectionThreadRequest
from src.api.models.responses import HistoricalCollectionThreadResponse
from src.config.settings import settings
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import ChainSelectionTieBreakLLMResponse, HistoricalCollectionThreadLLMResponse

from .audit import build_ai_audit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "historical_collection_thread"
PROMPT_TEMPLATE_VERSION = "v1"
BIDIRECTIONAL_PROMPT_TEMPLATE_VERSION = "v2"


historical_llm_client = LLMProviderWithFallback()

SYSTEM_PROMPT = """You classify historical accounts-receivable collection thread evidence.

You must separate deterministic facts from interpretation:
- Deterministic facts can show chronology, sender/recipient/contact changes, invoice mentions, rolling invoice exposure, and current Sage validation.
- You decide protocol interpretation: whether an outbound touch is a same-contact escalation, cross-contact escalation, debtor-reply response, manual follow-up, promise/remittance acknowledgement, non-collection, or unknown.
- Cross-contact changes can be facts, but they are not automatically escalation unless the conversation context supports escalation intent.
- An outbound response after a debtor reply is not escalation unless the chain context supports escalation.
- Current Sage/Silver validation controls what can be chased now. Do not make paid, closed, or currently non-chaseable invoices active.
- If multiple threads plausibly compete, recommend needs_review rather than forcing active.

For thread_collection_relevance mode, decide only whether the chronological thread is
collection-related. A collection-related thread contains an authored collection request,
overdue/payment follow-up, or a debtor response to that collection purpose. Invoice or
statement mentions alone are insufficient. Ignore quoted or forwarded text as authored
intent. Auto-replies, bounces, internal-only traffic, supplier/unrelated traffic, generic
acknowledgements, and insufficient or mixed evidence must be non_collection or uncertain.
When bidirectional_shadow is true, treat inbound messages from the debtor as role
``debtor_inbound``, manual internal outbound messages as ``internal_manual_outbound``,
messages correlated to an Outstanding AI draft as ``system_generated_outbound``, and
unresolvable messages as ``unknown``. Use both directions chronologically; an outbound
message is authored collection activity only when its role and content support that
purpose. Never infer authorship from quoted text.
Do not classify promises, disputes, remittances, insolvency, commitments, recipients, drafts,
chase policy, or active-thread selection in this mode. Current Sage status is context only and
does not prove historical collection purpose.

Return only JSON matching the schema."""


USER_PROMPT = """Mode: {mode}

Evidence JSON:
{payload}

Decide using the rules in the system prompt. For message_protocol mode, classify only the current message using prior chronological context. For debtor_thread_adjudication mode, recommend one active thread only when current open-overdue exposure and the evidence support it; otherwise use needs_review. For thread_collection_relevance mode, return relevance_label, confidence, signal_codes, evidence_message_ordinals, reason, and abstention_reason only."""

CHAIN_SELECTION_USER_PROMPT = """Mode: chain_selection_tiebreak

Candidate route facts (already deterministically eligible):
{payload}

Select only one supplied candidate key, choose continue_existing_chain, or abstain_manual_review.
Do not invent a candidate, invoice scope, recipient, policy, draft, or provider identifier.
Use evidence ordinals only. Mixed or insufficient evidence must abstain."""


class HistoricalCollectionThreadClassifier:
    async def classify(
        self, request: HistoricalCollectionThreadRequest
    ) -> HistoricalCollectionThreadResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        if request.mode == "chain_selection_tiebreak":
            user_prompt = CHAIN_SELECTION_USER_PROMPT.format(
                payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
            )
            response_schema = ChainSelectionTieBreakLLMResponse
        else:
            user_prompt = USER_PROMPT.format(
                mode=request.mode,
                payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str),
            )
            response_schema = HistoricalCollectionThreadLLMResponse
        prompt_version = (
            BIDIRECTIONAL_PROMPT_TEMPLATE_VERSION
            if request.mode == "thread_collection_relevance"
            and bool((request.guardrails or {}).get("bidirectional_shadow"))
            else PROMPT_TEMPLATE_VERSION
        )
        response = await historical_llm_client.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            response_schema=response_schema,
            caller="historical_collection_thread_tiebreak"
            if request.mode == "chain_selection_tiebreak"
            else "historical_collection_thread",
        )
        raw = json.loads(response.content)
        try:
            parsed = response_schema(**raw)
        except ValidationError as exc:
            logger.error("Historical collection-thread LLM response validation failed: %s", exc)
            raise LLMResponseInvalidError(
                message="LLM returned invalid historical collection-thread response",
                details={"validation_errors": exc.errors()},
            ) from exc
        if request.mode == "chain_selection_tiebreak":
            if parsed.action != "abstain_manual_review" and not parsed.selected_candidate_key:
                raise LLMResponseInvalidError(
                    message="LLM selected no candidate key for chain tie-break",
                    details={"required_field": "selected_candidate_key"},
                )
            return HistoricalCollectionThreadResponse(
                confidence=parsed.confidence,
                reason=parsed.reason,
                provider=response.provider,
                model=response.model,
                is_fallback=(response.provider != historical_llm_client.primary_provider_name),
                tokens_used=response.usage.get("total_tokens", 0),
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                selected_candidate_key=parsed.selected_candidate_key,
                selection_action=parsed.action,
                evidence_message_ordinals=parsed.evidence_message_ordinals,
                abstention_reason=parsed.abstention_reason,
                signal_codes=parsed.reason_codes,
                ai_audit=build_ai_audit(
                    response=response,
                    prompt_template_id="chain_selection_tiebreak",
                    prompt_template_version=prompt_version,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    prompt_input=prompt_input,
                    token_count=response.usage.get("total_tokens", 0),
                    prompt_tokens=response.usage.get("prompt_tokens", 0),
                    completion_tokens=response.usage.get("completion_tokens", 0),
                    inference_profile="classification",
                ),
            )
        if request.mode == "thread_collection_relevance" and not parsed.relevance_label:
            raise LLMResponseInvalidError(
                message="LLM returned no thread relevance label",
                details={"required_field": "relevance_label"},
            )
        if request.mode == "thread_collection_relevance" and any(
            (
                parsed.classification,
                parsed.protocol_touch_type,
                parsed.thread_actions,
                parsed.intent_details,
                parsed.commitment_acknowledgement_type,
            )
        ):
            raise LLMResponseInvalidError(
                message="LLM returned message/adjudication fields for thread relevance mode",
                details={"mode": request.mode},
            )

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
            thread_actions=parsed.thread_actions_dict(),
            guardrail_warnings=parsed.guardrail_warnings,
            secondary_intents=parsed.secondary_intents,
            intent_details=parsed.intent_details_payload(),
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=response.provider,
            model=response.model,
            is_fallback=(response.provider != historical_llm_client.primary_provider_name),
            ai_audit=build_ai_audit(
                response=response,
                prompt_template_id=(
                    "historical_thread_relevance"
                    if request.mode == "thread_collection_relevance"
                    else PROMPT_TEMPLATE_ID
                ),
                prompt_template_version=prompt_version,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_input=prompt_input,
                token_count=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                inference_profile="classification",
            ),
            relevance_label=parsed.relevance_label,
            signal_codes=parsed.signal_codes,
            evidence_message_ordinals=parsed.evidence_message_ordinals,
            abstention_reason=parsed.abstention_reason,
        )


historical_collection_thread_classifier = HistoricalCollectionThreadClassifier()
