"""AI-backed extraction of final sent-email invoice scope."""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests import AnalyzeSentDraftScopeRequest
from src.api.models.responses import (
    AnalyzeSentDraftScopeResponse,
    SentDraftInvoiceScopeDecision,
)
from src.llm import llm_client
from src.prompts._sanitize import sanitize_delimiter_tags
from src.prompts.sent_scope import (
    SENT_DRAFT_SCOPE_SYSTEM,
    SENT_DRAFT_SCOPE_TEMPLATE_ID,
    SENT_DRAFT_SCOPE_TEMPLATE_VERSION,
    SENT_DRAFT_SCOPE_USER,
)

from .audit import build_ai_audit

logger = logging.getLogger(__name__)


class _LLMScopeDecision(BaseModel):
    invoice_number: str
    obligation_id: str | None = None
    status: Literal[
        "retained_generated_invoice",
        "operator_added_invoice",
        "removed_generated_invoice",
        "not_present",
        "ambiguous",
    ]
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence: str | None = None


class _LLMScopeResponse(BaseModel):
    decisions: list[_LLMScopeDecision] = Field(default_factory=list)
    reasoning: str | None = None


def _norm_ref(value: str | None) -> str:
    return str(value or "").strip().upper()


def _json(values) -> str:
    return json.dumps(values, default=str, ensure_ascii=True, sort_keys=True)


class SentDraftScopeAnalyzer:
    """Analyze final sent email invoice scope with LLM + deterministic validation."""

    async def analyze(self, request: AnalyzeSentDraftScopeRequest) -> AnalyzeSentDraftScopeResponse:
        candidate_map = {
            _norm_ref(candidate.invoice_number): candidate
            for candidate in request.invoice_candidates
            if _norm_ref(candidate.invoice_number)
        }
        generated_refs = {
            _norm_ref(value) for value in request.generated.invoice_refs if _norm_ref(value)
        }

        if not candidate_map:
            return AnalyzeSentDraftScopeResponse(
                scope_extraction_status="review_required",
                review_recommended=True,
                review_reason_codes=["no_candidate_invoices"],
            )

        prompt_input = {
            "tenant_id": request.tenant_id,
            "party_id": request.party_id,
            "draft_id": request.draft_id,
            "touch_id": request.touch_id,
            "provider_message_id": request.provider_message_id,
            "sent_at": request.sent_at.isoformat() if request.sent_at else None,
            "generated_invoice_refs": sorted(generated_refs),
            "candidate_invoice_refs": sorted(candidate_map),
        }
        user_prompt = SENT_DRAFT_SCOPE_USER.format(
            generated_subject=sanitize_delimiter_tags(request.generated.subject or ""),
            generated_body=sanitize_delimiter_tags(
                request.generated.body_plain or request.generated.body_html or ""
            ),
            generated_invoice_refs=_json(sorted(generated_refs)),
            sent_subject=sanitize_delimiter_tags(request.sent.subject or ""),
            sent_body=sanitize_delimiter_tags(
                request.sent.body_plain or request.sent.body_html or ""
            ),
            candidate_invoices_json=_json(
                [
                    candidate.model_dump(mode="json", exclude_none=True)
                    for candidate in request.invoice_candidates
                ]
            ),
        )

        response = await llm_client.complete(
            system_prompt=SENT_DRAFT_SCOPE_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.0,
            response_schema=_LLMScopeResponse,
            caller="sent_scope_analysis",
        )
        tokens_used = response.usage.get("total_tokens", 0)
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)
        try:
            raw = json.loads(response.content)
            llm_result = _LLMScopeResponse(**raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error("Sent-scope LLM response validation failed: %s", exc)
            raise LLMResponseInvalidError(
                message="LLM returned invalid sent-scope analysis response",
                details={"error": str(exc)},
            )

        decisions = self._validated_decisions(llm_result.decisions, candidate_map, generated_refs)
        sent_refs = sorted(
            decision.invoice_number
            for decision in decisions
            if decision.status in {"retained_generated_invoice", "operator_added_invoice"}
        )
        retained = sorted(
            decision.invoice_number
            for decision in decisions
            if decision.status == "retained_generated_invoice"
        )
        operator_added = sorted(
            decision.invoice_number
            for decision in decisions
            if decision.status == "operator_added_invoice"
        )
        removed = sorted(
            decision.invoice_number
            for decision in decisions
            if decision.status == "removed_generated_invoice"
        )
        ambiguous = sorted(
            decision.invoice_number for decision in decisions if decision.status == "ambiguous"
        )
        review_codes = []
        if ambiguous:
            review_codes.append("ambiguous_sent_invoice_scope")
        if any(
            decision.confidence < 0.75 for decision in decisions if decision.status != "not_present"
        ):
            review_codes.append("low_confidence_sent_invoice_scope")
        if len(decisions) < len(candidate_map):
            review_codes.append("missing_candidate_decisions_repaired")

        confidence_values = [
            decision.confidence for decision in decisions if decision.status != "not_present"
        ]
        confidence = min(confidence_values) if confidence_values else 1.0
        scope_changed = bool(operator_added or removed or ambiguous)
        status = "review_required" if review_codes else "succeeded"
        audit = build_ai_audit(
            response=response,
            prompt_template_id=SENT_DRAFT_SCOPE_TEMPLATE_ID,
            prompt_template_version=SENT_DRAFT_SCOPE_TEMPLATE_VERSION,
            system_prompt=SENT_DRAFT_SCOPE_SYSTEM,
            user_prompt=user_prompt,
            prompt_input=prompt_input,
            token_count=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            inference_profile="sent_scope_analysis",
        )
        return AnalyzeSentDraftScopeResponse(
            invoice_refs_sent=sent_refs,
            invoice_refs_retained=retained,
            invoice_refs_operator_added=operator_added,
            invoice_refs_removed=removed,
            invoice_refs_ambiguous=ambiguous,
            invoice_scope_changed=scope_changed,
            scope_extraction_confidence=confidence,
            scope_extraction_status=status,
            review_recommended=bool(review_codes),
            review_reason_codes=review_codes,
            decisions=decisions,
            reasoning=llm_result.reasoning,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            provider=response.provider,
            model=response.model,
            is_fallback=response.is_fallback,
            ai_audit=audit,
        )

    def _validated_decisions(
        self,
        llm_decisions: list[_LLMScopeDecision],
        candidate_map: dict[str, object],
        generated_refs: set[str],
    ) -> list[SentDraftInvoiceScopeDecision]:
        by_ref: dict[str, _LLMScopeDecision] = {}
        for decision in llm_decisions:
            ref = _norm_ref(decision.invoice_number)
            if ref in candidate_map and ref not in by_ref:
                by_ref[ref] = decision

        validated: list[SentDraftInvoiceScopeDecision] = []
        for ref, candidate in candidate_map.items():
            raw = by_ref.get(ref)
            status = raw.status if raw else "not_present"
            if status == "retained_generated_invoice" and ref not in generated_refs:
                status = "operator_added_invoice"
            elif status == "operator_added_invoice" and ref in generated_refs:
                status = "retained_generated_invoice"
            elif status == "removed_generated_invoice" and ref not in generated_refs:
                status = "not_present"

            validated.append(
                SentDraftInvoiceScopeDecision(
                    invoice_number=ref,
                    obligation_id=getattr(candidate, "obligation_id", None),
                    status=status,
                    confidence=float(raw.confidence if raw else 1.0),
                    evidence=raw.evidence if raw else None,
                )
            )
        return validated


sent_scope_analyzer = SentDraftScopeAnalyzer()
