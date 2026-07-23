"""Evidence-grounded weekly overdue-report narrative summarisation."""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from src.api.errors import LLMResponseInvalidError
from src.api.models.requests.weekly_report import WeeklyOverdueReportSummaryRequest
from src.api.models.responses import WeeklyOverdueReportSummaryResponse
from src.config.settings import settings
from src.engine.audit import build_ai_audit
from src.llm.factory import LLMProviderWithFallback
from src.llm.schemas import WeeklyOverdueReportSummaryLLMResponse

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_ID = "weekly_overdue_report_summary"
PROMPT_TEMPLATE_VERSION = "v1"

_SYSTEM_PROMPT = """You prepare concise accounts-receivable notes for an internal weekly
overdue report used for approval, fact-checking, and follow-up planning.

The input is one debtor account. It contains current invoice facts plus a
chronological list of authored messages and operator notes. Email events use
the authored/unique body where the provider supplied it; older events may use
a bounded body fallback. The ordered events collectively represent the
retained account communication trail supplied for this report.

Return a JSON object only:
{
  "invoice_updates": [
    {
      "obligation_id": "an obligation_id supplied in invoices",
      "earlier_context": "material context before reporting_window_start",
      "period_activity": "what happened during the reporting window",
      "current_position": "the current verified position and blocker",
      "next_action": "one concrete internal next action",
      "evidence_ids": ["only supplied evidence_id values"]
    }
  ]
}

Rules:
- Return exactly one update for every supplied invoice and no other invoice.
- Use only supplied facts and evidence. Never infer payment, dispute,
  remittance, ownership, dates, amounts, references, or next steps.
- Distinguish debtor-authored statements, operator notes, and Outstanding AI
  outbound messages. direction=inbound means debtor-authored;
  direction=outbound means the collector/Outstanding AI authored it;
  direction=internal means an operator note. Never reverse those actors.
- Consider account-level events where relevant, but do not copy an
  invoice-specific event onto another invoice.
- Ignore unrelated operational correspondence. Include only evidence material
  to payment status, invoice blockers, commitments, remittances, disputes,
  collection contact, approval, fact-checking, or the next follow-up.
- "Earlier context" covers material events before reporting_window_start.
- "Period activity" covers reporting_window_start through
  reporting_window_end. If nothing material occurred, say "No material update."
- "Current position" uses the current invoice facts. A terminal/failed
  remittance verification is not an active remittance.
- "Next action" is operational and short. Do not recommend contacting a debtor
  when a supplied current control blocks chasing.
- If remittance_state starts with "cleared_", the remittance check is already
  complete and must not be recommended again. State the evidenced collection
  follow-up or review action instead.
- If you state a date, reproduce the exact supplied ISO date (YYYY-MM-DD).
  Never abbreviate, reformat, or calculate a date.
- Keep each field under 240 characters. Do not include email addresses,
  signatures, disclaimers, greetings, or long quotations.
- evidence_ids must contain the minimal supporting supplied evidence IDs.
- If evidence_truncated is true, do not claim the retained trail is exhaustive.
"""

_USER_PROMPT = "Weekly overdue-report evidence:\n{payload}"


def _parse_object(content: str) -> dict:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("weekly_report_summary_response_must_be_object")
    return parsed


def _telemetry(response) -> dict[str, object]:
    usage = response.usage if isinstance(getattr(response, "usage", None), dict) else {}
    return {
        "provider": str(getattr(response, "provider", "unknown") or "unknown"),
        "model": str(getattr(response, "model", "unknown") or "unknown"),
        "is_fallback": bool(getattr(response, "is_fallback", False)),
        "tokens_used": int(usage.get("total_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


class WeeklyOverdueReportSummarizer:
    def __init__(self) -> None:
        self._client = LLMProviderWithFallback(
            primary_provider="vertex",
            fallback_provider="openai",
        )

    async def summarize(
        self,
        request: WeeklyOverdueReportSummaryRequest,
    ) -> WeeklyOverdueReportSummaryResponse:
        prompt_input = request.model_dump(mode="json", exclude_none=True)
        user_prompt = _USER_PROMPT.format(
            payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        )
        response = await self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.classification_temperature,
            json_mode=True,
            caller="weekly_overdue_report_summary",
        )
        try:
            parsed = WeeklyOverdueReportSummaryLLMResponse(**_parse_object(response.content))
            expected = {invoice.obligation_id for invoice in request.invoices}
            actual = {item.obligation_id for item in parsed.invoice_updates}
            if actual != expected or len(parsed.invoice_updates) != len(expected):
                raise ValueError("weekly_report_summary_invoice_scope_mismatch")
            supplied_evidence = {event.evidence_id for event in request.evidence_events}
            if any(
                evidence_id not in supplied_evidence
                for item in parsed.invoice_updates
                for evidence_id in item.evidence_ids
            ):
                raise ValueError("weekly_report_summary_unknown_evidence_id")
            invoice_by_id = {invoice.obligation_id: invoice for invoice in request.invoices}
            for item in parsed.invoice_updates:
                invoice = invoice_by_id[item.obligation_id]
                if str(invoice.remittance_state or "").startswith("cleared_") and re.search(
                    r"\b(verify|confirm|check)\b.{0,30}\bremittance\b"
                    r"|\bremittance\b.{0,30}\b(verify|confirm|check)\b",
                    item.next_action,
                    flags=re.IGNORECASE,
                ):
                    raise ValueError("weekly_report_summary_reopens_cleared_remittance")
                for text in (
                    item.earlier_context,
                    item.period_activity,
                    item.current_position,
                    item.next_action,
                ):
                    if re.search(r"(?<!\d{4}-)\b\d{2}-\d{2}\b", text):
                        raise ValueError("weekly_report_summary_non_iso_date")
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Weekly overdue-report summary failed strict validation",
                extra={"error_type": type(exc).__name__},
            )
            raise LLMResponseInvalidError(
                message="LLM returned invalid weekly overdue-report summary",
                details={"telemetry": _telemetry(response)},
            ) from exc

        usage = response.usage if isinstance(response.usage, dict) else {}
        return WeeklyOverdueReportSummaryResponse(
            invoice_updates=[item.model_dump() for item in parsed.invoice_updates],
            tokens_used=int(usage.get("total_tokens") or 0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
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
                token_count=int(usage.get("total_tokens") or 0),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                inference_profile="classification",
            ),
        )


weekly_overdue_report_summarizer = WeeklyOverdueReportSummarizer()
