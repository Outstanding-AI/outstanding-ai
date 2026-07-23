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
PROMPT_TEMPLATE_VERSION = "v2"

_SYSTEM_PROMPT = """You prepare concise accounts-receivable notes for an internal weekly
overdue report used for approval, fact-checking, and follow-up planning.

The input is one debtor account. It contains current invoice facts plus a
chronological list of authored messages and operator notes. Email events use
the authored/unique body where the provider supplied it; older events may use
a bounded body fallback. The ordered events collectively represent the
retained account communication trail supplied for this report.

Return a JSON object only:
{
  "account_update": {
    "earlier_context": "material context before reporting_window_start",
    "period_activity": "what happened during the reporting window",
    "current_position": "the current verified account position and blockers",
    "next_action": "one concrete internal next action",
    "evidence_ids": ["only supplied evidence_id values"]
  }
}

Rules:
- Return one debtor-account update covering the supplied invoice portfolio.
- Name an invoice only when the distinction is operationally material; do not
  enumerate every invoice merely because it was supplied.
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
- Treat commitment_status, commitment_date, and commitment_amount as current
  invoice controls. State a supplied commitment explicitly in Current, using
  the word "commitment". A pending/active commitment blocks ordinary chasing;
  a broken commitment requires follow-up; a fulfilled/kept commitment is
  historical context and must not be described as still pending.
- amount_due is already the current Sage balance after allocated credits.
  Never subtract allocated_credit_amount again. Mention proven allocated
  credit references in Current when supplied.
- account_credit_positions are debtor-and-currency context, not invoice
  allocations. Never claim that unapplied credit reduced an invoice. When
  credit_review_required is true, make the next action an internal
  same-currency credit review.
- "Next action" is operational and short. Do not recommend contacting a debtor
  when a supplied current control blocks chasing.
- If remittance_state starts with "cleared_", the remittance check is already
  complete and no remittance action may be recommended. State the evidenced
  collection follow-up or another current review action instead.
- Use plain business language. Never expose input field names or machine status
  codes such as remittance_state, cleared_not_found, or requires_credit_review.
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


def _sanitize_business_text(value: str) -> str:
    """Translate known machine tokens and remove internal evidence handles."""

    text = str(value or "")
    text = re.sub(
        r"^\s*(Earlier|This week|Current|Next)(?:\s+position|\s+action)?\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    replacements = {
        "cleared_not_found": "payment evidence was not found in Sage",
        "cleared_rejected": "payment evidence was not verified",
        "cleared_failed": "payment evidence was not verified",
        "cleared_invalid": "payment evidence was not verified",
        "cleared_cancelled": "payment evidence was not verified",
        "cleared_declined": "payment evidence was not verified",
        "awaiting_verification": "payment evidence awaits verification",
        "remittance_state": "remittance status",
        "requires_credit_review": "credit review",
        "commitment_pending": "pending commitment",
        "promised": "commitment recorded",
        "amount_due": "amount due",
        "due_date": "due date",
        "days_overdue": "days overdue",
        "collection_status": "collection status",
        "allocated_credit_positions": "allocated credit positions",
        "allocated_credit_references": "allocated credit references",
        "remittance_check": "remittance check",
        "reporting_window": "reporting window",
    }
    for token, replacement in replacements.items():
        text = re.sub(rf"\b{re.escape(token)}\b", replacement, text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?i)(?:^|[;,]\s*)remittance(?:\s+(?:status|check))?\s*:?\s*(?:none|not recorded)\b",
        "",
        text,
    )
    text = re.sub(
        r"(?i)(?:^|[;,]\s*)allocated credit(?:s|\s+(?:references|positions))?\s*:?\s*(?:none|\[\])\b",
        "",
        text,
    )
    text = re.sub(r"\s*\((?:E\d{3})(?:\s*/\s*E\d{3})*\)", "", text)
    text = re.sub(r"\bE\d{3}\b", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\(\s*[–-]\s*\)", "", text)
    text = re.sub(r"(?:,\s*){2,}", ", ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = text.strip(" ,;")
    if text.endswith("..."):
        sentence_ends = list(re.finditer(r"\.(?=\s+[A-Z0-9])", text[:-3]))
        if sentence_ends and sentence_ends[-1].start() >= 40:
            text = text[: sentence_ends[-1].start() + 1]
        else:
            last_clause = text.rfind(";", 0, -3)
            if last_clause >= 40:
                text = f"{text[:last_clause].rstrip()}."
    if len(text) <= 240:
        return text
    clipped = text[:237].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}..."


def _append_business_fact(value: str, fact: str) -> str:
    fact = _sanitize_business_text(fact)
    available = 240 - len(fact) - 1
    if available <= 0:
        return fact[:240]
    prefix = _sanitize_business_text(value)
    if len(prefix) > available:
        prefix = prefix[:available].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{prefix} {fact}".strip()


def _validate_model_output(
    *,
    content: str,
    request: WeeklyOverdueReportSummaryRequest,
    evidence_id_map: dict[str, str],
) -> WeeklyOverdueReportSummaryLLMResponse:
    raw = _parse_object(content)
    account_update = raw.get("account_update")
    if isinstance(account_update, dict):
        for field in ("earlier_context", "period_activity", "current_position", "next_action"):
            if field in account_update:
                account_update[field] = _sanitize_business_text(account_update[field])
    parsed = WeeklyOverdueReportSummaryLLMResponse(**raw)
    supplied_evidence = set(evidence_id_map)
    parsed.account_update.evidence_ids = [
        evidence_id
        for evidence_id in parsed.account_update.evidence_ids
        if evidence_id in supplied_evidence
    ]
    allowed_dates = {
        request.reporting_window_start.isoformat(),
        request.reporting_window_end.isoformat(),
        request.generated_at.date().isoformat(),
        *(event.occurred_at.date().isoformat() for event in request.evidence_events),
        *(invoice.due_date.isoformat() for invoice in request.invoices if invoice.due_date),
        *(
            invoice.commitment_date.isoformat()
            for invoice in request.invoices
            if invoice.commitment_date
        ),
    }
    item = parsed.account_update
    active_remittance_exists = any(
        str(invoice.remittance_state or "") in {"awaiting_verification", "verified"}
        for invoice in request.invoices
    )
    if not active_remittance_exists and re.search(
        r"\bremittance\b",
        item.next_action,
        flags=re.IGNORECASE,
    ):
        item.next_action = "Review current collection controls before further follow-up."
    control_text = f"{item.current_position} {item.next_action}"
    if any(invoice.commitment_status for invoice in request.invoices) and not re.search(
        r"\bcommitment\b",
        control_text,
        flags=re.IGNORECASE,
    ):
        commitment_states = sorted(
            {
                str(invoice.commitment_status).replace("_", " ")
                for invoice in request.invoices
                if invoice.commitment_status
            }
        )
        item.current_position = _append_business_fact(
            item.current_position,
            f"Commitment status recorded: {', '.join(commitment_states)}.",
        )
    if (
        any(bool(invoice.allocated_credit_amount) for invoice in request.invoices)
        or bool(request.account_credit_positions)
    ) and not re.search(r"\bcredit\b", control_text, flags=re.IGNORECASE):
        credit_facts = [
            f"{position.currency} {position.unapplied_credit_amount:,.2f}"
            for position in request.account_credit_positions
        ]
        fact = (
            f"Unapplied credit requires internal review: {', '.join(credit_facts)}."
            if credit_facts
            else "Applied credit is already reflected in the current Sage balance."
        )
        item.current_position = _append_business_fact(item.current_position, fact)
    for text in (
        item.earlier_context,
        item.period_activity,
        item.current_position,
        item.next_action,
    ):
        if re.search(r"\b(remittance_state|cleared_[a-z_]+|requires_credit_review)\b", text):
            raise ValueError("weekly_report_summary_exposes_machine_status")
        if re.search(r"(?<!\d{4}-)\b\d{2}-\d{2}\b", text):
            raise ValueError("weekly_report_summary_non_iso_date")
        mentioned_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
        if not mentioned_dates.issubset(allowed_dates):
            raise ValueError("weekly_report_summary_unknown_date")
    return parsed


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
        evidence_id_map: dict[str, str] = {}
        for index, event in enumerate(prompt_input.get("evidence_events", []), start=1):
            prompt_evidence_id = f"E{index:03d}"
            evidence_id_map[prompt_evidence_id] = str(event["evidence_id"])
            event["evidence_id"] = prompt_evidence_id
        user_prompt = _USER_PROMPT.format(
            payload=json.dumps(prompt_input, ensure_ascii=True, sort_keys=True, default=str)
        )
        response = None
        parsed = None
        validation_error = None
        for attempt in range(2):
            correction = (
                ""
                if validation_error is None
                else (
                    "\nThe previous response failed strict validation with code "
                    f"{validation_error}. Correct that defect and return the full JSON object again."
                )
            )
            response = await self._client.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt + correction,
                temperature=settings.classification_temperature,
                json_mode=True,
                caller="weekly_overdue_report_summary",
            )
            try:
                parsed = _validate_model_output(
                    content=response.content,
                    request=request,
                    evidence_id_map=evidence_id_map,
                )
                break
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                validation_error = str(exc)
                logger.warning(
                    "Weekly overdue-report summary failed strict validation: %s",
                    validation_error,
                    extra={
                        "error_type": type(exc).__name__,
                        "attempt": attempt + 1,
                        "error_code": validation_error,
                    },
                )
        if parsed is None or response is None:
            raise LLMResponseInvalidError(
                message="LLM returned invalid weekly overdue-report summary",
                details={"telemetry": _telemetry(response)},
            )

        usage = response.usage if isinstance(response.usage, dict) else {}
        account_update = parsed.account_update.model_dump()
        account_update["evidence_ids"] = [
            evidence_id_map[evidence_id] for evidence_id in account_update["evidence_ids"]
        ]
        return WeeklyOverdueReportSummaryResponse(
            account_update=account_update,
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
