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
PROMPT_TEMPLATE_VERSION = "v3"

_SYSTEM_PROMPT = """You prepare concise accounts-receivable notes for an internal weekly
overdue report used for approval, fact-checking, and follow-up planning.

The input contains exactly one target invoice, its PO/sales-order/allocated-
credit lineage, and only the retained authored events mapped to or explicitly
naming that invoice. A supplied event can mention several invoices. Extract
only the clause about the target invoice and ignore every other invoice.

Return a JSON object only:
{
  "account_update": {
    "earlier_context": "material target-invoice history before reporting_window_start",
    "period_activity": "target-invoice activity during the reporting window",
    "current_position": "latest authored target-invoice position",
    "next_action": "specific target-invoice action supported by the supplied controls",
    "evidence_ids": ["only supplied evidence_id values"]
  }
}

Rules:
- Return one update for the single supplied target invoice.
- Never mention an invoice, PO, sales order, or credit reference listed in
  forbidden_references. A multi-invoice email is not permission to copy the
  other invoices into this row.
- Preserve the target invoice number and its supplied PO/sales-order/credit
  references exactly when they are materially relevant.
- Use only supplied facts and evidence. Never infer payment, dispute,
  remittance, ownership, dates, amounts, references, or next steps.
- Preserve contradictory authored statements rather than choosing one. For
  example, if one response says both "paid" and "under review" for the target
  invoice, state that the response is contradictory.
- Distinguish debtor-authored statements, operator notes, and Outstanding AI
  outbound messages. direction=inbound means debtor-authored;
  direction=outbound means the collector/Outstanding AI authored it;
  direction=internal means an operator note. Never reverse those actors.
- Ignore unrelated operational correspondence. Include only evidence material
  to this invoice's payment status, PO/order status, delivery or approval
  blocker, commitment, remittance, query, credit, collection contact, or
  follow-up.
- "Earlier context" covers material events before reporting_window_start.
- "Period activity" covers reporting_window_start through
  reporting_window_end. If nothing material occurred, say "No material update."
- Earlier and period fields must summarize authored evidence, not restate the
  current invoice fact. When a retained event exists in that period, include
  the event's supplied ISO date. Do not put amount_due, due_date,
  days_overdue, collection_status, PO, or sales-order facts in these fields
  unless the authored evidence itself states them.
- "Current position" means the latest authored position about this invoice,
  not a restatement of machine fields. Attribute debtor statements and
  operator notes accurately. If the debtor says paid while amount_due remains
  positive, report the debtor claim without asserting that payment cleared.
- Treat commitment fields as current invoice controls. Use the word
  "commitment"; do not call it a promise. A pending commitment blocks ordinary
  chasing; a broken commitment requires follow-up; a fulfilled commitment is
  historical and is not still pending.
- amount_due is the current balance after exact allocated credits. Never
  subtract allocated_credit_amount again and never apply debtor-level
  unapplied credit to this invoice.
- "Next action" must name this invoice and be operationally specific. Do not
  emit generic text such as "review collection controls" or "review evidence".
  Do not recommend debtor contact while a current control blocks chasing.
- If remittance_state starts with "cleared_", the remittance check is already
  complete and no remittance action may be recommended. State the evidenced
  collection follow-up or another current review action instead.
- Use plain business language. Never expose input field names or machine status
  codes such as remittance_state, cleared_not_found, or requires_credit_review.
- Never expose internal UUIDs, obligation IDs, party IDs, evidence IDs, or
  phrases such as "tied to obligation". Only business document references
  supplied on the target invoice may appear in narrative text.
- If you state a date, reproduce the exact supplied ISO date (YYYY-MM-DD).
  Never abbreviate, reformat, or calculate a date.
- Keep each field under 240 characters. Do not include email addresses,
  signatures, disclaimers, greetings, or long quotations.
- Paraphrase authored evidence into business language. Never copy transport
  prefixes such as "Received from debtor - internal forward context", sender
  addresses, reply subjects, or signature text into the result.
- evidence_ids must contain the minimal supporting supplied evidence IDs.
- If evidence_truncated is true, do not claim the retained trail is exhaustive.
- If no earlier or in-period target-invoice event exists, use the exact
  no-update sentence requested for that field. Do not fill it with facts from
  another invoice.
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
        "cleared_not_found": "payment evidence was not found in the accounting system",
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
        *(invoice.invoice_date.isoformat() for invoice in request.invoices if invoice.invoice_date),
        *(invoice.due_date.isoformat() for invoice in request.invoices if invoice.due_date),
        *(
            invoice.sales_order_date.isoformat()
            for invoice in request.invoices
            if invoice.sales_order_date
        ),
        *(
            invoice.commitment_date.isoformat()
            for invoice in request.invoices
            if invoice.commitment_date
        ),
    }
    item = parsed.account_update
    rendered = " ".join(
        (
            item.earlier_context,
            item.period_activity,
            item.current_position,
            item.next_action,
        )
    )
    for forbidden_reference in request.forbidden_references:
        if _contains_reference(rendered, forbidden_reference):
            raise ValueError("weekly_report_summary_cross_invoice_reference")
    if re.search(
        r"\b(review (?:the )?(?:current )?(?:collection )?controls|review (?:current )?evidence)\b",
        item.next_action,
        flags=re.IGNORECASE,
    ):
        raise ValueError("weekly_report_summary_generic_next_action")
    earlier_event_dates = {
        event.occurred_at.date().isoformat()
        for event in request.evidence_events
        if event.occurred_at.date() < request.reporting_window_start
    }
    period_event_dates = {
        event.occurred_at.date().isoformat()
        for event in request.evidence_events
        if request.reporting_window_start
        <= event.occurred_at.date()
        <= request.reporting_window_end
    }
    earlier_dates = earlier_event_dates | {
        value
        for event in request.evidence_events
        if event.occurred_at.date() < request.reporting_window_start
        for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", event.authored_text)
    }
    period_dates = period_event_dates | {
        value
        for event in request.evidence_events
        if request.reporting_window_start
        <= event.occurred_at.date()
        <= request.reporting_window_end
        for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", event.authored_text)
    }
    mentioned_earlier_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", item.earlier_context))
    mentioned_period_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", item.period_activity))
    if earlier_event_dates and not mentioned_earlier_dates.intersection(earlier_event_dates):
        raise ValueError("weekly_report_summary_missing_earlier_event_date")
    if not mentioned_earlier_dates.issubset(earlier_dates):
        raise ValueError("weekly_report_summary_non_evidence_earlier_date")
    if not earlier_dates:
        item.earlier_context = "No material earlier context."
    if period_event_dates and not mentioned_period_dates.intersection(period_event_dates):
        raise ValueError("weekly_report_summary_missing_period_event_date")
    if not mentioned_period_dates.issubset(period_dates):
        raise ValueError("weekly_report_summary_non_evidence_period_date")
    if not period_dates:
        item.period_activity = "No material update."
    for text in (
        item.earlier_context,
        item.period_activity,
        item.current_position,
        item.next_action,
    ):
        if re.search(r"\b(remittance_state|cleared_[a-z_]+|requires_credit_review)\b", text):
            raise ValueError("weekly_report_summary_exposes_machine_status")
        if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE):
            raise ValueError("weekly_report_summary_contains_email_address")
        if re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            text,
            flags=re.IGNORECASE,
        ):
            raise ValueError("weekly_report_summary_contains_internal_identifier")
        if re.search(
            r"\b(received from debtor\s*-\s*internal forward context|sent to debtor to)\b",
            text,
            flags=re.IGNORECASE,
        ):
            raise ValueError("weekly_report_summary_copies_transport_prefix")
        if re.search(r"(?<!\d{4}-)\b\d{2}-\d{2}\b", text):
            raise ValueError("weekly_report_summary_non_iso_date")
        mentioned_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
        if not mentioned_dates.issubset(allowed_dates):
            raise ValueError("weekly_report_summary_unknown_date")
    return parsed


def _contains_reference(text: str, reference: str) -> bool:
    normalized_text = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())
    normalized_reference = re.sub(r"[^A-Z0-9]", "", str(reference or "").upper())
    return len(normalized_reference) >= 4 and normalized_reference in normalized_text


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
        for attempt in range(3):
            correction = (
                ""
                if validation_error is None
                else (
                    "\nThe previous response failed strict validation with code "
                    f"{validation_error}. {_correction_for(validation_error)} "
                    "Return the full JSON object again."
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


def _correction_for(error: str) -> str:
    code = str(error or "")
    corrections = {
        "weekly_report_summary_missing_earlier_event_date": (
            "Summarize the earlier authored event and include its exact supplied ISO date."
        ),
        "weekly_report_summary_non_evidence_earlier_date": (
            "Earlier may contain only dates from earlier authored events, not invoice or due dates."
        ),
        "weekly_report_summary_missing_period_event_date": (
            "Summarize the in-period authored event and include its exact supplied ISO date."
        ),
        "weekly_report_summary_non_evidence_period_date": (
            "Period activity may contain only dates from in-period authored events."
        ),
        "weekly_report_summary_contains_email_address": (
            "Remove all email addresses and paraphrase the underlying business update."
        ),
        "weekly_report_summary_contains_internal_identifier": (
            "Remove internal UUIDs, obligation IDs, party IDs, and evidence handles. "
            "Use only the target invoice's supplied business document references."
        ),
        "weekly_report_summary_copies_transport_prefix": (
            "Remove mail transport prefixes and state only the business meaning."
        ),
        "weekly_report_summary_cross_invoice_reference": (
            "Remove every forbidden invoice, PO, sales-order, and credit reference."
        ),
        "weekly_report_summary_generic_next_action": (
            "Name the target invoice and give the specific supported next action."
        ),
    }
    return corrections.get(code, "Correct only the stated validation defect.")


weekly_overdue_report_summarizer = WeeklyOverdueReportSummarizer()
