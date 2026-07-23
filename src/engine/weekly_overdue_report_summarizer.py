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
PROMPT_TEMPLATE_VERSION = "v6"

_SYSTEM_PROMPT = """You prepare concise accounts-receivable notes for an internal weekly
overdue report used for approval, fact-checking, and follow-up planning.

The input contains exactly one target invoice, its PO/sales-order/allocated-
credit lineage, and only the retained authored events mapped to or explicitly
naming that invoice. A supplied event can mention several invoices. Extract
only the clause about the target invoice and ignore every other invoice.

Return a JSON object only:
{
  "material_updates": [
    {
      "evidence_id": "one supplied evidence_id",
      "summary": "the business meaning of that event for this invoice"
    }
  ]
}

Rules:
- Return one update for the single supplied target invoice.
- The application renders current accounting truth and the prior-week frozen
  accounting position deterministically. Summarize only authored evidence; do
  not restate prior_week_* or current invoice fields.
- "Earlier" is strictly the seven calendar days immediately before
  reporting_window_start. "This week" is reporting_window_start through
  reporting_window_end. Evidence outside those two windows must not be used.
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
- Return at most eight material updates, ordered oldest to newest.
- Each update must summarize exactly one supplied evidence event and carry that
  event's evidence_id. Do not combine several evidence IDs into one update.
- Do not return a routine reminder, statement, or follow-up send unless its
  authored text records a material commitment, remittance, query, credit,
  payment position, or blocker.
- Summaries must describe authored evidence, not restate current invoice facts.
  Attribute debtor statements, operator notes, and collector messages
  accurately. If the debtor says paid while amount_due remains positive,
  report the debtor claim without asserting that payment cleared.
- Treat commitment fields as current invoice controls. Use the word
  "commitment"; do not call it a promise. A pending commitment blocks ordinary
  chasing; a broken commitment requires follow-up; a fulfilled commitment is
  historical and is not still pending.
- amount_due is the current balance after exact allocated credits. Never
  subtract allocated_credit_amount again and never apply debtor-level
  unapplied credit to this invoice.
- If remittance_state starts with "cleared_", the remittance check is already
  complete. Do not describe it as still awaiting review.
- Use plain business language. Never expose input field names or machine status
  codes such as remittance_state, cleared_not_found, or requires_credit_review.
- Never expose internal UUIDs, obligation IDs, party IDs, evidence IDs, or
  phrases such as "tied to obligation". Only business document references
  supplied on the target invoice may appear in narrative text.
- If you state a date, reproduce the exact supplied ISO date (YYYY-MM-DD).
  Never abbreviate, reformat, or calculate a date.
- Keep each summary under 240 characters. Do not include email addresses,
  signatures, disclaimers, greetings, or long quotations.
- Do not recommend an action in the summary; the application determines the
  next action from current controls. Never mention "other invoices", "related
  invoices", or sibling documents.
- Paraphrase authored evidence into business language. Never copy transport
  prefixes such as "Received from debtor - internal forward context", sender
  addresses, reply subjects, or signature text into the result.
- If evidence_truncated is true, do not claim the retained trail is exhaustive.
- If no supplied event has material invoice-specific business meaning, return
  an empty material_updates list.
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
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?i)\s*[;,.]?\s*action\s*:\s*.*$", "", text)
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
    material_updates = raw.get("material_updates")
    if isinstance(material_updates, list):
        for update in material_updates:
            if isinstance(update, dict) and "summary" in update:
                update["summary"] = _sanitize_business_text(update["summary"])
    parsed = WeeklyOverdueReportSummaryLLMResponse(**raw)
    supplied_evidence = set(evidence_id_map)
    event_by_original_id = {event.evidence_id: event for event in request.evidence_events}
    seen: set[str] = set()
    validated_updates = []
    for item in parsed.material_updates:
        evidence_id = str(item.evidence_id)
        if evidence_id not in supplied_evidence:
            raise ValueError("weekly_report_summary_unknown_evidence")
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        original_id = evidence_id_map[evidence_id]
        event = event_by_original_id.get(original_id)
        if event is None:
            raise ValueError("weekly_report_summary_unknown_evidence")
        text = _remove_forbidden_reference_clauses(
            item.summary,
            request.forbidden_references,
        )
        item.summary = text
        if not text:
            continue
        if any(
            _contains_reference(text, forbidden_reference)
            for forbidden_reference in request.forbidden_references
        ):
            continue
        if re.search(
            r"\b(other|related|remaining|sibling|multiple)\s+(?:invoice|document)s?\b",
            text,
            re.I,
        ):
            continue
        if re.search(r"\b(remittance_state|cleared_[a-z_]+|requires_credit_review)\b", text):
            continue
        if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE):
            continue
        if re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"\b(received from debtor\s*-\s*internal forward context|sent to debtor to)\b",
            text,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(r"(?<!\d{4}-)\b\d{2}-\d{2}\b", text):
            continue
        allowed_dates = {
            event.occurred_at.date().isoformat(),
            *re.findall(r"\b\d{4}-\d{2}-\d{2}\b", event.authored_text),
        }
        mentioned_dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
        if not mentioned_dates.issubset(allowed_dates):
            continue
        if text:
            validated_updates.append(item)
    parsed.material_updates = validated_updates
    return parsed


def _remove_forbidden_reference_clauses(text: str, forbidden_references: list[str]) -> str:
    """Keep the target-invoice clause when one event also names sibling documents."""

    cleaned = re.sub(
        r"(?i)^.*?\b(?:multiple|other|related|remaining|sibling)\s+"
        r"(?:invoice|document)s?\s*:\s*",
        "",
        str(text or ""),
    )
    clauses = re.split(r"(?<=[.;])\s+|;\s*", cleaned)
    retained = [
        clause.strip()
        for clause in clauses
        if clause.strip()
        and not any(
            _contains_reference(clause, forbidden_reference)
            for forbidden_reference in forbidden_references
        )
    ]
    return _sanitize_business_text(" ".join(retained))


def _contains_reference(text: str, reference: str) -> bool:
    normalized_text = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())
    normalized_reference = re.sub(r"[^A-Z0-9]", "", str(reference or "").upper())
    if len(normalized_reference) >= 4 and normalized_reference in normalized_text:
        return True
    if normalized_reference.isdigit():
        wanted = normalized_reference.lstrip("0") or "0"
        return any(
            (candidate.lstrip("0") or "0") == wanted
            for candidate in re.findall(r"\d{4,}", str(text or ""))
        )
    return False


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
                response_schema=WeeklyOverdueReportSummaryLLMResponse,
                reasoning_effort="minimal",
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
        material_updates = [
            {
                "evidence_id": evidence_id_map[item.evidence_id],
                "summary": item.summary,
            }
            for item in parsed.material_updates
        ]
        return WeeklyOverdueReportSummaryResponse(
            material_updates=material_updates,
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
        "weekly_report_summary_unknown_evidence": (
            "Use only supplied evidence_id values and emit each at most once."
        ),
        "weekly_report_summary_unknown_date": (
            "Remove every date not written in the cited evidence event."
        ),
        "weekly_report_summary_non_iso_date": (
            "Remove abbreviated dates; use only exact supplied ISO dates when material."
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
        "weekly_report_summary_cross_invoice_language": (
            "Describe only the target invoice and do not refer to other or related invoices."
        ),
    }
    return corrections.get(code, "Correct only the stated validation defect.")


weekly_overdue_report_summarizer = WeeklyOverdueReportSummarizer()
