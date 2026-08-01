"""Evidence-grounded weekly overdue-report narrative summarisation."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

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
PROMPT_TEMPLATE_VERSION = "v10"

_SYSTEM_PROMPT = """You condense the latest invoice-specific debtor response
into a short factual update for an internal weekly overdue report.

The application separately renders the current Sage balance, overdue status,
and all confirmed chase dates. The supplied inbound evidence is a compact set
of typed workflow facts extracted from one debtor email for one invoice.

Return a JSON object only:
{
  "material_updates": [
    {
      "evidence_id": "one supplied evidence_id",
      "summary": "short debtor update grounded in the supplied facts"
    }
  ]
}

Rules:
- Return exactly one update for the single supplied inbound evidence event.
- Keep the summary to one plain phrase of at most 160 characters.
- Begin with exactly one of: "committed", "promised", "agreed", "advised",
  "confirmed", "reported", "sent", "provided", "supplied", "paid", "raised",
  "queried", "disputed", "requested", "asked", "stated", or "explained".
- State the newest material meaning: a payment commitment, remittance/payment
  claim, payment-timing statement, query/dispute, internal processing blocker,
  or document request.
- Use "committed" or "promised" only for commitment_to_pay facts. For
  payment_evidence use "reported", "sent", "advised", "confirmed", or
  "stated"; never upgrade a payment/remittance claim into a commitment.
- Prefer the most concrete grounded detail, such as a promised payment date,
  claimed payment date or amount, or the substance of a query.
- Translate machine-shaped facts into natural business language. Good forms
  include "committed to pay by 31 Jul 2026", "reported payment of GBP 1,000
  on 25 Jul 2026 and supplied remittance", and "queried the unit price against
  the purchase order".
- If update_kind is "already paid" and no safe amount or date is supplied, say
  "reported payment already made". If it is "remittance advice" without safe
  details, say "supplied remittance". If fact_type is document_request without
  a safe specific comment, say "requested a copy of the current invoice".
- Omit missing details. Never say unknown amount, unspecified date, missing
  detail, inbound evidence, obligation, or similar process language.
- Do not copy classification names, fact labels, fact statuses, JSON keys, or
  phrases such as pending, observed, verification pending, claimed date, fact
  value, or query or dispute into the summary.
- Do not use a colon or semicolon. Do not use the words message, email, chase,
  reminder, follow-up, sequence, level, stage, or touch.
- Do not state or infer the current balance, Sage status, days overdue, due
  date, chase date, final payment status, or next action.
- A fact status such as verification_pending means the debtor claimed payment;
  it does not prove that Sage has received or verified it.
- You may include an exact commitment/payment date or amount found in the
  supplied facts. Do not include invoice, PO, sales-order, bank, credit,
  evidence, mail, party, or internal identifiers.
- Do not mention a reminder number, reminder sequence, level, stage, touch
  index, escalation index, tone label, or phrases such as "first reminder".
- Do not say that the response was received; the application renders its date.
- Never mention an invoice, PO, sales order, or credit reference listed in
  forbidden_references. A multi-invoice message is not permission to copy
  another document into the phrase.
- Use only the supplied facts. Do not recommend a future action.
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
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", _humanize_iso_date, text)
    text = re.sub(r"\binfo\b", "information", text, flags=re.IGNORECASE)
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
    if len(text) <= 160:
        return text
    clipped = text[:157].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}..."


def _humanize_iso_date(match: re.Match[str]) -> str:
    try:
        return datetime.strptime(match.group(0), "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return match.group(0)


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
    rejection_codes: list[str] = []
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
        rejection_code = _update_rejection_code(
            text=text,
            event_authored_text=event.authored_text,
            forbidden_references=request.forbidden_references,
        )
        if rejection_code:
            rejection_codes.append(rejection_code)
            continue
        validated_updates.append(item)
    parsed.material_updates = validated_updates
    if len(parsed.material_updates) != 1:
        raise ValueError(
            rejection_codes[0]
            if rejection_codes
            else "weekly_report_summary_requires_one_debtor_update"
        )
    return parsed


def _update_rejection_code(
    *,
    text: str,
    event_authored_text: str,
    forbidden_references: list[str],
) -> str | None:
    if not text:
        return "weekly_report_summary_cross_invoice_reference"
    if any(_contains_reference(text, reference) for reference in forbidden_references):
        return "weekly_report_summary_cross_invoice_reference"
    if re.search(
        r"\b(other|related|remaining|sibling|multiple)\s+(?:invoice|document)s?\b",
        text,
        re.IGNORECASE,
    ):
        return "weekly_report_summary_cross_invoice_language"
    if re.search(r"\b(remittance_state|cleared_[a-z_]+|requires_credit_review)\b", text):
        return "weekly_report_summary_machine_terms"
    if re.search(
        r"\b(classification|fact\s+(?:type|subtype|status|value)|"
        r"verification\s+pending|claimed\s+date|query\s+or\s+dispute|observed|"
        r"evidence|obligation|unknown\s+amount|unspecified\s+date|missing\s+detail)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_machine_terms"
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE):
        return "weekly_report_summary_contains_email_address"
    if re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_contains_internal_identifier"
    if re.search(
        r"\b(received from debtor\s*-\s*internal forward context|sent to debtor to)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_copies_transport_prefix"
    if ":" in text or ";" in text:
        return "weekly_report_summary_invalid_punctuation"
    if re.search(
        r"\b(message|email|chase|reminders?|follow-up|sequence|level|stage|touch)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_chase_metadata"
    if not re.match(
        r"^(committed|promised|agreed|advised|confirmed|reported|sent|provided|"
        r"supplied|paid|raised|queried|disputed|requested|asked|stated|explained)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_invalid_opening"
    if not re.match(
        rf"^({'|'.join(_allowed_openings_for_event(event_authored_text))})\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "weekly_report_summary_opening_mismatches_fact"
    if not _numeric_values(text).issubset(_numeric_values(event_authored_text)):
        return "weekly_report_summary_ungrounded_number"
    return None


def _allowed_openings_for_event(authored_text: str) -> tuple[str, ...]:
    evidence = str(authored_text or "").lower()
    if "fact_type=commitment_to_pay" in evidence:
        return ("committed", "promised", "agreed", "stated")
    if "fact_type=payment_evidence" in evidence:
        return (
            "reported",
            "sent",
            "provided",
            "supplied",
            "paid",
            "advised",
            "confirmed",
            "stated",
        )
    if "fact_type=payment_timing_claim" in evidence:
        return ("advised", "reported", "stated", "disputed")
    return ("raised", "queried", "disputed", "requested", "asked", "explained", "stated")


def _numeric_values(text: str) -> set[str]:
    values: set[str] = set()
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text or "")):
        normalized = token.replace(",", "")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        values.add(normalized.lstrip("0") or "0")
    return values


def _remove_forbidden_reference_clauses(text: str, forbidden_references: list[str]) -> str:
    """Keep the target-invoice clause when one event also names sibling documents."""

    cleaned = str(text or "")
    for reference in forbidden_references:
        if reference:
            cleaned = re.sub(re.escape(reference), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?i)^.*?\b(?:multiple|other|related|remaining|sibling)\s+"
        r"(?:invoice|document)s?\s*:\s*",
        "",
        cleaned,
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
        "weekly_report_summary_machine_terms": (
            "Remove classification labels, fact labels, JSON keys, statuses, the word evidence, "
            "and phrases such as unknown amount or unspecified date. Omit missing details entirely."
        ),
        "weekly_report_summary_invalid_punctuation": (
            "Remove colons and semicolons and return one plain phrase."
        ),
        "weekly_report_summary_chase_metadata": (
            "Remove every mention of messages, emails, chases, reminders, follow-ups, sequence, level, stage, or touch."
        ),
        "weekly_report_summary_invalid_opening": (
            "Start with exactly one permitted past-tense verb from the system rules; do not add a subject before it."
        ),
        "weekly_report_summary_opening_mismatches_fact": (
            "Use a verb that matches the supplied fact type. Never describe a payment claim as a commitment."
        ),
        "weekly_report_summary_ungrounded_number": (
            "Remove every number, amount, or date that is not present in the supplied event facts."
        ),
        "weekly_report_summary_requires_one_debtor_update": (
            "Return exactly one update for the supplied inbound event. Summarize only "
            "the latest debtor commitment, remittance/payment claim, or query fact."
        ),
    }
    return corrections.get(code, "Correct only the stated validation defect.")


weekly_overdue_report_summarizer = WeeklyOverdueReportSummarizer()
