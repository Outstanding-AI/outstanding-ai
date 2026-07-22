"""Pure context-to-text helpers used by collection-draft prompt assembly.

These helpers deliberately do not call an LLM, access a provider, or read a
database. The generator owns prompt-section ordering; this module owns safe,
deterministic formatting of already supplied context facts.
"""

from __future__ import annotations

from typing import Any

from ._sanitize import sanitize_delimiter_tags

_PROMISE_TERMINAL_OUTCOMES = {
    "broken",
    "cancelled",
    "canceled",
    "clear",
    "cleared",
    "expired",
    "expired_unfulfilled",
    "fulfilled",
    "kept",
    "paid",
    "settled",
}


def safe_prompt_value(value: Any, *, max_length: int = 240) -> str:
    """Sanitize and bound one model-visible scalar value."""
    text = sanitize_delimiter_tags(str(value or ""))
    text = " ".join(text.split())
    return text[:max_length]


def as_dict(value: Any) -> dict[str, Any]:
    """Normalize Pydantic/dict-like prompt evidence into a mapping."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def float_value(value: Any) -> float:
    """Return a safe numeric prompt value without raising on sparse evidence."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def candidate_currency(request: Any, obligation: Any) -> str:
    """Resolve the document currency for a selected candidate obligation."""
    return (
        str(
            getattr(obligation, "currency", None)
            or getattr(obligation, "currency_code", None)
            or getattr(obligation, "document_currency_code", None)
            or getattr(obligation, "base_currency", None)
            or getattr(request.context.party, "currency", None)
            or getattr(request.context, "base_currency", "")
            or ""
        )
        .strip()
        .upper()
    )


def candidate_amount_after_credit(obligation: Any) -> float:
    """Select the Sage net balance when it was supplied, otherwise amount due."""
    net = getattr(obligation, "net_amount_due_after_credit_native", None)
    return float_value(net if net is not None else getattr(obligation, "amount_due", None))


def recent_history_is_inbound_reply_trigger(
    request: Any, recent_messages: list[dict[str, Any]]
) -> bool:
    """Return whether the latest supplied evidence makes this a reply follow-up."""
    if getattr(request, "trigger_classification", None):
        return True
    if not recent_messages:
        return False
    latest = recent_messages[0] if isinstance(recent_messages[0], dict) else {}
    return str(latest.get("direction") or "").lower() == "inbound" and bool(
        str(latest.get("classification") or "").strip()
    )


def collection_thread_temporal_lines(request: Any) -> list[str]:
    """Render bounded message-time invoice evidence without expanding demand scope."""
    evidence = getattr(request.context, "collection_thread_invoice_evidence", None) or []
    lines: list[str] = []
    for invoice in evidence[:12]:
        if not isinstance(invoice, dict):
            invoice = as_dict(invoice)
        invoice_number = safe_prompt_value(
            invoice.get("invoice_number") or invoice.get("invoice_ref_normalized") or "unknown",
            max_length=64,
        )
        current_state = safe_prompt_value(invoice.get("current_state") or "unknown", max_length=64)
        current_amount = invoice.get("current_amount_due")
        state_bits = []
        for state in (invoice.get("message_states") or [])[:3]:
            if not isinstance(state, dict):
                state = as_dict(state)
            as_of_state = safe_prompt_value(state.get("as_of_state") or "unknown", max_length=48)
            source = safe_prompt_value(state.get("as_of_source") or "unknown", max_length=48)
            confidence = safe_prompt_value(
                state.get("as_of_confidence") or "unknown", max_length=24
            )
            state_bits.append(f"{as_of_state} then ({source}, {confidence})")
        history = "; ".join(state_bits) if state_bits else "no message-time state"
        chase_flag = "yes" if invoice.get("will_be_chased_if_adopted") else "no"
        amount_suffix = (
            f", current amount due {current_amount}" if current_amount not in (None, "") else ""
        )
        lines.append(
            f"- {invoice_number}: {history}; current Sage state={current_state}{amount_suffix}; "
            f"current chase scope={chase_flag}"
        )
    return lines


def debtor_reply_promise_facts(request: Any) -> list[dict[str, str]]:
    """Return deduplicated current-reply and active-promise facts for a prompt."""
    facts: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_fact(
        *,
        source: str,
        promise_date: Any,
        amount: Any = None,
        invoice_refs: Any = None,
        excerpt: Any = None,
    ) -> None:
        if not promise_date:
            return
        date_text = safe_prompt_value(promise_date, max_length=32)
        amount_text = safe_prompt_value(amount, max_length=64) if amount not in (None, "") else ""
        refs = invoice_refs if isinstance(invoice_refs, list) else []
        ref_text = ", ".join(safe_prompt_value(ref, max_length=64) for ref in refs if ref)
        key = (source, date_text, ref_text)
        if key in seen:
            return
        seen.add(key)
        facts.append(
            {
                "source": source,
                "promise_date": date_text,
                "promise_amount": amount_text,
                "invoice_refs": ref_text,
                "excerpt": safe_prompt_value(excerpt, max_length=260) if excerpt else "",
            }
        )

    recent_messages = request.context.lane_recent_messages or request.context.recent_messages or []
    for message in recent_messages:
        if not isinstance(message, dict):
            continue
        classification = str(message.get("classification") or "").upper()
        direction = str(message.get("direction") or "").lower()
        has_promise = classification == "PROMISE_TO_PAY" or bool(message.get("promise_date"))
        if direction == "inbound" and has_promise:
            add_fact(
                source="current debtor reply",
                promise_date=message.get("promise_date"),
                amount=message.get("promise_amount"),
                invoice_refs=message.get("invoice_refs"),
                excerpt=message.get("body_snippet"),
            )

    for promise in getattr(request.context, "promises", []) or []:
        data = as_dict(promise)
        if str(data.get("outcome") or "pending").lower() in _PROMISE_TERMINAL_OUTCOMES:
            continue
        add_fact(
            source="active promise record",
            promise_date=data.get("promise_date"),
            amount=data.get("promise_amount"),
        )

    return facts


def protocol_decision_lines(lane_state: Any, request: Any) -> list[str]:
    """Render deterministic overdue-protocol facts for a selected lane."""
    if not isinstance(lane_state, dict):
        return []
    field_labels = (
        ("protocol_anchor_basis", "Anchor Basis"),
        ("protocol_anchor_date", "Anchor Date"),
        ("protocol_age_days", "Overdue Age Used"),
        ("protocol_selected_day", "Selected Protocol Day"),
        ("protocol_selected_level", "Selected Level"),
        ("protocol_selected_touch_index", "Selected Touch Index"),
        ("protocol_selected_tone", "Selected Tone"),
        ("protocol_intended_level", "Intended Level"),
        ("protocol_actual_sender_level", "Actual Sender Level"),
        ("protocol_fallback_reason", "Fallback Reason"),
        ("protocol_slot_key", "Protocol Slot"),
    )
    lines = [
        f"- {label}: {safe_prompt_value(lane_state[field])}"
        for field, label in field_labels
        if lane_state.get(field) not in (None, "")
    ]
    if runtime_tone := getattr(request, "tone", None):
        lines.append(f"- Runtime-Selected Tone: {safe_prompt_value(runtime_tone)}")
    sender = " / ".join(
        str(value)
        for value in (lane_state.get("current_sender_name"), lane_state.get("current_sender_email"))
        if value
    )
    if sender:
        lines.append(f"- Runtime-Selected Sender: {safe_prompt_value(sender)}")
    recipient = " / ".join(
        str(value)
        for value in (
            lane_state.get("current_recipient_name"),
            lane_state.get("current_recipient_email"),
        )
        if value
    )
    if recipient:
        lines.append(f"- Runtime-Selected Recipient: {safe_prompt_value(recipient)}")
    if lines:
        lines.append(
            "- Instruction: follow these protocol facts exactly. They are deterministic product decisions, "
            "not suggestions for the model to reinterpret."
        )
    return lines


def scheduled_prep_lines(lane_state: Any, request: Any) -> list[str]:
    """Render scheduler timing facts without exposing product internals."""
    values: dict[str, Any] = {}
    if isinstance(lane_state, dict):
        values.update(
            {
                "protocol_due_at": lane_state.get("protocol_due_at"),
                "not_before_at": lane_state.get("not_before_at"),
                "planned_send_at": lane_state.get("planned_send_at"),
                "is_forecast": lane_state.get("is_forecast"),
                "generation_policy_mode": lane_state.get("generation_policy_mode"),
            }
        )
    for context in getattr(request.context, "lane_contexts", []) or []:
        for field in (
            "protocol_due_at",
            "not_before_at",
            "planned_send_at",
            "is_forecast",
            "generation_policy_mode",
        ):
            value = getattr(context, field, None)
            if values.get(field) in (None, "") and value not in (None, ""):
                values[field] = value
    for field in ("protocol_due_at", "not_before_at", "planned_send_at", "is_scheduled_prep"):
        value = getattr(request.context, field, None)
        if values.get(field) in (None, "") and value not in (None, ""):
            values[field] = value
    if values.get("generation_policy_mode") != "scheduled_prep" and not values.get(
        "is_scheduled_prep"
    ):
        return []
    lines = []
    if values.get("is_forecast"):
        lines.append(
            "- Forecast Slot: yes; this draft is prepared early for an upcoming protocol-due action."
        )
    lines.append(
        "- Instruction: Scheduling/forecast dates are operational only. Never use a planned send, "
        "not-before, protocol-due, or forecast date as 'today', a prior-contact date, or any debtor-facing date. "
        "Use the decision cutoff for current-date wording and the invoice-specific sent-history section for prior "
        "outreach wording. Do not mention scheduling windows, forecasting, or internal policy to the debtor."
    )
    return lines


__all__ = [
    "as_dict",
    "candidate_amount_after_credit",
    "candidate_currency",
    "collection_thread_temporal_lines",
    "debtor_reply_promise_facts",
    "float_value",
    "protocol_decision_lines",
    "recent_history_is_inbound_reply_trigger",
    "safe_prompt_value",
    "scheduled_prep_lines",
]
