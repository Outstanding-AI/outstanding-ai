"""
Prompt construction helpers for the draft generator.

Module-level functions that build prompt sections for LLM draft generation.
Extracted from ``DraftGenerator`` to keep the orchestration class focused
on the generate/retry loop.
"""

import logging

from src.prompts._sanitize import sanitize_delimiter_tags

logger = logging.getLogger(__name__)

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


def _safe_prompt_value(value, *, max_length: int = 240) -> str:
    text = sanitize_delimiter_tags(str(value or ""))
    text = " ".join(text.split())
    return text[:max_length]


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _float_value(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_currency(request, obligation) -> str:
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


def _candidate_amount_after_credit(obligation) -> float:
    net = getattr(obligation, "net_amount_due_after_credit_native", None)
    if net is not None:
        return _float_value(net)
    return _float_value(getattr(obligation, "amount_due", None))


def _recent_history_is_inbound_reply_trigger(request, recent_msgs: list[dict]) -> bool:
    if getattr(request, "trigger_classification", None):
        return True
    if not recent_msgs:
        return False
    latest = recent_msgs[0] if isinstance(recent_msgs[0], dict) else {}
    return str(latest.get("direction") or "").lower() == "inbound" and bool(
        str(latest.get("classification") or "").strip()
    )


def _format_collection_thread_temporal_lines(request) -> list[str]:
    evidence = getattr(request.context, "collection_thread_invoice_evidence", None) or []
    lines: list[str] = []
    for invoice in evidence[:12]:
        if not isinstance(invoice, dict):
            invoice = _as_dict(invoice)
        invoice_number = _safe_prompt_value(
            invoice.get("invoice_number") or invoice.get("invoice_ref_normalized") or "unknown",
            max_length=64,
        )
        current_state = _safe_prompt_value(invoice.get("current_state") or "unknown", max_length=64)
        current_amount = invoice.get("current_amount_due")
        states = invoice.get("message_states") or []
        state_bits = []
        for state in states[:3]:
            if not isinstance(state, dict):
                state = _as_dict(state)
            as_of_state = _safe_prompt_value(state.get("as_of_state") or "unknown", max_length=48)
            source = _safe_prompt_value(state.get("as_of_source") or "unknown", max_length=48)
            confidence = _safe_prompt_value(
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


def _extract_debtor_reply_promise_facts(request) -> list[dict]:
    """Return promise facts from the current debtor reply and active promise state."""
    facts: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add_fact(*, source: str, promise_date, amount=None, invoice_refs=None, excerpt=None):
        if not promise_date:
            return
        date_text = _safe_prompt_value(promise_date, max_length=32)
        amount_text = _safe_prompt_value(amount, max_length=64) if amount not in (None, "") else ""
        refs = invoice_refs if isinstance(invoice_refs, list) else []
        ref_text = ", ".join(_safe_prompt_value(ref, max_length=64) for ref in refs if ref)
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
                "excerpt": _safe_prompt_value(excerpt, max_length=260) if excerpt else "",
            }
        )

    recent_msgs = request.context.lane_recent_messages or request.context.recent_messages or []
    for msg in recent_msgs:
        if not isinstance(msg, dict):
            continue
        classification = str(msg.get("classification") or "").upper()
        direction = str(msg.get("direction") or "").lower()
        has_promise = classification == "PROMISE_TO_PAY" or bool(msg.get("promise_date"))
        if direction == "inbound" and has_promise:
            add_fact(
                source="current debtor reply",
                promise_date=msg.get("promise_date"),
                amount=msg.get("promise_amount"),
                invoice_refs=msg.get("invoice_refs"),
                excerpt=msg.get("body_snippet"),
            )

    for promise in getattr(request.context, "promises", []) or []:
        data = _as_dict(promise)
        outcome = str(data.get("outcome") or "pending").lower()
        if outcome in _PROMISE_TERMINAL_OUTCOMES:
            continue
        add_fact(
            source="active promise record",
            promise_date=data.get("promise_date"),
            amount=data.get("promise_amount"),
        )

    return facts


def format_sender_persona(request) -> str:
    """Format sender persona context for prompt inclusion.

    Args:
        request: The generation request containing sender persona,
            name, title, and company.

    Returns:
        Multi-line string describing the sender persona for LLM
        prompt injection. Omits unavailable sender fields rather than
        creating placeholders for the model to fill.
    """
    company = request.sender_company or ""
    persona = request.sender_persona
    is_generic = persona and persona.is_generic_mailbox

    if is_generic:
        # Generic/shared mailbox — no personal identity
        name = request.sender_name or "Collections Team"
        lines = [
            f"- Mailbox Name: {name}",
            "- THIS IS A GENERIC/SHARED MAILBOX (e.g., accounts@, collections@)",
            "- Do NOT use a personal first-name greeting or personal sign-off",
            f"- Sign off as: 'Kind Regards, {name}'" + (f" — {company}" if company else ""),
            "- Use a professional, team-oriented voice (not individual personality)",
        ]
        if company:
            lines.insert(1, f"- Company: {company}")
        return "\n".join(lines)

    if not persona or not persona.communication_style:
        # No persona — use name/title/company if available
        name = request.sender_name or "Collections Team"
        parts = [f"Name: {name}"]
        if request.sender_title:
            parts.append(f"Title: {request.sender_title}")
        if company:
            parts.append(f"Company: {company}")
        parts.append(
            "No persona profile — use a neutral professional voice. "
            "For the sign-off, include only the sender fields listed above; "
            "omit any unavailable title or company."
        )
        return ", ".join(parts)

    lines = [
        f"- Name: {request.sender_name or persona.name}",
        f"- Title: {request.sender_title or persona.title or 'Team Member'}",
    ]
    if company:
        lines.append(f"- Company: {company}")
    if persona.level:
        lines.append(f"- Escalation Level: {persona.level} of 4")
    lines.extend(
        [
            f"- Communication Style: {persona.communication_style}",
            f"- Formality Level: {persona.formality_level}",
            f"- Emphasis: {persona.emphasis}",
        ]
    )
    return "\n".join(lines)


def build_extra_sections(request, behavior, candidate_obligations=None) -> str:
    """Build extended prompt sections for new context layers.

    Append optional context blocks to the user prompt:
    - Behaviour segment and profile metrics
    - Escalation history (prior senders for handoff narrative)
    - Sender style guidance and examples
    - Conversation history (recent inbound/outbound messages)
    - Tone preference override
    - Closure mode instructions
        - Invoice table placeholder instructions
        - Follow-up trigger classification guidance

    Args:
        request: The generation request with full case context.
        behavior: Party behaviour profile (or None).
        candidate_obligations: Upstream-sendable obligations for this draft.

    Returns:
        Concatenated string of all applicable prompt sections.
    """
    sections = []
    tracking = getattr(request.context, "communication_tracking", None)
    strict_sent_proof_type = getattr(tracking, "sent_proof_type", None) if tracking else None
    allow_thread_continuity = not tracking or (
        tracking.tracking_status == "tracked"
        and (
            tracking.send_confirmation_state is None
            or (
                tracking.send_confirmation_state == "confirmed"
                and strict_sent_proof_type
                in {"graph_sent_items_exact_oai", "message_trace", "purview_send_as"}
            )
        )
    )
    candidate_obligations = candidate_obligations or []

    if request.context.uses_current_datalake_contract():
        sections.append(
            "\n\n**Silver Application Decision Context:**\n"
            f"- Source Sync Run: {request.context.source_sync_run_id}\n"
            f"- Application Run: {request.context.application_run_id}\n"
            f"- Core Snapshot Watermark: {request.context.core_snapshot_watermark}\n"
            f"- Application Snapshot Watermark: {request.context.application_snapshot_watermark}\n"
            f"- Decision Cutoff: {request.context.application_decision_cutoff}\n"
            f"- Policy Snapshot: {request.context.policy_snapshot_id}\n"
            f"- Draft Candidate: {request.context.draft_candidate_id}\n"
            f"- Collection Basis: {request.context.chase_basis or request.context.collection_basis or 'overdue'}\n"
            "- Upstream has already selected sender, recipient, cadence, grace policy, escalation level, and candidate obligations."
        )

    if getattr(request.context, "collection_case_id", None):
        sections.append(
            "\n\n**Collection Case Decision Context:**\n"
            f"- Collection Case: {request.context.collection_case_id}\n"
            f"- Threading Strategy: {getattr(request.context, 'threading_strategy', None) or 'unknown'}\n"
            f"- Threading Mode: {getattr(request.context, 'threading_mode', None) or 'unknown'}\n"
            f"- Active Thread Subject: {getattr(request.context, 'active_thread_subject', None) or 'unknown'}\n"
            "- Current Sage/Silver Core obligations are the only demand scope. Historical case thread "
            "evidence is continuity context only."
        )

    if candidate_obligations:
        candidate_lines = []
        for obligation in candidate_obligations:
            inv = obligation.invoice_number or obligation.document_no or obligation.id
            status_bits = [
                f"id={obligation.id}",
                f"is_overdue={getattr(obligation, 'is_overdue', None)}",
                f"is_sendable={getattr(obligation, 'is_sendable', None)}",
                f"is_chase_eligible={getattr(obligation, 'is_chase_eligible', None)}",
            ]
            if getattr(obligation, "has_verified_purchase_order", False):
                status_bits.append(
                    "verified_po="
                    + str(getattr(obligation, "purchase_order_reference", None) or True)
                )
            if getattr(obligation, "has_verified_pod", False):
                status_bits.append(
                    "verified_pod=" + str(getattr(obligation, "pod_reference", None) or True)
                )
            candidate_lines.append(f"- {inv}: " + ", ".join(status_bits))
        sections.append(
            "\n\n**Draft Candidate Obligations:**\n"
            + "\n".join(candidate_lines)
            + "\nUse only these obligations for collection wording. Do not add invoices, widen the scope, "
            "or imply this is the debtor's full account unless these obligations are the full supplied scope."
        )

    excluded_lines = []
    excluded_seen = set()
    explicit_excluded = list(
        getattr(request.context, "excluded_source_disputed_obligations", None) or []
    )
    derived_excluded = []
    if not explicit_excluded:
        for obligation in getattr(request.context, "obligations", None) or []:
            source_query = str(getattr(obligation, "source_query_raw", None) or "").strip()
            if getattr(obligation, "is_source_disputed", False) or source_query:
                derived_excluded.append(
                    {
                        "invoice_number": obligation.invoice_number,
                        "document_no": obligation.document_no,
                        "id": obligation.id,
                        "source_query_raw": source_query,
                    }
                )
    for obligation in explicit_excluded or derived_excluded:
        if not isinstance(obligation, dict):
            continue
        inv = (
            obligation.get("invoice_number")
            or obligation.get("document_no")
            or obligation.get("id")
        )
        if not inv or str(inv) in excluded_seen:
            continue
        excluded_seen.add(str(inv))
        source_query = str(obligation.get("source_query_raw") or "").strip()
        excluded_lines.append(
            f"- {inv}: excluded, invoice dispute/source Sage query flag"
            + (f" ({_safe_prompt_value(source_query)})" if source_query else "")
        )
    if excluded_lines:
        sections.append(
            "\n\n**Excluded Source-Disputed Obligations:**\n"
            + "\n".join(excluded_lines)
            + "\nDo not ask for payment on these obligations unless the upstream context explicitly marks them cleared and sendable."
        )

    candidate_credit_context = getattr(request.context, "candidate_credit_context", None) or {}
    credit_positions = getattr(request.context, "party_credit_position_by_currency", None) or []
    invoice_credit_adjustments = getattr(request.context, "invoice_credit_adjustments", None) or []
    credit_review_flags = getattr(request.context, "credit_review_flags", None) or []
    if candidate_credit_context or credit_positions or invoice_credit_adjustments:
        credit_lines = []
        if candidate_credit_context:
            currency = str(candidate_credit_context.get("currency") or "").strip().upper()
            candidate_total = _float_value(candidate_credit_context.get("candidate_overdue_amount"))
            unapplied = _float_value(candidate_credit_context.get("unapplied_credit_amount"))
            net = _float_value(candidate_credit_context.get("net_candidate_amount"))
            invoice_refs = candidate_credit_context.get("invoice_refs") or []
            if currency and (candidate_total > 0 or unapplied > 0):
                credit_lines.append(
                    f"- Current draft scope {currency}: listed overdue invoices total {candidate_total:,.2f}; "
                    f"unapplied account credit {unapplied:,.2f}; net amount requiring payment "
                    f"for the listed invoices {net:,.2f}; invoices {', '.join(str(ref) for ref in invoice_refs)}"
                )
        candidate_totals_by_currency: dict[str, float] = {}
        for obligation in (
            candidate_obligations or getattr(request.context, "obligations", None) or []
        ):
            currency = _candidate_currency(request, obligation)
            if not currency:
                continue
            candidate_totals_by_currency[currency] = candidate_totals_by_currency.get(
                currency, 0.0
            ) + _candidate_amount_after_credit(obligation)
        positions_by_currency = {
            str(
                getattr(position, "currency_code", None)
                or getattr(request.context.party, "currency", "")
            )
            .strip()
            .upper(): position
            for position in credit_positions
        }
        for currency, candidate_total in sorted(candidate_totals_by_currency.items()):
            position = positions_by_currency.get(currency)
            unapplied = _float_value(getattr(position, "unapplied_credit_amount_native", 0.0))
            net = max(candidate_total - unapplied, 0.0)
            if candidate_total > 0 or unapplied > 0:
                credit_lines.append(
                    f"- Current draft scope {currency}: listed overdue invoices total {candidate_total:,.2f}; "
                    f"unapplied account credit {unapplied:,.2f}; net amount requiring payment "
                    f"for the listed invoices {net:,.2f}"
                )
        for position in credit_positions:
            currency = getattr(position, "currency_code", None) or getattr(
                request.context.party, "currency", ""
            )
            unapplied = _float_value(getattr(position, "unapplied_credit_amount_native", 0.0))
            overdue = _float_value(
                getattr(position, "recovery_eligible_overdue_amount_native", 0.0)
            )
            net = _float_value(getattr(position, "net_recovery_eligible_overdue_native", 0.0))
            if unapplied > 0 or overdue > 0:
                credit_lines.append(
                    f"- Party credit position background {currency}: overdue eligible for recovery {overdue:,.2f}; "
                    f"unapplied credit notes {unapplied:,.2f}; net requiring payment {net:,.2f}"
                )
        for adjustment in invoice_credit_adjustments:
            invoice = getattr(adjustment, "invoice_number", None) or getattr(
                adjustment, "obligation_id", ""
            )
            currency = getattr(adjustment, "currency_code", None) or getattr(
                request.context.party, "currency", ""
            )
            allocated = _float_value(getattr(adjustment, "allocated_credit_amount_native", 0.0))
            net = getattr(adjustment, "invoice_amount_due_after_credit_native", None)
            credit_lines.append(
                f"- Invoice {invoice}: Sage allocated credit {currency} {allocated:,.2f}"
                + (f"; net invoice balance {float(net):,.2f}" if net is not None else "")
            )
        if credit_lines:
            sections.append(
                "\n\n**Credit Note Context:**\n"
                + "\n".join(credit_lines)
                + "\nRules: allocated credit notes reduce only the Sage-linked invoice. "
                "Unapplied credit notes are account-level context: calculate debtor-facing net wording from "
                "the Current draft scope line, not from the wider Party credit position background. "
                "Do not carry credit/net figures forward from old sent-scope history. "
                "Do not claim account credit has been allocated to a specific invoice. Do not net across currencies. "
                "If the Current draft scope net amount is 0.00, write a neutral credit-allocation/account-update email: "
                "mention the listed invoices and unapplied credit, ask how the customer wants the credit allocated or "
                "whether payment/account update has already been arranged, and do not make a normal payment demand. "
                "If the Current draft scope line has unapplied account credit above 0.00 and net amount above 0.00, "
                "include this operator-style sentence with the exact currency, credit amount, and net amount from "
                "that Current draft scope line: "
                '"Our records show an unapplied credit of {currency} {credit_amount} on your account. '
                'This brings the net amount requiring payment for the invoices listed to {currency} {net_amount}."'
            )
        if credit_review_flags:
            sections.append(
                "\nCredit review flags: "
                + ", ".join(str(flag) for flag in credit_review_flags)
                + ". If credit fully covers recovery-eligible overdue, write a neutral allocation/update request, not a normal payment chase."
            )

    # Behaviour segment
    if behavior and behavior.behaviour_segment:
        sections.append(f"\n\n**Behaviour Segment:** {behavior.behaviour_segment}")
        if behavior.behaviour_profile and isinstance(behavior.behaviour_profile, dict):
            profile = behavior.behaviour_profile
            profile_lines = []
            for k in (
                "responsiveness_trend",
                "promise_fulfilment_rate",
                "dispute_frequency",
                "avg_response_time",
            ):
                if k in profile:
                    profile_lines.append(f"- {k.replace('_', ' ').title()}: {profile[k]}")
            if profile_lines:
                sections.append("\n".join(profile_lines))

    # Escalation level context
    escalation_level = getattr(request, "escalation_level", None)
    if escalation_level is not None:
        level_desc = {
            0: "automated first touch",
            1: "first contact",
            2: "follow-up",
            3: "escalation",
            4: "final escalation",
        }
        sections.append(
            f"\n**Current Escalation Level:** {escalation_level} ({level_desc.get(escalation_level, 'escalation')})"
        )
        if escalation_level == 0:
            sections.append(
                "THIS IS A LEVEL 0 AUTOMATED REMINDER — keep it simple, factual, and template-like."
            )

    lane_state = getattr(request.context, "lane", None)
    threading_strategy = (
        getattr(request.context, "threading_strategy", None) or "invoice_cohort_thread"
    )
    if lane_state:
        invoice_refs = ", ".join(lane_state.get("invoice_refs") or []) or "none"
        tone_ladder = ", ".join(lane_state.get("tone_ladder") or []) or "none"
        if threading_strategy == "single_active_debtor_thread":
            scope_rule = (
                "- Scope Rule: continue the single active debtor case thread. The invoice table and "
                "candidate obligations are the current chase scope for this case. Historical thread "
                "evidence may explain continuity, but it must not add invoices or amounts to the demand."
            )
            title = "Collection Case Scope Context"
        else:
            scope_rule = (
                "- Scope Rule: this email is for this lane/cohort only. Other lanes for the same debtor may exist "
                "and may be handled by different senders; do not merge or reference them unless listed here."
            )
            title = "Collection Lane Context"
        sections.append(
            f"\n\n**{title}:**\n"
            f"- Collection Lane: {request.context.collection_lane_id or lane_state.get('collection_lane_id') or 'unknown'}\n"
            f"- Current Level: {lane_state.get('current_level')} (entry level {lane_state.get('entry_level')})\n"
            f"- Mail Mode: {getattr(request.context, 'lane_mail_mode', None) or 'initial'}\n"
            f"- Scheduled Touch Index: {lane_state.get('scheduled_touch_index')} of {lane_state.get('max_touches_for_level')}\n"
            f"- Reminder Cadence (days): {lane_state.get('reminder_cadence_days_for_level')}\n"
            f"- Level Window (days): {lane_state.get('max_days_for_level')}\n"
            f"- Tone Ladder: {tone_ladder}\n"
            f"- Open Invoices: {invoice_refs}\n"
            f"- Outstanding Amount: {lane_state.get('outstanding_amount')}\n"
            f"- Suppression State: {lane_state.get('suppression_state') or 'none'}\n"
            f"{scope_rule}"
        )
        scheduled_touch = lane_state.get("scheduled_touch_index")
        if isinstance(scheduled_touch, int) and scheduled_touch > 1:
            sections.append(
                "\n\n**Debtor-Facing Prior Outreach Instruction:**\n"
                "- Include one short sentence that references prior outreach or the last contact date when supplied. "
                "Do not write as if this is a first contact. Do not expose the touch/reminder number to the debtor."
            )
        protocol_lines = _build_protocol_decision_lines(lane_state, request)
        if protocol_lines:
            sections.append(
                "\n\n**Protocol Decision (deterministic, do not override):**\n"
                + "\n".join(protocol_lines)
            )
        schedule_lines = _build_scheduled_prep_lines(lane_state, request)
        if schedule_lines:
            sections.append(
                "\n\n**Scheduled Prep Context (internal, do not mention):**\n"
                + "\n".join(schedule_lines)
            )
    elif request.context.lane_contexts:
        logger.warning(
            "LaneContextInfo.invoice_refs and outstanding_amount are deprecated; "
            "prefer CaseContext.lane for prompt construction."
        )
        lane_contexts = request.context.lane_contexts
        if len(lane_contexts) > 1 or getattr(request.context, "mode", None) == "multi_lane":
            lane_lines = []
            for lane in lane_contexts:
                invoice_refs = ", ".join(lane.invoice_refs) if lane.invoice_refs else "none"
                tone_ladder = (
                    ", ".join(lane.tone_ladder) if getattr(lane, "tone_ladder", None) else "none"
                )
                lane_lines.append(
                    f"- Lane {lane.lane_id}: level {lane.current_level}, "
                    f"touch {lane.scheduled_touch_index} of {lane.max_touches_for_level}, "
                    f"action={lane.action or 'collection'}, invoices={invoice_refs}, tone_ladder={tone_ladder}"
                )
            sections.append(
                "\n\n**Collection Scope Context:**\n"
                + (
                    "- Coverage Mode: single active debtor case. Multiple lane contexts are underlying invoice/cohort "
                    "state, but this draft continues one debtor case thread.\n"
                    if threading_strategy == "single_active_debtor_thread"
                    else "- Coverage Mode: multiple due recovery lanes are intentionally grouped because they share the "
                    "same debtor, recipient, sender, and protocol level.\n"
                )
                + "- Scope Rule: the invoice table is the authoritative scope for this draft. Include every listed "
                "sendable invoice, but do not claim this is the debtor's complete account if other lanes are not "
                "listed here.\n"
                "- Wording Rule: if lanes have different actions or touch indices, describe the request as a "
                "follow-up on the listed overdue invoices. Only say an invoice was previously reminded if the "
                "provided history proves it; otherwise keep the wording neutral.\n"
                + "\n".join(lane_lines)
            )
            max_touch = max(
                (int(getattr(lane, "scheduled_touch_index", 0) or 0) for lane in lane_contexts),
                default=0,
            )
            if max_touch > 1:
                sections.append(
                    "\n\n**Debtor-Facing Prior Outreach Instruction:**\n"
                    "- Include one short sentence that references prior outreach when supplied. "
                    "Do not write as if this is a first contact. Do not expose touch/reminder numbers to the debtor."
                )
        else:
            lane = lane_contexts[0]
            invoice_refs = ", ".join(lane.invoice_refs) if lane.invoice_refs else "none"
            tone_ladder = (
                ", ".join(lane.tone_ladder) if getattr(lane, "tone_ladder", None) else "none"
            )
            sections.append(
                "\n\n**Collection Lane Context:**\n"
                f"- Collection Lane: {lane.lane_id}\n"
                f"- Current Level: {lane.current_level} (entry level {lane.entry_level})\n"
                f"- Scheduled Touch Index: {lane.scheduled_touch_index} of {lane.max_touches_for_level}\n"
                f"- Reminder Cadence (days): {lane.reminder_cadence_days_for_level}\n"
                f"- Level Window (days): {lane.max_days_for_level}\n"
                f"- Tone Ladder: {tone_ladder}\n"
                f"- Open Invoices: {invoice_refs}\n"
                f"- Outstanding Amount: {lane.outstanding_amount}\n"
                + (
                    "- Scope Rule: continue the single active debtor case thread. The invoice table is the current "
                    "case chase scope; do not add invoices from history.\n"
                    if threading_strategy == "single_active_debtor_thread"
                    else "- Scope Rule: this email is for this lane/cohort only. Other lanes for the same debtor may exist "
                    "and may be handled by different senders; do not merge or reference them unless listed here."
                )
            )
            if int(getattr(lane, "scheduled_touch_index", 0) or 0) > 1:
                sections.append(
                    "\n\n**Debtor-Facing Prior Outreach Instruction:**\n"
                    "- Include one short sentence that references prior outreach when supplied. "
                    "Do not write as if this is a first contact. Do not expose touch/reminder numbers to the debtor."
                )

    lane_history = getattr(request.context, "lane_history", None)
    if lane_history:
        history_lines = []
        for event in lane_history[-8:]:
            detail = event.get("detail") or {}
            detail_bits = []
            if detail.get("mail_mode"):
                detail_bits.append(f"mail_mode={detail['mail_mode']}")
            if detail.get("tone_used"):
                detail_bits.append(f"tone={detail['tone_used']}")
            if detail.get("reason"):
                detail_bits.append(f"reason={detail['reason']}")
            if detail.get("replacement_reason"):
                detail_bits.append(f"replacement_reason={detail['replacement_reason']}")
            stale_changes = detail.get("stale_changes") or []
            if stale_changes:
                detail_bits.append(
                    "stale_changes=" + "; ".join(str(change) for change in stale_changes[:5])
                )
            suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
            history_lines.append(
                f"- {event.get('created_at', 'unknown')}: {event.get('event_type', 'event')} "
                f"level {event.get('from_level')}→{event.get('to_level')}{suffix}"
            )
        sections.append("\n\n**Lane History:**\n" + "\n".join(history_lines))

    actual_sent_scope_history = getattr(request.context, "actual_sent_scope_history", None) or []
    if actual_sent_scope_history:
        sent_scope_lines = []
        for history in actual_sent_scope_history[-6:]:
            sent_at = getattr(history, "sent_at", None)
            sent_at_str = sent_at.strftime("%Y-%m-%d") if sent_at else "unknown date"
            sent_refs = getattr(history, "invoice_refs_sent", None) or []
            generated_refs = getattr(history, "invoice_refs_generated", None) or []
            added_refs = getattr(history, "invoice_refs_added", None) or []
            removed_refs = getattr(history, "invoice_refs_removed", None) or []
            severity = getattr(history, "edit_severity", None) or "none"
            line = (
                f"- {sent_at_str}: actually sent invoices "
                f"{', '.join(sent_refs) if sent_refs else 'none recorded'}"
                f" (AI-generated scope: {', '.join(generated_refs) if generated_refs else 'none recorded'}; "
                f"edit severity: {severity})"
            )
            if added_refs:
                line += f"; operator added: {', '.join(added_refs)}"
            if removed_refs:
                line += f"; operator removed before send: {', '.join(removed_refs)}"
            if getattr(history, "payment_expectation_added", False):
                expectation_bits = [
                    getattr(history, "payment_expectation_kind", None) or "payment expectation"
                ]
                if getattr(history, "payment_expectation_date", None):
                    expectation_bits.append(f"date {getattr(history, 'payment_expectation_date')}")
                if getattr(history, "payment_expectation_amount", None) is not None:
                    expectation_bits.append(
                        f"amount {getattr(history, 'payment_expectation_amount')}"
                    )
                line += "; operator added payment expectation: " + ", ".join(expectation_bits)
            sent_scope_lines.append(line)
        sections.append(
            "\n\n**Actual Sent Scope History:**\n"
            + "\n".join(sent_scope_lines)
            + "\n\nUse this section only as evidence of what the debtor actually received after operator edits. "
            "The current invoice table remains the authoritative scope for this draft. Do not chase, demand, "
            "or re-add any invoice merely because it appears here if it is absent from the current invoice table. "
            "If the current invoice table differs from prior sent scope, do not reuse prior total, net amount, "
            "or credit arithmetic in debtor-facing copy. "
            "For wording such as 'we previously reminded you', rely on the actually sent invoices above, not on "
            "the AI-generated scope when an operator removed invoices before sending. Treat payment expectation "
            "flags here as prior-email evidence only; do not state a current promise/remittance unless the current "
            "promise or remittance context also supports it."
        )

    if request.context.sendable_obligation_ids or request.context.blocked_obligation_ids:
        sections.append(
            "\n\n**Lane Sendable Scope:**\n"
            f"- Sendable Obligations: {', '.join(request.context.sendable_obligation_ids or []) or 'none'}\n"
            f"- Blocked Obligations: {', '.join(request.context.blocked_obligation_ids or []) or 'none'}\n"
            f"- Blocked Reasons: {request.context.blocked_reasons_by_obligation_id or {}}"
        )

    reply_scope = getattr(request.context, "reply_scope", None) or {}
    if isinstance(reply_scope, dict) and reply_scope:
        scoped_invoices = [
            _safe_prompt_value(value, max_length=64)
            for value in (reply_scope.get("invoice_refs") or [])
            if str(value or "").strip()
        ]
        scoped_obligations = [
            _safe_prompt_value(value, max_length=64)
            for value in (reply_scope.get("obligation_ids") or [])
            if str(value or "").strip()
        ]
        sections.append(
            "\n\n**Reply Scope:**\n"
            f"- Scope Status: {_safe_prompt_value(reply_scope.get('scope_status'), max_length=64) or 'unknown'}\n"
            f"- Scoped Invoices: {', '.join(scoped_invoices) if scoped_invoices else 'none listed'}\n"
            f"- Scoped Obligations: {', '.join(scoped_obligations) if scoped_obligations else 'none listed'}\n"
            "- This reply must discuss only the scoped invoices/obligations above. Do not mention, chase, "
            "or ask for payment on unrelated open invoices.\n"
            "- If the triggering reply is a dispute, query, internal blocker, already-paid claim, promise, "
            "or remittance, acknowledge that the scoped item is being reviewed or noted; do not demand payment "
            "for that scoped item in this reply."
        )

    # Escalation history (all prior senders for handoff narrative)
    esc_history = request.context.escalation_history
    if esc_history:
        hist_lines = []
        for h in esc_history:
            # Level 0 senders are generic mailboxes — label as team, not person
            if h.get("level") == 0 or h.get("is_generic_mailbox"):
                hist_lines.append(
                    f"- Level 0: Accounts Team (generic mailbox) "
                    f"— {h['touch_count']} automated reminder(s), last on {h.get('last_touch_at', 'unknown')}"
                )
            else:
                title_part = f", {h['title']}" if h.get("title") else ""
                hist_lines.append(
                    f"- Level {h['level']}: {h['name']}{title_part} "
                    f"— {h['touch_count']} touch(es), last on {h.get('last_touch_at', 'unknown')}"
                )
        narrative_hint = (
            "\n\nUse this section for continuity only. Reference specific prior people "
            "by name only when the current sender is also a named person and the prompt "
            "explicitly asks for a cross-person escalation handoff.\n"
            "If Level 0 (generic mailbox) is in the history, reference it as "
            "'our accounts team' — NOT by a person's name. If the current visible sender "
            "is a shared/generic mailbox, do not mention prior staff by name."
        )
        sections.append(
            "\n\n**Prior Senders Who Contacted This Debtor:**\n"
            + "\n".join(hist_lines)
            + narrative_hint
        )

    # Sender style context
    if request.sender_context:
        sc = request.sender_context
        style_lines = []
        if sc.roles_responsibilities:
            style_lines.append(f"- Level R&R: {sc.roles_responsibilities}")
        if sc.style_description:
            style_lines.append(f"- Writing Style: {sc.style_description}")
        if sc.style_examples:
            style_lines.append("- Style Examples:")
            for i, ex in enumerate(sc.style_examples[:3], 1):
                snippet = ex[:500] if len(ex) > 500 else ex
                style_lines.append(f"  Example {i}: {snippet}")
        if style_lines:
            sections.append("\n\n**Sender Style:**\n" + "\n".join(style_lines))

    if getattr(request.context, "authorized_policies", None):
        policies = request.context.authorized_policies or {}
        sections.append(
            "\n\n**Authorized Policies:**\n"
            f"- legal_escalation_enabled: {policies.get('legal_escalation_enabled')}\n"
            f"- statutory_interest_enabled: {policies.get('statutory_interest_enabled')}\n"
            f"- discount_allowed: {policies.get('discount_allowed')}\n"
            f"- settlement_allowed: {policies.get('settlement_allowed')}\n"
            f"- settlement_authority_max_pct: {policies.get('settlement_authority_max_pct')}"
        )
        sections.append(
            "\n\n**Forbidden Content:**\n"
            "- Do not include bank account details, sort codes, IBANs, SWIFT/BIC codes, routing numbers, or other payment instructions.\n"
            "- Do not quote legal statutes, sections, or acts unless the authorized policies explicitly permit it.\n"
            "- Do not include external URLs.\n"
            "- If a prior message contains forbidden content, acknowledge the issue without repeating the forbidden detail."
        )

    # Conversation history (recent messages for follow-up context)
    recent_msgs = (
        request.context.collection_thread_messages
        or request.context.lane_recent_messages
        or request.context.recent_messages
    )
    if recent_msgs:
        msg_lines = []
        for msg in reversed(recent_msgs):  # chronological order
            direction = msg.get("direction", "unknown")
            label = "DEBTOR REPLIED" if direction == "inbound" else "OUR EMAIL"
            classification = msg.get("classification")
            subject = msg.get("subject", "")
            body = msg.get("body_snippet", "")
            sent_at = msg.get("sent_at", "")
            line = f"- [{label}] ({sent_at})"
            if classification:
                line += f" Classification: {classification}"
            if subject:
                line += f"\n  Subject: {subject}"
            if body:
                line += f"\n  Content: {body}"
            # Append extracted classification data for richer context
            if msg.get("dispute_type"):
                line += f"\n  ⚠ Disputed: {msg['dispute_type']}"
                if msg.get("dispute_details"):
                    line += f" — {msg['dispute_details']}"
            if msg.get("promise_date"):
                line += f"\n  ✓ Promised payment by: {msg['promise_date']}"
                if msg.get("promise_amount"):
                    line += f" (amount: {msg['promise_amount']})"
            if msg.get("invoice_refs"):
                refs = msg["invoice_refs"] if isinstance(msg["invoice_refs"], list) else []
                if refs:
                    line += f"\n  Invoices referenced: {', '.join(str(r) for r in refs)}"
            invoice_states = msg.get("invoice_states") if isinstance(msg, dict) else None
            if isinstance(invoice_states, list) and invoice_states:
                state_bits = []
                for state in invoice_states[:4]:
                    if not isinstance(state, dict):
                        state = _as_dict(state)
                    invoice = (
                        state.get("invoice_number") or state.get("invoice_ref_raw") or "unknown"
                    )
                    state_bits.append(
                        f"{invoice}: {state.get('as_of_state') or 'unknown'} then / "
                        f"{state.get('current_state') or 'unknown'} now "
                        f"({state.get('as_of_confidence') or 'unknown'} confidence)"
                    )
                line += "\n  Message-time invoice states: " + "; ".join(state_bits)
            # Debtor intent data from classifier
            if msg.get("claimed_amount"):
                claimed = f"\n  💰 Claimed payment: {msg['claimed_amount']}"
                if msg.get("claimed_date"):
                    claimed += f" on {msg['claimed_date']}"
                if msg.get("claimed_reference"):
                    claimed += f" (ref: {msg['claimed_reference']})"
                line += claimed
            if msg.get("disputed_amount"):
                line += f"\n  ⚖️ Disputed amount: {msg['disputed_amount']}"
            if msg.get("insolvency_type"):
                line += f"\n  🏛️ Insolvency: {msg['insolvency_type']}"
            msg_lines.append(line)
        if msg_lines:
            header = "\n\n**Recent Conversation History:**\n"
            is_reply_trigger = _recent_history_is_inbound_reply_trigger(request, recent_msgs)
            if allow_thread_continuity and is_reply_trigger:
                footer = (
                    "\n\nThis is a FOLLOW-UP email. You MUST acknowledge the debtor's most recent "
                    "response and build on it. Do NOT write a generic first-contact collection email."
                )
            elif allow_thread_continuity:
                footer = (
                    '\n\nUse the history above for thread continuity only. Do NOT say "thank you for '
                    'your reply" or imply the debtor recently responded unless the provided latest '
                    "message is an explicit inbound debtor reply trigger."
                )
            else:
                footer = (
                    "\n\nCommunication tracking is not fully confirmed for this thread. Use the history above only "
                    "when directly supported by the provided excerpts. Do not claim to have seen or received any "
                    "reply that is not explicitly present in context."
                )
            sections.append(header + "\n".join(msg_lines) + footer)

    temporal_lines = _format_collection_thread_temporal_lines(request)
    if temporal_lines:
        sections.append(
            "\n\n**Collection Thread Temporal Evidence:**\n"
            + "\n".join(temporal_lines)
            + "\n\nInstruction: use this evidence only to preserve continuity on the debtor case. "
            "Do not demand payment for any invoice that is not in the current candidate obligations. "
            "Do not use historical amounts as current demand amounts."
        )

    promise_facts = _extract_debtor_reply_promise_facts(request)
    if promise_facts:
        fact_lines = []
        for fact in promise_facts:
            line = f"- Source: {fact['source']}; promised payment date: {fact['promise_date']}"
            if fact["promise_amount"]:
                line += f"; promised amount: {fact['promise_amount']}"
            if fact["invoice_refs"]:
                line += f"; invoices: {fact['invoice_refs']}"
            if fact["excerpt"]:
                line += f'\n  Debtor wording: "{fact["excerpt"]}"'
            fact_lines.append(line)
        sections.append(
            "\n\n**Debtor Reply Promise Facts:**\n"
            + "\n".join(fact_lines)
            + "\n\nInstruction: this is a commitment acknowledgement. Acknowledge the promised "
            "date exactly, thank the debtor, and say we will look out for the payment and "
            "reconcile it once received. Do NOT ask for a payment date, payment timeline, "
            "payment status update, or whether payment can be expected when the promised "
            "date above is already known."
        )

    # Recent manual touchpoints (phone / SMS / letter / in-person / voicemail / other).
    # The metric-isolation discriminator ``touch_type`` is the source of truth for the
    # email-vs-manual split — NOT ``channel != 'email'`` — because the email section
    # above already covers AI emails. Backend filters out redacted manual rows at the
    # SQL layer (manual_status != 'redacted' on the AI context path only); we do not
    # re-filter here so the section reflects exactly what the operator's audit trail
    # shows in the timeline (minus redactions).
    recent_touches = getattr(request.context, "recent_touches", None) or []
    manual_touches = [t for t in recent_touches if getattr(t, "touch_type", None) == "manual_log"]
    if manual_touches:
        manual_lines = []
        # Show oldest-first so the prompt reads chronologically and the most
        # recent touch is the last thing the model sees before the directive.
        for touch in reversed(manual_touches):
            channel = (getattr(touch, "channel", None) or "other").replace("_", " ")
            direction = getattr(touch, "direction", None) or "outbound"
            sent_at = getattr(touch, "sent_at", None)
            sent_at_str = sent_at.strftime("%Y-%m-%d") if sent_at else "unknown date"
            operator = getattr(touch, "logged_by_user_name", None) or "an operator"
            purpose = (getattr(touch, "manual_purpose", None) or "general").replace("_", " ")
            notes = (getattr(touch, "manual_notes", None) or "").strip()
            linked_obligations = getattr(touch, "manual_obligations", None) or []
            invoice_refs = []
            for obligation in linked_obligations:
                invoice = None
                if isinstance(obligation, dict):
                    invoice = obligation.get("invoice_number") or obligation.get("obligation_id")
                else:
                    invoice = getattr(obligation, "invoice_number", None) or getattr(
                        obligation, "obligation_id", None
                    )
                if invoice:
                    invoice_refs.append(str(invoice))
            invoice_suffix = (
                f"; invoices: {', '.join(invoice_refs[:8])}" if invoice_refs else "; account-level"
            )
            if len(invoice_refs) > 8:
                invoice_suffix += f" +{len(invoice_refs) - 8} more"
            line = (
                f"- {sent_at_str} ({channel.capitalize()}, {direction}, {purpose}, "
                f"logged by {operator}{invoice_suffix})"
            )
            if notes:
                # Trim notes to keep the prompt window manageable; full text stays
                # in Silver. 800 chars is roughly the length of a detailed call
                # summary without dominating the prompt.
                snippet = notes if len(notes) <= 800 else notes[:800] + "…"
                line += f': "{snippet}"'
            manual_lines.append(line)
        sections.append(
            "\n\n**Recent Manual Touchpoints:**\n"
            + "\n".join(manual_lines)
            + "\n\nThese are operator-logged off-channel conversations. Treat them as business "
            "context for tone and factual continuity, not as proof that AI drove collection. "
            "Account-level notes apply to the debtor generally; invoice-linked notes apply only "
            "to the listed invoices. Only treat a manual note as collection-driving when the "
            "manual-intervention summary explicitly says so. If the effect is unknown, be "
            "conservative: do not claim AI ownership, do not invent commitments, and avoid "
            "escalating tone solely from that note. If a recent manual touchpoint includes a "
            "query update, commitment, or remittance note, reference dates / amounts verbatim. "
            "Do NOT chase queried invoices as normal collection items, and do NOT escalate tone "
            "if a live verbal commitment is in flight."
        )

    manual_intervention_summary = getattr(request.context, "manual_intervention_summary", None)
    if manual_intervention_summary:
        sections.append(
            "\n\n**Manual Intervention Interpretation:**\n"
            + str(manual_intervention_summary)
            + "\n\nUse this only to decide whether manual work should change attribution or tone. "
            "It does not create or settle invoice commitments by itself."
        )

    remittance_lines = []
    for remittance in getattr(request.context, "remittances", []) or []:
        received_at = getattr(remittance, "remittance_received_at", None)
        amount = getattr(remittance, "remittance_amount", None)
        reference = getattr(remittance, "bank_reference", None)
        outcome = getattr(remittance, "outcome", None) or "pending"
        line = f"- Remittance dated {received_at or 'unknown date'} ({outcome})"
        if amount is not None:
            line += f", amount {amount}"
        if reference:
            line += f", reference {reference}"
        remittance_lines.append(line)
    if remittance_lines:
        sections.append(
            "\n\n**Remittance Evidence:**\n"
            + "\n".join(remittance_lines)
            + "\n\nLive remittance evidence means do not chase the invoice as ignored. A broken or "
            "not-found remittance is scheduled collection context only; it must not be treated as "
            "an immediate debtor-reply trigger unless this request explicitly has a debtor reply "
            "classification. Ask for reconciliation help only if needed."
        )

    # last_response_snippet is deprecated; recent_messages is the canonical source.
    if not recent_msgs and request.context.communication:
        comm = request.context.communication
        if comm.__dict__.get("last_response_snippet"):
            logger.warning(
                "CommunicationInfo.last_response_snippet is deprecated and ignored; "
                "populate CaseContext.recent_messages[0].body_snippet instead."
            )

    # Customer segmentation context
    customer_type = (
        getattr(request.context.party, "customer_type", None)
        if request.context and request.context.party
        else None
    )
    size_bucket = (
        getattr(request.context.party, "size_bucket", None)
        if request.context and request.context.party
        else None
    )
    if not customer_type and behavior:
        customer_type = getattr(behavior, "customer_type", None)
    if not size_bucket and behavior:
        size_bucket = getattr(behavior, "size_bucket", None)
    if customer_type or size_bucket:
        sections.append(
            f"\n\n--- CUSTOMER SEGMENT ---\n"
            f"Customer type: {customer_type or 'unknown'}, Size: {size_bucket or 'unknown'}"
        )

    # Tone preference
    if request.tone_preference:
        sections.append(f"\n\n**Tone Preference:** {request.tone_preference}")

    # Closure mode
    if request.closure_mode:
        sections.append(
            "\n\n**CLOSURE EMAIL MODE**: This is a closure/thank-you email. "
            "The debtor has paid in full or the case is resolved. "
            "Use a grateful, relationship-preserving tone. "
            "Do NOT include any collection language, payment demands, "
            "or references to other invoices. Keep it brief and positive."
        )
    else:
        # Invoice table instruction (non-closure only)
        sections.append(
            "\n\nIMPORTANT: Do NOT write invoice numbers, amounts, or dates "
            "in the email body. Instead, include the exact placeholder "
            "{INVOICE_TABLE} where invoice details should appear. "
            "The system will replace this with a programmatic table. "
            "You may reference 'the invoices listed below' or "
            "'the outstanding items' in your prose."
        )

    # Follow-up trigger classification (explicit instruction for classification-aware drafts)
    if request.trigger_classification:
        sections.append(
            f"\n\n**FOLLOW-UP TRIGGER: {request.trigger_classification}**\n"
            f"This draft was triggered by the debtor's reply classified as "
            f"{request.trigger_classification}.\n"
            "Follow the Classification-Specific Follow-Up Guidance in your system "
            f"instructions for {request.trigger_classification}. "
            "Address what the debtor said directly."
        )

    # Sprint A item #3 follow-up (2026-04-28): for auto-replies
    # to payment claims (``payment_not_found`` / ``partial_payment_ack``),
    # surface the EXACT figures the debtor cited and the EXACT figures we
    # matched against. The model must use these — never invent values.
    follow_up = getattr(request, "follow_up_context", None)
    if follow_up is not None:
        facts: list[str] = []
        if follow_up.claimed_amount is not None:
            facts.append(f"- claimed_amount: {follow_up.claimed_amount}")
        if follow_up.claimed_date:
            facts.append(f"- claimed_date: {follow_up.claimed_date}")
        if follow_up.claimed_reference:
            facts.append(f"- claimed_reference: {follow_up.claimed_reference}")
        if follow_up.matched_amount is not None:
            facts.append(f"- matched_amount (found on our records): {follow_up.matched_amount}")
        if follow_up.residual_amount is not None:
            facts.append(f"- residual_amount (claim minus match): {follow_up.residual_amount}")
        if facts:
            sections.append(
                "\n\n**VERIFICATION FACTS (use VERBATIM — do NOT invent or round):**\n"
                + "\n".join(facts)
                + "\n\n"
                "Cite these values when acknowledging what the debtor told us "
                "and what our records show. If a value is missing above, do "
                "not fabricate it — phrase the email so the missing field is "
                "not required."
            )

    return "".join(sections)


def _build_protocol_decision_lines(lane_state, request) -> list[str]:
    """Render overdue-protocol routing facts when present on the lane context."""
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

    lines = []
    for field, label in field_labels:
        value = lane_state.get(field)
        if value is not None and value != "":
            lines.append(f"- {label}: {_safe_prompt_value(value)}")

    runtime_tone = getattr(request, "tone", None)
    if runtime_tone:
        lines.append(f"- Runtime-Selected Tone: {_safe_prompt_value(runtime_tone)}")

    sender_email = lane_state.get("current_sender_email")
    sender_name = lane_state.get("current_sender_name")
    if sender_email or sender_name:
        sender = " / ".join(str(v) for v in (sender_name, sender_email) if v)
        lines.append(f"- Runtime-Selected Sender: {_safe_prompt_value(sender)}")

    recipient_email = lane_state.get("current_recipient_email")
    recipient_name = lane_state.get("current_recipient_name")
    if recipient_email or recipient_name:
        recipient = " / ".join(str(v) for v in (recipient_name, recipient_email) if v)
        lines.append(f"- Runtime-Selected Recipient: {_safe_prompt_value(recipient)}")

    if lines:
        lines.append(
            "- Instruction: follow these protocol facts exactly. They are deterministic product decisions, "
            "not suggestions for the model to reinterpret."
        )
    return lines


def _build_scheduled_prep_lines(lane_state, request) -> list[str]:
    """Render scheduler timing facts without exposing product internals."""
    values = {}
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
    planned_send_at = (
        values.get("planned_send_at")
        or values.get("not_before_at")
        or values.get("protocol_due_at")
    )
    if planned_send_at:
        lines.append(f"- Planned Send Timing: {_safe_prompt_value(planned_send_at)}")
    if values.get("not_before_at"):
        lines.append(
            f"- Do Not Imply This Touch Occurred Before: {_safe_prompt_value(values['not_before_at'])}"
        )
    if values.get("is_forecast"):
        lines.append(
            "- Forecast Slot: yes; this draft is prepared early for an upcoming protocol-due action."
        )
    lines.append(
        "- Instruction: use the planned send timing only for temporal wording. Do not mention scheduling windows, "
        "forecasting, internal policy, or any 'send after' instruction to the debtor."
    )
    return lines
