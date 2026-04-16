"""
Prompt construction helpers for the draft generator.

Module-level functions that build prompt sections for LLM draft generation.
Extracted from ``DraftGenerator`` to keep the orchestration class focused
on the generate/retry loop.
"""


def format_sender_persona(request) -> str:
    """Format sender persona context for prompt inclusion.

    Args:
        request: The generation request containing sender persona,
            name, title, and company.

    Returns:
        Multi-line string describing the sender persona for LLM
        prompt injection.  Falls back to name/title placeholders
        when no persona profile is available.
    """
    company = request.sender_company or ""
    persona = request.sender_persona
    is_generic = persona and persona.is_generic_mailbox

    if is_generic:
        # Generic/shared mailbox — no personal identity
        name = persona.name if persona else request.sender_name or "Collections Team"
        lines = [
            f"- Mailbox Name: {name}",
            "- THIS IS A GENERIC/SHARED MAILBOX (e.g., accounts@, collections@)",
            "- Do NOT use a personal first-name greeting or personal sign-off",
            f"- Sign off as: 'Regards, {name}'" + (f" — {company}" if company else ""),
            "- Use a professional, team-oriented voice (not individual personality)",
        ]
        if company:
            lines.insert(1, f"- Company: {company}")
        return "\n".join(lines)

    if not persona or not persona.communication_style:
        # No persona — use name/title/company if available
        name = request.sender_name or "[SENDER_NAME]"
        title = request.sender_title or "[SENDER_TITLE]"
        company_line = f", Company: {company}" if company else ""
        return f"Name: {name}, Title: {title}{company_line} (no persona profile — use neutral professional voice)"

    lines = [
        f"- Name: {persona.name}",
        f"- Title: {persona.title or 'Team Member'}",
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


def build_extra_sections(request, behavior) -> str:
    """Build extended prompt sections for new context layers.

    Append optional context blocks to the user prompt:
    - Behaviour segment and profile metrics
    - Escalation history (prior senders for handoff narrative)
    - Sender style guidance and examples
    - Conversation history (recent inbound/outbound messages)
    - Last response snippet (fallback when no full history)
    - Tone preference override
    - Closure mode instructions
    - Invoice table placeholder instructions
    - Follow-up trigger classification guidance

    Args:
        request: The generation request with full case context.
        behavior: Party behaviour profile (or None).

    Returns:
        Concatenated string of all applicable prompt sections.
    """
    sections = []
    tracking = getattr(request.context, "communication_tracking", None)
    allow_thread_continuity = not tracking or (
        tracking.tracking_status == "tracked"
        and tracking.send_confirmation_state in (None, "confirmed")
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
            "\n\nWhen writing the handoff narrative, reference these SPECIFIC people "
            "by name and title. For L2+: mention the L1 sender. For L3+: you may say "
            "'Both [L1 name] and [L2 name] have reached out.'\n"
            "If Level 0 (generic mailbox) is in the history, reference it as "
            "'our accounts team' — NOT by a person's name."
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

    # Conversation history (recent messages for follow-up context)
    recent_msgs = request.context.recent_messages
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
            if allow_thread_continuity:
                footer = (
                    "\n\nThis is a FOLLOW-UP email. You MUST acknowledge the debtor's most recent "
                    "response and build on it. Do NOT write a generic first-contact collection email."
                )
            else:
                footer = (
                    "\n\nCommunication tracking is not fully confirmed for this thread. Use the history above only "
                    "when directly supported by the provided excerpts. Do not claim to have seen or received any "
                    "reply that is not explicitly present in context."
                )
            sections.append(header + "\n".join(msg_lines) + footer)

    # Last response snippet (fallback if no recent_messages)
    if not recent_msgs and request.context.communication:
        comm = request.context.communication
        if comm.last_response_snippet:
            footer = (
                "This is a FOLLOW-UP email. Acknowledge the debtor's response and build on it."
                if allow_thread_continuity
                else "Only reference this response if you can ground it directly in the snippet above."
            )
            sections.append(
                f"\n\n**Debtor's Last Response:**\n"
                f"- Type: {comm.last_response_type or 'Unknown'}\n"
                f"- Subject: {comm.last_response_subject or 'N/A'}\n"
                f"- Content: {comm.last_response_snippet}\n\n"
                f"{footer}"
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

    return "".join(sections)
