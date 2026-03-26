"""
Shared formatting helpers for the engine package.

Contains formatting functions used by both the draft generator and email
classifier to avoid duplication.
"""

from datetime import datetime


def format_industry_context(industry) -> str:
    """Format industry context for prompt inclusion.

    Full version used by the draft generator.  Includes payment cycle,
    DSO, escalation patience, communication style, dispute types,
    AI context notes, and current-quarter seasonal patterns.

    Args:
        industry: Industry profile object (or None) with fields
            like payment_cycle, typical_dso_days, escalation_patience,
            seasonal_patterns, and ai_context_notes.

    Returns:
        Multi-line string of industry context.  Includes
        current-quarter seasonal pattern when available.
    """
    if not industry:
        return "Not specified (general B2B collection)"

    lines = [
        f"- Industry: {industry.name} ({industry.code})",
        f"- Payment Norm: {industry.payment_cycle} (typical DSO: {industry.typical_dso_days} days)",
        f"- Escalation Approach: {industry.escalation_patience}",
        f"- Communication Style: {industry.preferred_tone}",
    ]

    if industry.common_dispute_types:
        lines.append(f"- Common Disputes: {', '.join(industry.common_dispute_types)}")

    if industry.ai_context_notes:
        lines.append(f"- Industry Notes: {industry.ai_context_notes}")

    if industry.seasonal_patterns:
        # Get current quarter
        quarter = f"Q{(datetime.now().month - 1) // 3 + 1}"
        if quarter in industry.seasonal_patterns:
            lines.append(f"- Current Season ({quarter}): {industry.seasonal_patterns[quarter]}")

    return "\n".join(lines)


def format_industry_context_for_classification(industry) -> str:
    """Format industry context for the classification prompt.

    Lighter version used by the email classifier.  Focuses on dispute
    types, hardship indicators, and handling notes rather than payment
    cycle and escalation details.

    Args:
        industry: Industry profile object (or None) with fields
            for common dispute types, hardship indicators, and
            handling notes.

    Returns:
        Multi-line string of industry context, or a generic
        fallback message when no industry profile exists.
    """
    if not industry:
        return "Not specified (general B2B collection)"

    lines = [
        f"- Industry: {industry.name} ({industry.code})",
    ]

    if industry.common_dispute_types:
        lines.append(f"- Common Dispute Types: {', '.join(industry.common_dispute_types)}")

    if industry.hardship_indicators:
        lines.append(f"- Industry Hardship Signals: {', '.join(industry.hardship_indicators)}")

    if industry.dispute_handling_notes:
        lines.append(f"- Dispute Notes: {industry.dispute_handling_notes}")

    if industry.hardship_handling_notes:
        lines.append(f"- Hardship Notes: {industry.hardship_handling_notes}")

    return "\n".join(lines)


def format_invoice_table(context) -> str:
    """Format per-invoice details for the classification prompt.

    Build a human-readable list of all outstanding obligations with
    invoice number, amount, due date, and days overdue.  When
    obligation-level collection statuses are available, append them
    so the LLM can distinguish open vs. disputed vs. promised
    invoices.

    Args:
        context: Case context containing obligations and optionally
            obligation_statuses.

    Returns:
        Formatted multi-line string of invoice details.
    """
    if not context.obligations:
        return "No outstanding invoices on record."

    currency = context.party.currency or "GBP"
    lines = []
    for o in context.obligations:
        inv_num = o.invoice_number or "—"
        due = o.due_date or "—"
        lines.append(
            f"- {inv_num}: {currency} {o.amount_due:,.2f} due {due} "
            f"({o.days_past_due} days overdue)"
        )

    # Include obligation-level collection statuses if available
    if context.obligation_statuses:
        status_map = {}
        for s in context.obligation_statuses:
            if isinstance(s, dict):
                oid = s.get("obligation_id")
                cs = s.get("collection_status", "open")
                if oid:
                    status_map[str(oid)] = cs

        if status_map:
            enhanced = []
            for i, o in enumerate(context.obligations):
                status = status_map.get(str(o.id), "open") if hasattr(o, "id") else "open"
                inv_num = o.invoice_number or "—"
                due = o.due_date or "—"
                enhanced.append(
                    f"- {inv_num}: {currency} {o.amount_due:,.2f} due {due} "
                    f"({o.days_past_due} days overdue) [status: {status}]"
                )
            return "\n".join(enhanced)

    return "\n".join(lines)
