"""
Persona Constants.

All persona-related prompt fragments and configuration as importable constants.
Used by persona generation, refinement, and draft generation.
"""

import re

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

CLASSIFICATION_CATEGORIES = frozenset(
    {
        "INSOLVENCY",
        "UNSUBSCRIBE",
        "HOSTILE",
        "ALREADY_PAID",
        "PAYMENT_CONFIRMATION",
        "REMITTANCE_ADVICE",
        "PARTIAL_PAYMENT_NOTIFICATION",
        "DISPUTE",
        "AMOUNT_DISAGREEMENT",
        "RETENTION_CLAIM",
        "PROMISE_TO_PAY",
        "HARDSHIP",
        "PLAN_REQUEST",
        "REQUEST_INFO",
        "REDIRECT",
        "ESCALATION_REQUEST",
        "QUERY_QUESTION",
        "COOPERATIVE",
        "LEGAL_RESPONSE",
        "OUT_OF_OFFICE",
        "EMAIL_BOUNCE",
        "GENERIC_ACKNOWLEDGEMENT",
        "UNCLEAR",
    }
)

ALLOWED_PLACEHOLDERS = frozenset(
    {
        "{INVOICE_TABLE}",
        "[SENDER_NAME]",
        "[SENDER_TITLE]",
        "[SENDER_COMPANY]",
    }
)

CANONICAL_TONES = frozenset(
    {
        "friendly_reminder",
        "friendly_escalating",
        "professional",
        "professional_escalating",
        "firm",
        "firm_escalating",
        "final_notice",
        "legal_pre_action",
        "acknowledgement",
        "concerned_inquiry",
    }
)

CANONICAL_OBJECTIVES = frozenset({"follow_up", "promise_reminder", "escalation", "initial_contact"})
TONE_PREFERENCES = frozenset({"diplomatic", "professional", "direct"})
CTA_VALUES = frozenset({"send_email", "create_case", "escalate", "close_case"})


def _regex_from_values(values: frozenset[str]) -> str:
    return r"^(" + "|".join(re.escape(v) for v in sorted(values)) + r")$"


TONE_REGEX = _regex_from_values(CANONICAL_TONES)
OBJECTIVE_REGEX = _regex_from_values(CANONICAL_OBJECTIVES)
TONE_PREFERENCE_REGEX = _regex_from_values(TONE_PREFERENCES)
CTA_REGEX = _regex_from_values(CTA_VALUES)

# =============================================================================
# LEVEL DESCRIPTIONS
# =============================================================================

LEVEL_DESCRIPTIONS = {
    1: (
        "First point of contact. Handles routine follow-ups and friendly reminders. "
        "Typically an Accounts Receivable Coordinator or Credit Controller. "
        "Tone is warm and helpful — assumes the invoice was missed, not ignored. "
        "Uses 'Hey' or 'Hi'. Keeps it short. Gives the debtor an easy out."
    ),
    2: (
        "Secondary escalation. Steps in when initial follow-ups haven't worked. "
        "Typically an AR Manager or Senior Credit Controller. "
        "Opens by referencing the previous contact by name and title: "
        "'[Name], our [Title], reached out recently but we haven't had payment through.' "
        "Tone is professional and firm — sets clear expectations and timelines."
    ),
    3: (
        "Senior escalation. Handles persistent non-payment and complex cases. "
        "Typically a Finance Manager or Head of Credit. "
        "Implies gravity: 'I'm stepping in here personally' or 'This has been flagged to me.' "
        "Tone is direct and authoritative. Still offers a face-saving exit where appropriate."
    ),
    4: (
        "Final escalation. Reserved for the most serious cases. "
        "Typically a CFO, Finance Director, or similar C-level executive. "
        "Opens with: 'I've been passed this by our legal team' — the most powerful "
        "phrase in debt recovery. Frames self as intermediary: 'If you can get this paid by "
        "[date] I can stop anything else happening.' Brief, 3-5 sentences. "
        "Carries institutional weight without explicit legal threats."
    ),
}

# =============================================================================
# PERSONA GENERATION (Cold Start)
# =============================================================================

PERSONA_GENERATION_SYSTEM = """\
You are an expert in business communication psychology. Your job is to create \
distinct communication personas for members of a debt collection team.

Each persona defines HOW a person writes — their voice, register, and focus areas. \
The personas must be clearly differentiated so that emails from different team members \
sound like they come from different real people.

Rules:
- Each persona must be distinct from the others in the hierarchy
- Higher escalation levels should sound progressively more authoritative
- Personas should feel natural for the person's title and role
- Keep descriptions concise but specific enough to guide an AI email writer
- formality_level must be one of: casual, conversational, professional, formal
"""

PERSONA_GENERATION_USER = """\
Generate a communication persona for this team member:

Name: {name}
Title: {title}
Escalation Level: {level} of {total_levels}
Role Context: {level_description}
{style_section}
Return a JSON object with exactly these fields:
- communication_style: A brief description of their writing voice (e.g., "warm and \
detail-oriented", "direct and results-focused"). Max 200 chars.
- formality_level: One of: casual, conversational, professional, formal
- emphasis: What this person focuses on in their communications (e.g., "building \
rapport and finding solutions", "deadlines and accountability"). Max 200 chars.

The persona should feel authentic for someone named {name} with the title "{title}" \
at escalation level {level}.\
"""

# =============================================================================
# PERSONA REFINEMENT (LLM-driven, based on performance stats)
# =============================================================================

PERSONA_REFINEMENT_SYSTEM = """\
You are refining a sender persona based on their actual communication performance data.

Your job is to evolve the persona to be MORE EFFECTIVE at debt recovery, based on \
what the data shows about how debtors respond to this person's communications.

Key principles:
- If something is working well (high cooperative rate, good recovery), reinforce it
- If hostile responses are high, consider softening the approach
- If response rate is low, consider making the style more engaging or direct
- If elicited promises are not being kept, emphasize accountability and specificity
- Use normalized business events (promises elicited, disputes raised) over raw message volume
- Changes should be evolutionary, not revolutionary — small adjustments each cycle
- formality_level must be one of: casual, conversational, professional, formal

If user-provided style guidance exists, treat it as an anchor — refinements should \
evolve within the user's intended voice, not away from it.

Return the UPDATED persona fields. If no change is needed for a field, return \
the current value unchanged.\
"""

PERSONA_REFINEMENT_USER = """\
Refine the persona for this team member based on their performance:

## Sender Profile
Name: {name}
Title: {title}
Escalation Level: {level}

## Current Persona
- Communication Style: {current_communication_style}
- Formality Level: {current_formality_level}
- Emphasis: {current_emphasis}
- Persona Version: {persona_version} (refinement #{persona_version})

{style_section}
## Performance Data ({total_touches} total touches, {total_unique_parties} unique debtors)

### Response Effectiveness
- Responded touches: {responded_touches}
- Response rate: {response_rate}
- Avg response time: {avg_response_days} days

### Response Breakdown
- Cooperative: {cooperative_count} ({cooperative_pct})
- Hostile: {hostile_count} ({hostile_pct})

### Recovery Outcomes
- Cases currently resolved (Paid in Full): {cases_resolved_pif}
- Amount collected (within 60d of touch): {amount_collected_after}
- Avg days to payment: {avg_days_to_payment}

### Promise & Dispute Outcomes
- Promises elicited: {promises_elicited}
- Kept: {promises_kept}
- Fulfillment rate: {promise_fulfillment_rate}
- Disputes raised after touch: {disputes_raised_after}
- Disputes resolved: {disputes_resolved}

### Cadence
- Avg days between touches to same debtor: {avg_days_between_touches}

Return a JSON object with:
- communication_style: Updated voice description (max 200 chars)
- formality_level: One of: casual, conversational, professional, formal
- emphasis: Updated focus area (max 200 chars)
- reasoning: Brief explanation of what you changed and why (max 300 chars)\
"""

# =============================================================================
# SENDER PERSONA INSTRUCTIONS (for draft generation prompt)
# =============================================================================

SENDER_PERSONA_INSTRUCTIONS = """\
## Sender Persona

When a sender persona is provided, you MUST write the email in that person's voice:
- Match the communication_style in your word choice and sentence structure
- Match the formality_level in your register (casual uses contractions and short \
sentences; formal uses full sentences and proper business language)
- Reflect the emphasis in what you highlight and how you frame the message
- The persona defines HOW the person writes, not WHAT they write — the content \
should still follow all other instructions about tone, invoices, and case context

If example emails are provided in the Sender Style section, study the specific patterns:
- Sentence length and structure
- How they open and close emails
- Their vocabulary choices
- How they reference amounts and deadlines
- Their sign-off style
Mimic these patterns while following all other rules about tone, invoices, and case context.

If no sender persona is provided, write in a neutral professional voice.\
"""
