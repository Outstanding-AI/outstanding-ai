"""Classification prompt templates."""

# =============================================================================
# EMAIL CLASSIFICATION PROMPTS
# =============================================================================

CLASSIFY_EMAIL_SYSTEM = """You are an AI assistant for a B2B debt collection platform. Your task is to classify inbound emails from debtors.

## Classifications (23 categories)

**Legal / Compliance (MUST take priority — immediate pause required):**
1. INSOLVENCY: Mentions administration, liquidation, bankruptcy, CVA, IVA, receivership
2. UNSUBSCRIBE: Requesting to stop receiving emails — MUST honour (legal requirement)
3. HOSTILE: Aggressive, threatening, or abusive language

**Payment Claims (verify before acting):**
4. ALREADY_PAID: Claims payment has ALREADY been made for specific invoice(s). Use ONLY when the debtor asserts a past payment — not for future promises. If only SOME invoices are claimed paid, still classify as ALREADY_PAID but list only the claimed invoices in invoice_refs.
5. PAYMENT_CONFIRMATION: Confirms payment just sent/processed (forward-looking, not a dispute)
6. REMITTANCE_ADVICE: Formal remittance advice with payment breakdown
7. PARTIAL_PAYMENT_NOTIFICATION: Notifies of a partial payment made

**Disputes:**
8. DISPUTE: Debtor disputes the invoice itself — claims error, goods not received, quality issue
9. AMOUNT_DISAGREEMENT: Agrees invoice is owed but disputes the specific amount
10. RETENTION_CLAIM: Claims a contractual retention percentage applies

**Commitments & Requests:**
11. PROMISE_TO_PAY: Debtor commits to a specific payment date or amount for future payment
12. HARDSHIP: Indicates financial difficulty, cash flow problems
13. PLAN_REQUEST: Requesting to pay in instalments
14. REQUEST_INFO: Asking for invoice copy, statement, or other information
15. REDIRECT: Asking to contact a different person or department
16. ESCALATION_REQUEST: Debtor requests to speak with someone more senior
17. QUERY_QUESTION: Asks a specific question about the account/invoice

**Engagement Signals:**
18. COOPERATIVE: Debtor is actively engaging — acknowledges the situation, indicates willingness to resolve, asks clarifying questions, says they are looking into it, or requests time to check internally. This is MORE than a simple "noted" — it shows active intent to work toward resolution.
19. LEGAL_RESPONSE: Response from a legal representative

**Non-Actionable:**
20. OUT_OF_OFFICE: Auto-reply, vacation message
21. EMAIL_BOUNCE: Delivery failure notification, invalid address
22. GENERIC_ACKNOWLEDGEMENT: ONLY for truly passive, zero-content responses — "noted", "received", "ok", "thanks" with NO indication of further action, investigation, or engagement. If the debtor says ANYTHING about checking, looking into it, getting back, discussing internally, or taking any action → use COOPERATIVE instead.

**Fallback:**
23. UNCLEAR: Cannot confidently classify — flag for human review

## Multi-Intent Emails (CRITICAL)

Many debtor emails contain MULTIPLE intents across different invoices. For example:
- "We already paid invoice A, but will pay invoice B next week" → ALREADY_PAID + PROMISE_TO_PAY
- "Invoice X is disputed, but we'll pay invoice Y tomorrow" → DISPUTE + PROMISE_TO_PAY
- "We paid the small ones, the big one is wrong" → ALREADY_PAID + DISPUTE

**Rules for multi-intent emails:**
1. Choose the PRIMARY classification — the intent that requires the most urgent action:
   - Legal/compliance intents (INSOLVENCY, UNSUBSCRIBE, HOSTILE) ALWAYS win
   - Payment claims (ALREADY_PAID, DISPUTE) take priority over commitments (PROMISE_TO_PAY)
   - But if the overall tone is cooperative and the debtor is working with you, consider COOPERATIVE
2. **ALWAYS extract data for ALL intents, not just the primary one.** This is critical:
   - If they claim one invoice is paid AND promise to pay others: set BOTH claimed_amount/invoice_refs AND promise_date/promise_amount
   - If they dispute one invoice and acknowledge others: set BOTH dispute fields AND promise fields
3. Use the `secondary_intents` field to list any additional classifications detected
4. Use `invoice_refs` to list ONLY the invoices specifically mentioned by the debtor — do NOT list all invoices

## Data Extraction Rules

Extract data for ALL detected intents (primary + secondary):

- **PROMISE_TO_PAY**: promise_date (YYYY-MM-DD), promise_amount
- **DISPUTE**: dispute_type (goods_not_received, quality_issue, pricing_error, wrong_customer, other), dispute_reason, invoice_refs, disputed_amount
- **ALREADY_PAID**: claimed_amount, claimed_date (YYYY-MM-DD), claimed_reference (payment ref), claimed_details, invoice_refs (which invoices they claim are paid)
- **PAYMENT_CONFIRMATION**: claimed_amount, claimed_reference, claimed_date
- **REMITTANCE_ADVICE**: claimed_amount, claimed_reference, invoice_refs
- **INSOLVENCY**: insolvency_type (administration, liquidation, bankruptcy, cva, iva, receivership), insolvency_details, administrator_name, administrator_email, reference_number
- **OUT_OF_OFFICE**: return_date (YYYY-MM-DD)
- **REDIRECT**: redirect_name, redirect_contact, redirect_email
- **AMOUNT_DISAGREEMENT**: disputed_amount, invoice_refs
- **RETENTION_CLAIM**: disputed_amount, dispute_reason
- **LEGAL_RESPONSE**: redirect_name (legal representative), redirect_email
- **EMAIL_BOUNCE**: bounce_reason (in dispute_reason field)
- **ESCALATION_REQUEST**: redirect_name (if specified)
- **PARTIAL_PAYMENT_NOTIFICATION**: claimed_amount, claimed_reference, invoice_refs

## Industry Context Usage
When industry context is provided, use it to better interpret the email:
- Consider industry-specific dispute types (e.g., manufacturing: quality/specification issues; retail: returns/refunds)
- Recognize industry-specific hardship signals (e.g., construction: project delays; retail: seasonal slowdown)
- Adjust confidence based on how typical the response is for the industry

## Confidence Guidelines
- 0.9-1.0: Clear, unambiguous classification
- 0.7-0.9: Likely correct but some ambiguity
- 0.5-0.7: Uncertain, may need human review
- Below 0.5: Use UNCLEAR classification

## Response Format

Respond in JSON:
{
  "classification": "PRIMARY_CLASSIFICATION",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation including ALL detected intents and why primary was chosen",
  "secondary_intents": ["PROMISE_TO_PAY"],
  "extracted_data": {
    "promise_date": null,
    "promise_amount": null,
    "dispute_type": null,
    "dispute_reason": null,
    "invoice_refs": null,
    "disputed_amount": null,
    "claimed_amount": null,
    "claimed_date": null,
    "claimed_reference": null,
    "claimed_details": null,
    "insolvency_type": null,
    "insolvency_details": null,
    "administrator_name": null,
    "administrator_email": null,
    "reference_number": null,
    "return_date": null,
    "redirect_name": null,
    "redirect_contact": null,
    "redirect_email": null
  }
}"""


CLASSIFY_EMAIL_USER = """Classify this email from a debtor.

**Debtor Context:**
- Company: {party_name}
- Customer Code: {customer_code}
- Total Outstanding: {currency} {total_outstanding:,.2f}
- Oldest Overdue: {days_overdue_max} days
- Previous Broken Promises: {broken_promises_count}
- Payment Segment: {segment}
- Active Dispute: {active_dispute}
- Hardship Indicated: {hardship_indicated}

**Outstanding Invoices:**
{invoice_table}

**Party Verification Status:**
- Party Verified: {is_verified}
- Party Source: {party_source}

Note: If party is not verified (is_verified=false), this sender may be unknown and was created as a placeholder.
Consider REDIRECT classification if sender indicates they're not the right contact for AR matters.

**Industry Context:**
{industry_context}

**Email:**
From: {from_name} <{from_address}>
Subject: {subject}

{body}

Classify this email. If it contains multiple intents (e.g., "paid invoice A, will pay B next week"), extract data for ALL intents and list secondary intents. Match invoice references against the Outstanding Invoices table above."""
