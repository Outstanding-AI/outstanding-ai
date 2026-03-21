# Memory vs. Context Enrichment Analysis

> Analysis of whether solvix-ai needs "memory" capabilities for AI agent workflows

**Date**: January 2025 (original), March 2026 (updated)
**Status**: Analysis Complete - Recommendations partially implemented
**Recommendation**: Context enrichment, NOT memory systems

---

## Executive Summary

After analyzing all three repositories (solvix-ai, Solvix, solvix-etl), the answer is clear:

**You don't need "memory" - you need to use the data you already have.**

The email content, thread history, and conversation context already exist in your data lake. The gap is that `ContextBuilder` in Django doesn't include this rich context when calling solvix-ai.

### Progress Since Original Analysis

Since this analysis was written, significant context enrichment has been implemented:

- **IndustryInfo model** added to CaseContext (industry-specific AI context)
- **Extended PartyInfo** with tone_override, grace_days_override, touch_cap_override, do_not_contact_until
- **TouchHistory and PromiseHistory** lists added to CaseContext
- **Sender persona injection** into draft generation prompts
- **Industry-aware gate evaluation** with escalation_patience

What's still missing: thread content (email bodies), dispute details, and payment plan adherence feedback.

---

## Background: The "Memory" Question

The question arose from reviewing common patterns for "agents with memory" that typically involve:
- Vector embeddings for retrieval
- Graph databases for relationships
- LLM calls for memorization
- Cron jobs for decay/consolidation

These patterns solve problems for **conversational chatbots** that need to:
- Remember user preferences across sessions
- Handle context window limits in long conversations
- Resolve contradictory information from different time periods

**solvix-ai is architecturally different** - it's a stateless microservice where:
- Each request (classify, generate-draft, evaluate-gates) is one-shot
- Complete context is passed in every request via `CaseContext`
- Django (Solvix) is the single source of truth for all data

---

## Current Data Architecture

### Email Data Flow
```
Microsoft Graph API (webhooks/batch)
    |
solvix-etl
    |
+-----------------------------------------------------+
| Bronze Layer (S3 Parquet)                           |
| - bronze_inbound_emails (FULL body_plain, body_html)|
| - bronze_outbound_emails (FULL content)             |
| - conversation_id, in_reply_to, references          |
+-----------------------------------------------------+
    |
+-----------------------------------------------------+
| Silver Layer (S3 Parquet)                           |
| - threads (grouped by conversation_id, linked to party)|
| - thread_messages (classification, extracted_* fields) |
| - bronze_email_id -> links to full content          |
+-----------------------------------------------------+
```

---

## Complete Data Audit

### Data That IS Being Passed to solvix-ai

| Category | Data Source | Status |
|----------|------------|--------|
| **Party/Debtor** | `parties` table | Full party context including credit_limit, on_hold |
| **Relationship** | `parties` table | relationship_tier, is_verified, source |
| **Behavior** | `parties` table | lifetime_value, avg_days_to_pay, on_time_rate, segment |
| **Case State** | `parties` table | case_state, days_in_state, broken_promises_count, hardship_indicated |
| **Invoices** | `obligations` table | Top invoices by days_past_due |
| **Communication Summary** | `touches` table | touch_count, last_touch_at, last_tone_used |
| **Recent Touches** | `touches` table | Last N touches with tone, sender_level, had_response |
| **Promises** | `promise_history` | promise_date, amount, outcome (kept/broken/pending) |
| **Tenant Settings** | `tenant_settings` | brand_tone, touch_cap, grace_days, etc. |
| **Party Overrides** | `parties` table | tone_override, grace_days_override, touch_cap_override |
| **Industry Context** | `industry_configs` | **NEW** - Full IndustryInfo with DSO, escalation patience, etc. |
| **Sender Persona** | `escalation_contacts` | **NEW** - communication_style, formality_level, emphasis |
| **Last Response** | `thread_messages` | last_response_at, last_response_type |

### Data That EXISTS But Is NOT Being Passed

| Category | Data Source | What's Missing |
|----------|------------|----------------|
| **Email Bodies** | `bronze_inbound_emails` | Actual text of debtor replies |
| **Outbound Content** | `bronze_outbound_emails` | What you previously sent |
| **Thread Grouping** | `threads` | Thread context and message sequence |
| **Promise Text** | `thread_messages` | "We'll pay GBP 1,000 by Friday" (source text) |
| **Dispute Details** | `thread_messages` | "Claims goods were damaged" (full details) |
| **Invoice Refs** | `thread_messages` | Which invoices debtor mentioned |
| **Redirect Info** | `thread_messages` | Alternate contact provided |
| **Evidence/Receipts** | `evidence` | Payment evidence NOT queried |
| **Allocations** | `allocations` | How payments map to invoices |
| **Payment Plans** | `payment_plans` | Active plan status and adherence |

---

## Impact on Draft Quality

### Current Draft Prompt Sees

```
**Debtor:** ABC Corp (customer_code: ABC001)
**Total Outstanding:** GBP 15,000.00
**Invoices:** INV-001: GBP 5,000 (45 days), INV-002: GBP 10,000 (30 days)
**Communication:** 3 touches, last contact 5 days ago, last tone: professional
**Last Response Type:** DISPUTE
**Broken Promises:** 2
**Industry:** Manufacturing (patient escalation, net60 payment cycle)
**Sender:** Sarah Williams, Credit Controller (warm and detail-oriented)
```

### What the Prompt COULD See (with further enrichment)

```
**Debtor:** ABC Corp (customer_code: ABC001)
**Total Outstanding:** GBP 15,000.00
**Invoices:** INV-001: GBP 5,000 (45 days), INV-002: GBP 10,000 (30 days)

**Conversation History (3 messages):**
1. [Jan 5] YOU: "Dear ABC Corp, we noticed INV-001 is now 30 days overdue..."
2. [Jan 8] DEBTOR: "We're having cash flow issues due to a delayed payment from our client.
   We can pay INV-002 now but need until Jan 20 for INV-001."
3. [Jan 10] YOU: "Thank you for the update. We've noted your commitment..."

**Active Dispute on INV-001:** "Claims goods arrived damaged, requesting credit note"

**Broken Promise #1:** "Committed to pay GBP 5,000 by Jan 20" - OUTCOME: broken
**Broken Promise #2:** "Said cheque was in post on Dec 15" - OUTCOME: broken
```

---

## Recommendation

### Principle: Enrich CaseContext, Don't Add Memory

**Keep solvix-ai stateless.** Extend ContextBuilder in Django to include thread content.

### Remaining Implementation Options

#### Option A: Direct Thread Content (Simple)

Extend ContextBuilder to join `thread_messages` -> `bronze_*_emails` and include recent message bodies.

**Pros**: Simple, uses existing data
**Cons**: Long threads = large context = token cost

#### Option B: Thread Summarization (Recommended for Scale)

Add a summarization step - either in Django or as a new solvix-ai endpoint.

#### Option C: Hybrid (Best of Both)

1. Pre-computed summaries for threads > 3 messages (ETL/Gold layer)
2. Real-time content for last 2-3 messages (ContextBuilder)
3. On-demand summarization for complex threads (solvix-ai endpoint)

---

## What NOT To Do

1. **Don't add vector stores to solvix-ai** - Your data is structured, not semantic
2. **Don't add graph memory** - Relationships are in your relational DB
3. **Don't add cron jobs for memory decay** - ETL handles data lifecycle
4. **Don't store conversation state in solvix-ai** - Django/data lake owns state
5. **Don't embed emails** - Use structured extraction from existing classifications

---

## Summary: Memory vs. Context Enrichment

| "Memory" Approach (Article) | Your Actual Need |
|----------------------------|------------------|
| Vector embeddings for retrieval | Direct SQL queries to existing tables |
| Graph databases for relationships | Foreign keys in S3 Parquet (DuckDB/Athena) |
| LLM calls for memorization | Data already extracted by solvix-ai classification |
| Cron jobs for decay/consolidation | ETL already handles data lifecycle |
| Stateful AI service | Stateless AI with richer context |

**Bottom line**: You have the data. You just need to include it in CaseContext.
