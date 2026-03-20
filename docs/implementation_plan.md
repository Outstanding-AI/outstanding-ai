# Complete Architecture Plan - Outstanding AI Improvements

> **Last Updated**: March 2026
> **Status**: Phases 1-2 partially complete, Phase 3-4 pending

## Existing Infrastructure Analysis

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| **Payment Plans** | `payment_plans` table (ETL) | Schema exists, not yet populated |
| **Plan Instalments** | `plan_instalments` table (ETL) | Schema exists, tracks each payment |
| **Promise Tracking** | `Party.promise_date`, `promise_amount`, `broken_promises_count` | Working |
| **Evidence Linking** | `plan_instalments.evidence_id` -> `evidence` table | Can link payments to instalments |
| **Classification** | `ThreadMessage.extracted_promise_date/amount` | AI extraction working |
| **Industry Type** | `IndustryInfo` model in solvix-ai, industry config in backend | **IMPLEMENTED** |
| **Sender Personas** | `PersonaGenerator` in solvix-ai, escalation_contacts in backend | **IMPLEMENTED** |
| **Gate Evaluation** | Deterministic 6-gate engine (no LLM) | **IMPLEMENTED** |
| **Service Auth** | `ServiceAuthMiddleware` with Bearer token | **IMPLEMENTED** |

### What's Missing

| Component | Gap | Impact |
|-----------|-----|--------|
| **Payment Plan Creation** | No workflow to create from AI offer | AI generates but can't persist |
| **Plan -> Payment Matching** | No ETL job to match Evidence to Instalments | Can't track adherence |
| **AI Memory Updates** | No pipeline for learning from outcomes | AI doesn't improve from feedback |
| **Langfuse Observability** | No tracing/monitoring of LLM calls | Can't measure quality at scale |
| **Tool Calling** | No tool use for real-time data lookups | AI can't check live balances |

---

## The Feedback Loop Problem

```
+-----------------------------------------------------------------------------+
|                        CURRENT STATE (BROKEN LOOP)                          |
+-----------------------------------------------------------------------------+

    AI Generates Draft               User Sends              Debtor Pays?
    ------------------------------------------------------------------------->
    "Pay GBP 500/month for 3 months"    Email sent              ??? (unknown)
                                                                   |
                                                                   v
                                                            Sage Records Payment
                                                                   |
                                                                   v
    <------------------------------------------------------------------
    NO FEEDBACK TO AI                                         Gap here!
    Next email doesn't know about plan

+-----------------------------------------------------------------------------+
|                       DESIRED STATE (CLOSED LOOP)                           |
+-----------------------------------------------------------------------------+

    AI Proposes Plan    User Approves    Plan Created    Instalments Generated
    -------------------------------------------------------------------------->
    "GBP 500/month x 3"    Click "Accept"   payment_plans   plan_instalments (3)
                           |                               | due_date each month
                           v                               v
                        Party.ai_memories <------- Sage Sync <--- Debtor Pays
                        "Plan active, 1/3 paid"     Evidence matches instalment
                           |
                           v
                        Next AI call knows about plan adherence
```

---

## Phase 1: Langfuse Local Setup (Observability)

**Goal**: Get LLM call observability working locally.
**Status**: NOT STARTED

### Implementation

```bash
# Start Langfuse locally
docker-compose -f docker-compose.langfuse.yml up -d

# Access dashboard at http://localhost:3000
# Create API keys in UI
```

#### Files to Create

| File | Purpose |
|------|---------|
| `docker-compose.langfuse.yml` | Local Langfuse stack |
| `src/observability/__init__.py` | Module init |
| `src/observability/tracing.py` | Trace decorators |
| `.env.example` additions | Langfuse config |

#### Integration Points

```python
# In classifier.py and generator.py
from src.observability.tracing import trace_llm_call

@trace_llm_call(name="classify_email")
async def classify(self, request):
    ...
```

---

## Phase 2: Industry Type Capture

**Status**: COMPLETED (solvix-ai side)

### What Was Implemented

1. **`IndustryInfo` model** in `src/api/models/requests.py` with:
   - code, name, typical_dso_days, alarm_dso_days, payment_cycle
   - escalation_patience (patient/standard/aggressive)
   - common_dispute_types, hardship_indicators
   - preferred_tone, ai_context_notes, seasonal_patterns
   - dispute_handling_notes, hardship_handling_notes, communication_notes

2. **Industry context in prompts** - Classification and draft generation prompts now include industry-specific context

3. **Industry-aware gate evaluation** - Escalation gate considers `industry.escalation_patience` to control allowed jump size

### What Remains (Backend/Frontend)

- [ ] Add industry management UI in Settings
- [ ] Add industry dropdown to onboarding wizard
- [ ] Auto-suggest industry from Companies House API (optional)

---

## Phase 3: Payment Plan Feedback Loop

**Status**: NOT STARTED

### Data Flow Design

```
+------------------------------------------------------------------------+
| STEP 1: AI Proposes Plan (solvix-ai)                                   |
+------------------------------------------------------------------------+
|  Draft Generation Response:                                             |
|  {                                                                      |
|    "body": "We can offer a payment plan of GBP 500/month...",          |
|    "proposed_plan": {           <--- NEW FIELD (not yet implemented)    |
|      "total_amount": 1500.00,                                          |
|      "instalment_count": 3,                                            |
|      "instalment_amount": 500.00,                                      |
|      "frequency": "monthly"                                            |
|    }                                                                    |
|  }                                                                      |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
| STEP 2: User Approves Plan (Django Backend)                            |
+------------------------------------------------------------------------+
|  POST /api/parties/{party_id}/payment-plans/                           |
|  -> Creates payment_plans row                                           |
|  -> Creates plan_instalments rows                                       |
|  -> Updates Party.case_state = "payment_plan"                          |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
| STEP 3: Payment Detection (ETL)                                         |
+------------------------------------------------------------------------+
|  ETL Job: match_payments_to_instalments                                 |
|  1. Match Evidence records to plan_instalments                          |
|  2. Mark missed instalments as 'overdue' after grace period            |
|  3. Mark completed plans                                                |
+------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------+
| STEP 4: AI Context Update (ETL + AI)                                    |
+------------------------------------------------------------------------+
|  Party.ai_memories updated with plan adherence data                     |
|  Next AI call includes this context automatically                       |
+------------------------------------------------------------------------+
```

### Required Changes

| Repo | Change | Effort |
|------|--------|--------|
| **solvix-ai** | Add `proposed_plan` to draft response | Low |
| **Outstanding AI (Django)** | Add PaymentPlan model + API endpoints | Medium |
| **solvix-etl** | Add payment matching job | Medium |
| **solvix-etl** | Add AI memory update job | Medium |
| **Frontend** | Plan approval UI in draft editor | Medium |

---

## Phase 4: Tools Implementation

**Status**: NOT STARTED

### Confirmed Tool Requirements

| Tool | Purpose | Data Source |
|------|---------|-------------|
| `get_current_balance` | Real-time balance for guardrails | Sage API or Silver layer |
| `get_payment_plan_status` | Check if plan exists, adherence | `payment_plans` table |
| `calculate_payment_plan` | Generate plan options | Business logic |
| `lookup_invoice` | Get invoice details | `obligations` table |
| `check_contact_allowed` | Regulatory/gate check | `Party` + `TenantSettings` |

### Tool Architecture

```python
# src/tools/base.py
from abc import ABC, abstractmethod

class AITool(ABC):
    """Base class for AI-callable tools."""
    name: str
    description: str
    parameters: dict  # JSON Schema

    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        pass
```

---

## Open Questions

### 1. Payment Plan Approval Flow

**Question**: Where should plan approval happen?

| Option | Description |
|--------|-------------|
| A | In Outlook (user edits draft, adds "PLAN APPROVED" marker) |
| B | In Outstanding AI UI (separate approval step after draft) |
| C | Automatic (if debtor responds with agreement) |

### 2. Payment Matching Tolerance

**Question**: How strict should payment matching be?

| Scenario | Exact Match | +/-5% Tolerance | +/-10% Tolerance |
|----------|-------------|-----------------|------------------|
| Instalment: GBP 500, Payment: GBP 500 | Match | Match | Match |
| Instalment: GBP 500, Payment: GBP 495 | No | Match | Match |
| Instalment: GBP 500, Payment: GBP 450 | No | No | Match |

### 3. AI Memory Storage

**Question**: Since Party is ETL-managed, where should `ai_memories` live?

| Option | Description |
|--------|-------------|
| A | New `party_ai_memories` table in ETL (clean separation) |
| B | JSON field added to `parties` table (simpler, needs ETL migration) |
| C | Separate Django-managed table with FK to Party (more flexibility) |

---

## Implementation Order (Revised)

```
Next:
+-- Phase 1: Langfuse local setup (observability)
+-- Phase 3: Payment plan creation API (backend + AI)

Later:
+-- Phase 4: Tools architecture scaffolding
+-- Phase 4: Implement get_current_balance, lookup_invoice
+-- Phase 3: ETL payment matching job
+-- Phase 3: AI memory updates
```

---

## Completed Work Summary

The following was implemented across recent commits:

1. **Industry Context** (solvix-ai + backend): Full `IndustryInfo` model with 13 fields, integrated into classification/generation prompts and gate evaluation (escalation_patience)

2. **Sender Persona System** (solvix-ai + backend): 4-level escalation hierarchy with cold start generation and performance-based refinement. Personas injected into draft generation prompts.

3. **Deterministic Gate Evaluation**: Replaced LLM-based gate evaluation with 6 deterministic Python gates for reliability and speed. Industry-aware escalation rules.

4. **Service Authentication**: Bearer token middleware for service-to-service calls. Public paths exempt.

5. **Extended Party Model**: Added tone_override, grace_days_override, touch_cap_override, do_not_contact_until, monthly_touch_count to PartyInfo.

6. **Security Hardening**: Prompt injection detection on custom_instructions, rate limiting via slowapi, CORS configuration, max_length constraints on all string fields.

7. **LLM Model Upgrades**: Gemini 2.5 Pro (primary), gpt-5-nano reasoning model (fallback) with appropriate token budgets.

8. **Docker Optimization**: Non-root user, uv cache mounts, graceful shutdown.
