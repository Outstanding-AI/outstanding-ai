# Outstanding AI Engine

Stateless AI microservice powering intelligent debt collection workflows.

## Project Identity

- **Purpose**: Email classification (23 categories), draft generation (5 tones), persona management. Gate evaluation lives in the Django backend (`services/gate_checker.py`); the AI Engine no longer evaluates gates.
- **Stack**: FastAPI + Vertex AI (`google-genai`) primary / OpenAI fallback / Anthropic optional
- **Port**: 8001

## Directory Structure

```
solvix-ai/
├── src/
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── models/          # Pydantic request/response models
│   │   │   ├── requests/    # Request models package (context, party, persona, validation)
│   │   │   └── responses.py
│   │   ├── routes/          # classify, generate, persona, health
│   │   ├── middleware.py    # RequestID + ServiceAuth
│   │   └── errors.py       # Custom exceptions
│   ├── config/
│   │   ├── settings.py      # Pydantic Settings from .env
│   │   └── constants.py     # Tone rules, voice principles
│   ├── engine/
│   │   ├── classifier.py    # Email classification
│   │   ├── generator.py     # Draft generation (orchestration: _assemble_prompt → _run_llm_with_guardrails → _build_response)
│   │   ├── generator_prompts.py # Prompt builders for draft generation
│   │   ├── formatters.py    # Shared formatting utilities
│   │   └── persona.py       # Persona generation/refinement
│   ├── guardrails/          # 7 parallel validators (includes ToneClampingGuardrail)
│   ├── llm/                 # Provider factory + implementations
│   ├── prompts/             # LLM prompt templates
│   ├── evals/               # LLM evaluation framework (batch.py, metrics.py, realtime.py)
│   └── utils/               # JSON extraction helpers
├── tests/                   # 75 tests
└── .claude/rules/           # Path-scoped context (see below)
```

## Context Loading (Path-Scoped Rules)

Domain knowledge loads automatically via `.claude/rules/` when working on matching paths:

| Rule | Paths | Content |
|------|-------|---------|
| `llm-providers.md` | src/llm/**, src/config/** | Provider hierarchy, model config, fallback chain |
| `guardrails.md` | src/guardrails/** | 7 validators (includes ToneClampingGuardrail), severity, follow-up exception |
| `classification.md` | src/engine/classifier.py | 23 categories, extraction, multi-intent |
| `generation.md` | src/engine/generator.py, src/engine/generator_prompts.py, src/engine/formatters.py | Tones, greeting, conciseness, voice rules |
| `api-routes.md` | src/api/** | Endpoints, schemas, middleware |
| `docs-reference.md` | docs/** | Index to API reference and cross-repo contracts |

File navigation: see `CODEBASE_MAP.md`. Detailed API reference: see `docs/API_REFERENCE.md`.

## Common Commands

```bash
# Development
make dev                    # Start FastAPI dev server
uvicorn src.main:app --reload --port 8001

# Testing
pytest tests/ -v --tb=short   # 75 tests
pytest tests/ -x               # Stop on first failure

# Linting
.venv/bin/ruff check src/
.venv/bin/ruff check --fix src/
```

## Environment Variables

```bash
# Required for Vertex AI (primary)
VERTEX_PROJECT_ID=production-493814
VERTEX_LOCATION=europe-west2
VERTEX_MODEL=gemini-2.5-flash
VERTEX_WIF_CONFIG_PATH=/app/infra/vertex-wif-config.json
# No static Google credentials — auth is Workload Identity Federation from the
# AWS ECS task role -> GCP STS -> Google SA impersonation. WIF runtime also needs
# AWS_CONTAINER_CREDENTIALS_RELATIVE_URI (auto-set on ECS) + AWS_REGION.

# Fallback
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-5-mini

# Optional
ANTHROPIC_API_KEY=<key>
ANTHROPIC_MODEL=claude-sonnet-4-20250514
ANTHROPIC_CLASSIFICATION_MODEL=claude-haiku-4-5-20251001

# Common
LLM_TEMPERATURE=0.3
SERVICE_AUTH_TOKEN=<token>
CORS_ORIGINS=http://localhost:8000
```

## Local vs Production

| Setting | Local | Production |
|---------|-------|-----------|
| `SERVICE_AUTH_TOKEN` | Empty (disabled) | Required (AWS Secrets Manager) |
| Vertex auth | ADC / local creds | AWS task role + WIF config file |
| Deployment | Docker Compose on localhost | ECS Fargate |
| Log level | `INFO` | `WARNING` |
| `IDLE_SHUTDOWN_SECONDS` | Not set (disabled) | e.g. `300` — background watchdog sends SIGTERM after idle period |
| ECS health check | N/A | `/ping` (NOT `/health` — `/health/llm` burns provider quota) |
| Auth bypass paths | `_PUBLIC_PATHS` in `src/api/middleware.py` | `/health`, `/ping`, `/docs`, `/openapi.json`, `/redoc` |

**ECS Fargate idle shutdown**: When `IDLE_SHUTDOWN_SECONDS` > 0, a background watchdog thread monitors time since last request and sends SIGTERM after the idle period expires. This allows the AI Engine container to shut down when unused, reducing Fargate costs.

## Related Repos

| Repo | Path | Relation |
|------|------|----------|
| Outstanding AI (Django) | `../Solvix` | Backend — calls AI Engine via HTTP, circuit breaker |
| solvix-etl | `../solvix-etl` | ETL — no direct integration |
| solvix_frontend | `../solvix_frontend` | Frontend — no direct integration |

## Lane-Only Escalation Protocol (April 2026)

- `GenerateDraftRequest` carries `collection_lane.tone_ladder: list[str]` as the deterministic per-touch ladder for the current level.
- Backend picks the exact `tone` from that ladder using `scheduled_touch_index` and passes that concrete tone to the AI request.
- AI does not choose a different tone. It uses `scheduled_touch_index`, `max_touches_for_level`, `last_reply_classification`, suppression history, `days_since_last_touch`, and `lane_history[]` to vary urgency/content framing while staying inside the backend-selected tone.
- `ToneClampingGuardrail` (7th guardrail, HIGH severity): validates the generated copy stays aligned with the requested tone and inside the current level's ladder.
- Ack drafts are tone-locked to `acknowledgement` regardless of level.
- Deleted upstream: `escalation_level` + `allowed_tones` as top-level request fields, AI-chosen within-range tone selection, `clamp_tone()`, `min_gap_days`.
- Level 0 prompt section: template-like reminders, team sign-off, no persona, factual subjects.
- L0→L1 handoff narrative: "Our accounts team has been in touch..." (references generic mailbox as team).
- Escalation history builder labels Level 0 senders as "Accounts Team (automated reminders)".
- `is_generic_mailbox` on `SenderPersona`: skips personal voice, uses team-oriented language.

## Communication Tracking Context (April 2026)

`GenerateDraftRequest.communication_tracking` (optional) conveys tracked-thread state from Django backend so the AI can calibrate continuity claims:

- `tracking_status`: `tracked | degraded | manual_only | closed`
- `tracking_reason`: `send_unconfirmed | visibility_lost | conversation_collision | reply_anchor_unresolved`
- `send_confirmation_state`: `pending | confirmed | unknown`
- `reply_anchor_email`: monitored Reply-To mailbox
- `is_ai_tracked_thread`: whether the thread was AI-originated and enrolled at push time

**Prompt rule**: Continuity language ("following up on your last reply", "as I mentioned in my previous message") is allowed ONLY when `tracking_status='tracked'` AND send is confirmed. If `degraded` or `manual_only`, the prompt must not assert continuity unless the actual prior message text is explicitly present in the rendered context. Claiming a reply we didn't observe = hallucination.

**Why**: BCC/shared-mailbox transport rules are tenant-side and unreliable. The backend can legitimately hold drafts whose send is unconfirmed — the AI must not overclaim chronology when the control plane says we're flying blind.

## Collection Lane Context (April 2026 — Single-Lane Only)

`GenerateDraftRequest` carries a `collection_lane` / `lane_context` block for every collection-mode draft. Runtime is lane-only — no bundling, no owner/guest semantics, no replacement flow.

Fields:
- `collection_lane_id` — UUID of the lane this draft represents.
- `current_level` / `entry_level` — lane's escalation level.
- `scheduled_touch_index` / `max_touches_for_level` / `reminder_cadence_days_for_level` / `max_days_for_level` — cadence state.
- `tone_ladder: list[str]` — deterministic per-touch ladder for the current level. Runtime has already selected the exact `tone` for the draft from this ladder.
- `invoice_refs[]` — obligation references in this lane's cohort.
- `outstanding_amount` — sum of open obligations in cohort.
- `lane_history[]` — last N `CollectionLaneEvent` rows (mail pushes + replies) for prompt continuity.
- `last_reply_classification` — most recent inbound intent for this lane, if any.

`mail_mode` on the request: `initial | reminder | escalation | ack | handoff_reply`.

**Prompt rules**:
- AI never chooses sender, level, suppression state, thread, invoice scope, or tone. Those are backend-owned.
- For `mail_mode="reminder"` with `scheduled_touch_index > 1`: reference prior unanswered outreach using `lane_history`; increase urgency through content (approaching escalation, consequences), not by switching away from the backend-selected tone.
- For `mail_mode="escalation"`: `scheduled_touch_index` has just reset to 1 and backend has already selected the new level's first tone — frame as a new sender escalation, not a continuation.
- For `mail_mode="ack"`: tone-locked to `acknowledgement`; one-shot acknowledgement, no collection asks.
- For `mail_mode="handoff_reply"`: acknowledge the redirect/new-contact request; sender has just changed but the lane ID stays the same.

Full contract: backend `docs/CONTRACTS.md` section 5 (Tone Contract) + section 8 (App DB Contract).

## CaseContext Schema Versioning (May 2026 — Silver Application transition)

Every `CaseContext` in every request (`/classify`, `/generate-draft`) accepts `schema_version: Literal[2, 3, 4]`. v1 (legacy Sage-keyed) has been **retired**.

- **`schema_version=2` / `3`** — canonical provider-agnostic contexts. Requires on every obligation: `id` (UUID) + `external_id` + `provider_type`. Requires on every party: `external_id` + `provider_type`.
- **`schema_version=4`** — Silver Application current-context transition payload. Requires draft lineage, a valid recipient email, and at least one sendable/chase-eligible obligation for normal draft generation.

`ObligationInfo.sage_id` field was **removed** from the Pydantic model.

**Lane-scope guardrail** (`src/guardrails/lane_scope.py:44-65`) unconditionally looks up blocked obligation IDs by `obligation.id` (canonical UUID). The schema_version=1 branch (`obligation.sage_id` lookup) has been deleted.

**Silver Application prompt boundary**: `CaseContext.uses_current_datalake_contract()` is strict to `return self.schema_version == 4` (`src/api/models/requests/context.py`). The previous `or any(source_sync_run_id, application_run_id, …)` short-circuit was deleted — opportunistic lineage fields on v2/v3 payloads must not trigger v4 prompt sections. V2/V3 traffic continues to flow through the unchanged code paths and is fully backwards-compatible.

**V4 additive fields** (verified at HEAD in `src/api/models/requests/context.py`):
- On `CaseContext`: `context_version`, `source_sync_run_id`, `application_run_id`, `core_snapshot_watermark`, `application_snapshot_watermark`, `application_decision_cutoff`, `input_silver_version_ids_json`, `input_silver_version_ids`, `policy_snapshot_id`, `draft_candidate_id`, `draft_generation_run_id`, `collection_basis`, `chase_basis`, `total_outstanding_amount`, `total_overdue_amount`, `outstanding_invoice_count`, `overdue_invoice_count`, plus `*_current` projection dicts (`party_communication_state_current`, `party_collection_state_current`, `party_behavior_profile_current`, `party_verification_state_current`, `obligation_collection_status_current`, `verification_tasks_current`, `payment_verifications_current`, `payment_verification_obligations_current`, `promise_history_current`, `promise_obligations_current`, `dispute_history_current`, `dispute_obligations_current`, `insolvency_history_current`, `sender_selection_events_current`, `recipient_selection_events_current`, `sender_performance_current`, `excluded_source_disputed_obligations`).
- On `ObligationInfo`: `silver_version_id`, `document_no`, `sage_transaction_urn`, `document_currency_code`, `is_outstanding`, `is_overdue`, `days_overdue`, `effective_grace_days`, `is_chase_eligible`, `source_query_raw`, `has_source_query_flag`, `is_source_disputed`, `source_dispute_type`, `source_dispute_observed_from`, `has_verified_purchase_order`, `has_verified_pod`, `procurement_context_status`, `purchase_order_reference`, `pod_reference`.
- `GenerateDraftRequest.validate_current_datalake_context` (in `src/api/models/requests/validation.py`) fails closed when `schema_version == 4` if any required lineage field is missing, no valid recipient email is available, or no eligible/sendable obligation passes the local `_is_sendable_candidate` mirror. The mirror now honors `context.sendable_obligation_ids` (May 2026 fix): when the upstream caller has populated that whitelist, only obligations whose `id` appears in it count as eligible — keeping the validation gate aligned with engine-side candidate selection. Outside that whitelist, the mirror still applies the same blocked/sendable/chase/source-dispute predicates as the engine.

## Guardrail Tightening (May 2026, schema_version=4)

The four content guardrails were tightened to use V4 fields when present (and remain backwards-compatible on V2/V3):

- **`lane_scope.py`** — accepts `kwargs["candidate_invoice_refs"]` for explicit candidate scoping. Reference matching is whitespace/symbol-insensitive via `_invoice_ref_variants`. Bare-digit fallback variants are now only added for cohort entries whose normalized form is itself digit-only; prefixed cohort entries (e.g. `INV-12345`) route through a length-equal bare-digit lookup so a body extraction of `1234` cannot collide with cohort `12345` (digit-prefix collision). Obligations marked `is_source_disputed`, `is_sendable=False`, `is_chase_eligible=False`, or carrying a non-empty `source_query_raw` are added to `blocked_invoice_refs` automatically.
- **`factual_grounding.py`** — adds `_validate_source_disputes_not_chased` (blocks chase wording targeting source-disputed invoices) and `_validate_procurement_grounding` (PO/POD claims require `has_verified_purchase_order`/`has_verified_pod=True`). The chase detector is now `_chases_invoice_ref(output, ref)`, which splits the draft into segments via `_segments` and only flags when chase wording and the invoice ref appear in the **same segment**. PO false-positive fix (May 2026): the procurement-claim regex now matches only the explicit forms `purchase order` or `po <number/ref/reference/#>` (e.g. `PO #123`, `PO ref 4567`) — bare `po` standalone (which previously fired on "PO Box") no longer triggers. Note: the source-disputed chase check still relies on segment-level chase-language proximity, so an informational mention of a disputed invoice in the same sentence as generic chase wording can still false-positive — this is a deliberate narrow surface, not a regression.
- **`numerical.py`** — `_validate_total_calculation` and `_validate_days_overdue` now operate on `_scoped_obligations(context, kwargs)`. When V4 is in use, totals are summed against `obligation.amount_due_base` (or `original_amount_base`) rather than `amount_due`; days-overdue checks prefer `obligation.days_overdue` over `days_past_due`.
- **`temporal.py`** — `_validate_promise_date_is_future` now uses `_decision_date(context)` derived from `context.application_decision_cutoff` instead of `date.today()`. Timezone-handling fix (May 2026): `_decision_date` now normalizes a `datetime` cutoff to UTC before calling `.date()` (assumes UTC for naïve inputs, then `.astimezone(timezone.utc).date()`). Same logic for ISO-string inputs. The previous implementation called `cutoff.date()` directly, dropping timezone information and collapsing tz-aware cutoffs to the local naïve date.

## AI Audit Metadata (May 2026)

`src/engine/audit.py` introduces `build_ai_audit(...)` which assembles an `AIAuditMetadata` (defined in `src/api/models/responses.py`) attached to `ClassifyResponse` and `GenerateDraftResponse`. Captures: `ai_provider`, `ai_model`, `ai_region` (Vertex location only), `prompt_template_id` + `prompt_template_version`, hashes of system + user prompts and the JSON-serialized `prompt_input` payload, `guardrail_pipeline_version`, lineage IDs (`policy_snapshot_id`, `draft_candidate_id`, `draft_generation_run_id`, `source_sync_run_id`, `application_run_id`), `input_silver_version_ids_json`, and token/latency metrics.

**WARNING — PII in prompt_input**: the `prompt_input` payload passed into `hash_payload` is a raw structure that may contain debtor names, email addresses, invoice numbers, and other PII. Only the SHA-256 hash is persisted on `AIAuditMetadata.prompt_input_hash`; do **not** add any code path that logs the unhashed `prompt_input` to CloudWatch, telemetry events, or audit tables. If a regression starts emitting raw `prompt_input` in logs, treat as a privacy incident.

## Mixed-Reply `intent_details` Rule (May 2026, prompt + validator)

`src/prompts/classification.py` instructs the LLM that for any debtor email containing multiple material intents, **every intent must carry its own `intent_details[*].extracted_data`** — invoice references must NOT be copied across intents (e.g. PROMISE_TO_PAY's `invoice_refs` must be the promised invoices only, not the disputed ones).

**Enforcement**: a `model_validator(mode="after")` named `validate_intent_details_scope` on `ClassificationLLMResponse` (`src/llm/schemas.py`) now enforces the rule at parse time. The validator:

1. Asserts `intent_details[0].intent` matches the top-level `classification`.
2. Rejects unknown intents (must be in `CLASSIFICATION_CATEGORIES`).
3. For any **secondary** entry whose intent is in the `MATERIAL_SCOPE_INTENTS` allow-list (`ALREADY_PAID`, `AMOUNT_DISAGREEMENT`, `DISPUTE`, `HARDSHIP`, `PARTIAL_PAYMENT_NOTIFICATION`, `PAYMENT_CONFIRMATION`, `PLAN_REQUEST`, `PROMISE_TO_PAY`, `REMITTANCE_ADVICE`, `RETENTION_CLAIM`), requires non-null `extracted_data`.
4. Rejects an `invoice_ref` that appears in more than one entry's `extracted_data.invoice_refs`.

If the LLM emits cross-intent invoice leakage or omits per-intent extraction for a material secondary intent, parsing fails with a `ValueError` and the classifier falls back to its retry/error path — invariant violations no longer pass through silently. Tests live in `tests/test_llm_schemas.py`.

**Telemetry**: classify/generate routes log the actual `schema_version` in every event (start/success/error).

**Long-term home**: backend-owned `solvix-contracts` package (`Solvix/contracts/src/solvix_contracts/ai/context.v2`) exports the shared v2 core while AI carries additive v3/v4 transition fields locally. Backend and AI are aligned to **`solvix-contracts==0.11.0`**; parity CI at `.github/workflows/contracts-version-parity.yml` should block any joint deploy if the AI pin drifts again.

## LLM Runtime Invariants (April 2026)

Non-obvious gotchas — violating these breaks production first-sync draft generation.

1. **Vertex client is per-call, not singleton.** `VertexProvider.complete()` builds a fresh `genai.Client(...)` inside the method body. The module-level singleton (the original shape) bound anyio `Event`/`Lock` primitives to whichever event loop first used the client. Guardrails run on a ThreadPoolExecutor with fresh event loops per worker thread — sharing the Client across loops produced `"Event is bound to a different event loop"` and `"Event loop is closed"` RuntimeErrors that the old `basicConfig` formatter silently dropped.
2. **Always `await client.aio.aclose()` in `finally`.** Per-call Clients leak an httpx connection pool each invocation without this. Close exceptions are swallowed to a WARN log so they don't mask the real call result.
3. **Every LLM call carries `caller="..."` kwarg.** `BaseLLMProvider.complete` takes `caller` as a keyword-only parameter; `LLMProviderWithFallback.complete` propagates it to both primary + fallback; all call sites (draft_generation, classification, persona_generation, persona_refinement, entity_verification, health_check) tag their requests. Metrics, `_primary_failures_by_caller` counter, and CloudWatch error logs all carry `caller` for attribution.
4. **`is_fallback: bool` on `LLMResponse`.** Factory sets `response.model_copy(update={"is_fallback": True})` on the fallback path; surfaces in logs as `used_fallback=true`.
5. **Provider error logs use `exc_info=True`** plus `extra={"caller", "error_type", "structured"}`. CloudWatch now shows the actual exception class + stack instead of "Vertex provider error" on its own.
6. **Logging formatter injects sentinels.** `src/main.py` registers `_DefaultExtrasFilter` that sets `caller/error_type/error` to `"-"` when the LogRecord lacks them — the formatter stays stable across both tagged and untagged records.
7. **Guardrail thread pool is 6 workers for 7 guardrails.** `src/guardrails/executor.py` `_guardrail_executor = ThreadPoolExecutor(max_workers=6)`. The CRITICAL guardrail runs first serially (fail-fast), then the remaining 6 fill the pool in parallel. Old 7-worker comment was wrong and the 7-thread config amplified the cross-loop primitive contention described in #1.
8. **Entity guardrail event-loop hygiene.** Worker runs `asyncio.new_event_loop()` + `set_event_loop(loop)`; `finally` MUST call `asyncio.set_event_loop(None)` before `loop.close()` so the next guardrail on the same worker thread doesn't pick up a dead loop via `get_event_loop()`.
9. **Entity prompt must separate debtor from sender identity.** `ENTITY_VALIDATION_PROMPT` lists EXPECTED DEBTOR ENTITIES and ALLOWED SENDER ENTITIES separately. The generator threads `sender_company`, `sender_name`, `sender_mailbox_name` into the guardrail `kwargs`. Without this split the guardrail flags valid drafts that mention the sender company in sign-offs (e.g. "your account with ESWL") as hallucinated unrelated companies.
10. **No application-level output-token caps.** Do not pass `max_tokens` / `max_output_tokens` in guardrail, draft, classification, or fallback calls. Rely on provider-native limits plus retry/defer behavior; explicit app caps caused `LengthFinishReasonError` and empty Vertex payload failures during first-sync draft generation.
11. **`solvix-contracts` pin parity.** Pin is mandatory for parity with backend + ETL. `PartyInfoV2.source` is a **required** field; semantically locked to canonical `provider_type`. All AI fixtures construct `CaseContext` with explicit `source=provider_type`. Runtime must never synthesize provider identity with hidden `or "sage_200"` / `or "microsoft_365"` fallbacks.

## Skills

Use `/debug-drafts` for guided draft generation debugging (prompt failures, guardrail blocks, LLM fallback).
Use `/debug-classification` for guided classification debugging (wrong category, missing extraction, multi-intent).

## After Code Changes — Keep Context Files in Sync

After any refactoring or feature work, proactively update these files before finishing:

| What changed | Update |
|---|---|
| File added/removed/renamed | `CODEBASE_MAP.md` |
| New/changed Pydantic model, prompt, or guardrail | `.claude/rules/<matching-rule>.md` |
| New API route or LLM provider | `.claude/rules/<matching-rule>.md` + `CODEBASE_MAP.md` + `docs/API_REFERENCE.md` |
| New command or env var | This file (`CLAUDE.md`) |
| New agent or skill | This file (`CLAUDE.md`) |

Rule matching: each `.claude/rules/` file has a `paths:` frontmatter — update the rule whose paths match the changed code.
