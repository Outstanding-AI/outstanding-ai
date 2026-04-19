# Outstanding AI Engine

Stateless AI microservice powering intelligent debt collection workflows.

## Project Identity

- **Purpose**: Email classification (23 categories), draft generation (5 tones), compliance gates, persona management
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
│   │   ├── routes/          # classify, generate, gates, persona, health
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
│   │   ├── gate_evaluator.py # Gate evaluation (DEPRECATED — gates in Django)
│   │   ├── escalation_validator.py # Escalation validation logic
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
| `gates.md` | src/engine/gate_evaluator.py, src/engine/escalation_validator.py | DEPRECATED — gates in Django |
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
# Required for local Vertex work
VERTEX_PROJECT_ID=production-493814
VERTEX_LOCATION=europe-west2
VERTEX_MODEL=gemini-2.5-flash

# Fallback
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-5-nano

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
