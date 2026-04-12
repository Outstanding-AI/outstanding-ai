# Outstanding AI Engine

Stateless AI microservice powering intelligent debt collection workflows.

## Project Identity

- **Purpose**: Email classification (23 categories), draft generation (5 tones), compliance gates, persona management
- **Stack**: FastAPI + LangChain with Gemini (primary) / OpenAI (fallback) / Anthropic (optional)
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
# Required
GEMINI_API_KEY=<key>
GEMINI_MODEL=gemini-2.5-pro

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
| LLM API keys | `.env` file | AWS Secrets Manager |
| Deployment | Docker Compose on localhost | ECS Fargate |
| Log level | `INFO` | `WARNING` |
| `IDLE_SHUTDOWN_SECONDS` | Not set (disabled) | e.g. `300` — background watchdog sends SIGTERM after idle period |
| ECS health check | N/A | `/ping` (NOT `/health` — `/health` calls Gemini and burns quota) |
| Auth bypass paths | `_PUBLIC_PATHS` in `src/api/middleware.py` | `/health`, `/ping`, `/docs`, `/openapi.json`, `/redoc` |

**ECS Fargate idle shutdown**: When `IDLE_SHUTDOWN_SECONDS` > 0, a background watchdog thread monitors time since last request and sends SIGTERM after the idle period expires. This allows the AI Engine container to shut down when unused, reducing Fargate costs.

## Related Repos

| Repo | Path | Relation |
|------|------|----------|
| Outstanding AI (Django) | `../Solvix` | Backend — calls AI Engine via HTTP, circuit breaker |
| solvix-etl | `../solvix-etl` | ETL — no direct integration |
| solvix_frontend | `../solvix_frontend` | Frontend — no direct integration |

## Escalation Protocol V2 Support

- `GenerateDraftRequest` has `escalation_level` (int 0-4) and `allowed_tones` (list[str]) fields
- Generator passes `tone`, `escalation_level`, `allowed_tones` to guardrail pipeline as kwargs
- `ToneClampingGuardrail` (7th guardrail, HIGH severity): validates tone is in `allowed_tones` for the level
- Level 0 prompt section: template-like reminders, team sign-off, no persona, factual subjects
- L0→L1 handoff narrative: "Our accounts team has been in touch..." (references generic mailbox as team)
- Escalation history builder labels Level 0 senders as "Accounts Team (automated reminders)"
- `is_generic_mailbox` on `SenderPersona`: skips personal voice, uses team-oriented language

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
