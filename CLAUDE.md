# Solvix AI Engine

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
│   │   ├── routes/          # classify, generate, gates, persona, health
│   │   ├── middleware.py    # RequestID + ServiceAuth
│   │   └── errors.py       # Custom exceptions
│   ├── config/
│   │   ├── settings.py      # Pydantic Settings from .env
│   │   └── constants.py     # Tone rules, voice principles
│   ├── engine/
│   │   ├── classifier.py    # Email classification
│   │   ├── generator.py     # Draft generation
│   │   ├── gate_evaluator.py # DEPRECATED — gates in Django
│   │   └── persona.py       # Persona generation/refinement
│   ├── guardrails/          # 6 parallel validators
│   ├── llm/                 # Provider factory + implementations
│   ├── prompts/             # LLM prompt templates
│   └── utils/               # JSON extraction helpers
├── tests/                   # 75 tests
└── .claude/rules/           # Path-scoped context (see below)
```

## Context Loading (Path-Scoped Rules)

Domain knowledge loads automatically via `.claude/rules/` when working on matching paths:

| Rule | Paths | Content |
|------|-------|---------|
| `llm-providers.md` | src/llm/**, src/config/** | Provider hierarchy, model config, fallback chain |
| `guardrails.md` | src/guardrails/** | 6 validators, severity, follow-up exception |
| `classification.md` | src/engine/classifier.py | 23 categories, extraction, multi-intent |
| `generation.md` | src/engine/generator.py | Tones, greeting, conciseness, voice rules |
| `gates.md` | src/engine/gate_evaluator.py | DEPRECATED — gates in Django |
| `api-routes.md` | src/api/** | Endpoints, schemas, middleware |

File navigation: see `CODEBASE_MAP.md`.

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

## Related Repos

| Repo | Path | Relation |
|------|------|----------|
| Solvix (Django) | `../Solvix` | Backend — calls AI Engine via HTTP, circuit breaker |
| solvix-etl | `../solvix-etl` | ETL — no direct integration |
| solvix_frontend | `../solvix_frontend` | Frontend — no direct integration |
