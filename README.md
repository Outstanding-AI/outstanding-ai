# Outstanding AI Engine

Stateless AI service for the Outstanding AI debt collection platform. Provides email classification, response draft generation, compliance gate evaluation, and sender persona management for automated collections workflows.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)

> **Documentation Hub:** For cross-repo platform context, start with [Architecture](../Solvix/docs/ARCHITECTURE.md), [Contracts](../Solvix/docs/CONTRACTS.md), and [Single Debtor Email Lineage](../Solvix/docs/SINGLE_DEBTOR_EMAIL_LINEAGE.md) in the backend docs.

---

## Features

- **Email Classification**: Classify inbound customer emails into 23 categories with extracted data
- **Draft Generation**: Generate contextual response drafts with `{INVOICE_TABLE}` placeholder, closure email mode, sender style injection, and classification-aware follow-ups via `trigger_classification`
- **Gate Evaluation**: Evaluate compliance gates (deterministic, deprecated — gates now run in Django backend)
- **Sender Persona Management**: Generate and refine sender personas for a 4-level escalation hierarchy
- **Guardrails Pipeline**: Validate AI outputs with 7 parallel guardrails (placeholder validation, factual grounding, numerical consistency, entity verification, temporal consistency, contextual coherence, tone clamping)
- **Triple LLM Support**: Primary Vertex AI (`gemini-2.5-flash`), fallback OpenAI `gpt-5-mini`, optional Anthropic Claude (Sonnet for drafts, Haiku for classification)
- **Service Authentication**: Bearer token auth for service-to-service calls
- **Rate Limiting**: Per-tenant rate limits via `X-Tenant-ID` header (falls back to IP for direct callers)
- **Robust JSON Parsing**: Multi-strategy JSON extraction from LLM responses (handles markdown blocks, trailing commas, etc.)
- **Versioned Context Contract**: Backend requests carry `schema_version`. The service accepts only the canonical `CaseContext` v2 contract — v1 (legacy Sage-keyed) has been retired (`schema_version: Literal[2]` only) per `solvix-contracts==0.2.0`.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Outstanding AI │────▶│  Outstanding AI   │────▶│ Vertex Gemini  │
│  Backend        │◀────│   Engine          │◀────│   (Primary)   │
│                 │     │   Port 8001       │     │   gpt-5-mini  │
└─────────────────┘     └──────────────────┘     │   (Fallback)  │
                               │                  └───────────────┘
                               │
                        ┌──────┴──────┐
                        │  Middleware  │
                        │  Auth + RID  │
                        │  Rate Limit  │
                        └─────────────┘
```

The AI Engine is stateless - it receives all context via HTTP requests and does not access the database directly.
Case-context payload shapes are shared cross-repo through the backend-owned `solvix-contracts` package.

> **Note:** Classification results produced by this service are stored on `thread_messages` in the data lake and are now visible to users via the Communication History API (Django `sage/api_views.py`) — displayed as color-coded badges in the frontend's CommunicationTimeline and RecentActivity components.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Simple liveness check (no LLM calls) |
| `/health` | GET | Shallow probe-safe health check (`{"status": "ok"}`) |
| `/health/llm` | GET | Deep provider-aware health check with LLM verification |
| `/classify` | POST | Classify inbound email into 23 categories |
| `/generate-draft` | POST | Generate response draft with optional sender persona |
| `/evaluate-gates` | POST | Evaluate compliance gates (deterministic, deprecated — gates now in Django) |
| `/generate-persona` | POST | Generate initial personas for escalation contacts |
| `/refine-persona` | POST | Refine persona based on performance data |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Google Cloud ADC for Vertex AI or an OpenAI API key for fallback

### Local Development (uv - Recommended)

```bash
# Clone and setup
cd solvix-ai

# Install dependencies with uv
make install

# Configure environment
cp .env.example .env
# Edit .env for your local setup. For Vertex, use ADC outside ECS.

# Run the server with auto-reload
make dev
# API: http://localhost:8001
# Health: http://localhost:8001/health
# Deep LLM health: http://localhost:8001/health/llm
# Ping: http://localhost:8001/ping
```

### Shared Contracts Dependency

`solvix-contracts` is installed from the backend repo tag `contracts-v0.2.0`. For normal local development, authenticate git once and install through uv:

```bash
gh auth setup-git
uv sync --extra dev
```

When iterating on backend contracts locally before a release tag exists, temporarily override the installed package inside this repo's virtualenv:

```bash
uv sync --extra dev
uv pip install --reinstall -e ../Solvix/contracts

# Reset back to the pinned release tag
uv sync --frozen --extra dev --reinstall-package solvix-contracts
```

Docker builds require a short-lived GitHub App installation token passed as a BuildKit secret:

```bash
CONTRACTS_READ_TOKEN=<installation-token> make docker-build
```

GitHub Actions uses a GitHub App, not a personal access token, to create that short-lived token. Configure the app with `contents: read` on `Outstanding-AI/outstanding-backend` and add these secrets to the AI repo or org environment:

- `CONTRACTS_APP_ID`
- `CONTRACTS_APP_PRIVATE_KEY`

### Docker

```bash
# Build and run
make docker-build
make docker-run

# Or with docker-compose
make docker-up
make docker-logs   # View logs
make docker-down   # Stop
```

---

## Makefile Commands

```bash
# Setup
make install          # Install dependencies (uv)
make pre-commit-install  # Install pre-commit hooks

# Development
make run              # Run server
make dev              # Run with auto-reload

# Testing
make test             # Run unit tests (mocked, no API calls)
make test-cov         # Run with coverage report
make test-live        # Print the manual live validation workflow

# Code Quality
make lint             # Run ruff linter
make format           # Format code
make clean            # Remove cache files

# Docker
make docker-build     # Build image
make docker-run       # Run container
make docker-up        # Start with docker-compose
make docker-down      # Stop docker-compose
make docker-logs      # View docker-compose logs
```

---

## Configuration

Environment variables (in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `VERTEX_PROJECT_ID` | GCP project for Vertex AI | `production-493814` |
| `VERTEX_LOCATION` | Vertex region | `europe-west2` |
| `VERTEX_MODEL` | Vertex model | `gemini-2.5-flash` |
| `VERTEX_WIF_CONFIG_PATH` | WIF config file in the container | `/app/infra/vertex-wif-config.json` |
| `OPENAI_API_KEY` | OpenAI API key | Fallback LLM |
| `OPENAI_MODEL` | OpenAI model | `gpt-5-mini` |
| `VERTEX_MAX_TOKENS` | Vertex max tokens | `65535` |
| `OPENAI_MAX_TOKENS` | OpenAI max tokens | `32768` |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional 3rd provider |
| `ANTHROPIC_MODEL` | Anthropic model (drafts) | `claude-sonnet-4-20250514` |
| `ANTHROPIC_CLASSIFICATION_MODEL` | Anthropic model (classify) | `claude-haiku-4-5-20251001` |
| `LLM_TIMEOUT_SECONDS` | Per-call timeout | `60` |
| `SERVICE_AUTH_TOKEN` | Bearer token for auth | Empty (disabled) |
| `CORS_ALLOWED_ORIGINS` | Comma-separated origins | Empty (all in debug) |
| `LOG_LEVEL` | Logging level | `INFO` |

## Classification Categories (23)

| Category | Description |
|----------|-------------|
| `INSOLVENCY` | Bankruptcy, administration, liquidation |
| `DISPUTE` | Invoice dispute, goods/services issue |
| `ALREADY_PAID` | Claims payment already made |
| `UNSUBSCRIBE` | Requests to stop contact |
| `HOSTILE` | Aggressive or threatening language |
| `PROMISE_TO_PAY` | Commits to specific payment |
| `HARDSHIP` | Financial difficulty |
| `PLAN_REQUEST` | Requests payment plan |
| `REDIRECT` | Directs to another person |
| `REQUEST_INFO` | Asks for more information |
| `OUT_OF_OFFICE` | Auto-reply |
| `COOPERATIVE` | Positive engagement |
| `UNCLEAR` | Cannot determine intent |
| `PAYMENT_CONFIRMATION` | Confirms payment made with details |
| `REMITTANCE_ADVICE` | Formal remittance document |
| `EMAIL_BOUNCE` | Delivery failure notification |
| `AMOUNT_DISAGREEMENT` | Disputes the amount owed |
| `RETENTION_CLAIM` | Claims contractual retention |
| `LEGAL_RESPONSE` | Response via legal representative |
| `GENERIC_ACKNOWLEDGEMENT` | Simple acknowledgement without action |
| `QUERY_QUESTION` | Asks a question about the account |
| `ESCALATION_REQUEST` | Debtor requests to speak to someone senior |
| `PARTIAL_PAYMENT` | Notifies of partial payment |

## Draft Tones

| Tone | Use Case |
|------|----------|
| `friendly_reminder` | First contact. Warm, brief nudge — not a lecture |
| `professional` | Standard business tone. State facts and what you need |
| `firm` | Direct, no pleasantries. Emphasizes obligation and deadlines |
| `final_notice` | Last attempt before legal referral. 3-5 sentences max. No softening |
| `concerned_inquiry` | Good customers with unusual behavior. Brief, genuine concern |

### Conciseness (CRITICAL)
- All drafts must read like a real person wrote them — 4-8 sentences, not paragraphs
- No filler ("I am writing to inform you..."), no template language
- Enforced in: `src/prompts/draft_generation.py`

### Legal Escalation (final_notice + high touch count)
- When `final_notice` AND `touch_count >= 5`: explicitly mention legal team referral
- No softening phrases. 3-5 sentences max, no pleasantries

## Gate Types

| Gate | Type | Description |
|------|------|-------------|
| `touch_cap` | Block | Maximum contacts per month |
| `cooling_off` | Block | Minimum days between touches; enforces do_not_contact_until |
| `dispute_active` | Block | Block if dispute pending |
| `hardship` | Warning | Special handling required (does not block) |
| `unsubscribe` | Block | Contact opted out |
| `escalation_appropriate` | Block | Valid escalation path (considers industry patience) |

Gate evaluation is **deterministic** (Python logic, no LLM calls) for reliability and speed.

## Guardrails

The guardrail pipeline validates AI-generated content before it's returned. Guardrails run in parallel using a thread pool for performance.

| Guardrail | Severity | Description |
|-----------|----------|-------------|
| `placeholder_validation` | Critical | Detects hallucinated `[CAPS]`/`{CAPS}` placeholders (whitelist: INVOICE_TABLE, SENDER_NAME/TITLE/COMPANY) |
| `factual_grounding` | Critical | Validates invoice numbers and amounts match context |
| `numerical_consistency` | Critical | Ensures calculations are correct (totals = sum of parts) |
| `entity_verification` | High | Verifies customer code and company name match |
| `temporal_consistency` | Medium | Validates date references are accurate |
| `contextual_coherence` | Low | Checks overall response coherence |
| `tone_clamping` | High | Validates AI-chosen tone is within level's `tone_ladder` range (lane protocol); rewrites/regenerates on drift |

**Blocking Behavior:**
- `Critical` and `High` severity failures block the output
- `Medium` severity issues generate warnings but allow the output
- `Low` severity issues are logged only

## Sender Personas

The persona system supports a 4-level escalation hierarchy:

| Level | Typical Role | Default Tone |
|-------|-------------|--------------|
| 1 | AR Coordinator / Credit Controller | friendly_reminder |
| 2 | AR Manager / Senior Credit Controller | professional |
| 3 | Finance Manager / Head of Credit | firm |
| 4 | CFO / Finance Director | final_notice |

Personas define **how** a person writes (communication_style, formality_level, emphasis), not **what** they write. They are:
- **Generated** via cold start when admin saves escalation hierarchy
- **Refined** based on performance data during sync cycles
- **Injected** into draft generation prompts to control voice

## Testing

### Unit tests (no API calls)

The default unit tests **mock** the LLM layer so they are fast, deterministic, and do **not** call external APIs.

```bash
make test
# or directly:
uv run pytest tests/ -v
```

### Test Suite

| Test File | Coverage |
| --------- | -------- |
| `test_api.py` | API endpoint routing and response formats |
| `test_classifier.py` | Email classification with all 23 categories |
| `test_generator.py` | Draft generation with 5 tone types |
| `test_gate_evaluator.py` | Gate evaluation + escalation validation |
| `test_guardrail_severities.py` | Guardrail severity level verification |
| `test_provider_metadata.py` | Provider/model metadata in responses |
| `test_vertex_provider.py` | Vertex provider, WIF auth wiring, fallback behavior |
| `test_aws_ecs_supplier.py` | ECS task-role metadata supplier |
| `test_guardrails/` | Individual guardrail and pipeline tests |
| `test_evals/` | Evaluation system tests |

### Live validation

The repo does not carry a dedicated live-integration test file today. Use
`make test-live` for the manual smoke-test checklist against a running service.

```bash
make test-live
```

## Project Structure

```
solvix-ai/
├── src/
│   │   ├── models/          # Pydantic request/response models
│   │   │   ├── requests/    # Request models package
│   │   │   │   ├── context.py   # CaseContext, ObligationContext
│   │   │   │   ├── party.py     # ClassifyRequest, GenerateDraftRequest
│   │   │   │   ├── persona.py   # Persona request models
│   │   │   │   └── validation.py # Shared validators
│   │   │   └── responses.py # ClassifyResponse, GuardrailValidation, persona responses, etc.
│   │   ├── routes/          # FastAPI route handlers
│   │   │   ├── classify.py
│   │   │   ├── generate.py
│   │   │   ├── gates.py
│   │   │   ├── health.py    # /ping + /health + /health/llm
│   │   │   └── persona.py   # /generate-persona + /refine-persona
│   │   ├── errors.py        # Custom API exceptions
│   │   └── middleware.py     # RequestIDMiddleware + ServiceAuthMiddleware
│   ├── config/
│   │   ├── settings.py      # Pydantic settings (LLM, auth, CORS, rate limits)
│   │   └── constants.py     # Persona prompts and level descriptions
│   ├── engine/              # Core AI logic
│   │   ├── classifier.py    # Email classification (23 categories)
│   │   ├── generator.py     # Draft generation orchestration
│   │   ├── generator_prompts.py # Prompt builders for draft generation
│   │   ├── formatters.py    # Shared formatting utilities
│   │   ├── gate_evaluator.py # Deterministic gate evaluation (6 gates, no LLM)
│   │   ├── escalation_validator.py # Escalation validation logic
│   │   └── persona.py       # Persona generation and refinement
│   ├── guardrails/          # Output validation
│   │   ├── base.py          # Base classes and result types
│   │   ├── pipeline.py      # Guardrail pipeline orchestration
│   │   ├── executor.py      # Parallel guardrail execution
│   │   ├── feedback.py      # Guardrail feedback generation
│   │   ├── factual_grounding.py  # Invoice/amount validation (CRITICAL)
│   │   ├── numerical.py     # Calculation verification (CRITICAL)
│   │   ├── entity.py        # Customer code/name validation (HIGH)
│   │   ├── temporal.py      # Date reference validation (MEDIUM)
│   │   ├── contextual.py    # Coherence checking (LOW)
│   │   └── tone_clamping.py # AI tone within level's tone_ladder range (HIGH)
│   ├── llm/
│   │   ├── base.py          # BaseLLMProvider abstract class
│   │   ├── factory.py       # LLM client factory (Vertex→OpenAI fallback)
│   │   ├── aws_ecs_supplier.py
│   │   ├── vertex_provider.py
│   │   ├── openai_provider.py
│   │   └── schemas.py       # LLM response validation schemas
│   ├── prompts/             # Prompt templates
│   │   ├── classification.py
│   │   └── draft_generation.py
│   ├── utils/
│   │   ├── json_extractor.py # Robust JSON parsing
│   │   └── metrics.py        # Metrics utilities
│   ├── evals/
│   │   ├── metrics.py        # Evaluation metrics
│   │   ├── batch.py          # Batch evaluation runner
│   │   └── realtime.py       # Real-time evaluation tracking
│   └── main.py              # FastAPI app entrypoint
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_api.py
│   ├── test_classifier.py
│   ├── test_generator.py
│   ├── test_gate_evaluator.py
│   ├── test_guardrail_severities.py
│   ├── test_provider_metadata.py
│   ├── test_vertex_provider.py
│   ├── test_aws_ecs_supplier.py
│   ├── test_guardrails/
│   └── test_evals/
├── docs/
│   ├── implementation_plan.md
│   └── memory_context_analysis.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Integration with Outstanding AI Backend

The Django backend integrates via `services/ai_engine.py`:

```python
from services.ai_engine import AIEngineClient

async with AIEngineClient() as client:
    # Classify email
    result = await client.classify_email(email_content, context)

    # Generate draft (with optional persona)
    draft = await client.generate_draft(context, persona, tone)

    # Check gates (deterministic, fast)
    gates = await client.evaluate_gates(context, action, tone)

    # Generate personas (cold start)
    personas = await client.generate_personas(contacts)

    # Refine persona (performance-based)
    refined = await client.refine_persona(contact, persona, performance)
```

### Authentication

When `SERVICE_AUTH_TOKEN` is set, all requests (except `/health`, `/ping`, `/docs`) must include:
```
Authorization: Bearer <token>
```

### Docker Connectivity

The Outstanding AI backend runs inside Docker and needs to connect to the AI Engine:

**macOS / Windows (Docker Desktop):**
```bash
AI_ENGINE_URL=http://host.docker.internal:8001
```

**Linux:**
```bash
AI_ENGINE_URL=http://172.17.0.1:8001
```

### Running with Outstanding AI Backend

1. **Start the AI Engine** (runs on host):
   ```bash
   cd solvix-ai
   make dev
   ```

2. **Start Outstanding AI Backend** (runs in Docker):
   ```bash
   cd Solvix
   make dev-backend
   ```

3. **Verify connectivity**:
   ```bash
   curl http://localhost:8001/ping
   ```

## License

Proprietary - Outstanding AI
