# Outstanding AI Engine — API Reference

Stateless FastAPI microservice at port 8001. Called exclusively by the Django backend via HTTP with circuit breaker.

---

## Authentication

| Setting | Local | Production |
|---------|-------|-----------|
| `SERVICE_AUTH_TOKEN` | Empty (auth disabled) | Required (AWS Secrets Manager) |

Token sent as `Authorization: Bearer <token>` header. Bypass paths (no auth required): `/health`, `/ping`, `/docs`, `/openapi.json`, `/redoc`.

---

## Endpoints

### POST /classify

Classify an inbound debtor email into one of 23 categories.

**Request** (`ClassifyRequest`):
- `case: CaseContext` — party, obligations, communication, behavior
- `email_body: str`
- `email_subject: str`
- `sender_email: str`
- `X-Tenant-ID` header — used for per-tenant rate limiting

**Response** (`ClassifyResponse`):
- `category: str` — one of 23 categories
- `confidence: float` — 0.0–1.0
- `secondary_categories: List[str]` — multi-intent categories
- `extracted_data: ExtractedData` — category-specific fields (promise_date, dispute_type, etc.)
- `ai_reasoning: str`
- `token_counts: dict`

---

### POST /generate-draft

Generate a collection email draft (5 tones, persona-aware, with guardrails).

**Request** (`GenerateDraftRequest`):
- `case: CaseContext`
- `sender: SenderContext` — persona and escalation level context
- `tone: str` — exact backend-selected tone for this draft (for example `friendly_reminder`, `professional`, `firm`, `final_notice`)
- `trigger_classification: Optional[str]` — follow-up context (e.g., payment_acknowledgement)
- `closure_mode: Optional[bool]` — relationship-preserving tone
- `X-Tenant-ID` header

**Response** (`GenerateDraftResponse`):
- `subject: str`
- `body_html: str` — HTML with `{INVOICE_TABLE}` placeholder (Django replaces)
- `body_plain: str` — plain text version
- `guardrail_validation: GuardrailValidation` — 7 individual check results
- `reasoning: dict` — tone_rationale, strategy, key_factors
- `primary_cta: str` — request_payment | request_call | offer_plan | request_timeline
- `follow_up_days: int`
- `invoices_referenced: List[str]`

---

### POST /generate-persona

Cold-start persona generation from escalation contacts.

**Request** (`GeneratePersonaRequest`):
- `escalation_contacts: List[EscalationContact]`
- `tenant_id: str`

**Response** (`GeneratePersonaResponse`):
- `persona: PersonaResult` — communication_style, formality_level, emphasis, style_description, style_examples

---

### POST /refine-persona

Refine persona based on performance data (10+ touches).

**Request** (`RefinePersonaRequest`):
- `current_persona: PersonaResult`
- `performance_data: dict`
- `tenant_id: str`

**Response** (`RefinePersonaResponse`):
- `persona: PersonaResult`

---

### POST /evaluate-gates (DEPRECATED)

> **DEPRECATED**: Gate evaluation moved to Django backend (`services/gate_checker.py`). This endpoint is kept for backward compatibility only and is NOT called in production.

---

### GET /ping

Liveness check — no LLM calls. Returns `{"status": "ok"}`.

### GET /health

Shallow health check — no LLM calls. Returns `{"status": "ok"}` and is safe for load balancers and lightweight probes.

### GET /health/llm

Deep provider-aware health check — calls configured LLM providers. Use this for operator diagnostics, not for ECS health checks.

---

## Rate Limiting

Per-tenant via `X-Tenant-ID` header. Falls back to IP address for direct callers.
Key function: `_get_tenant_key()` in `src/api/routes/classify.py` and `src/api/routes/generate.py`.

---

## ECS Idle Shutdown

When `IDLE_SHUTDOWN_SECONDS` > 0, a background watchdog thread monitors time since last request and sends SIGTERM after the idle period. Reduces Fargate costs when AI Engine is unused between syncs.

---

## Error Responses

Custom exceptions in `src/api/errors.py`:

| HTTP Code | Cause |
|-----------|-------|
| 422 | Request validation error (Pydantic) |
| 401 | Missing or invalid SERVICE_AUTH_TOKEN |
| 429 | Rate limit exceeded for tenant |
| 503 | LLM provider unavailable (all providers failed) |
| 500 | Internal error (check logs) |

---

## Source Files

| What | File |
|------|------|
| Request models | `src/api/models/requests/` (context.py, party.py, persona.py, validation.py) |
| Response models | `src/api/models/responses.py` |
| Route handlers | `src/api/routes/*.py` |
| Middleware | `src/api/middleware.py` (RequestID + ServiceAuth) |
| Error types | `src/api/errors.py` |
| CaseContext builder (caller) | Backend `services/context_builder.py` |
| Response mirror (caller) | Backend `services/ai_engine/schemas.py` (warn-only Pydantic) |

Cross-repo contract: see `Solvix/docs/CONTRACTS.md`.
