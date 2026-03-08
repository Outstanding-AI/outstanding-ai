# Solvix AI Engine

> Stateless AI microservice powering intelligent debt collection workflows

## Project Identity

- **Name**: Solvix AI Engine
- **Purpose**: Stateless AI microservice for debt collection workflows
- **Core Capabilities**: Email classification, draft generation, compliance gate evaluation, sender persona management
- **Architecture**: FastAPI + LangChain with Gemini (primary) / OpenAI (fallback) / Anthropic (optional)
- **Port**: 8001 (default)

---

## Directory Structure

```
solvix-ai/
├── src/
│   ├── main.py                      # FastAPI app entry point, registers all routes
│   ├── api/
│   │   ├── __init__.py
│   │   ├── errors.py                # Custom exceptions (ValidationError, LLMProviderError, etc.)
│   │   ├── middleware.py            # RequestIDMiddleware + ServiceAuthMiddleware
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py          # Pydantic models: ClassifyRequest, GenerateDraftRequest, persona models, etc.
│   │   │   └── responses.py         # Pydantic models: ClassifyResponse, GuardrailValidation, persona responses, etc.
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── classify.py          # POST /classify - email classification endpoint
│   │       ├── generate.py          # POST /generate-draft - draft generation endpoint
│   │       ├── gates.py             # POST /evaluate-gates - compliance gate endpoint
│   │       ├── health.py            # GET /ping, GET /health - health check endpoints
│   │       └── persona.py           # POST /generate-persona, POST /refine-persona
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # Pydantic Settings - loads from .env
│   │   └── constants.py             # Persona prompts and level descriptions
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── classifier.py            # EmailClassifier - 23-category classification
│   │   ├── generator.py             # DraftGenerator - collection email drafts with persona support
│   │   ├── gate_evaluator.py        # GateEvaluator - 6 deterministic compliance gates (no LLM)
│   │   └── persona.py              # PersonaGenerator - cold start generation and refinement
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── base.py                  # Base classes, GuardrailSeverity enum
│   │   ├── pipeline.py              # GuardrailPipeline - parallel execution (6 workers)
│   │   ├── placeholder.py           # Detects hallucinated placeholders (CRITICAL)
│   │   ├── factual_grounding.py     # Validates invoices/amounts exist in context (skips empty invoice numbers)
│   │   ├── numerical.py             # Validates calculations and totals
│   │   ├── entity.py                # Validates customer codes/names
│   │   ├── temporal.py              # Validates date references
│   │   └── contextual.py           # Validates overall coherence
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseLLMProvider abstract class
│   │   ├── factory.py               # LLMProviderWithFallback - Gemini→OpenAI (+ optional Anthropic)
│   │   ├── gemini_provider.py       # Gemini implementation via LangChain
│   │   ├── openai_provider.py       # OpenAI implementation via LangChain
│   │   ├── anthropic_provider.py    # Anthropic implementation (optional third provider)
│   │   └── schemas.py               # LLM response validation schemas (classification, draft, persona)
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── classification.py        # System/user prompts for classification
│   │   └── draft_generation.py      # System/user prompts for draft generation (with industry/persona)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── json_extractor.py        # Robust JSON parsing with 4 fallback strategies
│   │   └── metrics.py               # Metrics computation utilities
│   └── evals/
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics computation
│       ├── batch.py                 # Batch evaluation runner
│       └── realtime.py              # Real-time evaluation tracking
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_api.py                  # API endpoint tests
│   ├── test_classifier.py           # Classification engine tests
│   ├── test_generator.py            # Draft generation tests
│   ├── test_gate_evaluator.py       # Gate evaluation tests
│   ├── test_guardrail_severities.py # Guardrail severity tests
│   ├── test_provider_metadata.py    # Provider metadata tests
│   ├── test_live_integration.py     # Real LLM API tests (requires API keys)
│   ├── test_llm_providers.py        # Provider tests
│   ├── test_guardrails/             # Individual guardrail tests
│   │   ├── test_factual_grounding.py
│   │   └── test_pipeline.py
│   └── test_evals/                  # Evaluation system tests
│       └── test_realtime.py
├── docs/
│   ├── implementation_plan.md       # Architecture plan for future improvements
│   └── memory_context_analysis.md   # Analysis: context enrichment vs memory
├── Dockerfile                       # Production build (non-root user, uv sync)
├── docker-compose.yml               # Local development setup
├── pyproject.toml                   # Dependencies and project metadata
├── Makefile                         # Development commands
├── .env.example                     # Environment template
└── README.md                        # Project documentation
```

---

## Detailed File Descriptions

### Entry Point & Configuration

#### `src/main.py`
Creates FastAPI app, registers routes from `api/routes/`, configures CORS, includes middleware (ServiceAuth + RequestID), sets up rate limiting via slowapi, and registers global exception handlers. This is the application entry point that Uvicorn runs.

#### `src/config/settings.py`
Pydantic Settings class that loads configuration from `.env`:
- `api_host`, `api_port`, `debug` - Server configuration
- `cors_allowed_origins` - Comma-separated CORS origins
- `llm_provider` - "gemini" or "openai"
- `gemini_api_key`, `gemini_model`, `gemini_temperature`, `gemini_max_tokens` - Gemini config
- `openai_api_key`, `openai_model`, `openai_temperature`, `openai_max_tokens` - OpenAI config
- `anthropic_api_key`, `anthropic_model`, `anthropic_temperature`, `anthropic_classification_model` - Anthropic config (optional)
- `llm_timeout_seconds`, `llm_max_retries` - Reliability settings
- `service_auth_token` - Service-to-service authentication
- `rate_limit_classify`, `rate_limit_generate`, `rate_limit_gates` - Per-IP rate limits

#### `src/config/constants.py`
Persona-related prompt constants:
- `LEVEL_DESCRIPTIONS` - 4-level escalation hierarchy descriptions
- `PERSONA_GENERATION_SYSTEM/USER` - Cold start persona generation prompts
- `PERSONA_REFINEMENT_SYSTEM/USER` - Performance-based refinement prompts
- `SENDER_PERSONA_INSTRUCTIONS` - Instructions for draft generation with personas

---

### API Layer (`src/api/`)

#### `errors.py`
Custom exceptions with HTTP status codes:
- `SolvixBaseError` - Base class with error_code, details, status_code
- `ValidationError` - 400 Bad Request
- `LLMProviderError` - 503 Service Unavailable
- `LLMResponseInvalidError` - 500 JSON parsing failed
- `LLMTimeoutError` - 504 Gateway Timeout
- `LLMRateLimitedError` - 429 Too Many Requests
- `ErrorCode` - Enum of error codes
- `ErrorResponse` - Structured error response model

#### `middleware.py`
Two middleware classes:

**RequestIDMiddleware**:
- Assigns UUID to each request (or accepts client-provided `X-Request-ID`)
- Stores in context variable for access throughout request
- Logs request start/end with duration in milliseconds
- Adds `X-Request-ID` header to response

**ServiceAuthMiddleware**:
- Enforces Bearer token authentication when `SERVICE_AUTH_TOKEN` is set
- Public paths exempt: `/health`, `/ping`, `/docs`, `/openapi.json`, `/redoc`
- Returns 401 if token missing or invalid
- Disabled when no token configured (local development)

#### `models/requests.py`
Input Pydantic models:
- **EmailContent**: subject, body, from_address, from_name, received_at (with max_length constraints)
- **PartyInfo**: party_id, customer_code, name, country_code, currency, credit_limit, on_hold, relationship_tier, tone_override, grace_days_override, touch_cap_override, do_not_contact_until, monthly_touch_count, is_verified, source
- **BehaviorInfo**: lifetime_value, avg_days_to_pay, on_time_rate, partial_payment_rate, segment
- **ObligationInfo**: invoice_number, original_amount, amount_due, due_date, days_past_due, state
- **CommunicationInfo**: touch_count, last_touch_at, last_touch_channel, last_sender_level, last_tone_used, last_response_at, last_response_type
- **TouchHistory**: sent_at, tone, sender_level, had_response
- **PromiseHistory**: promise_date, promise_amount, outcome
- **IndustryInfo**: code, name, typical_dso_days, alarm_dso_days, payment_cycle, escalation_patience, common_dispute_types, hardship_indicators, preferred_tone, ai_context_notes, seasonal_patterns, dispute_handling_notes, hardship_handling_notes, communication_notes
- **CaseContext**: Aggregates all above with case_state, days_in_state, active_dispute, hardship_indicated, broken_promises_count, brand_tone, touch_cap, touch_interval_days, grace_days, promise_grace_days, do_not_contact_until, monthly_touch_count, relationship_tier, unsubscribe_requested, industry, obligation_statuses (list), obligation_snapshot (list), recent_messages (list), currency_symbol (str)
- **SenderPersona**: name, title, communication_style, formality_level, emphasis
- **PersonaContact**: name, title, level (1-4)
- **SenderPerformanceStats**: 25+ performance metrics for persona refinement
- **GeneratePersonaRequest**: contacts list, total_levels
- **RefinePersonaRequest**: contact info, current_persona, performance stats, persona_version
- **ClassifyRequest**: email, context
- **GenerateDraftRequest**: context, sender_persona, sender_name, sender_title, tone, objective, custom_instructions (with prompt injection detection)
- **EvaluateGatesRequest**: context, proposed_action, proposed_tone
- **EvaluateGatesBatchRequest**: contexts (max 100), proposed_action, proposed_tone

**Security**: `custom_instructions` field has prompt injection validation that checks for patterns like "ignore previous", "system prompt", "act as", etc.

#### `models/responses.py`
Output Pydantic models:
- **ExtractedData**: promise_date/amount, dispute_type/reason/invoice_refs/disputed_amount, claimed_amount/date/reference/details, insolvency_type/details/administrator, return_date, redirect_name/contact/email
- **GuardrailValidation**: all_passed, guardrails_run, guardrails_passed, blocking_failures, warnings, factual_accuracy
- **ClassifyResponse**: classification, confidence, reasoning, extracted_data, guardrail_validation, tokens_used, provider, model, is_fallback
- **PersonaResult**: name, level, communication_style, formality_level, emphasis
- **GeneratePersonaResponse**: personas list
- **RefinePersonaResponse**: communication_style, formality_level, emphasis, reasoning
- **GenerateDraftResponse**: subject, body (HTML), tone_used, invoices_referenced, guardrail_validation, tokens_used, provider, model, is_fallback
- **GateResult**: passed, reason, current_value, threshold
- **EvaluateGatesResponse**: allowed, gate_results dict, recommended_action, tokens_used, provider, model, is_fallback
- **PartyGateResult**: party_id, customer_code, allowed, gate_results, recommended_action, blocking_gate
- **EvaluateGatesBatchResponse**: total, allowed_count, blocked_count, results
- **HealthResponse**: status, version, provider, model, fallback_provider, fallback_model, fallback_count, model_available, fallback_available, uptime_seconds

#### `routes/classify.py`
POST /classify endpoint - Accepts ClassifyRequest, invokes EmailClassifier, returns ClassifyResponse

#### `routes/generate.py`
POST /generate-draft endpoint - Accepts GenerateDraftRequest (with optional sender_persona), invokes DraftGenerator, returns GenerateDraftResponse

#### `routes/gates.py`
POST /evaluate-gates endpoint - Accepts EvaluateGatesRequest, invokes GateEvaluator (deterministic), returns EvaluateGatesResponse

#### `routes/health.py`
Two health check endpoints:
- **GET /ping** - Simple liveness check (no LLM calls, use for Docker HEALTHCHECK)
- **GET /health** - Full health check with LLM provider verification (makes actual API calls, use sparingly)

#### `routes/persona.py`
- **POST /generate-persona** - Generate initial personas for escalation contacts (cold start)
- **POST /refine-persona** - Refine a persona based on performance data (LLM-driven)

---

### Engine Layer (`src/engine/`)

#### `classifier.py` - EmailClassifier
Classifies inbound customer emails into 23 categories with confidence scoring.

**Categories** (in priority order):
1. INSOLVENCY - Bankruptcy, liquidation, administration
2. DISPUTE - Invoice/amount/service disputes
3. ALREADY_PAID - Claims payment was already made
4. PAYMENT_CONFIRMATION - Confirms payment with reference/amount
5. REMITTANCE_ADVICE - Formal remittance details
6. UNSUBSCRIBE - Request to stop communications
7. HOSTILE - Threatening, abusive language
8. PROMISE_TO_PAY - Commitment to pay by date
9. HARDSHIP - Financial difficulty claims
10. PLAN_REQUEST - Request for payment plan
11. REDIRECT - Points to different contact
12. REQUEST_INFO - Asks for invoice copies, statements
13. AMOUNT_DISAGREEMENT - Disputes specific amounts (not the invoice itself)
14. RETENTION_CLAIM - Retention percentage withheld
15. LEGAL_RESPONSE - Response from legal representative
16. OUT_OF_OFFICE - Auto-reply, vacation
17. EMAIL_BOUNCE - Delivery failure notification
18. COOPERATIVE - Willing to engage positively
19. GENERIC_ACKNOWLEDGEMENT - Simple acknowledgement without action
20. QUERY_QUESTION - Asks a question about the debt/account
21. ESCALATION_REQUEST - Debtor requests to speak with someone senior
22. PARTIAL_PAYMENT_NOTIFICATION - Notifies of partial payment made
23. UNCLEAR - Cannot determine intent

**Extracted Data**:
- promise_date, promise_amount (for PROMISE_TO_PAY)
- dispute_type, dispute_reason, invoice_refs, disputed_amount (for DISPUTE)
- claimed_amount, claimed_date, claimed_reference (for ALREADY_PAID, PAYMENT_CONFIRMATION, PARTIAL_PAYMENT_NOTIFICATION)
- insolvency_type, administrator_name/email (for INSOLVENCY)
- return_date (for OUT_OF_OFFICE)
- redirect_name, redirect_contact, redirect_email (for REDIRECT)

**Temperature**: 0.2 (low for consistency)

#### `generator.py` - DraftGenerator
Generates professional collection email drafts with 5 tone levels and optional sender persona.

**Tones**:
1. friendly_reminder - Soft, first touch
2. professional - Standard business tone
3. firm - Clear urgency, consequences mentioned
4. final_notice - Last chance before escalation
5. concerned_inquiry - For hardship/dispute cases

**Adaptations**:
- Adjusts for relationship_tier (VIP gets softer language)
- Adjusts for verification status (unverified parties get identity confirmation)
- Uses industry context for tone/escalation calibration
- Writes in sender persona voice when provided

**Output**: HTML format with `<p>` tags (not `<br>`), `{INVOICE_TABLE}` placeholder for invoice details

**Behaviour Segment Adaptation**: Prompt includes instructions per segment (ghost, escalation_responsive, strategic_non_payer, dispute_delayer, first_time_late, reliable_late_payer, genuine_hardship, habitual_slow_payer)

**Context Fields**: max_days_overdue and obligation_count passed to prompt for urgency calibration

**Empty Invoice Handling**: Obligations with no invoice_number shown as "(no invoice number)" in prompt context

**Temperature**: 0.7 (higher for creativity)

**Metrics Logged**: latency_ms, llm_latency_ms, guardrail_latency_ms, retry_count, total_tokens

#### `gate_evaluator.py` - GateEvaluator
Evaluates 6 compliance gates using **deterministic Python logic** (no LLM calls) for reliability and speed.

**Gates**:
1. **touch_cap** - Monthly contact limit not exceeded
2. **cooling_off** - Minimum days between touches respected; also enforces do_not_contact_until hold dates
3. **dispute_active** - Blocks if active dispute
4. **hardship** - Special handling if hardship indicated (warning, not blocking)
5. **unsubscribe** - Blocks if party opted out
6. **escalation_appropriate** - Validates tone escalation path; considers industry escalation_patience (patient/standard/aggressive)

**Behavior**:
- Returns allowed=true/false based on all gates
- Hardship is a warning, not a block
- Failed gates return recommended_action (alternative approach)
- Reports `provider: "deterministic"`, `model: "rule_engine"`

#### `persona.py` - PersonaGenerator
Manages sender persona lifecycle:

**Cold Start Generation** (`generate_personas`):
- Takes list of contacts with name, title, level (1-4)
- LLM generates communication_style, formality_level, emphasis for each
- Temperature: 0.7 (creative)

**Performance-Based Refinement** (`refine_persona`):
- Takes current persona + 25+ performance metrics
- LLM suggests evolutionary adjustments based on debtor response patterns
- Temperature: 0.5 (more conservative)

**4-Level Escalation Hierarchy**:
- Level 1: Coordinator/Controller (friendly_reminder tone)
- Level 2: AR Manager (professional tone)
- Level 3: Finance Manager/Head of Credit (firm tone)
- Level 4: CFO/Finance Director (final_notice tone)

---

### Guardrails Layer (`src/guardrails/`)

Validates AI outputs against context to prevent hallucinations.

#### `base.py`
Defines severity levels and base classes:
- **GuardrailSeverity**: CRITICAL, HIGH, MEDIUM, LOW
- **GuardrailResult**: Individual validation result with expected/found/details
- **GuardrailPipelineResult**: Aggregated results with blocking status
- **BaseGuardrail**: Abstract base class for all guardrails

#### `pipeline.py` - GuardrailPipeline
Runs 6 guardrails in parallel using ThreadPoolExecutor (6 workers).
- Supports sequential mode with fail-fast on critical failures
- Sorts results by severity: CRITICAL → HIGH → MEDIUM → LOW
- Blocks output if any CRITICAL or HIGH guardrail fails
- Separates warnings (MEDIUM/LOW) from blocking failures

#### `placeholder.py` - PlaceholderValidationGuardrail (CRITICAL)
- Detects hallucinated placeholders like `[CONTACT_NAME]`, `[COMPANY_PHONE]`, `{DEADLINE_DATE}`
- Whitelist: `{INVOICE_TABLE}`, `[SENDER_NAME]`, `[SENDER_TITLE]`, `[SENDER_COMPANY]`
- Pure regex, deterministic, runs first (cheapest guardrail)

#### `factual_grounding.py` - FactualGroundingGuardrail (CRITICAL)
- Validates invoice numbers mentioned exist in context
- **Skips obligations with empty/null invoice numbers** (won't crash on None.upper())
- Validates monetary amounts match obligations or their sum
- Flexible invoice matching (INV-123, #123, Invoice 123)
- Amount matching with rounding tolerance

#### `numerical.py` - NumericalConsistencyGuardrail (CRITICAL)
- Validates total calculations equal sum of parts
- Validates stated days overdue match context values
- Checks percentages if mentioned
- Tolerance: ±1 day for timing differences

#### `entity.py` - EntityVerificationGuardrail (HIGH)
- Validates customer code matches context exactly
- Validates company name matches (case-insensitive)
- Detects mismatches that indicate hallucination

#### `temporal.py` - TemporalConsistencyGuardrail (MEDIUM)
- Validates date references are accurate
- Checks due date references
- Validates recency claims against actual timestamps

#### `contextual.py` - ContextualCoherenceGuardrail (LOW)
- Validates overall response makes sense
- Checks for contradictions
- Ensures tone matches case state
- Verifies action recommendations align with gates

---

### LLM Layer (`src/llm/`)

#### `base.py`
Abstract `BaseLLMProvider` with:
- `complete(system_prompt, user_prompt, temperature, max_tokens)` - Main completion method
- `health_check()` - Validates API connectivity
- `LLMResponse` - Standardized response with content, usage stats

#### `factory.py` - LLMProviderWithFallback
- Lazy-initializes providers on first use
- Automatic fallback from Gemini → OpenAI on failure
- Tracks fallback_count for monitoring
- Logs fallback reasons for debugging

#### `gemini_provider.py`
- Uses `ChatGoogleGenerativeAI` from LangChain
- Model: `gemini-2.5-pro` (configurable)
- JSON mode via `response_mime_type: application/json`
- Default temperature: 0.3, max tokens: 8192

#### `openai_provider.py`
- Uses `ChatOpenAI` from LangChain
- Model: `gpt-5-nano` (reasoning model, configurable)
- JSON mode via `response_format: json_object`
- Default temperature: 0.3, max tokens: 32768 (high for reasoning models - reasoning tokens consume from this budget)

#### `schemas.py`
Pydantic schemas for LLM response validation:
- `LLMExtractedData` - Extracted fields for all classification categories
- `ClassificationLLMResponse` - Validates classification in 23 categories, confidence 0-1
- `DraftGenerationLLMResponse` - Validates subject and body strings
- `PersonaLLMResponse` - Validates communication_style, formality_level, emphasis
- `PersonaRefinementLLMResponse` - Validates updated persona fields + reasoning

---

### Prompts (`src/prompts/`)

#### `classification.py`
- **System Prompt**: Defines 23 categories in priority order with extraction rules
- **User Prompt**: Includes debtor context, party verification status, industry context, and email content
- Output: JSON with classification, confidence, reasoning, extracted_data

#### `draft_generation.py`
- **System Prompt**: Defines 5 tones, relationship tier adjustments, verification handling, industry context usage, behaviour segment usage (10 segments with specific instructions), sender persona instructions, greeting style (Hello/Hi — never Dear), placeholder rules
- **User Prompt**: Includes debtor info, invoice list (top 10 by days overdue), communication history, behavioural context (segment, on_time_rate, avg_days_to_pay, max_days_overdue, obligation_count), industry context, sender persona
- Output: JSON with subject, HTML body, reasoning (tone_rationale, strategy, key_factors), primary_cta, follow_up_days, invoices_referenced
- **Greeting Rule**: Always "Hello" or "Hi" — never "Dear". friendly/concerned tones prefer "Hi", professional/firm/final tones prefer "Hello"
- **Placeholder Rules**: Only allowed: `{INVOICE_TABLE}`, `[SENDER_NAME]`, `[SENDER_TITLE]`, `[SENDER_COMPANY]` — no invented placeholders

**Note**: Gate evaluation prompts no longer exist - gates are deterministic (no LLM).
Persona prompts are in `src/config/constants.py`.

---

### Utils (`src/utils/`)

#### `json_extractor.py`
`extract_json()` function with 4 fallback strategies to handle LLM quirks:
1. **Direct Parse**: Try parsing content as-is
2. **Strip Markdown**: Remove ```json code blocks
3. **Find JSON Object**: Use brace matching to extract JSON from text
4. **Clean Content**: Remove trailing commas, then retry

Handles: Markdown code blocks, BOM characters, trailing commas, nested JSON, mixed text

#### `metrics.py`
Metrics computation utilities for tracking performance.

---

### Evals (`src/evals/`)

#### `metrics.py`
Evaluation metrics computation for measuring AI quality

#### `batch.py`
Batch evaluation runner for testing against datasets

#### `realtime.py`
Real-time evaluation tracking for production monitoring

---

### Tests (`tests/`)

#### `conftest.py`
Shared pytest fixtures:
- sample_email_content, sample_party_info, sample_obligations
- sample_case_context, sample_classify_request
- Mock LLM responses

#### `test_api.py`
API endpoint tests for request/response models and status codes

#### `test_classifier.py`
Classification engine tests for all 23 categories, extracted data parsing, guardrail integration

#### `test_generator.py`
Draft generation tests for 5 tone types, HTML formatting, invoice reference tracking

#### `test_gate_evaluator.py`
Gate evaluation tests for 6 deterministic gates, failure recommendations

#### `test_guardrail_severities.py`
Tests verifying correct severity levels for each guardrail

#### `test_provider_metadata.py`
Tests verifying provider/model metadata in responses

#### `test_llm_providers.py`
Provider tests for Gemini, OpenAI, and fallback mechanism

#### `test_live_integration.py`
Real LLM API tests (requires actual API keys, makes network calls)

---

## API Endpoints

### GET /ping
Simple liveness check (no LLM calls). Use for Docker health checks.

**Response**:
```json
{
  "status": "ok",
  "uptime_seconds": 3600
}
```

### GET /health
Full health check with LLM provider verification. Makes actual API calls - use sparingly.

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "fallback_provider": "openai",
  "fallback_model": "gpt-5-nano",
  "fallback_count": 0,
  "model_available": true,
  "fallback_available": true,
  "uptime_seconds": 3600
}
```

### POST /classify
Classify inbound customer email.

**Request**:
```json
{
  "email": {
    "subject": "RE: Invoice #123",
    "body": "I already paid this last week...",
    "from_address": "john@company.com"
  },
  "context": { /* CaseContext object */ }
}
```

**Response**:
```json
{
  "classification": "ALREADY_PAID",
  "confidence": 0.95,
  "reasoning": "Customer explicitly states payment was made...",
  "extracted_data": {
    "claimed_amount": 500.00,
    "claimed_date": "2026-02-20"
  },
  "guardrail_validation": {
    "all_passed": true,
    "guardrails_run": 5,
    "guardrails_passed": 5,
    "blocking_failures": [],
    "warnings": [],
    "factual_accuracy": 1.0
  },
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "is_fallback": false
}
```

### POST /generate-draft
Generate collection email draft with optional sender persona.

**Request**:
```json
{
  "context": { /* CaseContext object */ },
  "sender_persona": {
    "name": "Sarah Williams",
    "title": "Credit Controller",
    "communication_style": "warm and detail-oriented",
    "formality_level": "professional",
    "emphasis": "building rapport and finding solutions"
  },
  "tone": "professional",
  "objective": "follow_up",
  "custom_instructions": "Mention the 10% early payment discount"
}
```

**Response**:
```json
{
  "subject": "Outstanding Balance - ABC Corp",
  "body": "<p>Hello John,</p><p>We hope this message finds you well...</p>",
  "tone_used": "professional",
  "invoices_referenced": ["INV-001", "INV-002"],
  "guardrail_validation": {
    "all_passed": true,
    "guardrails_run": 5,
    "guardrails_passed": 5,
    "blocking_failures": [],
    "warnings": [],
    "factual_accuracy": 1.0
  },
  "provider": "gemini",
  "model": "gemini-2.5-pro",
  "is_fallback": false
}
```

### POST /evaluate-gates
Evaluate compliance gates before action (deterministic, no LLM).

**Request**:
```json
{
  "context": { /* CaseContext object */ },
  "proposed_action": "send_email",
  "proposed_tone": "professional"
}
```

**Response**:
```json
{
  "allowed": true,
  "gate_results": {
    "touch_cap": {"passed": true, "reason": "Monthly touches (3) below cap (10)"},
    "cooling_off": {"passed": true, "reason": "Days since last touch (5) meets minimum interval (3)"},
    "dispute_active": {"passed": true, "reason": "No active dispute"},
    "hardship": {"passed": true, "reason": "No hardship indicated"},
    "unsubscribe": {"passed": true, "reason": "No unsubscribe request"},
    "escalation_appropriate": {"passed": true, "reason": "Single-step escalation from 'friendly_reminder' to 'professional'"}
  },
  "recommended_action": null,
  "provider": "deterministic",
  "model": "rule_engine",
  "is_fallback": false
}
```

### POST /generate-persona
Generate initial personas for escalation contacts (cold start).

**Request**:
```json
{
  "contacts": [
    {"name": "Sarah Williams", "title": "Credit Controller", "level": 1},
    {"name": "James Chen", "title": "AR Manager", "level": 2}
  ],
  "total_levels": 4
}
```

**Response**:
```json
{
  "personas": [
    {
      "name": "Sarah Williams",
      "level": 1,
      "communication_style": "warm and detail-oriented",
      "formality_level": "conversational",
      "emphasis": "building rapport and finding solutions"
    },
    {
      "name": "James Chen",
      "level": 2,
      "communication_style": "direct and results-focused",
      "formality_level": "professional",
      "emphasis": "clear expectations and timelines"
    }
  ]
}
```

### POST /refine-persona
Refine a persona based on performance data.

**Request**:
```json
{
  "name": "Sarah Williams",
  "title": "Credit Controller",
  "level": 1,
  "current_persona": {
    "name": "Sarah Williams",
    "communication_style": "warm and detail-oriented",
    "formality_level": "conversational",
    "emphasis": "building rapport"
  },
  "performance": {
    "total_touches": 150,
    "response_rate": 0.45,
    "cooperative_count": 30,
    "hostile_count": 2,
    "promises_kept": 20,
    "promises_broken": 5
  },
  "persona_version": 1
}
```

**Response**:
```json
{
  "communication_style": "warm but accountability-focused",
  "formality_level": "professional",
  "emphasis": "building rapport while setting clear deadlines",
  "reasoning": "High promise breakage (20%) suggests need for stronger follow-through language"
}
```

---

## Flow Scenarios

### Flow 1: Email Classification

```
1. Inbound email arrives at Django backend (Solvix)
2. Django calls POST /classify with:
   - email (subject, body, from_address)
   - context (party info, obligations, communication history, industry)
3. classify.py route handler invokes EmailClassifier
4. Classifier builds prompt with all context including industry
5. LLM returns JSON with classification, confidence, reasoning, extracted_data
6. json_extractor parses response (handles markdown, trailing commas)
7. Guardrails pipeline validates reasoning against context
8. Response returned with classification + guardrail_validation
9. Django stores classification and takes appropriate action
```

### Flow 2: Draft Generation

```
1. Collection agent requests email draft in Django
2. Django calls POST /generate-draft with:
   - context (party, obligations, communication history, industry)
   - sender_persona (optional - from escalation hierarchy)
   - tone (friendly_reminder/professional/firm/final_notice/concerned_inquiry)
   - objective (optional: follow_up/promise_reminder/escalation/initial_contact)
   - custom_instructions (optional, prompt-injection validated)
3. generate.py route handler invokes DraftGenerator
4. Generator calculates total outstanding, sorts invoices
5. Builds detailed prompt with invoice list, last touch info, industry context, persona
6. LLM returns JSON with subject and HTML body
7. json_extractor parses response
8. Guardrails validate: invoice numbers exist, amounts correct, names match
9. Response returned with draft + invoices_referenced + guardrail_validation
10. Django presents draft to agent for review/send
```

### Flow 3: Gate Evaluation (Before Any Action)

```
1. Before taking collection action, Django checks gates
2. Django calls POST /evaluate-gates with:
   - context (includes monthly_touch_count, last_touch_at, do_not_contact_until, etc.)
   - proposed_action (send_email, create_case, escalate, close_case)
   - proposed_tone (optional)
3. gates.py route handler invokes GateEvaluator
4. Evaluator runs 6 deterministic Python checks (no LLM call):
   - Touch cap, cooling off, dispute, hardship, unsubscribe, escalation
5. Escalation gate considers industry.escalation_patience
6. Returns allowed (true/false) + individual gate results
7. If not allowed, returns recommended_action
8. Django either proceeds with action or shows blocked message
```

### Flow 4: LLM Fallback

```
1. Any LLM endpoint makes call via LLMProviderWithFallback
2. Factory tries Gemini first (primary provider)
3. If Gemini fails (timeout, rate limit, error):
   - Logs failure with reason
   - Automatically switches to OpenAI
   - Increments fallback_count for monitoring
4. If OpenAI also fails, raises LLMProviderError (503)
5. Health endpoint reports fallback_count for monitoring
```

### Flow 5: Guardrails Blocking

```
1. LLM generates response (classification reasoning or draft body)
2. GuardrailPipeline runs 5 guardrails in parallel:
   - FactualGrounding (CRITICAL): Checks invoices/amounts
   - Numerical (CRITICAL): Checks calculations
   - Entity (HIGH): Checks customer codes
   - Temporal (MEDIUM): Checks dates
   - Contextual (LOW): Checks coherence
3. Results sorted by severity
4. If any CRITICAL or HIGH guardrail fails:
   - guardrail_validation.all_passed = false
   - blocking_failures populated with failure details
5. MEDIUM/LOW failures added to warnings array
6. Consumer (Django) can decide whether to show draft or regenerate
```

### Flow 6: Persona Lifecycle

```
1. Admin saves escalation hierarchy in Solvix frontend
2. Django calls POST /generate-persona with contacts list
3. LLM generates personas for each contact (cold start)
4. Django stores personas in escalation_contacts table
5. During sync cycle, for senders with sufficient data:
   - Django calls POST /refine-persona with current persona + performance stats
   - LLM suggests evolutionary adjustments
   - Django updates persona (incrementing persona_version)
6. When generating drafts, Django passes sender_persona in request
7. DraftGenerator writes email in that persona's voice
```

---

## Integration with Solvix Ecosystem

### Solvix (Django Backend)

**Repository**: Main collection platform backend
**Connection**: HTTP calls to solvix-ai endpoints with Bearer token auth

```python
# In Solvix: services/ai_engine.py
class AIEngineClient:
    base_url = "http://host.docker.internal:8001"  # or Linux IP
    headers = {"Authorization": f"Bearer {SERVICE_AUTH_TOKEN}"}

    async def classify_email(self, email, context) -> ClassifyResponse:
        return await self._post("/classify", {...})

    async def generate_draft(self, context, persona, tone) -> GenerateDraftResponse:
        return await self._post("/generate-draft", {...})

    async def evaluate_gates(self, context, action, tone) -> EvaluateGatesResponse:
        return await self._post("/evaluate-gates", {...})

    async def generate_persona(self, contacts) -> GeneratePersonaResponse:
        return await self._post("/generate-persona", {...})

    async def refine_persona(self, contact, persona, performance) -> RefinePersonaResponse:
        return await self._post("/refine-persona", {...})
```

**Data Flow**:
1. Django receives email or triggers action
2. Builds CaseContext from database (Party, Obligation, Communication, Industry models)
3. Calls solvix-ai endpoint (with auth token)
4. Processes response, stores results
5. Updates case state based on classification/gates

### How Backend Triggers AI Work (Post-Migration)
The Solvix backend no longer uses Celery. Instead:
1. Backend creates a `BackgroundJob` row with `queue="ai"` and `task_name="ai.generate_single_draft"`
2. The worker (`run_worker --queue ai`) claims the job via SELECT FOR UPDATE SKIP LOCKED
3. The task handler in `Solvix/organizations/task_handlers.py` calls this AI Engine via HTTP
4. Results are stored in Silver tables (drafts, thread_messages.classification)
5. For batch drafts: a `group_id` groups multiple `ai.generate_single_draft` jobs;
   `ai.check_group_complete` fires when all finish, triggering Gold refresh

### Task Names That Call This Service
| Task Name | Calls Endpoint | Purpose |
|-----------|---------------|---------|
| `ai.generate_single_draft` | `POST /generate-draft` | Generate one collection email |
| `ai.process_email_classification` | `POST /classify` | Classify inbound email |
| Gate evaluation | `POST /evaluate-gates` | Called inline (not via job queue) |
| Persona generation | `POST /generate-persona` | Called during escalation setup |
| Persona refinement | `POST /refine-persona` | Called during sync cycle |

### solvix-etl

**Repository**: Data pipeline for importing collection data
**Relationship**: Prepares the data that flows into Solvix → solvix-ai

**Data prepared by ETL**:
- Party information (debtors, customer codes, industry)
- Obligations (invoices from Sage/accounting systems)
- Historical communication data
- Payment history and behavior metrics

**Flow**:
```
Source Systems (Sage) → solvix-etl → Solvix DB → solvix-ai context
```

### solvix_frontend

**Repository**: React/Next.js UI for collection teams
**Relationship**: Consumes AI results via Django API

**UI displays**:
- Email classification badges and confidence
- AI-generated drafts for agent review
- Gate status (allowed/blocked with reasons)
- Guardrail validation warnings
- Sender persona management (escalation hierarchy)

**Flow**:
```
solvix_frontend → Solvix API → (internally calls solvix-ai) → Response to UI
```

---

## Docker Connectivity

When running in Docker:
- **macOS/Windows**: `http://host.docker.internal:8001`
- **Linux**: `http://172.17.0.1:8001` or host's actual IP

---

## Development Commands

```bash
# Setup
make install              # Install dependencies with uv
make pre-commit-install   # Install pre-commit hooks
cp .env.example .env      # Create config, add GEMINI_API_KEY

# Run locally
make dev                  # uvicorn with --reload on port 8001
make run                  # Production mode without reload

# Docker
make docker-build         # Build Docker image
make docker-run           # Run Docker container
make docker-up            # Start with docker-compose
make docker-logs          # View container logs
make docker-down          # Stop containers

# Testing
make test                 # Run unit tests (mocked LLM)
make test-cov             # Run with coverage report
make test-live            # Run live integration tests (requires API key)

# Code quality
make lint                 # Run ruff linter
make format               # Auto-format code
make clean                # Remove __pycache__, .pytest_cache
```

---

## Key Conventions

1. **Temperature Settings**:
   - Classification: 0.2 (consistency)
   - Gates: N/A (deterministic, no LLM)
   - Generation: 0.7 (creativity)
   - Persona generation: 0.7 (creative)
   - Persona refinement: 0.5 (conservative)

2. **Email Format**: HTML with `<p>` tags, not `<br>`

3. **Guardrails First**: All AI output validated before returning

4. **Robust JSON**: Always use `extract_json()`, never raw `json.loads()`

5. **Request IDs**: Every request gets UUID for tracing

6. **Async/Await**: All I/O operations are async

7. **Lazy Initialization**: LLM providers only initialize on first use

8. **Singleton Engines**: Classifier, Generator, GateEvaluator, PersonaGenerator are module-level singletons

9. **Deterministic Gates**: Gate evaluation uses Python logic, not LLM calls

10. **Service Auth**: Bearer token required in production (configurable via `SERVICE_AUTH_TOKEN`)

11. **Rate Limiting**: Per-IP rate limits via slowapi (100/minute defaults)

12. **Prompt Injection Prevention**: custom_instructions validated against known attack patterns

---

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your-key-here

# Optional (for fallback)
OPENAI_API_KEY=your-key-here

# LLM Configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-pro
GEMINI_TEMPERATURE=0.3
GEMINI_MAX_TOKENS=8192
OPENAI_MODEL=gpt-5-nano
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=32768

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
DEBUG=false
LOG_LEVEL=INFO

# Security
SERVICE_AUTH_TOKEN=            # Bearer token for service-to-service auth
CORS_ALLOWED_ORIGINS=          # Comma-separated origins (empty = allow all in debug)

# Reliability
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=3

# Rate Limiting (per-IP)
# RATE_LIMIT_CLASSIFY=100/minute
# RATE_LIMIT_GENERATE=100/minute
# RATE_LIMIT_GATES=100/minute
```

---

## Tech Stack

- **Language**: Python 3.12
- **Web Framework**: FastAPI 0.109+, Uvicorn 0.27+
- **LLM Integration**: LangChain (langchain>=0.3.0)
  - LangChain Google GenAI (Gemini provider)
  - LangChain OpenAI (OpenAI provider)
- **Data Validation**: Pydantic 2.5+, Pydantic Settings 2.1+
- **Rate Limiting**: slowapi
- **Concurrency**: asyncio, ThreadPoolExecutor (for parallel guardrails)
- **Package Manager**: uv (modern Python package manager)
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Code Quality**: ruff (linting and formatting), pre-commit
- **Containerization**: Docker, Docker Compose
