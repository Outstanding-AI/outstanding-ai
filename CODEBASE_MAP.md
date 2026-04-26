# Outstanding AI Engine — Codebase Map

Concept → file navigation index.

## Engine (Core AI Logic)

| Concept | File |
|---------|------|
| Email classification | `src/engine/classifier.py` |
| Draft generation (orchestration) | `src/engine/generator.py` | `DraftGenerator.generate()` orchestrates `_assemble_prompt`, `_run_llm_with_guardrails`, `_build_response`; internal dataclasses: `_TokenTotals`, `_TimingInfo`, `_PromptContext` |
| Draft prompt builders | `src/engine/generator_prompts.py` |
| Shared formatters | `src/engine/formatters.py` |
| Persona management | `src/engine/persona.py` |

> Gate evaluation is **backend-only** — see `Solvix/services/gate_checker.py`. The AI Engine no longer hosts a gate evaluator.

## Prompts

| Concept | File |
|---------|------|
| Classification prompt | `src/prompts/classification.py` |
| Draft generation prompt | `src/prompts/draft_generation.py` |
| Prompt sanitization helpers | `src/prompts/_sanitize.py` |

## Guardrails

| Concept | File |
|---------|------|
| Pipeline orchestrator | `src/guardrails/pipeline.py` (orchestration), `executor.py` (execution), `feedback.py` (feedback) |
| Base validator | `src/guardrails/base.py` |
| Placeholder detection | `src/guardrails/placeholder.py` |
| Factual grounding | `src/guardrails/factual_grounding.py` |
| Numerical consistency | `src/guardrails/numerical.py` |
| Entity verification | `src/guardrails/entity.py` |
| Temporal consistency | `src/guardrails/temporal.py` |
| Contextual coherence | `src/guardrails/contextual.py` |
| Tone clamping (v2) | `src/guardrails/tone_clamping.py` |

## LLM Providers

| Concept | File |
|---------|------|
| Provider factory | `src/llm/factory.py` |
| Vertex provider | `src/llm/vertex_provider.py` |
| AWS ECS WIF credential supplier | `src/llm/aws_ecs_supplier.py` |
| OpenAI provider | `src/llm/openai_provider.py` |
| Anthropic provider | `src/llm/anthropic_provider.py` |

## API Layer

| Concept | File |
|---------|------|
| Classification endpoint | `src/api/routes/classify.py` |
| Generation endpoint | `src/api/routes/generate.py` |
| Persona endpoints | `src/api/routes/persona.py` |
| Health checks | `src/api/routes/health.py` |
| Request models | `src/api/models/requests/` (package: context.py, party.py, persona.py, validation.py) |
| Response models | `src/api/models/responses.py` |
| Middleware | `src/api/middleware.py` |
| Error types | `src/api/errors.py` |

## Config & Utils

| Concept | File |
|---------|------|
| Settings (Pydantic) | `src/config/settings.py` |
| Constants | `src/config/constants.py` |
| JSON extractor | `src/utils/json_extractor.py` |
| App entry point | `src/main.py` |
