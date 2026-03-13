# solvix-ai — Codebase Map

Concept → file navigation index.

## Engine (Core AI Logic)

| Concept | File |
|---------|------|
| Email classification | `src/engine/classifier.py` |
| Draft generation | `src/engine/generator.py` |
| Gate evaluation (DEPRECATED) | `src/engine/gate_evaluator.py` |
| Persona management | `src/engine/persona.py` |

## Prompts

| Concept | File |
|---------|------|
| Classification prompt | `src/prompts/classification.py` |
| Draft generation prompt | `src/prompts/draft_generation.py` |
| Gate evaluation prompt | `src/prompts/gate_evaluation.py` |
| Persona prompts | `src/prompts/persona.py` |

## Guardrails

| Concept | File |
|---------|------|
| Pipeline orchestrator | `src/guardrails/pipeline.py` |
| Base validator | `src/guardrails/base.py` |
| Placeholder detection | `src/guardrails/placeholder.py` |
| Factual grounding | `src/guardrails/factual_grounding.py` |
| Numerical consistency | `src/guardrails/numerical.py` |
| Entity verification | `src/guardrails/entity.py` |
| Temporal consistency | `src/guardrails/temporal.py` |
| Contextual coherence | `src/guardrails/contextual.py` |

## LLM Providers

| Concept | File |
|---------|------|
| Provider factory | `src/llm/factory.py` |
| Gemini provider | `src/llm/gemini_provider.py` |
| OpenAI provider | `src/llm/openai_provider.py` |
| Anthropic provider | `src/llm/anthropic_provider.py` |

## API Layer

| Concept | File |
|---------|------|
| Classification endpoint | `src/api/routes/classify.py` |
| Generation endpoint | `src/api/routes/generate.py` |
| Gates endpoint | `src/api/routes/gates.py` |
| Persona endpoints | `src/api/routes/persona.py` |
| Health checks | `src/api/routes/health.py` |
| Request models | `src/api/models/requests.py` |
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
