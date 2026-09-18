"""Run bounded DeepSeek scoring over frozen, sealed V3 local evaluation packets.

Requests and labels are owner-only files outside Git. This runner writes only
opaque packet identifiers, model telemetry, dispositions, and aggregate event
metrics. It never persists message/attachment text or evidence excerpts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from solvix_contracts.ai import MailSemanticEvidenceRequestV3

from src.config.settings import settings
from src.engine.mail_semantic_evidence_v3 import MailSemanticEvidenceInterpreterV3
from src.llm.base import LLMRateLimitedError


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _family_set(label: dict[str, Any]) -> set[str]:
    return {
        str(assertion["event_family"])
        for assertion in label.get("assertions") or []
        if isinstance(assertion, dict) and assertion.get("event_family")
    }


def _tuple_set(label: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(assertion["event_family"]),
            str(assertion.get("polarity") or ""),
            str(assertion.get("transition") or ""),
        )
        for assertion in label.get("assertions") or []
        if isinstance(assertion, dict) and assertion.get("event_family")
    }


def _metrics(
    expected: Counter[Any], predicted: Counter[Any], matched: int
) -> dict[str, float | int]:
    expected_total = sum(expected.values())
    predicted_total = sum(predicted.values())
    false_positives = predicted_total - matched
    false_negatives = expected_total - matched
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / expected_total if expected_total else 0.0
    return {
        "true_positives": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
    }


async def _run(
    request_rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, concurrency: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings.enable_mail_semantic_evidence_v3 = True
    interpreter = MailSemanticEvidenceInterpreterV3()
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        packet_id = str(row.get("packet_id") or "")
        label = labels.get(packet_id)
        if not packet_id or label is None:
            raise ValueError("request packet is missing a frozen score label")
        request = MailSemanticEvidenceRequestV3.model_validate(row["request"])
        async with semaphore:
            response = await interpreter.interpret(request)
        expected_families = _family_set(label)
        expected_tuples = _tuple_set(label)
        predicted_families = {event.family for event in response.semantic_events}
        predicted_tuples = {
            (event.family, event.polarity, event.transition) for event in response.semantic_events
        }
        invocation = (
            response.operation_summary.invocations[0]
            if response.operation_summary.invocations
            else None
        )
        return {
            "packet_id": packet_id,
            "scope": label.get("scope"),
            "expected_event_families": sorted(expected_families),
            "predicted_event_families": sorted(predicted_families),
            "expected_event_tuples": [list(item) for item in sorted(expected_tuples)],
            "predicted_event_tuples": [list(item) for item in sorted(predicted_tuples)],
            "disposition": response.disposition,
            "reason_codes": response.reason_codes,
            "family_exact_match": expected_families == predicted_families,
            "tuple_exact_match": expected_tuples == predicted_tuples,
            "provider": invocation.provider if invocation else None,
            "model": invocation.model if invocation else None,
            "prompt_tokens": response.operation_summary.prompt_tokens,
            "completion_tokens": response.operation_summary.completion_tokens,
            "reasoning_tokens": response.operation_summary.reasoning_tokens,
            "total_tokens": response.operation_summary.total_tokens,
        }

    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    expected_family_counter: Counter[str] = Counter()
    predicted_family_counter: Counter[str] = Counter()
    expected_tuple_counter: Counter[tuple[str, str, str]] = Counter()
    predicted_tuple_counter: Counter[tuple[str, str, str]] = Counter()
    family_matches = tuple_matches = family_exact = tuple_exact = accepted = 0
    token_totals: Counter[str] = Counter()
    skipped_after_rate_limit = 0
    for index, request_row in enumerate(request_rows):
        try:
            row = await one(request_row)
        except LLMRateLimitedError as exc:
            errors.append(
                {
                    "packet_id": str(request_row.get("packet_id") or ""),
                    "error_type": type(exc).__name__,
                    "error_code": str(exc)[:120],
                }
            )
            skipped_after_rate_limit = len(request_rows) - index - 1
            break
        except Exception as exc:
            errors.append(
                {
                    "packet_id": str(request_row.get("packet_id") or ""),
                    "error_type": type(exc).__name__,
                    "error_code": str(exc)[:120],
                }
            )
            continue
        result_rows.append(row)
        expected_families = set(row["expected_event_families"])
        predicted_families = set(row["predicted_event_families"])
        expected_tuples = {tuple(value) for value in row["expected_event_tuples"]}
        predicted_tuples = {tuple(value) for value in row["predicted_event_tuples"]}
        expected_family_counter.update(expected_families)
        predicted_family_counter.update(predicted_families)
        expected_tuple_counter.update(expected_tuples)
        predicted_tuple_counter.update(predicted_tuples)
        family_matches += len(expected_families & predicted_families)
        tuple_matches += len(expected_tuples & predicted_tuples)
        family_exact += int(bool(row["family_exact_match"]))
        tuple_exact += int(bool(row["tuple_exact_match"]))
        accepted += int(row["disposition"] == "accepted")
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            token_totals[field] += int(row[field] or 0)
    completed = len(result_rows)
    return result_rows, {
        "evaluated_packets": len(request_rows),
        "completed_packets": completed,
        "provider_errors": len(errors),
        "skipped_after_rate_limit": skipped_after_rate_limit,
        "accepted_packets": accepted,
        "family_exact_accuracy": family_exact / completed if completed else 0.0,
        "tuple_exact_accuracy": tuple_exact / completed if completed else 0.0,
        "family_metrics": _metrics(
            expected_family_counter, predicted_family_counter, family_matches
        ),
        "tuple_metrics": _metrics(expected_tuple_counter, predicted_tuple_counter, tuple_matches),
        "expected_family_counts": dict(sorted(expected_family_counter.items())),
        "predicted_family_counts": dict(sorted(predicted_family_counter.items())),
        "token_totals": dict(token_totals),
        "errors": errors,
        "scope_note": (
            "Strict positive-event subset only: this is not exhaustive-negative, debtor-identity, "
            "invoice-allocation, settlement, or operational-state validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 4:
        raise ValueError("concurrency must be between 1 and 4")
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    labels = {
        str(label["packet_id"]): label
        for label in json.loads(args.labels.read_text(encoding="utf-8"))["labels"]
    }
    if args.offset < 0:
        raise ValueError("offset must not be negative")
    requests = requests[args.offset :]
    if args.max_packets is not None:
        if args.max_packets < 1:
            raise ValueError("max-packets must be positive")
        requests = requests[: args.max_packets]
    result_rows, summary = asyncio.run(_run(requests, labels, concurrency=args.concurrency))
    _write_private(
        args.output,
        {
            "schema_version": "full-trail-v3-eval.v1",
            "requests_path": str(args.requests),
            "labels_path": str(args.labels),
            "offset": args.offset,
            "result_packets": result_rows,
            "summary": summary,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluated_packets": summary["evaluated_packets"],
                "provider_errors": summary["provider_errors"],
            }
        )
    )


if __name__ == "__main__":
    main()
