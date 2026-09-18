"""Evaluate sealed V3 mail interpretation as binary presence then multi-event classification.

The frozen label file separates scoreable positive/negative/unknown messages.
Stage one scores only binary-eligible messages. Stage two scores event-family
multiplicity only for scoreable positive messages, and canonical tuples only for
the explicitly eligible event instances. Owner-only inputs and outputs never
persist message or attachment text.
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


def _binary_metrics(*, tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    total = tp + fp + fn + tn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _event_metrics(
    expected: Counter[Any], predicted: Counter[Any], matched: int
) -> dict[str, float | int]:
    expected_total = sum(expected.values())
    predicted_total = sum(predicted.values())
    fp = predicted_total - matched
    fn = expected_total - matched
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / expected_total if expected_total else 0.0
    return {
        "true_positive_instances": matched,
        "false_positive_instances": fp,
        "false_negative_instances": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _expected_families(label: dict[str, Any]) -> Counter[str]:
    return Counter(
        str(instance["canonical_contract_tuple"]["event_family"])
        for instance in label.get("event_instances") or []
        if isinstance(instance, dict) and instance.get("evidence_supported")
    )


def _expected_tuples(label: dict[str, Any]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (
            str(instance["canonical_contract_tuple"]["event_family"]),
            str(instance["canonical_contract_tuple"]["polarity"]),
            str(instance["canonical_contract_tuple"]["transition"]),
        )
        for instance in label.get("event_instances") or []
        if isinstance(instance, dict) and instance.get("canonical_tuple_scoring_mask")
    )


def _counter_intersection_count(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left & right).values())


async def _run(
    request_rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings.enable_mail_semantic_evidence_v3 = True
    interpreter = MailSemanticEvidenceInterpreterV3()
    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    binary = Counter()
    expected_family = Counter()
    predicted_family = Counter()
    expected_tuple = Counter()
    predicted_tuple = Counter()
    family_matched = tuple_matched = 0
    family_exact_records = tuple_exact_records = family_records = tuple_records = 0
    unknown_binary_records = 0
    token_totals = Counter()
    skipped_after_rate_limit = 0

    for index, row in enumerate(request_rows):
        packet_id = str(row.get("packet_id") or "")
        label = labels.get(packet_id)
        if not packet_id or label is None:
            raise ValueError("request packet is missing a two-stage score label")
        request = MailSemanticEvidenceRequestV3.model_validate(row["request"])
        try:
            response = await interpreter.interpret(request)
        except LLMRateLimitedError as exc:
            errors.append(
                {
                    "packet_id": packet_id,
                    "error_type": type(exc).__name__,
                    "error_code": str(exc)[:120],
                }
            )
            skipped_after_rate_limit = len(request_rows) - index - 1
            break
        except Exception as exc:
            errors.append(
                {
                    "packet_id": packet_id,
                    "error_type": type(exc).__name__,
                    "error_code": str(exc)[:120],
                }
            )
            continue

        predicted_events = list(response.semantic_events)
        predicted_present = bool(predicted_events)
        binary_scoreable = bool(label.get("binary_scoreable"))
        gold_present = label.get("event_present")
        if binary_scoreable and isinstance(gold_present, bool):
            if gold_present and predicted_present:
                binary["tp"] += 1
            elif gold_present:
                binary["fn"] += 1
            elif predicted_present:
                binary["fp"] += 1
            else:
                binary["tn"] += 1
        else:
            unknown_binary_records += 1

        expected_families = _expected_families(label)
        predicted_families = Counter(event.family for event in predicted_events)
        expected_tuples = _expected_tuples(label)
        predicted_tuples = Counter(
            (event.family, event.polarity, event.transition) for event in predicted_events
        )
        family_scoreable = bool(gold_present) and bool(expected_families)
        tuple_scoreable = bool(label.get("stage_two_complete_record_scoring_mask")) and bool(
            expected_tuples
        )
        if family_scoreable:
            family_records += 1
            expected_family.update(expected_families)
            predicted_family.update(predicted_families)
            family_matched += _counter_intersection_count(expected_families, predicted_families)
            family_exact_records += int(expected_families == predicted_families)
        if tuple_scoreable:
            tuple_records += 1
            expected_tuple.update(expected_tuples)
            predicted_tuple.update(predicted_tuples)
            tuple_matched += _counter_intersection_count(expected_tuples, predicted_tuples)
            tuple_exact_records += int(expected_tuples == predicted_tuples)

        invocation = (
            response.operation_summary.invocations[0]
            if response.operation_summary.invocations
            else None
        )
        result_rows.append(
            {
                "packet_id": packet_id,
                "binary_scoreable": binary_scoreable,
                "gold_event_present": gold_present,
                "predicted_event_present": predicted_present,
                "family_scoreable": family_scoreable,
                "tuple_scoreable": tuple_scoreable,
                "expected_family_instances": dict(expected_families),
                "predicted_family_instances": dict(predicted_families),
                "expected_tuple_instances": [
                    {"tuple": list(key), "count": value}
                    for key, value in sorted(expected_tuples.items())
                ],
                "predicted_tuple_instances": [
                    {"tuple": list(key), "count": value}
                    for key, value in sorted(predicted_tuples.items())
                ],
                "disposition": response.disposition,
                "reason_codes": response.reason_codes,
                "provider": invocation.provider if invocation else None,
                "model": invocation.model if invocation else None,
                "prompt_tokens": response.operation_summary.prompt_tokens,
                "completion_tokens": response.operation_summary.completion_tokens,
                "reasoning_tokens": response.operation_summary.reasoning_tokens,
                "total_tokens": response.operation_summary.total_tokens,
            }
        )
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            token_totals[field] += int(response.operation_summary.model_dump().get(field) or 0)

    return result_rows, {
        "evaluated_packets": len(request_rows),
        "completed_packets": len(result_rows),
        "provider_errors": len(errors),
        "skipped_after_rate_limit": skipped_after_rate_limit,
        "stage_one_binary_eligible_packets": sum(binary.values()),
        "stage_one_unknown_packets": unknown_binary_records,
        "stage_one_binary_metrics": _binary_metrics(
            tp=binary["tp"], fp=binary["fp"], fn=binary["fn"], tn=binary["tn"]
        ),
        "stage_two_family_scoreable_positive_packets": family_records,
        "stage_two_family_multiset_exact_accuracy": family_exact_records / family_records
        if family_records
        else 0.0,
        "stage_two_family_instance_metrics": _event_metrics(
            expected_family, predicted_family, family_matched
        ),
        "stage_two_tuple_scoreable_positive_packets": tuple_records,
        "stage_two_tuple_multiset_exact_accuracy": tuple_exact_records / tuple_records
        if tuple_records
        else 0.0,
        "stage_two_tuple_instance_metrics": _event_metrics(
            expected_tuple, predicted_tuple, tuple_matched
        ),
        "expected_family_instance_counts": dict(sorted(expected_family.items())),
        "predicted_family_instance_counts": dict(sorted(predicted_family.items())),
        "token_totals": dict(token_totals),
        "errors": errors,
        "scope_note": (
            "Stage one scores only explicitly adjudicated positive/negative messages; unknown messages are "
            "excluded. Stage two scores event-instance multiplicity only on scoreable positive messages. "
            "Identity, allocation, settlement, and operational-state accuracy remain out of scope."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if args.offset < 0:
        raise ValueError("offset must not be negative")
    requests = json.loads(args.requests.read_text(encoding="utf-8"))[args.offset :]
    if args.max_packets is not None:
        if args.max_packets < 1:
            raise ValueError("max-packets must be positive")
        requests = requests[: args.max_packets]
    labels = {
        str(label["packet_id"]): label
        for label in json.loads(args.labels.read_text(encoding="utf-8"))["labels"]
    }
    result_rows, summary = asyncio.run(_run(requests, labels))
    _write_private(
        args.output,
        {
            "schema_version": "full-trail-v3-two-stage-eval.v1",
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
