from __future__ import annotations

from pathlib import Path

from src.common.sql_attribution import athena_attribution_comment, sanitize_sql_label


def test_sanitize_sql_label_removes_unsafe_characters() -> None:
    assert sanitize_sql_label("ai.lake reader /* raw */") == "ai.lake_reader_raw"


def test_ai_athena_attribution_comment_defaults_to_lake_reader_unknown() -> None:
    comment = athena_attribution_comment(source=None)

    assert comment.startswith("/* solvix_sql:v1;runtime=ai;component=lake_reader;")
    assert "source=ai.lake_reader.unknown" in comment
    assert "query_class=select" in comment


def test_ai_athena_attribution_comment_includes_context() -> None:
    comment = athena_attribution_comment(
        source="ai.lake_reader.draft_context",
        tenant_id="tenant-1",
        sync_run_id="sync-1",
    )

    assert "source=ai.lake_reader.draft_context" in comment
    assert "tenant=tenant-1" in comment
    assert "sync_run_id=sync-1" in comment


def test_regional_lake_reader_does_not_define_private_attribution_helpers() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "src/lake/regional_reader.py").read_text()

    assert "_sanitize_sql_label" not in text
    assert "_athena_attribution_comment" not in text
