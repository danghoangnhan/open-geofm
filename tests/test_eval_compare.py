"""Tests for `open_geofm.eval.compare`.

The helper scans a VLMEvalKit `--work-dir` tree and pivots the
``*_score.json`` files into the headline Markdown table. The tests use
synthetic JSON in `tmp_path`; no real VLMEvalKit run required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from open_geofm.eval.compare import (
    BenchmarkScore,
    app,
    parse_score_json,
    scan_work_dir,
    to_delta_table,
    to_markdown_table,
)

# ---------------------------------------------------------------------------
# parse_score_json
# ---------------------------------------------------------------------------


def test_parse_score_json_picks_overall_key(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"Overall": 42.5, "GPS": 38.0}))
    score, metric, extras = parse_score_json(p)
    assert score == 42.5
    assert metric == "Overall"
    assert extras == {"GPS": 38.0}


def test_parse_score_json_prefers_overall_over_accuracy(tmp_path: Path) -> None:
    """When both `Overall` and `Accuracy` are present, `Overall` wins."""
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"Accuracy": 30.0, "Overall": 35.0}))
    score, metric, _ = parse_score_json(p)
    assert (score, metric) == (35.0, "Overall")


def test_parse_score_json_falls_back_to_accuracy(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"Accuracy": 30.0, "Detail": "ok"}))
    score, metric, _ = parse_score_json(p)
    assert (score, metric) == (30.0, "Accuracy")


def test_parse_score_json_averages_nested_leaves(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"per_task": {"a": 10.0, "b": 20.0, "c": 30.0}}))
    score, metric, _ = parse_score_json(p)
    assert score == 20.0
    assert metric == "mean(leaves)"


def test_parse_score_json_scalar_value(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps(67.3))
    score, metric, _ = parse_score_json(p)
    assert (score, metric) == (67.3, "value")


def test_parse_score_json_unknown_schema_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"weird": ["a", "b"], "notes": "..."}))
    score, metric, _ = parse_score_json(p)
    assert score is None
    assert metric == "unknown"


# ---------------------------------------------------------------------------
# scan_work_dir
# ---------------------------------------------------------------------------


def _write_score(work_dir: Path, model: str, benchmark: str, score: float) -> Path:
    model_dir = work_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{benchmark}_{model}_score.json"
    path.write_text(json.dumps({"Overall": score}))
    return path


def test_scan_work_dir_finds_score_files(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_base", "MathVista_MINI", 20.0)
    _write_score(tmp_path, "qwen2vl_2b_geofm", "MathVista_MINI", 31.5)
    _write_score(tmp_path, "qwen2vl_2b_geofm", "GeoQA", 48.0)

    scores = scan_work_dir(tmp_path)
    assert len(scores) == 3
    keys = {(s.model, s.benchmark, s.score) for s in scores}
    assert keys == {
        ("qwen2vl_2b_base", "MathVista_MINI", 20.0),
        ("qwen2vl_2b_geofm", "MathVista_MINI", 31.5),
        ("qwen2vl_2b_geofm", "GeoQA", 48.0),
    }


def test_scan_work_dir_logs_and_keeps_unknown_schema(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "m1" / "b1_m1_score.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"weird": ["x", "y"]}))
    with caplog.at_level("WARNING"):
        scores = scan_work_dir(tmp_path)
    assert len(scores) == 1
    assert scores[0].score is None
    assert any("no recognised metric" in r.message for r in caplog.records)


def test_scan_work_dir_handles_missing_subdir(tmp_path: Path) -> None:
    """A directly-nested score file (no model subdir) still parses; model defaults to dir name."""
    p = tmp_path / "MathVista_MINI_score.json"
    p.write_text(json.dumps({"Overall": 12.0}))
    scores = scan_work_dir(tmp_path)
    assert len(scores) == 1
    assert scores[0].benchmark == "MathVista_MINI"
    # Model resolution falls back to the parent-dir name (`tmp_path.name`).
    assert scores[0].score == 12.0


# ---------------------------------------------------------------------------
# Markdown table formatting
# ---------------------------------------------------------------------------


def _make_scores(rows: list[tuple[str, str, float | None]]) -> list[BenchmarkScore]:
    return [
        BenchmarkScore(model=m, benchmark=b, score=s, metric="Overall") for m, b, s in rows
    ]


def test_to_markdown_table_pivots_models_x_benchmarks() -> None:
    scores = _make_scores(
        [
            ("m1", "b1", 10.0),
            ("m1", "b2", 20.0),
            ("m2", "b1", 15.0),
        ]
    )
    table = to_markdown_table(scores)
    lines = table.splitlines()
    assert lines[0] == "| Model | b1 | b2 |"
    assert lines[1] == "|---|---|---|"
    # m1 row has scores for both benchmarks; m2 only has b1.
    m1 = next(line for line in lines if line.startswith("| m1 "))
    m2 = next(line for line in lines if line.startswith("| m2 "))
    assert "10.00" in m1 and "20.00" in m1
    assert "15.00" in m2 and "—" in m2


def test_to_markdown_table_handles_empty_input() -> None:
    assert to_markdown_table([]) == "_(no results)_"


def test_to_delta_table_shows_signed_deltas() -> None:
    scores = _make_scores(
        [
            ("base", "MathVista", 20.0),
            ("base", "GeoQA", 40.0),
            ("geofm", "MathVista", 30.0),
            ("geofm", "GeoQA", 48.0),
        ]
    )
    table = to_delta_table(scores, baseline_model="base")
    assert "30.00 (+10.0)" in table
    assert "48.00 (+8.0)" in table
    # Baseline row marked.
    assert "**base** (base)" in table


def test_to_delta_table_shows_negative_delta() -> None:
    scores = _make_scores(
        [
            ("base", "MathVista", 50.0),
            ("worse", "MathVista", 45.0),
        ]
    )
    table = to_delta_table(scores, baseline_model="base")
    assert "45.00 (-5.0)" in table


def test_to_delta_table_raises_when_baseline_missing() -> None:
    scores = _make_scores([("m1", "b1", 10.0)])
    with pytest.raises(ValueError, match="baseline 'missing' not found"):
        to_delta_table(scores, baseline_model="missing")


def test_to_delta_table_handles_missing_cell() -> None:
    """If the baseline doesn't have a score for a benchmark, the delta column
    should fall back to the raw cell instead of crashing."""
    scores = _make_scores(
        [
            ("base", "b1", 10.0),
            ("geofm", "b1", 12.0),
            ("geofm", "b2", 50.0),  # base has no score for b2
        ]
    )
    table = to_delta_table(scores, baseline_model="base")
    assert "12.00 (+2.0)" in table
    # b2 cell for geofm: raw value, no delta annotation.
    geofm_row = next(line for line in table.splitlines() if line.startswith("| geofm "))
    assert "50.00" in geofm_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_table_to_stdout(tmp_path: Path) -> None:
    _write_score(tmp_path, "m1", "b1", 10.0)
    runner = CliRunner()
    result = runner.invoke(app, [str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "10.00" in result.output


def test_cli_writes_to_out_file(tmp_path: Path) -> None:
    _write_score(tmp_path, "m1", "b1", 10.0)
    out_path = tmp_path / "results.md"
    runner = CliRunner()
    result = runner.invoke(app, [str(tmp_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "10.00" in out_path.read_text()


def test_cli_baseline_produces_delta_table(tmp_path: Path) -> None:
    _write_score(tmp_path, "base", "b1", 10.0)
    _write_score(tmp_path, "geofm", "b1", 15.0)
    runner = CliRunner()
    result = runner.invoke(app, [str(tmp_path), "--baseline", "base"])
    assert result.exit_code == 0, result.output
    assert "(+5.0)" in result.output
    assert "**base** (base)" in result.output


def test_cli_empty_workdir_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, [str(tmp_path)])
    assert result.exit_code == 1
    assert "No *_score.json" in result.output


def test_cli_help_works_without_torch() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Typer surfaces the subcommand args; the CLI advertises both flags.
    out = result.output.lower()
    assert "baseline" in out
    assert "work" in out
