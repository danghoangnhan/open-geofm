"""Tests for `open_geofm.ablate.sweep`.

The sweep utilities parse model-name suffix tokens (`_10k`, `_r16`, `_gmbl`)
to slice a VLMEvalKit work-dir into a Phase 9 ablation table + plot. Tests
exercise the parser corner-cases, the Markdown pivot, and the matplotlib
output (file is created, contains a PNG header).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_geofm.ablate import (
    SweepPoint,
    plot_sweep,
    scale_curve,
    to_markdown_sweep,
)
from open_geofm.ablate.sweep import (
    _parse_rank,
    _parse_renderer,
    _parse_scale,
    lora_rank_curve,
    renderer_split,
)
from open_geofm.eval.compare import scan_work_dir

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("qwen2vl_2b_lora_5k", ("qwen2vl_2b_lora", 5000.0, "5k")),
        ("qwen2vl_2b_lora_10K", ("qwen2vl_2b_lora", 10000.0, "10k")),
        ("qwen2vl_2b_lora_1000", ("qwen2vl_2b_lora", 1000.0, "1000")),
        ("qwen2vl_7b_lora_20k", ("qwen2vl_7b_lora", 20000.0, "20k")),
    ],
)
def test_parse_scale_handles_k_suffix(name: str, expected) -> None:
    assert _parse_scale(name) == expected


def test_parse_scale_does_not_swallow_rank_token() -> None:
    """`_r10` is a LoRA rank token, NOT a 10-sample data scale. The scale parser
    must reject it (the rank parser will pick it up)."""
    # `r10` ends in `10` but the underscore-anchored regex requires `_<digits>(k?)`
    # — `_r10` has a leading `r`, which the digit-only group can't match.
    assert _parse_scale("qwen2vl_2b_lora_r10") is None


def test_parse_scale_returns_none_on_unsuffixed_name() -> None:
    assert _parse_scale("qwen2vl_2b_lora") is None


def test_parse_rank_extracts_rank() -> None:
    assert _parse_rank("qwen2vl_2b_lora_r16") == ("qwen2vl_2b_lora", 16.0, "r16")
    assert _parse_rank("qwen2vl_2b_lora_r64") == ("qwen2vl_2b_lora", 64.0, "r64")


def test_parse_rank_returns_none_on_data_scale_token() -> None:
    assert _parse_rank("qwen2vl_2b_lora_10k") is None


def test_parse_renderer_distinguishes_mpl_and_gmbl() -> None:
    assert _parse_renderer("qwen2vl_2b_lora_mpl") == ("qwen2vl_2b_lora", 0.0, "mpl")
    assert _parse_renderer("qwen2vl_2b_lora_gmbl") == ("qwen2vl_2b_lora", 1.0, "gmbl")


def test_parse_renderer_returns_none_on_other_suffixes() -> None:
    assert _parse_renderer("qwen2vl_2b_lora_10k") is None
    assert _parse_renderer("qwen2vl_2b_lora_r16") is None


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _write_score(work_dir: Path, model: str, benchmark: str, score: float) -> None:
    model_dir = work_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{benchmark}_{model}_score.json").write_text(
        json.dumps({"Overall": score})
    )


def test_scale_curve_groups_by_base_config(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_5k", "MathVista_MINI", 25.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_10k", "MathVista_MINI", 30.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_20k", "MathVista_MINI", 33.0)
    _write_score(tmp_path, "qwen2vl_7b_lora_10k", "MathVista_MINI", 48.0)
    # Different benchmark — must be filtered out.
    _write_score(tmp_path, "qwen2vl_2b_lora_10k", "GeoQA", 99.0)

    points = scale_curve(scan_work_dir(tmp_path), benchmark="MathVista_MINI")
    assert {(p.base_config, p.axis_value, p.score) for p in points} == {
        ("qwen2vl_2b_lora", 5000.0, 25.0),
        ("qwen2vl_2b_lora", 10000.0, 30.0),
        ("qwen2vl_2b_lora", 20000.0, 33.0),
        ("qwen2vl_7b_lora", 10000.0, 48.0),
    }


def test_scale_curve_sorted_by_base_then_axis(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_20k", "MathVista_MINI", 33.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_5k", "MathVista_MINI", 25.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_10k", "MathVista_MINI", 30.0)
    points = scale_curve(scan_work_dir(tmp_path), benchmark="MathVista_MINI")
    # All same base, sorted by axis_value ascending.
    assert [p.axis_value for p in points] == [5000.0, 10000.0, 20000.0]


def test_scale_curve_drops_models_without_axis_suffix(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora", "MathVista_MINI", 20.0)  # baseline (no scale tag)
    _write_score(tmp_path, "qwen2vl_2b_lora_5k", "MathVista_MINI", 25.0)
    points = scale_curve(scan_work_dir(tmp_path), benchmark="MathVista_MINI")
    assert len(points) == 1
    assert points[0].axis_value == 5000.0


def test_renderer_split_finds_two_arms(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_mpl", "GeoQA", 42.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_gmbl", "GeoQA", 46.0)
    points = renderer_split(scan_work_dir(tmp_path), benchmark="GeoQA")
    assert {(p.axis_label, p.score) for p in points} == {("mpl", 42.0), ("gmbl", 46.0)}


def test_lora_rank_curve_picks_r_token(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_r8", "MathVista_MINI", 28.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_r16", "MathVista_MINI", 31.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_r32", "MathVista_MINI", 32.0)
    points = lora_rank_curve(scan_work_dir(tmp_path), benchmark="MathVista_MINI")
    assert [p.axis_value for p in points] == [8.0, 16.0, 32.0]
    assert [p.score for p in points] == [28.0, 31.0, 32.0]


def test_slice_drops_none_scores(tmp_path: Path) -> None:
    # Write a malformed score JSON — `parse_score_json` returns `score=None`.
    work = tmp_path / "qwen2vl_2b_lora_5k"
    work.mkdir(parents=True)
    (work / "MathVista_MINI_qwen2vl_2b_lora_5k_score.json").write_text(
        json.dumps({"unknown_key": "no number here"})
    )
    points = scale_curve(scan_work_dir(tmp_path), benchmark="MathVista_MINI")
    assert points == []


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_to_markdown_sweep_orders_axis_numerically() -> None:
    """The axis labels in the header must be in *numeric* order — `5k` before
    `10k` even though `10k` sorts first lexicographically."""
    points = [
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 20000.0, "20k", 33.0),
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 5000.0, "5k", 25.0),
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 10000.0, "10k", 30.0),
    ]
    table = to_markdown_sweep(points)
    header = table.splitlines()[0]
    assert header.index("5k") < header.index("10k") < header.index("20k")
    # And the row of scores follows the same order.
    row = next(line for line in table.splitlines() if line.startswith("| qwen2vl_2b_lora"))
    assert row.index("25.00") < row.index("30.00") < row.index("33.00")


def test_to_markdown_sweep_handles_missing_cells() -> None:
    """Two base configs, one with a hole on the 20k column. The hole becomes `—`."""
    points = [
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 5000.0, "5k", 25.0),
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 20000.0, "20k", 33.0),
        SweepPoint("qwen2vl_7b_lora", "MathVista_MINI", 5000.0, "5k", 45.0),
    ]
    table = to_markdown_sweep(points)
    row_7b = next(line for line in table.splitlines() if line.startswith("| qwen2vl_7b_lora"))
    assert "45.00" in row_7b
    assert "—" in row_7b


def test_to_markdown_sweep_empty_input() -> None:
    assert to_markdown_sweep([]) == "_(no results)_"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def test_plot_sweep_writes_png(tmp_path: Path) -> None:
    points = [
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 5000.0, "5k", 25.0),
        SweepPoint("qwen2vl_2b_lora", "MathVista_MINI", 10000.0, "10k", 30.0),
    ]
    out = tmp_path / "scale.png"
    fig = plot_sweep(points, title="t", xlabel="x", ylabel="y", out_path=out, log_x=True)
    assert out.exists() and out.stat().st_size > 0
    # PNG header is the 8 bytes \x89PNG\r\n\x1a\n.
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # Returned figure should be usable too.
    assert fig is not None
