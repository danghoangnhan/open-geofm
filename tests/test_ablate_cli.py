"""Tests for the `scripts/05_ablate.py` Typer CLI.

Walks the real CLI surface — pseudo end-to-end with a synthetic work-dir.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "_ablate_cli", REPO_ROOT / "scripts" / "05_ablate.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


def _write_score(work_dir: Path, model: str, benchmark: str, score: float) -> None:
    p = work_dir / model
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{benchmark}_{model}_score.json").write_text(json.dumps({"Overall": score}))


def test_scale_subcommand_prints_table(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_5k", "MathVista_MINI", 25.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_10k", "MathVista_MINI", 30.0)
    runner = CliRunner()
    result = runner.invoke(_load_cli(), ["scale", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "5k" in result.output and "10k" in result.output
    assert "25.00" in result.output and "30.00" in result.output


def test_scale_subcommand_writes_files(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_5k", "MathVista_MINI", 25.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_10k", "MathVista_MINI", 30.0)
    out_md = tmp_path / "out" / "scale.md"
    out_png = tmp_path / "out" / "scale.png"
    runner = CliRunner()
    result = runner.invoke(
        _load_cli(),
        ["scale", str(tmp_path), "--out-md", str(out_md), "--out-png", str(out_png)],
    )
    assert result.exit_code == 0, result.output
    assert out_md.exists()
    assert out_png.exists()
    assert "5k" in out_md.read_text()


def test_renderer_subcommand_runs(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_mpl", "GeoQA", 42.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_gmbl", "GeoQA", 46.0)
    runner = CliRunner()
    result = runner.invoke(_load_cli(), ["renderer", str(tmp_path), "--benchmark", "GeoQA"])
    assert result.exit_code == 0, result.output
    assert "mpl" in result.output and "gmbl" in result.output
    assert "42.00" in result.output and "46.00" in result.output


def test_rank_subcommand_runs(tmp_path: Path) -> None:
    _write_score(tmp_path, "qwen2vl_2b_lora_r8", "MathVista_MINI", 28.0)
    _write_score(tmp_path, "qwen2vl_2b_lora_r16", "MathVista_MINI", 31.0)
    runner = CliRunner()
    result = runner.invoke(_load_cli(), ["rank", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "r8" in result.output and "r16" in result.output


def test_scale_subcommand_exits_when_no_scores(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(_load_cli(), ["scale", str(tmp_path)])
    assert result.exit_code == 1
    assert "No *_score.json" in result.output


def test_scale_subcommand_exits_when_no_axis_match(tmp_path: Path) -> None:
    # Has scores but no recognised data-scale token.
    _write_score(tmp_path, "qwen2vl_2b_lora", "MathVista_MINI", 20.0)
    runner = CliRunner()
    result = runner.invoke(_load_cli(), ["scale", str(tmp_path)])
    assert result.exit_code == 2
    assert "axis" in result.output.lower() or "suffixes" in result.output.lower()


@pytest.mark.parametrize("cmd", ["scale", "renderer", "rank"])
def test_subcommand_help_works(cmd: str) -> None:
    runner = CliRunner()
    result = runner.invoke(_load_cli(), [cmd, "--help"])
    assert result.exit_code == 0
    assert cmd in result.output.lower() or "benchmark" in result.output.lower()
