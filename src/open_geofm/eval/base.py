"""Evaluation interfaces + the CPU-only score-comparison report object.

`Evaluator` is the benchmark-run seam (concrete impl drives VLMEvalKit, in
`_vlmeval_backend.py`). `ScoreComparer` is pure-stdlib report logic wrapping the
free functions in `compare.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import EvalConfig
from .compare import BenchmarkScore, scan_work_dir, to_delta_table, to_markdown_table


@runtime_checkable
class Evaluator(Protocol):
    """Run a benchmark eval and return parsed scores."""

    def evaluate(self) -> list[BenchmarkScore]: ...


class ScoreComparer:
    """Scan a VLMEvalKit work-dir and pivot scores into a Markdown table."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config = config or EvalConfig()

    def scan(self, work_dir: Path) -> list[BenchmarkScore]:
        return scan_work_dir(work_dir)

    def table(self, scores: list[BenchmarkScore], *, baseline: str | None = None) -> str:
        return to_delta_table(scores, baseline) if baseline else to_markdown_table(scores)
