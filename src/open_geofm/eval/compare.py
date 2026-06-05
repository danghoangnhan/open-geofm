"""Compare VLMEvalKit run outputs and emit the headline Markdown table.

Blueprint §2 Phase 8 + Phase 9. VLMEvalKit's `--work-dir` writes one
subdirectory per `--model` it was invoked with; each subdirectory holds a
``<benchmark>_<model>_score.json`` per benchmark. After running:

    bash scripts/04_eval.sh qwen2vl_2b_base   outputs/qwen2vl-2b-lora     # the base model
    bash scripts/04_eval.sh qwen2vl_2b_geofm  outputs/qwen2vl-2b-lora      # post-finetune
    bash scripts/04_eval.sh qwen2vl_7b_geofm  outputs/qwen2vl-7b-lora

…you want one table: rows=models, cols=benchmarks, deltas vs baseline.
This module produces that table — same headline figure used in the README
and `wiki/06-Evaluation.md`. No torch / vllm import: pure JSON + stdlib.

CLI::

    python -m open_geofm.eval.compare outputs/eval                     # raw scores
    python -m open_geofm.eval.compare outputs/eval --baseline qwen2vl_2b_base
    python -m open_geofm.eval.compare outputs/eval --baseline qwen2vl_2b_base --out results.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, no_args_is_help=True)

# Order matters: first hit wins. "Overall" is VLMEvalKit's canonical name for
# the headline number on MathVista / GeoQA; the rest are common fallbacks
# across benchmark adapters.
_PREFERRED_METRIC_KEYS: tuple[str, ...] = (
    "Overall",
    "overall",
    "Accuracy",
    "accuracy",
    "Average",
    "average",
    "score",
    "Score",
)


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """One (model, benchmark) result row."""

    model: str
    benchmark: str
    score: float | None
    metric: str
    extras: dict[str, float] = field(default_factory=dict)
    source: Path | None = None


def _parse_score_data(data: Any) -> tuple[float | None, str, dict[str, float]]:
    """Core of `parse_score_json`, operating on already-parsed JSON so callers
    that already have the dict (e.g. `scan_work_dir`) don't re-read the file."""
    if isinstance(data, int | float):
        return float(data), "value", {}
    if not isinstance(data, dict):
        return None, "unknown", {}

    # Single-level: pick the first preferred key.
    for key in _PREFERRED_METRIC_KEYS:
        if key in data and isinstance(data[key], int | float):
            extras = {
                k: float(v) for k, v in data.items() if k != key and isinstance(v, int | float)
            }
            return float(data[key]), key, extras

    # Two-level: average ONLY the nested-dict leaves. Top-level scalars are
    # metadata (sample counts, seeds) and must not pollute the headline mean
    # (bug #24: the old code pooled both levels).
    leaves: list[float] = []
    for v in data.values():
        if isinstance(v, dict):
            leaves.extend(float(w) for w in v.values() if isinstance(w, int | float))
    if leaves:
        return sum(leaves) / len(leaves), "mean(leaves)", {}

    return None, "unknown", {}


def parse_score_json(path: Path) -> tuple[float | None, str, dict[str, float]]:
    """Parse a VLMEvalKit ``*_score.json`` payload -> ``(score, metric, extras)``.

    ``score`` is None when no recognised metric is present; callers can still
    surface the path so the user can inspect the raw JSON.
    """
    return _parse_score_data(json.loads(path.read_text()))


def _model_and_benchmark(path: Path, root: Path) -> tuple[str, str]:
    """Infer ``(model, benchmark)`` from a ``*_score.json`` under ``root``.

    Expected layout:
        ``<root>/<model>/<benchmark>_<model>_score.json``
    Falls back to:
      * directory name when the file isn't nested,
      * filename stem (sans ``_score``) when no ``_<model>`` suffix is found.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)

    parts = rel.parts
    model = parts[-2] if len(parts) >= 2 else path.parent.name
    if not model or model in (".", str(root)):
        model = "unknown"

    name = path.stem
    if name.endswith("_score"):
        name = name[: -len("_score")]
    suffix = f"_{model}"
    benchmark = name[: -len(suffix)] if name.endswith(suffix) else name
    return model, benchmark


def scan_work_dir(work_dir: Path) -> list[BenchmarkScore]:
    """Walk ``work_dir`` recursively and parse every ``*_score.json`` found."""
    work_dir = Path(work_dir)
    out: list[BenchmarkScore] = []
    for path in sorted(work_dir.rglob("*_score.json")):
        # Read + parse exactly once (bug #25: the warning branch used to re-read).
        data = json.loads(path.read_text())
        score, metric, extras = _parse_score_data(data)
        model, benchmark = _model_and_benchmark(path, work_dir)
        if score is None:
            keys = list(data.keys()) if isinstance(data, dict) else data
            log.warning("%s: no recognised metric (keys=%s); leaving score=None", path, keys)
        out.append(
            BenchmarkScore(
                model=model,
                benchmark=benchmark,
                score=score,
                metric=metric,
                extras=extras,
                source=path,
            )
        )
    return out


def _format_cell(score: float | None) -> str:
    return "—" if score is None else f"{score:.2f}"


def _format_delta_cell(score: float | None, baseline_score: float | None) -> str:
    if score is None:
        return "—"
    if baseline_score is None:
        return f"{score:.2f}"
    delta = score - baseline_score
    sign = "+" if delta >= 0 else ""
    return f"{score:.2f} ({sign}{delta:.1f})"


def to_markdown_table(scores: list[BenchmarkScore]) -> str:
    """Pivot ``scores`` into a Markdown table: rows = model, cols = benchmark."""
    if not scores:
        return "_(no results)_"
    models = sorted({s.model for s in scores})
    benchmarks = sorted({s.benchmark for s in scores})
    by_key = {(s.model, s.benchmark): s for s in scores}

    header = "| Model | " + " | ".join(benchmarks) + " |"
    sep = "|" + "|".join(["---"] * (len(benchmarks) + 1)) + "|"
    rows = [header, sep]
    for m in models:
        cells = [m]
        for b in benchmarks:
            s = by_key.get((m, b))
            cells.append(_format_cell(s.score if s else None))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def to_delta_table(scores: list[BenchmarkScore], baseline_model: str) -> str:
    """Same pivot, but every non-baseline cell shows ``score (Δ vs baseline)``."""
    if not scores:
        return "_(no results)_"
    models = sorted({s.model for s in scores})
    if baseline_model not in models:
        raise ValueError(f"baseline {baseline_model!r} not found; available: {sorted(models)}")
    benchmarks = sorted({s.benchmark for s in scores})
    by_key = {(s.model, s.benchmark): s for s in scores}

    header = "| Model | " + " | ".join(benchmarks) + " |"
    sep = "|" + "|".join(["---"] * (len(benchmarks) + 1)) + "|"
    rows = [header, sep]

    # Baseline first (annotated), then the rest in sorted order.
    base_cells = [f"**{baseline_model}** (base)"]
    for b in benchmarks:
        s = by_key.get((baseline_model, b))
        base_cells.append(_format_cell(s.score if s else None))
    rows.append("| " + " | ".join(base_cells) + " |")

    for m in models:
        if m == baseline_model:
            continue
        cells = [m]
        for b in benchmarks:
            s = by_key.get((m, b))
            base_s = by_key.get((baseline_model, b))
            cells.append(
                _format_delta_cell(
                    s.score if s else None,
                    base_s.score if base_s else None,
                )
            )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def compare(
    work_dir: Path = typer.Argument(..., help="VLMEvalKit work-dir root."),
    baseline: str | None = typer.Option(None, help="Model name to use as the baseline (Δ table)."),
    out: Path | None = typer.Option(None, help="Write table here; stdout if omitted."),
) -> None:
    """Scan a VLMEvalKit work-dir and emit a Markdown comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    scores = scan_work_dir(work_dir)
    if not scores:
        typer.echo(f"No *_score.json files found under {work_dir}.", err=True)
        raise typer.Exit(code=1)

    table = (
        to_delta_table(scores, baseline_model=baseline) if baseline else to_markdown_table(scores)
    )

    if out is not None:
        out.write_text(table + "\n")
        typer.echo(f"wrote {out}")
    else:
        typer.echo(table)


def _coerce_extras(d: dict[str, Any]) -> dict[str, float]:
    """Strict cast for type-checked downstream use; drops non-numeric entries."""
    return {k: float(v) for k, v in d.items() if isinstance(v, int | float)}


if __name__ == "__main__":
    app()
