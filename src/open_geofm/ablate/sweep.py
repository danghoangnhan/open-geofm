"""Slice a VLMEvalKit work-dir along one ablation axis.

Blueprint §2 Phase 9. Each model name in the work-dir encodes ``<base>_<axis>``
where ``<axis>`` is one of:

* data-scale token: ``1k``, ``5k``, ``10k``, ``20k`` (`scale_curve`)
* renderer token: ``mpl`` / ``gmbl`` (`renderer_split`)
* lora rank token: ``r8`` / ``r16`` / ``r32`` / ``r64`` (`lora_rank_curve`)

The parser strips the axis token off the end, yielding the *base config* —
that's the grouping key for the plot (one line per base config, x-axis =
axis token, y-axis = score on the chosen benchmark).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..eval.compare import BenchmarkScore

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One (base_config, benchmark, axis_value, score) row."""

    base_config: str
    benchmark: str
    axis_value: float
    axis_label: str
    score: float


# ---------------------------------------------------------------------------
# Axis parsers
# ---------------------------------------------------------------------------


# Matches the suffix `_5k`, `_10k`, `_500`, `_2K`. The fragment is anchored to
# end-of-string so we don't accidentally swallow `r10` (an rank-10 LoRA).
_SCALE_RE = re.compile(r"_(\d+)([kK]?)$")
# `_r8`, `_r16`, `_r32`, `_r64` (LoRA rank token, distinct from data scale).
_RANK_RE = re.compile(r"_r(\d+)$")
# `_mpl` (matplotlib) | `_gmbl` (GMBL-style) renderer token.
_RENDERER_RE = re.compile(r"_(mpl|gmbl)$")


def _parse_scale(model_name: str) -> tuple[str, float, str] | None:
    """``qwen2vl_2b_lora_10k`` -> ``("qwen2vl_2b_lora", 10000.0, "10k")``."""
    m = _SCALE_RE.search(model_name)
    if not m:
        return None
    n = float(m.group(1)) * (1000 if m.group(2) else 1)
    return model_name[: m.start()], n, m.group(0).lstrip("_").lower()


def _parse_rank(model_name: str) -> tuple[str, float, str] | None:
    m = _RANK_RE.search(model_name)
    if not m:
        return None
    return model_name[: m.start()], float(m.group(1)), f"r{m.group(1)}"


_RENDERER_VALUES = {"mpl": 0.0, "gmbl": 1.0}


def _parse_renderer(model_name: str) -> tuple[str, float, str] | None:
    m = _RENDERER_RE.search(model_name)
    if not m:
        return None
    token = m.group(1)
    return model_name[: m.start()], _RENDERER_VALUES[token], token


_AXIS_PARSERS = {
    "scale": _parse_scale,
    "rank": _parse_rank,
    "renderer": _parse_renderer,
}


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def _slice(
    scores: list[BenchmarkScore],
    *,
    axis: str,
    benchmark: str,
) -> list[SweepPoint]:
    """Filter scores to a single benchmark and parse the axis token."""
    parser = _AXIS_PARSERS[axis]
    points: list[SweepPoint] = []
    for s in scores:
        if s.benchmark != benchmark or s.score is None:
            continue
        parsed = parser(s.model)
        if parsed is None:
            continue
        base, val, label = parsed
        points.append(
            SweepPoint(
                base_config=base,
                benchmark=s.benchmark,
                axis_value=val,
                axis_label=label,
                score=s.score,
            )
        )
    points.sort(key=lambda p: (p.base_config, p.axis_value))
    return points


def scale_curve(scores: list[BenchmarkScore], *, benchmark: str) -> list[SweepPoint]:
    """Headline Phase-9 figure: accuracy vs. dataset size (1K / 5K / 10K / 20K)."""
    return _slice(scores, axis="scale", benchmark=benchmark)


def renderer_split(scores: list[BenchmarkScore], *, benchmark: str) -> list[SweepPoint]:
    """Headline matplotlib-vs-GMBL ablation: two points per base config."""
    return _slice(scores, axis="renderer", benchmark=benchmark)


def lora_rank_curve(scores: list[BenchmarkScore], *, benchmark: str) -> list[SweepPoint]:
    """LoRA rank sweep: r=8 / r=16 / r=32 / r=64."""
    return _slice(scores, axis="rank", benchmark=benchmark)


# ---------------------------------------------------------------------------
# Output: Markdown table + matplotlib plot
# ---------------------------------------------------------------------------


def to_markdown_sweep(points: list[SweepPoint]) -> str:
    """Pivot one sweep into a Markdown table: rows = base config, cols = axis label.

    Cells: `score.2f`. Missing combinations: `—`.
    """
    if not points:
        return "_(no results)_"
    bases = sorted({p.base_config for p in points})
    # Preserve numeric order on the x-axis (not lexicographic — `2k` before `10k`).
    axis_pairs = sorted({(p.axis_value, p.axis_label) for p in points})
    axis_labels = [label for _, label in axis_pairs]
    by_key = {(p.base_config, p.axis_label): p for p in points}

    header = "| Model | " + " | ".join(axis_labels) + " |"
    sep = "|" + "|".join(["---"] * (len(axis_labels) + 1)) + "|"
    rows = [header, sep]
    for b in bases:
        cells = [b]
        for label in axis_labels:
            hit = by_key.get((b, label))
            cells.append("—" if hit is None else f"{hit.score:.2f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def plot_sweep(
    points: list[SweepPoint],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path | None = None,
    log_x: bool = False,
):
    """Render a sweep into a matplotlib `Figure`.

    One line per base config, scored against the same benchmark. Saves to
    `out_path` if given (and returns the figure either way so notebooks can
    `display(fig)`). The plot is deliberately minimal — readers regularly
    re-style the curves for the blog post.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bases = sorted({p.base_config for p in points})
    for base in bases:
        xs = [p.axis_value for p in points if p.base_config == base]
        ys = [p.score for p in points if p.base_config == base]
        labels = [p.axis_label for p in points if p.base_config == base]
        ax.plot(xs, ys, marker="o", label=base)
        for xv, yv, lab in zip(xs, ys, labels, strict=True):
            ax.annotate(lab, (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    return fig
