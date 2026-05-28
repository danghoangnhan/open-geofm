"""Phase 9 ablation driver.

Reads a VLMEvalKit work-dir, slices it along one of the documented axes
(data-scale curve, renderer ablation, LoRA-rank sweep) and writes both a
Markdown table and a matplotlib PNG.

Usage::

    # Data-scale curve on MathVista-MINI for the 2B-LoRA configs.
    uv run python scripts/05_ablate.py scale outputs/eval \\
        --benchmark MathVista_MINI \\
        --out-md outputs/ablate/scale_mathvista.md \\
        --out-png outputs/ablate/scale_mathvista.png

    # Renderer ablation on GeoQA.
    uv run python scripts/05_ablate.py renderer outputs/eval \\
        --benchmark GeoQA --out-md outputs/ablate/renderer_geoqa.md

    # LoRA rank sweep on MathVista-MINI.
    uv run python scripts/05_ablate.py rank outputs/eval \\
        --benchmark MathVista_MINI --out-png outputs/ablate/rank_curve.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from open_geofm.ablate.sweep import (
    lora_rank_curve,
    plot_sweep,
    renderer_split,
    scale_curve,
    to_markdown_sweep,
)
from open_geofm.eval.compare import scan_work_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _common(
    work_dir: Path,
    benchmark: str,
    out_md: Path | None,
    out_png: Path | None,
    slicer,
    *,
    title: str,
    xlabel: str,
    log_x: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    scores = scan_work_dir(work_dir)
    if not scores:
        typer.echo(f"No *_score.json files found under {work_dir}.", err=True)
        raise typer.Exit(code=1)

    points = slicer(scores, benchmark=benchmark)
    if not points:
        typer.echo(
            f"No points after slicing benchmark={benchmark!r} along this axis. "
            f"Check the model-name suffixes (e.g. ``..._10k`` for scale).",
            err=True,
        )
        raise typer.Exit(code=2)

    table = to_markdown_sweep(points)
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table + "\n")
        typer.echo(f"wrote {out_md}")
    else:
        typer.echo(table)

    if out_png is not None:
        plot_sweep(
            points,
            title=title,
            xlabel=xlabel,
            ylabel=f"{benchmark} score",
            out_path=out_png,
            log_x=log_x,
        )
        typer.echo(f"wrote {out_png}")


@app.command()
def scale(
    work_dir: Path = typer.Argument(..., help="VLMEvalKit work-dir root."),
    benchmark: str = typer.Option("MathVista_MINI", help="Benchmark to slice on."),
    out_md: Path | None = typer.Option(None, "--out-md", help="Markdown table output."),
    out_png: Path | None = typer.Option(None, "--out-png", help="Plot output."),
) -> None:
    """Data-scale curve (5k / 10k / 20k tokens at the end of the model name)."""
    _common(
        work_dir,
        benchmark,
        out_md,
        out_png,
        scale_curve,
        title="Data-scale curve (open-geofm-mini)",
        xlabel="dataset size (samples)",
        log_x=True,
    )


@app.command()
def renderer(
    work_dir: Path = typer.Argument(..., help="VLMEvalKit work-dir root."),
    benchmark: str = typer.Option("MathVista_MINI", help="Benchmark to slice on."),
    out_md: Path | None = typer.Option(None, "--out-md", help="Markdown table output."),
    out_png: Path | None = typer.Option(None, "--out-png", help="Plot output."),
) -> None:
    """Renderer ablation (`_mpl` vs `_gmbl` tokens)."""
    _common(
        work_dir,
        benchmark,
        out_md,
        out_png,
        renderer_split,
        title="Renderer ablation (matplotlib vs GMBL-style)",
        xlabel="renderer (0=matplotlib, 1=GMBL)",
        log_x=False,
    )


@app.command()
def rank(
    work_dir: Path = typer.Argument(..., help="VLMEvalKit work-dir root."),
    benchmark: str = typer.Option("MathVista_MINI", help="Benchmark to slice on."),
    out_md: Path | None = typer.Option(None, "--out-md", help="Markdown table output."),
    out_png: Path | None = typer.Option(None, "--out-png", help="Plot output."),
) -> None:
    """LoRA-rank sweep (`_r8` / `_r16` / `_r32` / `_r64` tokens)."""
    _common(
        work_dir,
        benchmark,
        out_md,
        out_png,
        lora_rank_curve,
        title="LoRA rank sweep",
        xlabel="LoRA rank r",
        log_x=True,
    )


if __name__ == "__main__":
    app()
