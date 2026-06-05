"""Unified open-geofm CLI.

A thin typer app over the config + registry + pipeline layers. The generation
command builds an `AppConfig`-driven `Pipeline` (Phase 2 -> 4 -> 6); `backends`
and `config` are introspection helpers that need no GPU / formalgeo data.

Run with ``python -m open_geofm.cli ...``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from . import registry
from .config import AppConfig
from .factory import build_generation_pipeline
from .formal.loader import FormalGeo7KSource
from .pipeline import PipelineContext

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_config(config_path: Path | None) -> AppConfig:
    return AppConfig.from_yaml(config_path) if config_path else AppConfig.default()


@app.command()
def backends() -> None:
    """List every registered stage backend (the registry contents)."""
    for reg in (
        registry.SOURCES,
        registry.SOLVERS,
        registry.GATHERERS,
        registry.RENDERERS,
        registry.LLM_CLIENTS,
        registry.VERIFIERS,
        registry.BUILDERS,
        registry.TRAINERS,
        registry.EVALUATORS,
    ):
        typer.echo(f"{reg.kind:>10}: {', '.join(reg.available())}")


@app.command()
def config(config_path: Path | None = typer.Option(None, "--config", help="YAML to load.")) -> None:
    """Print the resolved config tree (defaults + any YAML overrides)."""
    typer.echo(json.dumps(_load_config(config_path).model_dump(mode="json"), indent=2))


@app.command()
def generate(
    n: int = typer.Option(100, help="Total synthetic samples to generate."),
    config_path: Path | None = typer.Option(None, "--config", help="YAML config."),
    out: Path = typer.Option(Path("data/open-geofm-mini"), help="Output dataset directory."),
    seed_start: int = typer.Option(1, help="First FormalGeo7K PID."),
    seed_end: int = typer.Option(7000, help="Last PID (inclusive)."),
) -> None:
    """Run the Phase 2->4->6 generation pipeline (needs FormalGeo7K data)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = _load_config(config_path)
    source = FormalGeo7KSource(cfg.formalgeo)
    seeds = (source.load(pid) for pid in range(seed_start, seed_end + 1))
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_generation_pipeline(cfg, seeds, target_n=n)
    ctx = pipeline.run(PipelineContext(image_dir=image_dir, out_dir=out))
    typer.echo(f"done: {len(ctx.rendered)} samples -> {out}")


if __name__ == "__main__":
    app()
