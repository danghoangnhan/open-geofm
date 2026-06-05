"""Pipeline orchestration spine.

`Stage` is the run-contract ABC; `Pipeline` folds an ordered list of stages over
a typed `PipelineContext`. Stages receive already-constructed collaborators
(typed by their stage Protocol) and are pure with respect to the context they
thread — so this module imports no concrete backend and stays CPU-clean
(`import open_geofm.pipeline` pulls in only stdlib + config + registry).

The generation stages (`SampleStage` → `RenderStage` → `BuildStage`) mirror the
paper's Phase 2 → 4 → 6 flow. Heavier stages (train/eval) are driven directly by
their own CLI for now; they plug into the same `Stage` contract when wired.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dataset.builder import make_sample_id

if TYPE_CHECKING:
    from .config import DatasetConfig
    from .sampling.algorithm1 import Algorithm1Runner, SyntheticSample

log = logging.getLogger("open_geofm.pipeline")


# ---------------------------------------------------------------------------
# Value objects flowing through the pipeline.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RenderedSample:
    """A verified sample paired with the path of its rendered diagram."""

    sample: SyntheticSample
    image_path: Path


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """The accumulating state threaded through every stage."""

    synthetic: tuple[SyntheticSample, ...] = ()
    rendered: tuple[RenderedSample, ...] = ()
    dataset_artifact: Any = None
    image_dir: Path | None = None
    out_dir: Path | None = None

    def with_(self, **changes: Any) -> PipelineContext:
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Stage contract + runner.
# ---------------------------------------------------------------------------


class Stage(ABC):
    """One pipeline step: ``run(ctx) -> ctx``."""

    name: str

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext: ...


class Pipeline:
    """Folds an ordered list of stages over a context, with per-stage timing."""

    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = tuple(stages)

    def run(self, ctx: PipelineContext | None = None) -> PipelineContext:
        ctx = ctx or PipelineContext()
        for stage in self.stages:
            t0 = time.time()
            ctx = stage.run(ctx)
            log.info("stage %s done in %.2fs", stage.name, time.time() - t0)
        return ctx


# ---------------------------------------------------------------------------
# Concrete generation stages (Phase 2 -> 4 -> 6).
# Collaborators are duck-typed by their stage Protocol; no backend imported here.
# ---------------------------------------------------------------------------


class SampleStage(Stage):
    """Run Algorithm 1 over a fixed set of seeds, capped at `target_n` samples.

    Each accepted sample is assigned a stable, monotonically increasing `index`
    so the renderer and dataset builder agree on its id (the fix for the
    renderer/builder filename mismatch).
    """

    name = "sample"

    def __init__(self, runner: Algorithm1Runner, seeds: Iterable[Any], target_n: int) -> None:
        self.runner = runner
        self.seeds = seeds
        self.target_n = target_n

    def run(self, ctx: PipelineContext) -> PipelineContext:
        accepted: list[SyntheticSample] = []
        for problem in self.seeds:
            for sample in self.runner.run_for_seed(problem):
                accepted.append(replace(sample, index=len(accepted)))
                if len(accepted) >= self.target_n:
                    return ctx.with_(synthetic=tuple(accepted))
        return ctx.with_(synthetic=tuple(accepted))


class RenderStage(Stage):
    """Render each synthetic sample to ``<image_dir>/<sample_id>.png``."""

    name = "render"

    def __init__(self, renderer: Any, dataset_config: DatasetConfig, *, seed_base: int = 0) -> None:
        self.renderer = renderer
        self.dataset_config = dataset_config
        self.seed_base = seed_base

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.image_dir is not None, "RenderStage requires ctx.image_dir"
        rendered: list[RenderedSample] = []
        for sample in ctx.synthetic:
            sample_id = make_sample_id(sample, self.dataset_config)
            path = ctx.image_dir / f"{sample_id}{self.dataset_config.image_ext}"
            # Per-sample render seed → distinct augmentation per sample.
            img = self.renderer.render(
                sample.construction_cdl,
                sample.drawable_image_cdl(),
                seed=self.seed_base + (sample.index or 0),
            )
            img.save(path)
            rendered.append(RenderedSample(sample=sample, image_path=path))
        return ctx.with_(rendered=tuple(rendered))


class BuildStage(Stage):
    """Assemble the rendered samples into an HF dataset (+ JSONL sidecar)."""

    name = "build"

    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.image_dir is not None and ctx.out_dir is not None
        artifact = self.builder.build(
            [r.sample for r in ctx.rendered], image_dir=ctx.image_dir, out_dir=ctx.out_dir
        )
        return ctx.with_(dataset_artifact=artifact)
