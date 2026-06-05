"""Build pipelines + stage collaborators from an `AppConfig`.

This is the wiring layer: it reads the config's backend selectors, resolves each
concrete class through the registry (the single importlib boundary), and
composes them into a `Pipeline`. Kept separate from `pipeline.py` so the
orchestration spine imports no concrete backend.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import registry
from .config import AppConfig
from .nlg.rewriter import TemplateProblemRewriter
from .pipeline import BuildStage, Pipeline, RenderStage, SampleStage
from .sampling.algorithm1 import Algorithm1Runner


def build_source(config: AppConfig) -> Any:
    return registry.SOURCES.create(config.pipeline.source_backend, config.formalgeo)


def build_solver(config: AppConfig) -> Any:
    return registry.SOLVERS.create(config.pipeline.solver_backend, config.fgps)


def build_gatherer(config: AppConfig) -> Any:
    return registry.GATHERERS.create(config.pipeline.gatherer_backend, config.sampling)


def build_rewriter(config: AppConfig) -> TemplateProblemRewriter:
    """Template rewriter, with an LLM client injected only in `mode='llm'`."""
    llm = (
        registry.LLM_CLIENTS.create(config.pipeline.llm_client_backend)
        if config.nlg.mode == "llm"
        else None
    )
    return TemplateProblemRewriter(config.nlg, llm=llm)


def build_verifier(config: AppConfig) -> Any:
    return registry.VERIFIERS.create(config.pipeline.verifier_backend, config.verify)


def build_renderer(config: AppConfig) -> Any:
    return registry.RENDERERS.create(config.pipeline.renderer_backend, config.render)


def build_builder(config: AppConfig) -> Any:
    return registry.BUILDERS.create(config.pipeline.builder_backend, config.dataset)


def build_runner(config: AppConfig) -> Algorithm1Runner:
    """Assemble Algorithm 1 from config-selected collaborators."""
    return Algorithm1Runner(
        gatherer=build_gatherer(config),
        solver=build_solver(config),
        rewriter=build_rewriter(config),
        verifier=build_verifier(config),
        config=config.sampling,
    )


def build_generation_pipeline(
    config: AppConfig, seeds: Iterable[Any], *, target_n: int
) -> Pipeline:
    """The Phase 2->4->6 generation pipeline (sample -> render -> build)."""
    return Pipeline(
        [
            SampleStage(build_runner(config), seeds, target_n),
            RenderStage(build_renderer(config), config.dataset),
            BuildStage(build_builder(config)),
        ]
    )
