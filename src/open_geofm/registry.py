"""The single import-isolation boundary.

Each `StageRegistry` maps a config-selected backend name to a
``(module_path, class_name)`` pair of **strings**. The concrete backend module
is imported — eagerly, at its own module top — only when `create()` runs
`importlib.import_module(module_path)` for the selected name. Therefore:

  * importing `open_geofm.registry` pulls in NO heavy dependency (the entries
    are plain strings);
  * a backend's heavy imports (`import torch`, `from formalgeo ... import`,
    `from vllm import LLM`) live eagerly at the top of its `_*_backend.py` and
    enter the process only when that backend is selected.

This is the *only* place where "which module is imported" is deferred. Nothing
imports a backend inside a function body.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _Entry:
    module_path: str
    class_name: str


class StageRegistry:
    """Name → concrete class resolver for one pipeline-stage kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str, module_path: str, class_name: str) -> None:
        self._entries[name] = _Entry(module_path, class_name)

    def resolve(self, name: str) -> type:
        """Import the backend module and return its class (no instantiation)."""
        try:
            entry = self._entries[name]
        except KeyError:
            raise ValueError(
                f"Unknown {self.kind} backend {name!r}; available: {self.available()}"
            ) from None
        module = importlib.import_module(entry.module_path)
        return getattr(module, entry.class_name)

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Resolve + instantiate the selected backend."""
        return self.resolve(name)(*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._entries)


# ---------------------------------------------------------------------------
# One registry per stage kind. Entries are STRINGS — importing this module
# imports none of the referenced backends.
# ---------------------------------------------------------------------------

SOURCES = StageRegistry("source")
SOURCES.register("formalgeo7k", "open_geofm.formal.loader", "FormalGeo7KSource")

SOLVERS = StageRegistry("solver")
SOLVERS.register("fgps", "open_geofm.formal.solver", "FGPSSolver")

GATHERERS = StageRegistry("gatherer")
GATHERERS.register("formalgeo", "open_geofm.sampling.gather_metrics", "FormalGeoGatherer")

RENDERERS = StageRegistry("renderer")
RENDERERS.register("mpl", "open_geofm.render.matplotlib_renderer", "MatplotlibRenderer")
RENDERERS.register("gmbl", "open_geofm.render.gmbl_renderer", "GmblRenderer")

LLM_CLIENTS = StageRegistry("llm_client")
LLM_CLIENTS.register("vllm", "open_geofm.nlg.rewrite_local", "LocalRewriter")
LLM_CLIENTS.register("hf", "open_geofm.nlg.rewrite_hf", "HFRewriter")
LLM_CLIENTS.register("openai", "open_geofm.nlg.rewrite_openai", "OpenAIRewriter")

VERIFIERS = StageRegistry("verifier")
VERIFIERS.register("sympy", "open_geofm.nlg.verify", "SympyVerifier")

BUILDERS = StageRegistry("builder")
BUILDERS.register("hf", "open_geofm.dataset.builder", "HFDatasetBuilder")

TRAINERS = StageRegistry("trainer")
TRAINERS.register("qwen_vl", "open_geofm.train._trl_backend", "QwenVLTrainer")

EVALUATORS = StageRegistry("evaluator")
EVALUATORS.register("vlmevalkit", "open_geofm.eval._vlmeval_backend", "VLMEvalKitEvaluator")
