"""Load FormalGeo7K problems by PID: the `FormalGeo7KSource` class + functional
shims (`load_problem` / `iter_problems`).

`formalgeo` is imported eagerly at module top — this is a concrete backend
(reached via the registry / the `formal` extra), not part of the base graph.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]

from ..config import (
    DEFAULT_DATA_DIRNAME,
    DEFAULT_DATA_ENV_VAR,
    DEFAULT_DATASET_NAME,
    DEFAULT_LOADER_CACHE_SIZE,
    DEFAULT_PROBLEM_COUNT_KEY,
    FormalGeoConfig,
)
from .cdl import Problem

__all__ = ["FormalGeo7KSource", "iter_problems", "load_problem"]


def _resolve_root(root: Path | str | None) -> Path:
    """Resolve the FormalGeo7K root: explicit arg → env var → repo `data/`."""
    if root is not None:
        return Path(root)
    env = os.environ.get(DEFAULT_DATA_ENV_VAR)
    if env:
        return Path(env)
    # loader.py lives at src/open_geofm/formal/loader.py → repo root is parents[3].
    return Path(__file__).resolve().parents[3] / DEFAULT_DATA_DIRNAME


def _problem_from_raw(raw: dict) -> Problem:
    """Convert a FormalGeo7K problem JSON into our `Problem` dataclass."""
    return Problem(
        pid=int(raw["problem_id"]),
        construction_cdl=tuple(raw.get("construction_cdl") or ()),
        text_cdl=tuple(raw.get("text_cdl") or ()),
        image_cdl=tuple(raw.get("image_cdl") or ()),
        goal_cdl=raw.get("goal_cdl") or "",
        theorem_seqs=tuple(raw.get("theorem_seqs") or ()),
        answer=raw.get("problem_answer"),
    )


@lru_cache(maxsize=DEFAULT_LOADER_CACHE_SIZE)
def _get_loader(datasets_root: str, dataset_name: str) -> DatasetLoader:
    """Cached DatasetLoader per (root, dataset)."""
    return DatasetLoader(dataset_name, datasets_root)


def load_problem(
    pid: int,
    root: Path | str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Problem:
    """Load problem `pid` from the FormalGeo7K dataset."""
    root = _resolve_root(root)
    raw = _get_loader(str(root), dataset_name).get_problem(pid)
    return _problem_from_raw(raw)


def iter_problems(
    root: Path | str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Iterator[Problem]:
    """Yield every FormalGeo7K `Problem` in PID order."""
    root = _resolve_root(root)
    loader = _get_loader(str(root), dataset_name)
    n = loader.info[DEFAULT_PROBLEM_COUNT_KEY]
    for pid in range(1, n + 1):
        yield _problem_from_raw(loader.get_problem(pid))


class FormalGeo7KSource:
    """Concrete `ProblemSource` over the FormalGeo7K corpus. Holds one
    DatasetLoader as instance state (replacing the lru_cache-on-string hack)."""

    def __init__(
        self,
        config: FormalGeoConfig | None = None,
        *,
        root: Path | str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.config = config or FormalGeoConfig()
        self.root = _resolve_root(root)
        self.dataset_name = dataset_name or self.config.dataset_name
        self._loader = DatasetLoader(self.dataset_name, str(self.root))

    def load(self, pid: int) -> Problem:
        return _problem_from_raw(self._loader.get_problem(pid))

    @property
    def problem_count(self) -> int:
        return int(self._loader.info[self.config.problem_count_key])

    def iter_problems(self) -> Iterator[Problem]:
        for pid in range(self.config.first_pid, self.problem_count + 1):
            yield self.load(pid)
