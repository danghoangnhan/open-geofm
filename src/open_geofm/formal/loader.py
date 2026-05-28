"""Load FormalGeo7K problems by PID.

Blueprint §2 Phase 1. Wraps the `formalgeo` PyPI package (BitSecret/formalgeo).
CPU-only.

Default data root: ``$OPEN_GEOFM_DATA / formalgeo7k_v2``. Run
``scripts/01_download_formalgeo7k.sh`` once to populate it (~521 MB download).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from .cdl import Problem

_DEFAULT_DATASET = "formalgeo7k_v2"


def _resolve_root(root: Path | str | None) -> Path:
    """Resolve the FormalGeo7K root: explicit arg → env var → repo `data/` default."""
    if root is not None:
        return Path(root)
    env = os.environ.get("OPEN_GEOFM_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data"


@lru_cache(maxsize=4)
def _get_loader(datasets_root: str, dataset_name: str):
    """Cached DatasetLoader instance. Lazy import keeps `formalgeo` out of the
    base venv's import graph (it's a 'formal' extra)."""
    from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]

    return DatasetLoader(dataset_name, datasets_root)


def load_problem(
    pid: int,
    root: Path | str | None = None,
    dataset_name: str = _DEFAULT_DATASET,
) -> Problem:
    """Load problem `pid` from the FormalGeo7K dataset.

    Args:
        pid: 1-indexed FormalGeo7K problem id (FormalGeo7K uses 1..7000).
        root: directory that *contains* `<dataset_name>/`. Defaults to
            `$OPEN_GEOFM_DATA` or the repo's `data/`.
        dataset_name: `formalgeo7k_v2` (default) or `formalgeo7k_v1`.
    """
    root = _resolve_root(root)
    raw = _get_loader(str(root), dataset_name).get_problem(pid)
    return _problem_from_raw(raw)


def iter_problems(
    root: Path | str | None = None,
    dataset_name: str = _DEFAULT_DATASET,
) -> Iterator[Problem]:
    """Yield every FormalGeo7K `Problem` in PID order."""
    root = _resolve_root(root)
    loader = _get_loader(str(root), dataset_name)
    n = loader.info["problem_number"]
    for pid in range(1, n + 1):
        yield _problem_from_raw(loader.get_problem(pid))


def _problem_from_raw(raw: dict) -> Problem:
    """Convert a FormalGeo7K problem JSON into our `Problem` dataclass.

    The JSON schema (FormalGeo7K v2) is::

        {"problem_id": int,
         "construction_cdl": list[str],
         "text_cdl": list[str],
         "image_cdl": list[str],
         "goal_cdl": str,
         "problem_answer": str,
         "theorem_seqs": list[str],
         ...}
    """
    return Problem(
        pid=int(raw["problem_id"]),
        construction_cdl=tuple(raw.get("construction_cdl") or ()),
        text_cdl=tuple(raw.get("text_cdl") or ()),
        image_cdl=tuple(raw.get("image_cdl") or ()),
        goal_cdl=raw.get("goal_cdl") or "",
        theorem_seqs=tuple(raw.get("theorem_seqs") or ()),
        answer=raw.get("problem_answer"),
    )
