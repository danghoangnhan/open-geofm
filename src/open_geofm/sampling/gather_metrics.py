"""BFS over the FormalGeo theorems to enumerate `M_all` for a seed problem.

Drives `formalgeo.solver.interactive.Interactor` forward: each round applies
every theorem in the GDL; stop when a round adds no new condition or `max_depth`
rounds elapse. Conditions are snapshotted to CDL via `inverse_parse_logic_to_cdl`
and canonicalised to `Equal(...)` form.

`formalgeo` is imported eagerly at module top (concrete backend / `formal`
extra). The `inverse_parse` module is imported *as a module* so `_bfs_collect`
reads the function as an attribute at call time (keeps it patchable in tests
without an in-function import).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import formalgeo.parse.inverse_parse_m2f as _m2f  # type: ignore[import-not-found]
from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]
from formalgeo.solver.interactive import Interactor  # type: ignore[import-not-found]
from func_timeout import FunctionTimedOut, func_timeout

from ..config import (
    DEFAULT_BFS_MAX_DEPTH,
    DEFAULT_BFS_TIMEOUT_S,
    DEFAULT_DATASET_NAME,
    DEFAULT_INTERACTOR_CACHE_SIZE,
    SamplingConfig,
)
from ..formal.cdl import Problem
from ..formal.cdl import goal_metric_for as goal_metric_for
from ..formal.cdl import to_equal_form as _to_equal_form
from ..formal.cdl import value_of as value_of
from ..formal.loader import _resolve_root

log = logging.getLogger(__name__)

__all__ = ["FormalGeoGatherer", "gather_metric_info", "goal_metric_for", "value_of"]


@lru_cache(maxsize=DEFAULT_INTERACTOR_CACHE_SIZE)
def _interactor(datasets_root: str, dataset_name: str) -> Interactor:
    """Cached `Interactor`. `load_problem` resets state so reuse is safe."""
    loader = DatasetLoader(dataset_name, datasets_root)
    return Interactor(loader.predicate_GDL, loader.theorem_GDL)


def _bfs_collect(interactor, problem_cdl: dict, max_depth: int) -> tuple[str, ...]:
    """Load the problem, expand for `max_depth` rounds, snapshot canonical CDL."""
    interactor.load_problem(problem_cdl)
    theorem_names = list(interactor.parsed_theorem_GDL.keys())
    for round_idx in range(max_depth):
        updated = False
        for t_name in theorem_names:
            try:
                if interactor.apply_theorem_by_name(t_name):
                    updated = True
            except Exception as e:
                log.debug("theorem %s raised at round %d: %s", t_name, round_idx, e)
        if not updated:
            break

    snapshot = _m2f.inverse_parse_logic_to_cdl(interactor.problem)
    seen: set[str] = set()
    out: list[str] = []
    for step in sorted(snapshot):
        for cdl in snapshot[step]:
            canon = _to_equal_form(cdl)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(canon)
    return tuple(out)


def gather_metric_info(
    problem: Problem,
    max_depth: int = DEFAULT_BFS_MAX_DEPTH,
    timeout_s: float = DEFAULT_BFS_TIMEOUT_S,
    *,
    root: Path | str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> tuple[str, ...]:
    """Return `M_all`: every metric derivable from `problem` via theorem BFS, up
    to `max_depth` rounds / `timeout_s` seconds. Returns () on timeout/error."""
    root_path = _resolve_root(root)
    interactor = _interactor(str(root_path), dataset_name)
    problem_cdl = problem.to_fgps_cdl(require_answer=False)
    try:
        return func_timeout(timeout_s, _bfs_collect, args=(interactor, problem_cdl, max_depth))
    except FunctionTimedOut:
        log.info("gather_metric_info timed out for pid=%s after %.1fs", problem.pid, timeout_s)
        return ()
    except Exception as e:
        log.warning("gather_metric_info failed for pid=%s: %s", problem.pid, e)
        return ()


class FormalGeoGatherer:
    """Concrete `MetricGatherer` driving the theorem-application BFS. Holds one
    Interactor as instance state; knobs come from `SamplingConfig`."""

    def __init__(
        self,
        config: SamplingConfig | None = None,
        *,
        root: Path | str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.config = config or SamplingConfig()
        self.root = _resolve_root(root)
        self.dataset_name = dataset_name or DEFAULT_DATASET_NAME
        loader = DatasetLoader(self.dataset_name, str(self.root))
        self._interactor = Interactor(loader.predicate_GDL, loader.theorem_GDL)

    def gather(self, problem: Problem) -> tuple[str, ...]:
        problem_cdl = problem.to_fgps_cdl(require_answer=False)
        try:
            return func_timeout(
                self.config.bfs_timeout_s,
                _bfs_collect,
                args=(self._interactor, problem_cdl, self.config.bfs_max_depth),
            )
        except FunctionTimedOut:
            log.info("gather timed out for pid=%s", problem.pid)
            return ()
        except Exception as e:
            log.warning("gather failed for pid=%s: %s", problem.pid, e)
            return ()
