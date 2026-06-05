"""FGPS symbolic solver: the `FGPSSolver` class + a backward-compatible `solve`.

Wraps BitSecret/FGPS forward/backward search. `formalgeo` is imported eagerly at
module top — this module is a concrete backend (reached via the registry / the
`formal` extra), never part of the base CPU import graph.

`FGPSSolver` is the clean `Solver` implementation; `solve()` / `_searcher()` /
`_problem_cdl()` remain as thin functional shims for existing callers and tests.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import formalgeo.parse.inverse_parse_m2f as _m2f  # type: ignore[import-not-found]
from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]
from formalgeo.solver.backward_search import BackwardSearcher  # type: ignore[import-not-found]
from formalgeo.solver.forward_search import ForwardSearcher  # type: ignore[import-not-found]
from func_timeout import FunctionTimedOut, func_timeout

from ..config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_DATASET_NAME,
    DEFAULT_FGPS_SEARCH_STRATEGY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SEARCHER_CACHE_SIZE,
    DEFAULT_SOLVE_STRATEGY,
    DEFAULT_SOLVE_TIMEOUT_S,
    FGPSConfig,
)
from .base import SolverResult
from .cdl import Problem, to_equal_form
from .loader import _resolve_root

log = logging.getLogger(__name__)

__all__ = ["FGPSSolver", "SolverResult", "solve"]


def build_searcher(
    *,
    predicate_gdl,
    theorem_gdl,
    strategy: str,
    search_strategy: str,
    max_depth: int,
    beam_size: int,
    theorem_usage_stats: dict,
):
    """Construct an FGPS searcher, forwarding ALL hyperparameters (the previous
    `_searcher` ignored max_depth/beam_size and hardcoded 15/20)."""
    cls = BackwardSearcher if strategy == "backward" else ForwardSearcher
    return cls(
        predicate_GDL=predicate_gdl,
        theorem_GDL=theorem_gdl,
        strategy=search_strategy,
        max_depth=max_depth,
        beam_size=beam_size,
        t_info=theorem_usage_stats,
    )


def _run_search(searcher, problem_cdl: dict) -> tuple[bool, list]:
    """init_search + search as one func_timeout target."""
    searcher.init_search(problem_cdl)
    return searcher.search()


def _extract_answer(searcher) -> str | None:
    """Pull the verified answer off a solved FGPS searcher."""
    fg_goal = getattr(searcher.problem, "goal", None)
    if fg_goal is None:
        return None
    # solved_answer holds numbers for algebra/equal goals; .answer holds the
    # logical-goal payload (tuple of point labels) for non-algebra goals.
    if fg_goal.solved_answer is not None:
        return str(fg_goal.solved_answer)
    if fg_goal.answer is not None:
        return str(fg_goal.answer)
    return None


def _derived_metrics(searcher) -> tuple[str, ...]:
    """Best-effort: snapshot the conditions FGPS proved into canonical
    `Equal(...)` CDL strings, so the Algorithm 1 fallback has a structured source
    (the paper's 'last valid inference'). Returns () if the snapshot fails."""
    try:
        snapshot = _m2f.inverse_parse_logic_to_cdl(searcher.problem)
    except Exception:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for step in sorted(snapshot):
        for cdl in snapshot[step]:
            canon = to_equal_form(cdl)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return tuple(out)


class FGPSSolver:
    """Concrete `Solver` over FGPS. Hyperparameters come from `FGPSConfig`; the
    searcher is built once and reused across `solve()` calls."""

    def __init__(
        self,
        config: FGPSConfig | None = None,
        *,
        root: Path | str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.config = config or FGPSConfig()
        self.root = _resolve_root(root)
        self.dataset_name = dataset_name or DEFAULT_DATASET_NAME
        loader = DatasetLoader(self.dataset_name, str(self.root))
        self._searcher = build_searcher(
            predicate_gdl=loader.predicate_GDL,
            theorem_gdl=loader.theorem_GDL,
            strategy=self.config.strategy,
            search_strategy=self.config.search_strategy,
            max_depth=self.config.max_depth,
            beam_size=self.config.beam_size,
            theorem_usage_stats=self.config.theorem_usage_stats,
        )

    def solve(self, problem: Problem, candidate_answer: str | None = None) -> SolverResult:
        problem_cdl = problem.to_fgps_cdl(
            candidate_answer,
            unused_goal_cdl=self.config.unused_goal_cdl,
            stub_answer=self.config.stub_answer,
        )
        try:
            solved, seqs = func_timeout(
                self.config.timeout_s, _run_search, args=(self._searcher, problem_cdl)
            )
        except FunctionTimedOut:
            return SolverResult(False, None, (), timed_out=True)
        except Exception as e:
            log.warning("FGPS solve failed for pid=%s: %s", problem.pid, e)
            return SolverResult(False, None, (), error=str(e))
        if not solved:
            return SolverResult(False, None, ())
        return SolverResult(
            solved=True,
            answer=_extract_answer(self._searcher),
            theorem_seqs=tuple(str(s) for s in (seqs or ())),
            derived_metrics=_derived_metrics(self._searcher),
        )


# ---------------------------------------------------------------------------
# Backward-compatible functional shims.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=DEFAULT_SEARCHER_CACHE_SIZE)
def _searcher(
    datasets_root: str,
    dataset_name: str,
    strategy: Literal["forward", "backward"],
    max_depth: int,
    beam_size: int,
):
    """Cached searcher per (dataset, strategy, depth, beam). FIX: forwards
    max_depth/beam_size to the constructor (previously hardcoded 15/20)."""
    loader = DatasetLoader(dataset_name, datasets_root)
    return build_searcher(
        predicate_gdl=loader.predicate_GDL,
        theorem_gdl=loader.theorem_GDL,
        strategy=strategy,
        search_strategy=DEFAULT_FGPS_SEARCH_STRATEGY,
        max_depth=max_depth,
        beam_size=beam_size,
        theorem_usage_stats={},
    )


def _problem_cdl(problem: Problem, candidate_answer: str | None) -> dict:
    """`Problem` -> FGPS dict (requires an answer to verify)."""
    return problem.to_fgps_cdl(candidate_answer, require_answer=True)


def solve(
    problem: Problem,
    strategy: Literal["forward", "backward"] = DEFAULT_SOLVE_STRATEGY,
    timeout_s: float = DEFAULT_SOLVE_TIMEOUT_S,
    *,
    candidate_answer: str | None = None,
    root: Path | str | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    max_depth: int = DEFAULT_MAX_DEPTH,
    beam_size: int = DEFAULT_BEAM_SIZE,
) -> SolverResult:
    """Functional shim: run FGPS on `problem` and return the verified answer."""
    root_path = _resolve_root(root)
    searcher = _searcher(str(root_path), dataset_name, strategy, max_depth, beam_size)
    problem_cdl = _problem_cdl(problem, candidate_answer)
    try:
        solved, seqs = func_timeout(timeout_s, _run_search, args=(searcher, problem_cdl))
    except FunctionTimedOut:
        return SolverResult(False, None, (), timed_out=True)
    except Exception as e:
        log.warning("FGPS solve failed for pid=%s: %s", problem.pid, e)
        return SolverResult(False, None, (), error=str(e))
    if not solved:
        return SolverResult(False, None, ())
    return SolverResult(
        solved=True,
        answer=_extract_answer(searcher),
        theorem_seqs=tuple(str(s) for s in (seqs or ())),
        derived_metrics=_derived_metrics(searcher),
    )
