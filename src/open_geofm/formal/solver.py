"""Wraps BitSecret/FGPS for forward/backward symbolic search.

Blueprint §2 Phase 1 + Phase 3. CPU-only. 15s timeout per call (FGPS auto_run hangs
on some seeds).

Backed by `formalgeo.solver.{forward_search,backward_search}` from the
`formalgeo` PyPI package. The dataset's predicate / theorem GDLs are reused
across calls via an `lru_cache`-d searcher factory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from func_timeout import FunctionTimedOut, func_timeout

from .cdl import Problem
from .loader import _resolve_root

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Outcome of a single solver call."""

    solved: bool
    answer: str | None
    theorem_seqs: tuple[str, ...]
    timed_out: bool = False
    error: str | None = None


@lru_cache(maxsize=4)
def _searcher(
    datasets_root: str,
    dataset_name: str,
    strategy: Literal["forward", "backward"],
    max_depth: int,
    beam_size: int,
):
    """Cached searcher per (dataset, strategy). Each call to `init_search` resets
    its internal state, so a single instance is safe to reuse across problems."""
    from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]
    from formalgeo.solver.backward_search import (  # type: ignore[import-not-found]
        BackwardSearcher,
    )
    from formalgeo.solver.forward_search import (  # type: ignore[import-not-found]
        ForwardSearcher,
    )

    loader = DatasetLoader(dataset_name, datasets_root)
    cls = BackwardSearcher if strategy == "backward" else ForwardSearcher
    # `t_info={}` keeps every theorem in the search space (no pruning); FGPS
    # ships per-theorem usage stats but they're an optimisation, not required.
    return cls(
        predicate_GDL=loader.predicate_GDL,
        theorem_GDL=loader.theorem_GDL,
        strategy="bfs",
        max_depth=15,
        beam_size=20,
        t_info={},
    )


def _problem_cdl(problem: Problem, candidate_answer: str | None) -> dict:
    """`Problem` -> the dict shape that FGPS's `parse_problem_cdl` expects.

    FGPS is an *answer-verifier*: it proves that `goal.item == problem_answer`.
    For seed problems (Phase 1) the answer comes from the dataset; for
    synthesised problems (Phase 2 Algorithm 1) the driver derives a candidate
    answer from `gather_metric_info`'s BFS and passes it in.
    """
    answer = candidate_answer if candidate_answer is not None else problem.answer
    if answer is None:
        raise ValueError(
            "FGPS requires a candidate answer to verify; pass `candidate_answer=...` "
            "or set `Problem.answer`."
        )
    return {
        "problem_id": problem.pid,
        "construction_cdl": list(problem.construction_cdl),
        "text_cdl": list(problem.text_cdl),
        "image_cdl": list(problem.image_cdl),
        "goal_cdl": problem.goal_cdl,
        "problem_answer": str(answer),
    }


def _run_search(searcher, problem_cdl: dict) -> tuple[bool, list]:
    """Wrap searcher.init_search + .search into a single call so func_timeout
    only needs one target."""
    searcher.init_search(problem_cdl)
    return searcher.search()


def solve(
    problem: Problem,
    strategy: Literal["forward", "backward"] = "backward",
    timeout_s: float = 15.0,
    *,
    candidate_answer: str | None = None,
    root: Path | str | None = None,
    dataset_name: str = "formalgeo7k_v2",
    max_depth: int = 15,
    beam_size: int = 20,
) -> SolverResult:
    """Run FGPS on `problem` and return the verified answer + proof trace.

    FGPS is an answer-verifier; pass `candidate_answer` for synthesised problems
    (Algorithm 1 derives this from `gather_metric_info`). For seed problems the
    answer is read from `problem.answer`.

    Times out at `timeout_s` (FGPS occasionally hangs on pathological seeds).
    Errors are caught and surfaced as `SolverResult(solved=False, error=...)`
    so the Phase 2 driver can keep going.
    """
    root_path = _resolve_root(root)
    searcher = _searcher(str(root_path), dataset_name, strategy, max_depth, beam_size)
    problem_cdl = _problem_cdl(problem, candidate_answer)

    try:
        solved, seqs = func_timeout(timeout_s, _run_search, args=(searcher, problem_cdl))
    except FunctionTimedOut:
        return SolverResult(solved=False, answer=None, theorem_seqs=(), timed_out=True)
    except Exception as e:
        log.warning("FGPS solve failed for pid=%s: %s", problem.pid, e)
        return SolverResult(solved=False, answer=None, theorem_seqs=(), error=str(e))

    if not solved:
        return SolverResult(solved=False, answer=None, theorem_seqs=())

    answer = None
    fg_goal = getattr(searcher.problem, "goal", None)
    if fg_goal is not None:
        # solved_answer holds numbers for algebra/equal goals; .answer holds the
        # logical-goal payload (tuple of point labels) for non-algebra goals.
        if fg_goal.solved_answer is not None:
            answer = str(fg_goal.solved_answer)
        elif fg_goal.answer is not None:
            answer = str(fg_goal.answer)
    return SolverResult(
        solved=True,
        answer=answer,
        theorem_seqs=tuple(str(s) for s in (seqs or ())),
    )


# Silence the noisy "DEBUG" prints FGPS emits to stdout when its `debug` flag
# isn't suppressed at the env level.
os.environ.setdefault("FORMALGEO_VERBOSE", "0")
