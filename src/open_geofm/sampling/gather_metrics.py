"""BFS over the FormalGeo theorems to enumerate `M_all` for a seed problem.

Blueprint §2 Phase 2:
    "We sample a random number n (where n ≤ min(|M_p|, |M_all| − |M_p|)).
     Next, we replace n metric conditions from M_p with n new conditions sampled
     from the remaining metric set M_all − M_p ..."

We drive `formalgeo.solver.interactive.Interactor` forward: each round, apply
every theorem in the GDL via `apply_theorem_by_name`; stop when a round adds no
new conditions or `max_depth` rounds have elapsed. Conditions are snapshotted
back to CDL strings via `inverse_parse_logic_to_cdl` (predicate-level
deduplication is built in).

FormalGeo7K v2's `theorem_GDL.json` contains ~234 theorems (the paper's "196"
is the v1 number; both work the same way).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from func_timeout import FunctionTimedOut, func_timeout

from ..formal.cdl import Problem
from ..formal.loader import _resolve_root

log = logging.getLogger(__name__)

# FGPS emits derived metrics in `Value(<predicate>(...),<n>)` form, but
# FormalGeo7K seed text_cdl uses `Equal(<predicate>(...),<n>)`. Canonicalise to
# `Equal(...)` so the Algorithm 1 swap can compare strings and `value_of` can
# extract RHS uniformly.
_VALUE_RE = re.compile(r"^Value\((.+),([^,]+)\)$")
_EQUAL_RE = re.compile(r"^Equal\((.+),([^,]+)\)$")


def _to_equal_form(cdl: str) -> str:
    """`Value(LengthOfLine(AB),5)` -> `Equal(LengthOfLine(AB),5)`. Identity otherwise."""
    m = _VALUE_RE.match(cdl)
    if not m:
        return cdl
    return f"Equal({m.group(1)},{m.group(2)})"


def value_of(cdl: str) -> str | None:
    """Pull the RHS out of an `Equal(X,v)` / `Value(X,v)` metric. Returns the
    raw value token (may be a number or a symbolic expression like ``x+21``);
    None if the CDL isn't value-bearing."""
    for rx in (_EQUAL_RE, _VALUE_RE):
        m = rx.match(cdl)
        if m:
            return m.group(2).strip()
    return None


def goal_metric_for(cdl: str) -> str:
    """`Equal(LengthOfLine(AB),5)` -> `Value(LengthOfLine(AB))` — the goal-CDL
    form FGPS expects (predicate name + arg, no value)."""
    for rx in (_EQUAL_RE, _VALUE_RE):
        m = rx.match(cdl)
        if m:
            return f"Value({m.group(1)})"
    return cdl


@lru_cache(maxsize=2)
def _interactor(datasets_root: str, dataset_name: str):
    """Cached `Interactor`. `load_problem` resets state so it's safe to reuse."""
    from formalgeo.data.data import DatasetLoader  # type: ignore[import-not-found]
    from formalgeo.solver.interactive import Interactor  # type: ignore[import-not-found]

    loader = DatasetLoader(dataset_name, datasets_root)
    return Interactor(loader.predicate_GDL, loader.theorem_GDL)


def _problem_cdl(problem: Problem) -> dict:
    """`Problem` -> dict shape FGPS's `parse_problem_cdl` expects.

    `problem_answer` is required by the parser but unused for forward expansion,
    so we pass a stub `"0"` when the seed answer is missing (e.g. synthesised
    Algorithm 1 candidates).
    """
    return {
        "problem_id": problem.pid,
        "construction_cdl": list(problem.construction_cdl),
        "text_cdl": list(problem.text_cdl),
        "image_cdl": list(problem.image_cdl),
        "goal_cdl": problem.goal_cdl or "Value(unused)",
        "problem_answer": str(problem.answer) if problem.answer is not None else "0",
    }


def _bfs_collect(interactor, problem_cdl: dict, max_depth: int) -> tuple[str, ...]:
    """Body of the BFS: load problem, expand for `max_depth` rounds, snapshot CDL."""
    from formalgeo.parse.inverse_parse_m2f import (  # type: ignore[import-not-found]
        inverse_parse_logic_to_cdl,
    )

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

    snapshot = inverse_parse_logic_to_cdl(interactor.problem)
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
    max_depth: int = 1,
    timeout_s: float = 30.0,
    *,
    root: Path | str | None = None,
    dataset_name: str = "formalgeo7k_v2",
) -> tuple[str, ...]:
    """Return `M_all`: every metric condition derivable from `problem`'s
    construction via BFS over theorem applications, up to `max_depth` rounds or
    `timeout_s` seconds (whichever comes first).

    On timeout the partial result is *not* returned (the underlying mutable
    state may be inconsistent); caller gets an empty tuple and can fall back.

    Defaults: `max_depth=1, timeout_s=30`. One round applies all ~234 theorems
    once and is enough for >90% of FormalGeo7K seeds to produce a usable
    `|M_all| > |M_p|`. Heavier seeds with many premises (e.g. circle problems
    with 20+ metrics) can blow the timeout — those samples just get dropped by
    the Algorithm 1 driver, matching the blueprint's "8-core multiprocessing
    target: ~5K samples/hr" expectation.
    """
    root_path = _resolve_root(root)
    interactor = _interactor(str(root_path), dataset_name)
    problem_cdl = _problem_cdl(problem)

    try:
        return func_timeout(
            timeout_s,
            _bfs_collect,
            args=(interactor, problem_cdl, max_depth),
        )
    except FunctionTimedOut:
        log.info("gather_metric_info timed out for pid=%s after %.1fs", problem.pid, timeout_s)
        return ()
    except Exception as e:
        log.warning("gather_metric_info failed for pid=%s: %s", problem.pid, e)
        return ()
