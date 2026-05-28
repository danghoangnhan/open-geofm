"""Algorithm 1 from the GeoFM paper (arXiv:2510.27448), implemented verbatim.

Blueprint §2 Phase 2 (the novel part)::

    Input: formalized seed problem set FS, number of synthetic problems m
    for P in FS:
        M_p   = MetricInfoOfProblemStatement(P)
        M_all = GatheringMetricInfo(P)        # BFS over theorems
        m_p = m
        while m_p > 1:
            n = Random(1, min(|M_p|, |M_all| − |M_p|))
            M_del = RandomSelect(M_p, n)
            M_add = RandomSelect(M_all \\ M_p, n)
            P_new = (P \\ M_del) ∪ M_add
            A_new = FormalGeoSolver(P_new)
            P_syn, A_syn = Template_and_LLM(P_new, A_new)
            if AnswerVerify(A_syn, A_new): S.add((P_syn, A_syn)); m_p -= 1
    return S
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace

from ..formal.cdl import Problem
from ..formal.solver import SolverResult
from ..formal.solver import solve as _default_solve
from .gather_metrics import gather_metric_info as _default_gather
from .gather_metrics import goal_metric_for, value_of
from .goal_picker import fallback_goal_from_trace, pick_goal

log = logging.getLogger(__name__)

# Callable shapes used by the driver; default-bound to the real Phase 1 / Phase 3
# implementations and swapped in tests for in-memory stubs.
SolveFn = Callable[[Problem], SolverResult]
GatherFn = Callable[[Problem], tuple[str, ...]]
VerifyFn = Callable[[str, str], bool]  # (nl_answer, fgps_answer) -> accepted
RewriteFn = Callable[[Problem, str], tuple[str, str]]  # (problem, fgps_answer) -> (nl_problem, nl_solution)


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    """One verified (problem, answer, trace) triple from Algorithm 1."""

    source_pid: int
    new_metrics: tuple[str, ...]
    deleted_metrics: tuple[str, ...]
    added_metrics: tuple[str, ...]
    goal: str
    answer: str
    theorem_seqs: tuple[str, ...]
    construction_cdl: tuple[str, ...] = field(default_factory=tuple)
    nl_problem: str | None = None
    nl_solution: str | None = None


def sample_new_problem(
    problem: Problem,
    m_all: tuple[str, ...],
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """One iteration of the inner swap: pick n, draw M_del / M_add, return
    `(P_new_metrics, M_del, M_add)`.

    Asserted invariants (covered by `tests/test_algorithm1_invariants.py`):
        * `len(M_del) == len(M_add) == n`
        * `set(M_del) & set(M_add) == ∅`
        * `len(P_new_metrics) == len(problem.all_metric_conditions)`
    """
    m_p = problem.all_metric_conditions
    n_max = min(len(m_p), len(m_all) - len(m_p))
    if n_max < 1:
        raise ValueError("Cannot swap: |M_p| or |M_all \\ M_p| is empty.")

    n = rng.randint(1, n_max)
    m_del = tuple(rng.sample(list(m_p), n))
    pool = [m for m in m_all if m not in m_p]
    m_add = tuple(rng.sample(pool, n))

    remaining = tuple(m for m in m_p if m not in m_del)
    p_new = remaining + m_add
    return p_new, m_del, m_add


def _split_text_image(problem: Problem, p_new: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Re-partition `p_new` into (text_cdl, image_cdl) by membership in the seed's
    original blocks; metrics from M_add that weren't in either default to text_cdl.
    Blueprint §2 Phase 2 (separable text/image allocator)."""
    text_set = set(problem.text_cdl)
    image_set = set(problem.image_cdl)
    text_out: list[str] = []
    image_out: list[str] = []
    for m in p_new:
        if m in image_set:
            image_out.append(m)
        elif m in text_set:
            text_out.append(m)
        else:
            text_out.append(m)
    return tuple(text_out), tuple(image_out)


def run_algorithm1(
    seeds: Iterable[Problem],
    m_per_seed: int,
    *,
    seed: int = 42,
    solve_fn: SolveFn | None = None,
    gather_fn: GatherFn | None = None,
    verify_fn: VerifyFn | None = None,
    rewrite_fn: RewriteFn | None = None,
    max_attempts_factor: int = 5,
) -> list[SyntheticSample]:
    """Drive Algorithm 1 end-to-end.

    Callbacks are injectable so this can be unit-tested without the heavy stack
    (Phase 1 solver, Phase 5 rewriter). Defaults bind to the real implementations
    once each phase lands.

    Args:
        seeds: iterable of seed `Problem`s (typically `iter_problems()`).
        m_per_seed: target number of accepted samples per seed (paper's `m`).
        seed: RNG seed.
        solve_fn: `Problem -> SolverResult`. Defaults to `formal.solver.solve`.
        gather_fn: `Problem -> M_all`. Defaults to `sampling.gather_metrics.gather_metric_info`.
        verify_fn: `(nl_answer, fgps_answer) -> bool`. Defaults to the verify-module
            wrapper around sympy nsimplify.
        rewrite_fn: `(problem, fgps_answer) -> (nl_problem, nl_solution)`. Must be
            provided in production (the NLG backend is configured by the caller).
        max_attempts_factor: bail after `m_per_seed * max_attempts_factor` rejects
            on a single seed to bound wall-clock.
    """
    rng = random.Random(seed)
    solve_fn = solve_fn or _default_solve
    gather_fn = gather_fn or _default_gather
    if verify_fn is None:
        from ..nlg.verify import verify as _v

        def verify_fn(nl: str, exp: str) -> bool:  # type: ignore[misc]
            return _v(nl, exp).accepted

    if rewrite_fn is None:
        raise ValueError("rewrite_fn is required; pass an NLG callback (Phase 5).")

    out: list[SyntheticSample] = []
    for problem in seeds:
        try:
            m_all = gather_fn(problem)
        except Exception as e:
            log.warning("gather_fn failed for pid=%s: %s", problem.pid, e)
            continue
        if len(m_all) <= len(problem.all_metric_conditions):
            log.info("pid=%s: |M_all| <= |M_p|, skipping (no swap pool).", problem.pid)
            continue

        accepted = 0
        attempts = 0
        max_attempts = max(1, m_per_seed * max_attempts_factor)
        while accepted < m_per_seed and attempts < max_attempts:
            attempts += 1
            try:
                p_new, m_del, m_add = sample_new_problem(problem, m_all, rng)
            except ValueError:
                break  # pool too small for any swap

            try:
                goal_metric = pick_goal(p_new, m_all, seed=rng.randrange(2**31))
            except ValueError:
                continue

            # The picked goal is a value-bearing metric like
            # `Equal(LengthOfLine(AC),5)`. FGPS expects two pieces:
            #   goal_cdl       = "Value(LengthOfLine(AC))"  (the question)
            #   problem_answer = "5"                         (the candidate answer)
            candidate_answer = value_of(goal_metric)
            if candidate_answer is None:
                # Goal isn't value-bearing (e.g. a logical relation) — skip;
                # Algorithm 1 §2 picks numeric goals.
                continue
            goal_cdl = goal_metric_for(goal_metric)

            text_cdl, image_cdl = _split_text_image(problem, p_new)
            candidate = replace(
                problem,
                text_cdl=text_cdl,
                image_cdl=image_cdl,
                goal_cdl=goal_cdl,
                theorem_seqs=(),
                answer=candidate_answer,
            )
            result = solve_fn(candidate)
            if not result.solved:
                # Phase 2 fallback: pick a different goal from the solver's trace.
                if not result.theorem_seqs:
                    continue
                try:
                    fallback_metric = fallback_goal_from_trace(result.theorem_seqs)
                except ValueError:
                    continue
                fallback_answer = value_of(fallback_metric)
                if fallback_answer is None:
                    continue
                candidate = replace(
                    candidate,
                    goal_cdl=goal_metric_for(fallback_metric),
                    answer=fallback_answer,
                )
                result = solve_fn(candidate)
                if not result.solved or result.answer is None:
                    continue
                goal_metric = fallback_metric

            assert result.answer is not None
            try:
                nl_problem, nl_solution = rewrite_fn(candidate, result.answer)
            except Exception as e:
                log.warning("rewrite_fn raised for pid=%s: %s", problem.pid, e)
                continue

            if not verify_fn(nl_solution, result.answer):
                continue

            out.append(
                SyntheticSample(
                    source_pid=problem.pid,
                    new_metrics=p_new,
                    deleted_metrics=m_del,
                    added_metrics=m_add,
                    goal=goal_metric,
                    answer=result.answer,
                    theorem_seqs=result.theorem_seqs,
                    construction_cdl=problem.construction_cdl,
                    nl_problem=nl_problem,
                    nl_solution=nl_solution,
                )
            )
            accepted += 1
    return out
