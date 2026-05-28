"""Algorithm 1 swap invariants.

Blueprint §2 Phase 2 — these invariants must hold for every iteration:
    * len(M_del) == len(M_add) == n,  1 <= n <= min(|M_p|, |M_all \\ M_p|)
    * M_del ⊆ M_p  and  M_add ⊆ M_all \\ M_p
    * M_del ∩ M_add == ∅
    * |P_new_metrics| == |M_p|         (the swap is size-preserving)
    * No element of M_add is in M_del

These tests can run today against the in-process `sample_new_problem` function,
which is the only Algorithm 1 piece independent of the (Phase 1) FGPS solver.
"""

from __future__ import annotations

import random

import pytest

from open_geofm.formal.cdl import Problem
from open_geofm.formal.solver import SolverResult
from open_geofm.sampling.algorithm1 import run_algorithm1, sample_new_problem
from open_geofm.sampling.goal_picker import fallback_goal_from_trace, pick_goal


@pytest.mark.parametrize("rng_seed", list(range(20)))
def test_swap_invariants(toy_problem: Problem, toy_m_all: tuple[str, ...], rng_seed: int) -> None:
    rng = random.Random(rng_seed)
    p_new, m_del, m_add = sample_new_problem(toy_problem, toy_m_all, rng)

    m_p = toy_problem.all_metric_conditions
    n = len(m_del)

    # 1. equal-size swap
    assert len(m_add) == n
    # 2. at least one swap; at most the size of the smaller pool
    assert 1 <= n <= min(len(m_p), len(toy_m_all) - len(m_p))
    # 3. M_del comes from M_p
    assert set(m_del).issubset(set(m_p))
    # 4. M_add comes from M_all \ M_p
    assert set(m_add).issubset(set(toy_m_all) - set(m_p))
    # 5. disjoint
    assert set(m_del).isdisjoint(set(m_add))
    # 6. size-preserving
    assert len(p_new) == len(m_p)
    # 7. P_new = (P \ M_del) ∪ M_add
    assert set(p_new) == (set(m_p) - set(m_del)) | set(m_add)


def test_swap_raises_when_pool_empty() -> None:
    """If `M_all == M_p`, there's nothing to swap in -> ValueError."""
    p = Problem(
        pid=99,
        construction_cdl=(),
        text_cdl=("a", "b"),
        image_cdl=(),
        goal_cdl="g",
    )
    rng = random.Random(0)
    with pytest.raises(ValueError):
        sample_new_problem(p, p.all_metric_conditions, rng)


def test_goal_picker_excludes_problem_statement(
    toy_problem: Problem, toy_m_all: tuple[str, ...]
) -> None:
    goal = pick_goal(toy_problem.all_metric_conditions, toy_m_all, seed=0)
    assert goal in toy_m_all
    assert goal not in toy_problem.all_metric_conditions


def test_goal_picker_raises_when_no_candidates() -> None:
    with pytest.raises(ValueError):
        pick_goal(("a", "b"), ("a", "b"), seed=0)


def test_fallback_goal_from_trace_extracts_rhs() -> None:
    trace = (
        "perpendicular_to_right_angle(1,2) -> MeasureOfAngle(ABC) = 90",
        "pythagorean(3,4,5) -> LengthOfLine(AC) = 5",
    )
    assert fallback_goal_from_trace(trace) == "LengthOfLine(AC) = 5"


def test_fallback_goal_from_trace_rejects_empty() -> None:
    with pytest.raises(ValueError):
        fallback_goal_from_trace(())


def test_run_algorithm1_driver_smoke(toy_problem: Problem, toy_m_all: tuple[str, ...]) -> None:
    """Driver: with stubbed solve/gather/verify/rewrite, the loop should produce
    `m_per_seed` accepted samples whose invariants match Algorithm 1."""

    def fake_solve(p: Problem) -> SolverResult:
        return SolverResult(solved=True, answer="5", theorem_seqs=("axiom -> done",))

    def fake_gather(_: Problem) -> tuple[str, ...]:
        return toy_m_all

    def fake_verify(nl_answer: str, fgps_answer: str) -> bool:
        return True

    def fake_rewrite(p: Problem, fgps_answer: str) -> tuple[str, str]:
        return (f"problem for pid={p.pid}", f"solution = {fgps_answer}")

    samples = run_algorithm1(
        [toy_problem],
        m_per_seed=3,
        seed=7,
        solve_fn=fake_solve,
        gather_fn=fake_gather,
        verify_fn=fake_verify,
        rewrite_fn=fake_rewrite,
    )
    assert len(samples) == 3
    for s in samples:
        assert s.source_pid == toy_problem.pid
        assert len(s.deleted_metrics) == len(s.added_metrics)
        assert set(s.deleted_metrics).isdisjoint(s.added_metrics)
        assert s.goal not in s.new_metrics
        assert s.nl_problem and s.nl_solution


def test_run_algorithm1_falls_back_on_unsolvable_goal(
    toy_problem: Problem, toy_m_all: tuple[str, ...]
) -> None:
    """If the first solve fails but produces a trace, the driver should retry
    with the fallback goal extracted from the trace."""
    calls: list[str] = []

    def fake_solve(p: Problem) -> SolverResult:
        calls.append(p.goal_cdl)
        # First call: not solved but trace present. Subsequent calls: solved.
        if len(calls) == 1:
            return SolverResult(
                solved=False, answer=None, theorem_seqs=("pyth -> Equal(LengthOfLine(AC),5)",)
            )
        return SolverResult(solved=True, answer="5", theorem_seqs=("done",))

    samples = run_algorithm1(
        [toy_problem],
        m_per_seed=1,
        seed=0,
        solve_fn=fake_solve,
        gather_fn=lambda _: toy_m_all,
        verify_fn=lambda *_: True,
        rewrite_fn=lambda p, a: ("p", "s"),
    )
    assert len(samples) == 1
    assert samples[0].goal == "Equal(LengthOfLine(AC),5)"
