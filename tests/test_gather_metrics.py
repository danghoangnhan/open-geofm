"""Phase 2 BFS tests.

Fast tests stub the Interactor so we exercise the BFS control flow without
loading the 234-theorem GDL. A gated `@slow` integration test runs the real BFS
on a known-fast seed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_geofm.formal.cdl import Problem
from open_geofm.sampling import gather_metrics
from open_geofm.sampling.gather_metrics import _to_equal_form, goal_metric_for, value_of


def test_to_equal_form_canonicalises_value_form() -> None:
    assert _to_equal_form("Value(LengthOfLine(AB),5)") == "Equal(LengthOfLine(AB),5)"
    # Pass-through for already-canonical or non-value-bearing items.
    assert _to_equal_form("Equal(LengthOfLine(AB),5)") == "Equal(LengthOfLine(AB),5)"
    assert _to_equal_form("PerpendicularBetweenLine(AB,BC)") == "PerpendicularBetweenLine(AB,BC)"


def test_value_of_extracts_rhs() -> None:
    assert value_of("Equal(LengthOfLine(AB),5)") == "5"
    assert value_of("Value(MeasureOfAngle(ABC),90)") == "90"
    # Non-value-bearing returns None.
    assert value_of("PerpendicularBetweenLine(AB,BC)") is None


def test_goal_metric_for_drops_value() -> None:
    assert goal_metric_for("Equal(LengthOfLine(AB),5)") == "Value(LengthOfLine(AB))"
    assert goal_metric_for("Value(LengthOfLine(AB),5)") == "Value(LengthOfLine(AB))"


class _FakeProblem:
    def __init__(self) -> None:
        self.condition = type("C", (), {"items": {}, "ids_of_step": {}})()
        self.parsed_predicate_GDL = {"Preset": {"Construction": set(), "BasicEntity": set()}}


class _FakeInteractor:
    """Stand-in for `formalgeo.solver.interactive.Interactor` — exposes just the
    surface `_bfs_collect` calls."""

    def __init__(self, derived_per_round: list[list[str]]):
        self._derived = derived_per_round
        self._round = 0
        self.parsed_theorem_GDL = {f"t_{i}": None for i in range(3)}
        self.problem = _FakeProblem()
        self.loaded_with: dict | None = None

    def load_problem(self, problem_cdl: dict) -> None:
        self.loaded_with = problem_cdl
        self._round = 0

    def apply_theorem_by_name(self, t_name: str) -> bool:
        # Update only once per round (on the first theorem).
        if t_name == next(iter(self.parsed_theorem_GDL)):
            self._round += 1
            return self._round <= len(self._derived)
        return False


def test_bfs_stops_when_no_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_bfs_collect` should break out of its loop once a round adds nothing."""
    fake = _FakeInteractor(derived_per_round=[["X"]])

    def fake_inverse_parse(problem):
        return {0: ["A"], 1: ["B", "B"]}  # `B` deduplicated by the BFS

    # Patch the lazy import inside the BFS body.
    import formalgeo.parse.inverse_parse_m2f as m2f  # type: ignore[import-not-found]

    monkeypatch.setattr(m2f, "inverse_parse_logic_to_cdl", fake_inverse_parse)
    out = gather_metrics._bfs_collect(fake, {"problem_id": 1}, max_depth=5)
    # Two unique CDL strings, deduped across steps and within a step.
    assert out == ("A", "B")
    # Round counter advanced once (the round that added "X"), then stopped.
    assert fake._round == 2  # one productive round, then the empty round breaks the loop


def test_bfs_respects_max_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every round produces an update, `_bfs_collect` should still stop at `max_depth`."""
    fake = _FakeInteractor(derived_per_round=[["x"]] * 10)  # always-updates

    import formalgeo.parse.inverse_parse_m2f as m2f  # type: ignore[import-not-found]

    monkeypatch.setattr(m2f, "inverse_parse_logic_to_cdl", lambda p: {0: []})
    gather_metrics._bfs_collect(fake, {"problem_id": 1}, max_depth=3)
    assert fake._round == 3


def test_gather_metric_info_returns_empty_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long-running BFS should be cut off and return ()."""

    def slow_bfs(*args, **kwargs):
        import time

        time.sleep(5)

    monkeypatch.setattr(gather_metrics, "_bfs_collect", slow_bfs)
    monkeypatch.setattr(gather_metrics, "_interactor", lambda *a, **k: object())
    p = Problem(
        pid=1,
        construction_cdl=(),
        text_cdl=(),
        image_cdl=(),
        goal_cdl="Value(x)",
        answer="0",
    )
    out = gather_metrics.gather_metric_info(p, timeout_s=0.2)
    assert out == ()


# Integration: against the real GDL, on a known-fast seed (pid=200, 8 metrics).
_REAL_ROOT = Path(os.environ.get("OPEN_GEOFM_DATA", "data"))
_HAS_REAL = (_REAL_ROOT / "formalgeo7k_v2" / "info.json").exists() or (
    Path(__file__).resolve().parents[1] / "data" / "formalgeo7k_v2" / "info.json"
).exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL, reason="FormalGeo7K not downloaded.")
def test_gather_metric_info_real_pid_200() -> None:
    import warnings

    from open_geofm.formal.loader import load_problem

    warnings.filterwarnings("ignore")
    p = load_problem(200)
    m_all = gather_metrics.gather_metric_info(p, max_depth=1, timeout_s=90)
    assert len(m_all) > len(p.all_metric_conditions), (
        f"BFS produced {len(m_all)} metrics; expected > seed's {len(p.all_metric_conditions)}"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL, reason="FormalGeo7K not downloaded.")
def test_algorithm1_end_to_end_real_seed() -> None:
    """End-to-end: BFS → swap → goal-pick → FGPS-verify → SyntheticSample.

    Drives the full Phase 2 pipeline on a known-fast seed and asserts at least
    one accepted sample comes back with a non-empty answer + new metrics that
    actually differ from the seed's.
    """
    import warnings

    from open_geofm.formal.loader import load_problem
    from open_geofm.formal.solver import solve as fgps_solve
    from open_geofm.sampling.algorithm1 import run_algorithm1

    warnings.filterwarnings("ignore")
    seed = load_problem(200)

    samples = run_algorithm1(
        [seed],
        m_per_seed=1,
        seed=7,
        solve_fn=lambda p: fgps_solve(p, timeout_s=15),
        gather_fn=lambda p: gather_metrics.gather_metric_info(p, max_depth=1, timeout_s=60),
        verify_fn=lambda *_: True,
        rewrite_fn=lambda p, a: (f"Q for pid={p.pid}", f"A = {a}"),
        max_attempts_factor=10,
    )
    assert len(samples) >= 1
    s = samples[0]
    assert s.source_pid == seed.pid
    assert s.answer is not None
    assert set(s.deleted_metrics).isdisjoint(s.added_metrics)
    assert s.goal not in s.new_metrics
