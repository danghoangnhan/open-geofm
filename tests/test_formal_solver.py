"""Phase 1 FGPS solver wrapper tests.

The fast tests check the wrapper's error/timeout handling without actually
running FGPS (which needs the 521 MB dataset). The slow integration test —
gated by `OPEN_GEOFM_DATA` and `pytest -m slow` — solves a real seed problem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_geofm.formal import solver
from open_geofm.formal.cdl import Problem
from open_geofm.formal.solver import SolverResult, _problem_cdl, solve


def test_problem_cdl_requires_answer() -> None:
    p = Problem(pid=1, construction_cdl=(), text_cdl=(), image_cdl=(), goal_cdl="Value(x)")
    with pytest.raises(ValueError, match="candidate answer"):
        _problem_cdl(p, candidate_answer=None)


def test_problem_cdl_uses_problem_answer_when_no_candidate() -> None:
    p = Problem(
        pid=1, construction_cdl=(), text_cdl=(), image_cdl=(), goal_cdl="Value(x)", answer="42"
    )
    cdl = _problem_cdl(p, candidate_answer=None)
    assert cdl["problem_answer"] == "42"


def test_problem_cdl_candidate_overrides_problem_answer() -> None:
    p = Problem(
        pid=1, construction_cdl=(), text_cdl=(), image_cdl=(), goal_cdl="Value(x)", answer="42"
    )
    cdl = _problem_cdl(p, candidate_answer="13")
    assert cdl["problem_answer"] == "13"


def test_solve_returns_error_result_when_fgps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If FGPS itself raises (e.g. missing GDL), we surface it on the result."""

    class _Boom:
        def init_search(self, *_):
            raise RuntimeError("simulated")

        def search(self):
            raise RuntimeError("unreachable")

    monkeypatch.setattr(solver, "_searcher", lambda *a, **k: _Boom())
    p = Problem(
        pid=99,
        construction_cdl=(),
        text_cdl=(),
        image_cdl=(),
        goal_cdl="Value(x)",
        answer="1",
    )
    r = solve(p, timeout_s=5)
    assert isinstance(r, SolverResult)
    assert r.solved is False
    assert r.error and "simulated" in r.error


# Integration: actually run FGPS on a known-easy pid.
_REAL_ROOT = Path(os.environ.get("OPEN_GEOFM_DATA", "data"))
_HAS_REAL = (_REAL_ROOT / "formalgeo7k_v2" / "info.json").exists() or (
    Path(__file__).resolve().parents[1] / "data" / "formalgeo7k_v2" / "info.json"
).exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not _HAS_REAL, reason="FormalGeo7K not downloaded.")
def test_solve_real_seed_pid_10() -> None:
    import warnings

    from open_geofm.formal.loader import load_problem

    warnings.filterwarnings("ignore")
    p = load_problem(10)
    r = solve(p, timeout_s=60)
    assert r.solved is True
    assert r.answer == "5"
    assert len(r.theorem_seqs) >= 1
