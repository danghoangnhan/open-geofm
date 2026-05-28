"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from open_geofm.formal.cdl import Problem


@pytest.fixture()
def toy_problem() -> Problem:
    """A minimal stand-in for a FormalGeo problem so unit tests don't need
    the real `formalgeo` data to run.
    """
    return Problem(
        pid=1,
        construction_cdl=("Triangle(A,B,C)",),
        text_cdl=("Equal(LengthOfLine(AB),3)", "Equal(LengthOfLine(BC),4)"),
        image_cdl=("PerpendicularBetweenLine(AB,BC)",),
        goal_cdl="Value(LengthOfLine(AC))",
        answer="5",
    )


@pytest.fixture()
def toy_m_all() -> tuple[str, ...]:
    """Stand-in `M_all` (the BFS-expanded metric set), in FormalGeo7K v2's
    canonical `Equal(<predicate>,<value>)` form. Strict superset of the seed's
    metrics."""
    return (
        "Equal(LengthOfLine(AB),3)",
        "Equal(LengthOfLine(BC),4)",
        "PerpendicularBetweenLine(AB,BC)",
        "Equal(LengthOfLine(AC),5)",
        "Equal(MeasureOfAngle(ABC),90)",
        "Equal(MeasureOfAngle(BAC),53.13)",
        "Equal(MeasureOfAngle(BCA),36.87)",
        "Equal(AreaOfTriangle(A,B,C),6)",
    )
