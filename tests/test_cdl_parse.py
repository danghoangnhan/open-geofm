"""Tests for the CDL parser and Problem dataclass.

`parse_cdl` is a Phase 1 stub - those tests are marked xfail until then.
The Problem dataclass invariants are testable today.
"""

from __future__ import annotations

import pytest

from open_geofm.formal.cdl import Problem, parse_cdl


def test_problem_all_metric_conditions_is_text_plus_image(toy_problem: Problem) -> None:
    assert toy_problem.all_metric_conditions == toy_problem.text_cdl + toy_problem.image_cdl


def test_problem_is_immutable(toy_problem: Problem) -> None:
    with pytest.raises((AttributeError, TypeError)):
        toy_problem.answer = "wrong"  # type: ignore[misc]


def test_parse_cdl_ignores_comments_and_blanks() -> None:
    raw = """
    # this is a comment
    Triangle(A,B,C)

    LengthOfLine(AB) = 3
    """
    assert parse_cdl(raw) == ("Triangle(A,B,C)", "LengthOfLine(AB) = 3")


def test_parse_cdl_strips_trailing_comment() -> None:
    assert parse_cdl("LengthOfLine(AB) = 3  # given") == ("LengthOfLine(AB) = 3",)


def test_parse_cdl_empty_returns_empty_tuple() -> None:
    assert parse_cdl("") == ()
    assert parse_cdl("# only a comment\n\n") == ()
