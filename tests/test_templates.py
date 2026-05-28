"""Phase 5 step-1 template-draft tests."""

from __future__ import annotations

from open_geofm.nlg.templates import _parse_statement, draft_nl


def test_parse_value_bearing_statement() -> None:
    pred, args, value = _parse_statement("LengthOfLine(AB) = 3")
    assert pred == "LengthOfLine"
    assert args == ("A", "B")
    assert value == "3"


def test_parse_polygon() -> None:
    pred, args, value = _parse_statement("Triangle(A,B,C)")
    assert pred == "Triangle"
    assert args == ("A", "B", "C")
    assert value is None


def test_parse_angle_unsplits_three_letter_token() -> None:
    pred, args, value = _parse_statement("MeasureOfAngle(ABC) = 90")
    assert pred == "MeasureOfAngle"
    assert args == ("A", "B", "C")
    assert value == "90"


def test_draft_nl_renders_known_predicates() -> None:
    conditions = (
        "Triangle(A,B,C)",
        "LengthOfLine(AB) = 3",
        "PerpendicularBetweenLine(AB,BC)",
    )
    text = draft_nl(conditions, "LengthOfLine(AC)", seed=0)
    # The literal CDL fragments shouldn't leak into the NL draft.
    assert "Triangle(A,B,C)" not in text
    assert "LengthOfLine(AB)" not in text
    # The goal token *as a CDL fragment* shouldn't either — the goal template
    # rewrites it as a question.
    assert "LengthOfLine(AC)" not in text
    assert "A" in text and "B" in text and "C" in text


def test_draft_nl_passes_unknown_predicates_through() -> None:
    text = draft_nl(("UnknownPredicate(X,Y)",), "LengthOfLine(XY)", seed=0)
    assert "UnknownPredicate(X,Y)" in text


def test_draft_nl_is_deterministic_with_seed() -> None:
    conds = ("LengthOfLine(AB) = 3", "LengthOfLine(BC) = 4")
    a = draft_nl(conds, "LengthOfLine(AC)", seed=42)
    b = draft_nl(conds, "LengthOfLine(AC)", seed=42)
    assert a == b
