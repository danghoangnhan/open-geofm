"""Predicate-to-NL templates for each FormalGeo predicate.

Blueprint §2 Phase 5 (Step 1 of the two-step NLG): a draft is stitched from
predicate templates, then (optionally) smoothed by an LLM. For each predicate we
keep a few English variants; the rewriter picks one and smooths it.

Slot names match the order points appear in the CDL arg list (`{a}`, `{b}`, …);
`{points}` is the full point list for variadic predicates; `{value}` is the RHS.
"""

from __future__ import annotations

import random
import re

# Predicates whose args are two 2-letter line tokens (e.g. `Parallel(AB,CD)`),
# which must be split into four single points to fill the {a}{b}∥{c}{d} slots.
_LINE_PAIR_PREDICATES = frozenset({"Parallel", "ParallelBetweenLine", "PerpendicularBetweenLine"})
# Predicates rendered with the full {points} list rather than fixed slots.
_VARIADIC_PREDICATES = frozenset({"Polygon", "Shape", "Collinear", "Cocircular"})

TEMPLATES: dict[str, list[str]] = {
    "Point": ["Point {a} is given.", "Let {a} be a point.", "There is a point {a}."],
    "Line": ["Line {a}{b} is drawn.", "{a}{b} is a line segment.", "Segment {a}{b} is given."],
    "Triangle": [
        "Triangle {a}{b}{c} is given.",
        "Consider triangle {a}{b}{c}.",
        "{a}{b}{c} forms a triangle.",
    ],
    "Quadrilateral": [
        "Quadrilateral {a}{b}{c}{d} is given.",
        "Consider quadrilateral {a}{b}{c}{d}.",
        "{a}{b}{c}{d} is a four-sided figure.",
    ],
    "Polygon": ["Polygon {points} is given.", "Consider polygon {points}."],
    "Shape": ["Shape {points} is given.", "Consider the figure {points}."],
    "Collinear": [
        "Points {points} are collinear.",
        "{points} lie on a common line.",
    ],
    "Cocircular": ["Points {points} are concyclic.", "{points} lie on a common circle."],
    "Parallel": [
        "Line {a}{b} is parallel to line {c}{d}.",
        "{a}{b} ∥ {c}{d}.",
        "Segments {a}{b} and {c}{d} are parallel.",
    ],
    "ParallelBetweenLine": ["Line {a}{b} is parallel to line {c}{d}.", "{a}{b} ∥ {c}{d}."],
    "PerpendicularBetweenLine": [
        "Line {a}{b} is perpendicular to line {c}{d}.",
        "{a}{b} ⊥ {c}{d}.",
        "{a}{b} meets {c}{d} at a right angle.",
    ],
    "LengthOfLine": [
        "The length of {a}{b} is {value}.",
        "|{a}{b}| = {value}.",
        "Segment {a}{b} measures {value}.",
    ],
    "MeasureOfAngle": [
        "The measure of ∠{a}{b}{c} is {value}°.",
        "∠{a}{b}{c} = {value}°.",
        "Angle {a}{b}{c} measures {value} degrees.",
    ],
    "AreaOfTriangle": ["The area of triangle {a}{b}{c} is {value}.", "[△{a}{b}{c}] = {value}."],
    "Equal": ["{a} equals {b}.", "{a} = {b}."],
}

GOAL_TEMPLATES: dict[str, list[str]] = {
    "LengthOfLine": [
        "Find the length of {a}{b}.",
        "What is |{a}{b}|?",
        "Compute the length of segment {a}{b}.",
    ],
    "MeasureOfAngle": [
        "Find the measure of ∠{a}{b}{c}.",
        "What is the size of angle {a}{b}{c}?",
    ],
    "AreaOfTriangle": ["Find the area of triangle {a}{b}{c}.", "What is the area of △{a}{b}{c}?"],
}

_SLOT_NAMES = ("a", "b", "c", "d", "e", "f")
_STMT_RE = re.compile(r"\b([A-Z][A-Za-z]*)\(([^)]*)\)\s*(?:=\s*([-\d.]+|[^)]+?))?\s*$")
_BRACE_RE = re.compile(r"\{[a-z]+\}")


def _parse_statement(stmt: str) -> tuple[str, tuple[str, ...], str | None] | None:
    """Return ``(predicate, args, value)`` or `None` if the line doesn't match."""
    m = _STMT_RE.match(stmt.strip())
    if not m:
        return None
    pred = m.group(1)
    raw_args = [t.strip() for t in m.group(2).split(",") if t.strip()]
    # Packed single-token forms: `LengthOfLine(AB)`, `MeasureOfAngle(ABC)`.
    if pred in {"LengthOfLine", "Line"} and len(raw_args) == 1 and len(raw_args[0]) == 2:
        raw_args = list(raw_args[0])
    elif pred == "MeasureOfAngle" and len(raw_args) == 1 and len(raw_args[0]) == 3:
        raw_args = list(raw_args[0])
    elif pred in _LINE_PAIR_PREDICATES:
        # `Parallel(AB,CD)` -> ('A','B','C','D') so the four point slots fill
        # (fix for the brace-leak on line-pair predicates, bug #12).
        split: list[str] = []
        for t in raw_args:
            if len(t) == 2 and t.isalpha():
                split.extend(t)
            else:
                split.append(t)
        raw_args = split
    return pred, tuple(raw_args), m.group(3)


def _fill(template: str, args: tuple[str, ...], value: str | None) -> str | None:
    """Fill a template; return None if any slot is left unfilled (so the caller
    can fall back to the raw statement instead of leaking literal braces)."""
    slots: dict[str, str] = {name: arg for name, arg in zip(_SLOT_NAMES, args, strict=False)}
    slots["points"] = ", ".join(args)  # variadic predicates (fix bug #13)
    if value is not None:
        slots["value"] = value
    try:
        filled = template.format(**slots)
    except (KeyError, IndexError):
        return None
    return None if _BRACE_RE.search(filled) else filled


def draft_nl(metric_conditions: tuple[str, ...], goal: str, *, seed: int | None = None) -> str:
    """Stitch a draft NL problem statement from the conditions + goal.

    Unknown predicates (and any whose template can't be fully filled) pass
    through verbatim so the downstream rewriter can clean them up — but a literal
    `{slot}` never reaches the output.
    """
    rng = random.Random(seed)
    out: list[str] = []
    for stmt in metric_conditions:
        parsed = _parse_statement(stmt)
        if parsed is None:
            out.append(stmt)
            continue
        pred, args, value = parsed
        choices = TEMPLATES.get(pred)
        filled = _fill(rng.choice(choices), args, value) if choices else None
        out.append(filled if filled is not None else stmt)

    parsed_goal = _parse_statement(goal)
    if parsed_goal is None:
        out.append(goal)
    else:
        pred, args, _ = parsed_goal
        choices = GOAL_TEMPLATES.get(pred)
        filled = _fill(rng.choice(choices), args, None) if choices else None
        out.append(filled if filled is not None else f"Find {goal}.")
    return " ".join(out)
