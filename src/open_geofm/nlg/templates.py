"""Predicate-to-NL templates for each FormalGeo predicate.

Blueprint §2 Phase 5 (Step 1 of the two-step NLG):
    "for each formal language expression in FormalGeo, we use GPT-4o to generate
     20 corresponding natural language templates, which are then manually
     reviewed and corrected."

For each predicate we keep 3-6 English variants; the rewriter (Step 2) picks
one and smooths it into natural prose. The seed set below covers the predicates
used by the conftest toy + the most common ~20 FormalGeo predicates so the
draft step is meaningful for the headline phase-6 dataset. Extend as new
predicates appear in M_all during Phase 2 BFS.

Template slot names match the order points appear in the CDL arg list.
"""

from __future__ import annotations

import random
import re

TEMPLATES: dict[str, list[str]] = {
    "Point": [
        "Point {a} is given.",
        "Let {a} be a point.",
        "There is a point {a}.",
    ],
    "Line": [
        "Line {a}{b} is drawn.",
        "{a}{b} is a line segment.",
        "Segment {a}{b} is given.",
    ],
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
    "Polygon": [
        "Polygon {a} is given.",
        "Consider polygon {a}.",
    ],
    "Collinear": [
        "Points {a}, {b}, and {c} are collinear.",
        "{a}, {b}, {c} lie on a common line.",
    ],
    "Parallel": [
        "Line {a}{b} is parallel to line {c}{d}.",
        "{a}{b} ∥ {c}{d}.",
        "Segments {a}{b} and {c}{d} are parallel.",
    ],
    "ParallelBetweenLine": [
        "Line {a}{b} is parallel to line {c}{d}.",
        "{a}{b} ∥ {c}{d}.",
    ],
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
    "AreaOfTriangle": [
        "The area of triangle {a}{b}{c} is {value}.",
        "[△{a}{b}{c}] = {value}.",
    ],
    "Equal": [
        "{a} equals {b}.",
        "{a} = {b}.",
    ],
}

# Goal templates: the goal is a *question*, not a statement.
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
    "AreaOfTriangle": [
        "Find the area of triangle {a}{b}{c}.",
        "What is the area of △{a}{b}{c}?",
    ],
}

_SLOT_NAMES = ("a", "b", "c", "d", "e", "f")
_STMT_RE = re.compile(r"\b([A-Z][A-Za-z]*)\(([^)]*)\)\s*(?:=\s*([-\d.]+|[^)]+?))?\s*$")


def _parse_statement(stmt: str) -> tuple[str, tuple[str, ...], str | None] | None:
    """Return ``(predicate, args, value)`` or `None` if the line doesn't match."""
    m = _STMT_RE.match(stmt.strip())
    if not m:
        return None
    pred = m.group(1)
    raw_args = [t.strip() for t in m.group(2).split(",") if t.strip()]
    # `LengthOfLine(AB)` packs both points into one token — split it.
    if pred in {"LengthOfLine", "Line"} and len(raw_args) == 1 and len(raw_args[0]) == 2:
        raw_args = list(raw_args[0])
    if pred == "MeasureOfAngle" and len(raw_args) == 1 and len(raw_args[0]) == 3:
        raw_args = list(raw_args[0])
    return pred, tuple(raw_args), m.group(3)


def _fill(template: str, args: tuple[str, ...], value: str | None) -> str:
    slots = {name: arg for name, arg in zip(_SLOT_NAMES, args, strict=False)}
    if value is not None:
        slots["value"] = value
    try:
        return template.format(**slots)
    except KeyError:
        # Missing slot → return template unchanged so the downstream rewriter can
        # still smooth it (and so we don't silently drop the statement).
        return template


def draft_nl(metric_conditions: tuple[str, ...], goal: str, *, seed: int | None = None) -> str:
    """Stitch a draft NL problem statement from the conditions + goal using TEMPLATES.

    Unknown predicates are passed through verbatim (the rewriter can clean them
    up). Output: one sentence per condition, then the goal sentence.
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
        if not choices:
            out.append(stmt)
            continue
        out.append(_fill(rng.choice(choices), args, value))

    parsed_goal = _parse_statement(goal)
    if parsed_goal is None:
        out.append(goal)
    else:
        pred, args, _ = parsed_goal
        choices = GOAL_TEMPLATES.get(pred)
        if choices:
            out.append(_fill(rng.choice(choices), args, None))
        else:
            out.append(f"Find {goal}.")
    return " ".join(out)
