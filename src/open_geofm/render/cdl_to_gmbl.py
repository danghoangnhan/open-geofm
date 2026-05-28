"""FormalGeo CDL -> GMBL constraint list.

Blueprint §2 Phase 4: the paper builds a mapping table from FormalGeo predicates
to GMBL constraints; we mirror that table here. A "constraint dict" has the
shape::

    {"type": "<gmbl_kind>", "args": (...), "value": float | None}

A downstream layout pass (`gmbl_renderer._solve_layout`) reads these and
minimises an L2 residual to place points.
"""

from __future__ import annotations

import re

# FormalGeo predicate -> (arity in points, GMBL constraint kind, value-bearing?).
# Arity tells the parser how many point letters to read from the argument string.
# `value_bearing` constraints expect ``= <number>`` on the right-hand side.
PREDICATE_TO_GMBL: dict[str, tuple[int, str, bool]] = {
    "Point":                    (1, "point",          False),
    "Line":                     (2, "edge",           False),
    "Triangle":                 (3, "polygon",        False),
    "Quadrilateral":            (4, "polygon",        False),
    "Polygon":                  (-1, "polygon",       False),  # variable arity
    "Shape":                    (-1, "polygon",       False),
    "Collinear":                (3, "collinear",      False),
    "Parallel":                 (4, "parallel",       False),
    "ParallelBetweenLine":      (4, "parallel",       False),
    "PerpendicularBetweenLine": (4, "perpendicular",  False),
    "LengthOfLine":             (2, "length",         True),
    "MeasureOfAngle":           (3, "angle",          True),
    "AreaOfTriangle":           (3, "area_triangle",  True),
    "Equal":                    (-1, "equal",         False),
}

_PRED_RE = re.compile(r"\b([A-Z][A-Za-z]*)\(([^)]*)\)\s*(?:=\s*([-\d.]+))?")


def translate(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
) -> list[dict]:
    """Translate CDL statements into a list of GMBL constraint dicts."""
    out: list[dict] = []
    for stmt in tuple(construction_cdl) + tuple(image_cdl):
        for pred, args, val in _PRED_RE.findall(stmt):
            spec = PREDICATE_TO_GMBL.get(pred)
            if spec is None:
                continue
            _, kind, value_bearing = spec
            tokens = [t.strip() for t in args.split(",") if t.strip()]
            # `LengthOfLine(AB)` packs both points into one token — split it.
            if len(tokens) == 1 and len(tokens[0]) > 1 and tokens[0].isalpha():
                tokens = list(tokens[0])
            value: float | None = None
            if value_bearing and val:
                try:
                    value = float(val)
                except ValueError:
                    value = None
            out.append({"type": kind, "args": tuple(tokens), "value": value})
    return out
