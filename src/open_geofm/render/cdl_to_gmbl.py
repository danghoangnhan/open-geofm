"""FormalGeo CDL -> typed GMBL constraint list.

A real (balanced-paren) parser replaces the old flat regex, so nested predicates
like ``Equal(LengthOfLine(AB),LengthOfLine(CD))`` parse correctly (bug #2) and
line-pair predicates split into single points (bug #3). Each `ConstraintKind`
owns its `residual` (the Strategy), so adding a kind needs no monolithic
if/elif — and the previously-declared-but-unimplemented area/equal kinds now
contribute to layout.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ConstraintKind(Enum):
    POINT = "point"
    EDGE = "edge"
    POLYGON = "polygon"
    COLLINEAR = "collinear"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    LENGTH = "length"
    ANGLE = "angle"
    AREA_TRIANGLE = "area_triangle"
    EQUAL_LENGTH = "equal_length"


# Predicate name -> (ConstraintKind, value_bearing?).
PREDICATE_TO_KIND: dict[str, tuple[ConstraintKind, bool]] = {
    "Point": (ConstraintKind.POINT, False),
    "Line": (ConstraintKind.EDGE, False),
    "Triangle": (ConstraintKind.POLYGON, False),
    "Quadrilateral": (ConstraintKind.POLYGON, False),
    "Polygon": (ConstraintKind.POLYGON, False),
    "Shape": (ConstraintKind.POLYGON, False),
    "Collinear": (ConstraintKind.COLLINEAR, False),
    "Parallel": (ConstraintKind.PARALLEL, False),
    "ParallelBetweenLine": (ConstraintKind.PARALLEL, False),
    "PerpendicularBetweenLine": (ConstraintKind.PERPENDICULAR, False),
    "LengthOfLine": (ConstraintKind.LENGTH, True),
    "MeasureOfAngle": (ConstraintKind.ANGLE, True),
    "AreaOfTriangle": (ConstraintKind.AREA_TRIANGLE, True),
}

_PRED_RE = re.compile(r"^([A-Za-z]\w*)\((.*)\)$")


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _parse(s: str):
    """Recursively parse ``Name(arg, ...)`` -> ``(name, [arg|subtree, ...])``."""
    m = _PRED_RE.match(s.strip())
    if not m:
        return None
    name, inner = m.group(1), m.group(2)
    args = [_parse(a) or a for a in _split_top_level(inner)]
    return (name, args)


def _points(token) -> list[str]:
    """Expand a point token: ``'AB'`` -> ['A','B'], ``'A'`` -> ['A']."""
    if isinstance(token, str) and token.isalpha():
        return list(token)
    return [token] if isinstance(token, str) else []


def _float(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class GmblConstraint:
    """A typed GMBL constraint. `args` are single point names; `value` is the
    numeric RHS where applicable."""

    kind: ConstraintKind
    args: tuple[str, ...]
    value: float | None = None

    def residual(self, pts: np.ndarray, idx: dict[str, int], length_scale: float) -> float:
        """Squared residual of this constraint under the current layout, or 0
        when it carries no geometric metric / references an unplaced point."""
        if any(a not in idx for a in self.args):
            return 0.0

        def p(name: str) -> np.ndarray:
            return pts[idx[name]]

        k = self.kind
        if k is ConstraintKind.LENGTH and self.value is not None and len(self.args) >= 2:
            target = self.value / length_scale
            return float((np.linalg.norm(p(self.args[1]) - p(self.args[0])) - target) ** 2)
        if k is ConstraintKind.EQUAL_LENGTH and len(self.args) >= 4:
            a, b, c, d = self.args[:4]
            return float((np.linalg.norm(p(b) - p(a)) - np.linalg.norm(p(d) - p(c))) ** 2)
        if k is ConstraintKind.PERPENDICULAR and len(self.args) >= 4:
            v1, v2 = p(self.args[1]) - p(self.args[0]), p(self.args[3]) - p(self.args[2])
            n1, n2 = np.linalg.norm(v1) or 1.0, np.linalg.norm(v2) or 1.0
            return float((v1 @ v2) / (n1 * n2)) ** 2
        if k is ConstraintKind.PARALLEL and len(self.args) >= 4:
            v1, v2 = p(self.args[1]) - p(self.args[0]), p(self.args[3]) - p(self.args[2])
            n1, n2 = np.linalg.norm(v1) or 1.0, np.linalg.norm(v2) or 1.0
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            return float(cross / (n1 * n2)) ** 2
        if k is ConstraintKind.COLLINEAR and len(self.args) >= 3:
            v1, v2 = p(self.args[1]) - p(self.args[0]), p(self.args[2]) - p(self.args[0])
            return float(v1[0] * v2[1] - v1[1] * v2[0]) ** 2
        if k is ConstraintKind.ANGLE and self.value is not None and len(self.args) >= 3:
            v1, v2 = p(self.args[0]) - p(self.args[1]), p(self.args[2]) - p(self.args[1])
            n1, n2 = np.linalg.norm(v1) or 1.0, np.linalg.norm(v2) or 1.0
            return float((v1 @ v2) / (n1 * n2) - math.cos(math.radians(self.value))) ** 2
        if k is ConstraintKind.AREA_TRIANGLE and self.value is not None and len(self.args) >= 3:
            v1, v2 = p(self.args[1]) - p(self.args[0]), p(self.args[2]) - p(self.args[0])
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            return float((area - self.value / (length_scale**2)) ** 2)
        return 0.0


def _constraints_from_tree(name: str, args: list, value: float | None) -> list[GmblConstraint]:
    """Turn one parsed predicate tree into zero or more `GmblConstraint`s."""
    # Equal(<pred>, <value-or-pred>) — value-bearing or equal-length.
    if name == "Equal" and len(args) == 2:
        lhs, rhs = args
        if isinstance(lhs, tuple):
            inner_name, inner_args = lhs
            if isinstance(rhs, tuple) and inner_name == "LengthOfLine" and rhs[0] == "LengthOfLine":
                pts = _flatten_points(inner_args) + _flatten_points(rhs[1])
                if len(pts) >= 4:
                    return [GmblConstraint(ConstraintKind.EQUAL_LENGTH, tuple(pts[:4]))]
            val = _float(rhs) if isinstance(rhs, str) else None
            return _constraints_from_tree(inner_name, inner_args, val)
        return []

    spec = PREDICATE_TO_KIND.get(name)
    if spec is None:
        return []
    kind, _ = spec
    pts = tuple(_flatten_points(args))
    if not pts:
        return []
    return [GmblConstraint(kind, pts, value)]


def _flatten_points(args: list) -> list[str]:
    out: list[str] = []
    for a in args:
        if isinstance(a, str):
            out.extend(_points(a))
    return out


def translate(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
) -> list[GmblConstraint]:
    """Translate CDL statements into typed GMBL constraints."""
    out: list[GmblConstraint] = []
    for stmt in tuple(construction_cdl) + tuple(image_cdl):
        # Top-level `Pred(...) = value` infix form (image_cdl style).
        lhs, value = stmt, None
        if "=" in stmt and "(" in stmt:
            head, _, tail = stmt.partition("=")
            if head.count("(") == head.count(")"):
                lhs, value = head.strip(), _float(tail.strip())
        tree = _parse(lhs)
        if tree is None:
            continue
        out.extend(_constraints_from_tree(tree[0], tree[1], value))
    return out


@dataclass(frozen=True, slots=True)
class Figure:
    """Derived drawing data shared by both renderers."""

    points: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    length_labels: list[tuple[str, str, float]] = field(default_factory=list)
    angle_labels: list[tuple[str, float]] = field(default_factory=list)


def figure_from_constraints(constraints: list[GmblConstraint]) -> Figure:
    """Collect points / edges / labels from constraints for the draw pass."""
    points: list[str] = []
    seen: set[str] = set()

    def add(pt: str) -> None:
        if pt not in seen:
            seen.add(pt)
            points.append(pt)

    edges: list[tuple[str, str]] = []
    length_labels: list[tuple[str, str, float]] = []
    angle_labels: list[tuple[str, float]] = []
    for c in constraints:
        for a in c.args:
            add(a)
        if c.kind is ConstraintKind.POLYGON:
            verts = list(c.args)
            edges.extend(zip(verts, verts[1:] + verts[:1], strict=False))
        elif c.kind is ConstraintKind.EDGE and len(c.args) >= 2:
            edges.append((c.args[0], c.args[1]))
        if c.kind is ConstraintKind.LENGTH and c.value is not None and len(c.args) >= 2:
            length_labels.append((c.args[0], c.args[1], c.value))
        if c.kind is ConstraintKind.ANGLE and c.value is not None and len(c.args) >= 3:
            angle_labels.append((c.args[1], c.value))
    return Figure(
        points=points, edges=edges, length_labels=length_labels, angle_labels=angle_labels
    )
