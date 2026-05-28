"""Baseline matplotlib renderer.

Blueprint §2 Phase 4. Recipe:
    * white background, anti-aliased
    * 0.8 px black strokes (RGBA)
    * CMU-Serif font 10-14 pt with 1-2 px label jitter
    * dashed light-gray auxiliary lines
    * random ±5° rotation augmentation
    * resolution: shorter edge ∈ {112, 224, 336}

This is the "obviously plotted" baseline used in the Phase 9 renderer ablation
against the GMBL-style renderer. Layout heuristic (no constraint solving):
points named in `Triangle(...)`, `Quadrilateral(...)`, `Polygon(...)` or
`Shape(...)` predicates are placed as a regular n-gon; freestanding `Line(AB)`
edges connect points already placed by a polygon predicate (or fall back to
spreading the remaining points on a surrounding circle). `LengthOfLine(AB) = v`
and `MeasureOfAngle(ABC) = v` from `image_cdl` are drawn as edge / angle labels.
"""

from __future__ import annotations

import io
import math
import random
import re
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")  # headless: required inside Docker + CI
import matplotlib.pyplot as plt
from PIL import Image

# Match `Triangle(A,B,C)`, `Polygon(A,B,C,D)`, `Quadrilateral(A,B,C,D)`, `Shape(A,B,C)`.
_POLY_RE = re.compile(r"\b(?:Triangle|Quadrilateral|Polygon|Shape)\(([^)]+)\)")
# Match `Line(AB)` or `LengthOfLine(AB) = 3`.
_LINE_RE = re.compile(r"\bLine\(([A-Z][A-Z])\)")
_LENGTH_RE = re.compile(r"\bLengthOfLine\(([A-Z][A-Z])\)\s*=\s*([-\d.]+)")
_ANGLE_RE = re.compile(r"\bMeasureOfAngle\(([A-Z]{3})\)\s*=\s*([-\d.]+)")


def _parse_points_and_edges(
    statements: Iterable[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Pull point names + edges out of a CDL block. Returns (points_in_order, edges)."""
    points: list[str] = []
    edges: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_point(p: str) -> None:
        if p not in seen:
            seen.add(p)
            points.append(p)

    for stmt in statements:
        for poly in _POLY_RE.findall(stmt):
            verts = [v.strip() for v in poly.split(",") if v.strip()]
            for v in verts:
                add_point(v)
            for a, b in zip(verts, verts[1:] + verts[:1], strict=False):
                edges.append((a, b))
        for line in _LINE_RE.findall(stmt):
            a, b = line[0], line[1]
            add_point(a)
            add_point(b)
            edges.append((a, b))
    return points, edges


def _layout_regular(points: list[str], radius: float = 1.0) -> dict[str, tuple[float, float]]:
    """Place `points` on a regular n-gon centred at origin."""
    n = len(points)
    if n == 0:
        return {}
    # Start at top, go clockwise, so the first vertex sits where readers expect.
    coords: dict[str, tuple[float, float]] = {}
    for i, p in enumerate(points):
        theta = math.pi / 2 - 2 * math.pi * i / n
        coords[p] = (radius * math.cos(theta), radius * math.sin(theta))
    return coords


def _edge_label(coords: dict[str, tuple[float, float]], a: str, b: str, text: str) -> tuple[float, float, float]:
    """Midpoint + perpendicular nudge for an edge label."""
    (ax, ay), (bx, by) = coords[a], coords[b]
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.0
    # Perpendicular unit vector (rotate 90° ccw), nudge outward.
    nx, ny = -dy / norm, dx / norm
    return mx + 0.08 * nx, my + 0.08 * ny, math.degrees(math.atan2(dy, dx))


def render(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
    *,
    shorter_edge_px: int = 224,
    seed: int | None = None,
) -> Image.Image:
    """Render `(construction_cdl, image_cdl)` to a `PIL.Image`.

    `shorter_edge_px` controls the output resolution; the image is square so
    both dimensions equal `shorter_edge_px`.
    """
    rng = random.Random(seed)

    all_stmts = tuple(construction_cdl) + tuple(image_cdl)
    points, edges = _parse_points_and_edges(all_stmts)
    coords = _layout_regular(points)

    # ±5° rotation augmentation (blueprint §2 Phase 4 recipe).
    rot = math.radians(rng.uniform(-5.0, 5.0))
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    coords = {p: (x * cos_r - y * sin_r, x * sin_r + y * cos_r) for p, (x, y) in coords.items()}

    dpi = 100
    side_in = shorter_edge_px / dpi
    fig, ax = plt.subplots(figsize=(side_in, side_in), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.patch.set_facecolor("white")

    # Edges: 0.8px black anti-aliased strokes.
    drawn_edges: set[frozenset[str]] = set()
    for a, b in edges:
        key = frozenset((a, b))
        if key in drawn_edges or a not in coords or b not in coords:
            continue
        drawn_edges.add(key)
        (ax_x, ay), (bx, by) = coords[a], coords[b]
        ax.plot([ax_x, bx], [ay, by], color="black", linewidth=0.8, antialiased=True)

    # Vertex labels with small jitter.
    for p, (x, y) in coords.items():
        jx, jy = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
        ax.text(
            x + jx * 1.5,
            y + jy * 1.5 + 0.06,
            p,
            family="serif",
            fontsize=12,
            ha="center",
            va="bottom",
        )

    # Length annotations from image_cdl (and, defensively, construction_cdl).
    for stmt in all_stmts:
        m = _LENGTH_RE.search(stmt)
        if not m:
            continue
        a, b = m.group(1)[0], m.group(1)[1]
        if a not in coords or b not in coords:
            continue
        lx, ly, _ = _edge_label(coords, a, b, m.group(2))
        ax.text(lx, ly, m.group(2), family="serif", fontsize=10, ha="center", va="center")

    # Angle labels: drop near the vertex (middle character of the 3-letter angle).
    for stmt in all_stmts:
        m = _ANGLE_RE.search(stmt)
        if not m:
            continue
        vertex = m.group(1)[1]
        if vertex not in coords:
            continue
        vx, vy = coords[vertex]
        ax.text(vx + 0.12, vy - 0.12, m.group(2) + "°", family="serif", fontsize=9, ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # Force the shorter edge to the requested size; produces a square crop/resize.
    img = img.resize((shorter_edge_px, shorter_edge_px), Image.LANCZOS)
    return img
