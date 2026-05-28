"""GMBL-inspired renderer (re-implementation of Krueger et al. 2021b's GMBL idea).

Blueprint §2 Phase 4. Pipeline:
    1. `cdl_to_gmbl.translate(construction_cdl, image_cdl)` -> GMBL constraint list
    2. `scipy.optimize.minimize` over sum-of-squares of constraints -> point coords
    3. PIL stroke renderer with hand-tuned aesthetics (textbook look)

This is a faithful re-implementation, not a port (the original GMBL source is
not publicly mirrored — see blueprint §8 risk #3). The renderer ablation in
Phase 9 quantifies how much the layout matters relative to the matplotlib
baseline.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

from .cdl_to_gmbl import translate


def _initial_layout(points: list[str], rng: random.Random) -> np.ndarray:
    """Place points on a perturbed regular n-gon. Returns a flat array of length 2n."""
    n = len(points)
    coords = np.empty((n, 2), dtype=float)
    for i in range(n):
        theta = math.pi / 2 - 2 * math.pi * i / max(n, 1)
        coords[i] = (math.cos(theta), math.sin(theta))
    coords += rng.uniform(-0.05, 0.05) * np.random.default_rng(rng.randrange(2**31)).standard_normal(coords.shape)
    return coords.flatten()


def _residuals(
    flat: np.ndarray, idx: dict[str, int], constraints: list[dict[str, Any]]
) -> float:
    """Sum of squared residuals across all known constraint kinds."""
    pts = flat.reshape(-1, 2)
    total = 0.0

    def vec(a: str, b: str) -> np.ndarray:
        return pts[idx[b]] - pts[idx[a]]

    for c in constraints:
        kind = c["type"]
        args = c["args"]
        if any(a not in idx for a in args):
            continue
        if kind == "length" and c["value"] is not None:
            # Scale FGPS length values into our [-1, 1]-ish layout space.
            v = vec(args[0], args[1])
            target = float(c["value"]) / 5.0
            total += (np.linalg.norm(v) - target) ** 2
        elif kind == "perpendicular":
            v1 = vec(args[0], args[1])
            v2 = vec(args[2], args[3])
            n1 = np.linalg.norm(v1) or 1.0
            n2 = np.linalg.norm(v2) or 1.0
            total += float((v1 @ v2) / (n1 * n2)) ** 2
        elif kind == "parallel":
            v1 = vec(args[0], args[1])
            v2 = vec(args[2], args[3])
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            n1 = np.linalg.norm(v1) or 1.0
            n2 = np.linalg.norm(v2) or 1.0
            total += float(cross / (n1 * n2)) ** 2
        elif kind == "collinear" and len(args) >= 3:
            a, b, c2 = args[0], args[1], args[2]
            v1 = pts[idx[b]] - pts[idx[a]]
            v2 = pts[idx[c2]] - pts[idx[a]]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            total += float(cross) ** 2
        elif kind == "angle" and c["value"] is not None and len(args) == 3:
            a, b, c2 = args
            v1 = pts[idx[a]] - pts[idx[b]]
            v2 = pts[idx[c2]] - pts[idx[b]]
            n1 = np.linalg.norm(v1) or 1.0
            n2 = np.linalg.norm(v2) or 1.0
            cos_target = math.cos(math.radians(float(c["value"])))
            total += float((v1 @ v2) / (n1 * n2) - cos_target) ** 2
    return total


def _solve_layout(
    constraints: list[dict[str, Any]], rng: random.Random
) -> dict[str, tuple[float, float]]:
    """Numerical layout: collect point names, set up an L2 objective, minimise."""
    point_set: list[str] = []
    seen: set[str] = set()
    for c in constraints:
        for a in c["args"]:
            if isinstance(a, str) and len(a) == 1 and a not in seen:
                seen.add(a)
                point_set.append(a)
    if not point_set:
        return {}
    idx = {p: i for i, p in enumerate(point_set)}
    x0 = _initial_layout(point_set, rng)
    res = minimize(_residuals, x0, args=(idx, constraints), method="L-BFGS-B")
    coords = res.x.reshape(-1, 2)
    # Normalise into [-1, 1]^2 so the renderer can map to canvas pixels.
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = (maxs - mins).max() or 1.0
    centre = (mins + maxs) / 2.0
    coords = (coords - centre) / (span / 2.0) * 0.85
    return {p: (float(coords[idx[p]][0]), float(coords[idx[p]][1])) for p in point_set}


def _draw(
    coords: dict[str, tuple[float, float]],
    constraints: list[dict[str, Any]],
    canvas_px: int,
) -> Image.Image:
    """Hand-tuned PIL renderer: ivory background, ~2px ink strokes, serif labels."""
    img = Image.new("RGB", (canvas_px, canvas_px), color=(252, 250, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSerif.ttf", size=max(11, canvas_px // 22))
    except OSError:
        font = ImageFont.load_default()

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        x, y = p
        return (canvas_px * (0.5 + 0.45 * x), canvas_px * (0.5 - 0.45 * y))

    drawn_edges: set[frozenset[str]] = set()

    def draw_edge(a: str, b: str, value: float | None = None) -> None:
        key = frozenset((a, b))
        if key in drawn_edges or a not in coords or b not in coords:
            return
        drawn_edges.add(key)
        pa, pb = to_px(coords[a]), to_px(coords[b])
        draw.line([pa, pb], fill=(20, 20, 20), width=2)
        if value is not None:
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            label = f"{value:g}"
            draw.text((mx + 8 * nx, my + 8 * ny), label, font=font, fill=(20, 20, 20))

    for c in constraints:
        if c["type"] == "polygon":
            verts = list(c["args"])
            for a, b in zip(verts, verts[1:] + verts[:1], strict=False):
                draw_edge(a, b)
        elif c["type"] == "edge":
            draw_edge(c["args"][0], c["args"][1])
        elif c["type"] == "length" and len(c["args"]) >= 2:
            draw_edge(c["args"][0], c["args"][1], c["value"])

    # Vertex dots + labels.
    for p, xy in coords.items():
        px_xy = to_px(xy)
        r = max(2, canvas_px // 80)
        draw.ellipse(
            [px_xy[0] - r, px_xy[1] - r, px_xy[0] + r, px_xy[1] + r], fill=(20, 20, 20)
        )
        draw.text((px_xy[0] + r + 2, px_xy[1] - r - 14), p, font=font, fill=(20, 20, 20))

    return img


def render(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
    *,
    shorter_edge_px: int = 336,
    seed: int | None = None,
) -> Image.Image:
    """Render `(construction_cdl, image_cdl)` via the GMBL-style pipeline."""
    rng = random.Random(seed)
    constraints = translate(construction_cdl, image_cdl)
    coords = _solve_layout(constraints, rng)
    if not coords:
        # No recognised constraints → blank ivory canvas (matches `_draw` background)
        # so the smoke test still gets a valid PIL.Image at the right size.
        return Image.new("RGB", (shorter_edge_px, shorter_edge_px), color=(252, 250, 245))
    return _draw(coords, constraints, shorter_edge_px)
