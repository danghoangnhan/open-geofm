"""GMBL-inspired renderer (re-implementation of Krueger et al. 2021b's GMBL idea).

Pipeline: CDL -> typed GMBL constraints -> `scipy.optimize.minimize` over the
sum of per-constraint residuals -> PIL stroke renderer with textbook aesthetics.
All tunables come from `RenderConfig`; nothing is hardcoded inline.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize

from ..config import DEFAULT_GMBL_RESOLUTION, RenderConfig
from .cdl_to_gmbl import GmblConstraint, figure_from_constraints, translate


def _initial_layout(n: int, rng: random.Random, jitter: float) -> np.ndarray:
    coords = np.empty((n, 2), dtype=float)
    for i in range(n):
        theta = math.pi / 2 - 2 * math.pi * i / max(n, 1)
        coords[i] = (math.cos(theta), math.sin(theta))
    gen = np.random.default_rng(rng.randrange(2**31))
    coords += jitter * gen.standard_normal(coords.shape)
    return coords.flatten()


def _residuals(
    flat: np.ndarray, idx: dict[str, int], constraints: list[GmblConstraint], length_scale: float
) -> float:
    pts = flat.reshape(-1, 2)
    return sum(c.residual(pts, idx, length_scale) for c in constraints)


def _solve_layout(
    constraints: list[GmblConstraint], rng: random.Random, config: RenderConfig
) -> dict[str, tuple[float, float]]:
    point_set: list[str] = []
    seen: set[str] = set()
    for c in constraints:
        for a in c.args:
            if a not in seen:
                seen.add(a)
                point_set.append(a)
    if not point_set:
        return {}
    idx = {p: i for i, p in enumerate(point_set)}
    x0 = _initial_layout(len(point_set), rng, config.init_jitter)
    res = minimize(
        _residuals,
        x0,
        args=(idx, constraints, config.length_layout_scale),
        method=config.optimizer_method,
    )
    coords = res.x.reshape(-1, 2)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    span = (maxs - mins).max() or 1.0
    centre = (mins + maxs) / 2.0
    coords = (coords - centre) / (span / 2.0) * config.layout_fill_fraction
    return {p: (float(coords[idx[p]][0]), float(coords[idx[p]][1])) for p in point_set}


def _draw(
    coords: dict[str, tuple[float, float]],
    constraints: list[GmblConstraint],
    canvas_px: int,
    config: RenderConfig,
) -> Image.Image:
    img = Image.new("RGB", (canvas_px, canvas_px), color=config.bg_rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            config.font_file, size=max(config.font_pt_min, canvas_px // config.font_pt_divisor)
        )
    except OSError:
        font = ImageFont.load_default()

    margin = config.canvas_margin

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        x, y = p
        return (canvas_px * (0.5 + margin * x), canvas_px * (0.5 - margin * y))

    drawn: set[frozenset[str]] = set()

    def draw_edge(a: str, b: str, value: float | None = None) -> None:
        key = frozenset((a, b))
        if key in drawn or a not in coords or b not in coords:
            return
        drawn.add(key)
        pa, pb = to_px(coords[a]), to_px(coords[b])
        draw.line([pa, pb], fill=config.ink_rgb, width=2)
        if value is not None:
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            off = config.gmbl_label_offset_px
            draw.text((mx + off * nx, my + off * ny), f"{value:g}", font=font, fill=config.ink_rgb)

    figure = figure_from_constraints(constraints)
    for a, b in figure.edges:
        draw_edge(a, b)
    for a, b, value in figure.length_labels:
        draw_edge(a, b, value)

    r = max(2, canvas_px // config.vertex_radius_divisor)
    for p, xy in coords.items():
        px, py = to_px(xy)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=config.ink_rgb)
        draw.text((px + r + 2, py - r - 14), p, font=font, fill=config.ink_rgb)
    return img


def render(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
    *,
    shorter_edge_px: int = DEFAULT_GMBL_RESOLUTION,
    seed: int | None = None,
    config: RenderConfig | None = None,
) -> Image.Image:
    """Render `(construction_cdl, image_cdl)` via the GMBL-style pipeline."""
    config = config or RenderConfig.gmbl_textbook()
    rng = random.Random(seed)
    constraints = translate(construction_cdl, image_cdl)
    coords = _solve_layout(constraints, rng, config)
    if not coords:
        return Image.new("RGB", (shorter_edge_px, shorter_edge_px), color=config.bg_rgb)
    return _draw(coords, constraints, shorter_edge_px, config)


class GmblRenderer:
    """Concrete `DiagramRenderer` (GMBL-style)."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig.gmbl_textbook()

    def render(
        self,
        construction_cdl: tuple[str, ...],
        image_cdl: tuple[str, ...],
        *,
        seed: int | None = None,
    ) -> Image.Image:
        return render(
            construction_cdl,
            image_cdl,
            shorter_edge_px=self.config.resolution,
            seed=seed,
            config=self.config,
        )
