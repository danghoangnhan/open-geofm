"""Baseline matplotlib renderer (the "obviously plotted" Phase-9 ablation arm).

Shares the fixed CDL parser (`cdl_to_gmbl`) with the GMBL renderer, so both see
the same points/edges/labels. Layout is a plain regular n-gon (no constraint
solving). All aesthetics come from `RenderConfig`.
"""

from __future__ import annotations

import io
import math
import random

import matplotlib

matplotlib.use("Agg")  # headless: required inside Docker + CI
import matplotlib.pyplot as plt
from PIL import Image

from ..config import DEFAULT_MPL_RESOLUTION, RenderConfig
from .cdl_to_gmbl import figure_from_constraints, translate


def _layout_regular(points: list[str], radius: float) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    n = len(points)
    for i, p in enumerate(points):
        theta = math.pi / 2 - 2 * math.pi * i / max(n, 1)
        coords[p] = (radius * math.cos(theta), radius * math.sin(theta))
    return coords


def _rgb01(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in rgb)  # type: ignore[return-value]


def render(
    construction_cdl: tuple[str, ...],
    image_cdl: tuple[str, ...],
    *,
    shorter_edge_px: int = DEFAULT_MPL_RESOLUTION,
    seed: int | None = None,
    config: RenderConfig | None = None,
) -> Image.Image:
    """Render `(construction_cdl, image_cdl)` to a square `PIL.Image`."""
    config = config or RenderConfig.matplotlib_baseline()
    rng = random.Random(seed)

    figure = figure_from_constraints(translate(construction_cdl, image_cdl))
    coords = _layout_regular(figure.points, config.layout_radius)

    rot = math.radians(rng.uniform(-config.rotation_deg, config.rotation_deg))
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    coords = {p: (x * cos_r - y * sin_r, x * sin_r + y * cos_r) for p, (x, y) in coords.items()}

    dpi = config.dpi
    side_in = shorter_edge_px / dpi
    fig, ax = plt.subplots(figsize=(side_in, side_in), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-config.axis_limit, config.axis_limit)
    ax.set_ylim(-config.axis_limit, config.axis_limit)
    fig.patch.set_facecolor("white")
    ink = _rgb01(config.ink_rgb)

    drawn: set[frozenset[str]] = set()
    for a, b in figure.edges:
        key = frozenset((a, b))
        if key in drawn or a not in coords or b not in coords:
            continue
        drawn.add(key)
        (ax_x, ay), (bx, by) = coords[a], coords[b]
        ax.plot([ax_x, bx], [ay, by], color=ink, linewidth=config.stroke_width, antialiased=True)

    for p, (x, y) in coords.items():
        jx, jy = rng.uniform(-config.label_jitter, config.label_jitter), rng.uniform(
            -config.label_jitter, config.label_jitter
        )
        ax.text(
            x + jx * 1.5,
            y + jy * 1.5 + config.vertex_label_dy,
            p,
            family=config.font_family,
            fontsize=config.vertex_font_pt,
            ha="center",
            va="bottom",
        )

    for a, b, value in figure.length_labels:
        if a not in coords or b not in coords:
            continue
        (ax_x, ay), (bx, by) = coords[a], coords[b]
        mx, my = (ax_x + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax_x, by - ay
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        ax.text(
            mx + config.edge_label_offset * nx,
            my + config.edge_label_offset * ny,
            f"{value:g}",
            family=config.font_family,
            fontsize=config.length_font_pt,
            ha="center",
            va="center",
        )

    for vertex, value in figure.angle_labels:
        if vertex not in coords:
            continue
        vx, vy = coords[vertex]
        ax.text(
            vx + config.angle_label_offset,
            vy - config.angle_label_offset,
            f"{value:g}{config.degree_symbol}",
            family=config.font_family,
            fontsize=config.angle_font_pt,
            ha="center",
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=config.pad_inches)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize((shorter_edge_px, shorter_edge_px), Image.LANCZOS)


class MatplotlibRenderer:
    """Concrete `DiagramRenderer` (matplotlib baseline)."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig.matplotlib_baseline()

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
