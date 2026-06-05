"""Renderer interface (matplotlib/PIL-free, so it stays import-light)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DiagramRenderer(Protocol):
    """Render `(construction_cdl, image_cdl)` to a `PIL.Image`."""

    def render(
        self,
        construction_cdl: tuple[str, ...],
        image_cdl: tuple[str, ...],
        *,
        seed: int | None = None,
    ): ...
