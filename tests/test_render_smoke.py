"""Renderer smoke tests.

Both renderers are Phase 4 stubs - the tests are xfail until then. They lock in
the *interface*: every renderer takes `(construction_cdl, image_cdl)` and
returns a `PIL.Image`.
"""

from __future__ import annotations

from PIL import Image

from open_geofm.render import gmbl_renderer, matplotlib_renderer

_CONSTRUCTION = ("Triangle(A,B,C)",)
_IMAGE = ("LengthOfLine(AB) = 3", "LengthOfLine(BC) = 4")


def test_matplotlib_renderer_returns_image() -> None:
    img = matplotlib_renderer.render(_CONSTRUCTION, _IMAGE, shorter_edge_px=224, seed=0)
    assert isinstance(img, Image.Image)
    assert min(img.size) == 224
    assert img.mode == "RGB"


def test_matplotlib_renderer_resolution_options() -> None:
    for edge in (112, 224, 336):
        img = matplotlib_renderer.render(_CONSTRUCTION, _IMAGE, shorter_edge_px=edge, seed=0)
        assert img.size == (edge, edge)


def test_gmbl_renderer_returns_image() -> None:
    img = gmbl_renderer.render(_CONSTRUCTION, _IMAGE, shorter_edge_px=336, seed=0)
    assert isinstance(img, Image.Image)
    assert min(img.size) == 336


def test_gmbl_renderer_handles_unknown_predicates() -> None:
    """Statements with no recognised predicates should still return a valid image."""
    img = gmbl_renderer.render(("Garbage(X)",), (), shorter_edge_px=112, seed=0)
    assert img.size == (112, 112)
