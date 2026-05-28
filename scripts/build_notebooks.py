"""Materialize the pedagogical notebooks from in-repo Python specs.

Each notebook's content lives in a small `_specs/` Python file as a list of
`(kind, source)` tuples. This script imports the spec, builds the notebook
via `nbformat`, executes every cell (CPU-only kernel), and writes it to
``notebooks/<name>.ipynb`` with outputs preserved (so the .ipynb renders on
GitHub without a re-run).

Usage::

    uv run python scripts/build_notebooks.py                  # build all
    uv run python scripts/build_notebooks.py 01 03           # selected only
    uv run python scripts/build_notebooks.py --no-execute    # build w/o running

Add a notebook by dropping a `<NN>_<slug>.py` file into
``scripts/notebook_specs/`` that exposes ``TITLE`` and ``CELLS``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "scripts" / "notebook_specs"
OUT_DIR = REPO_ROOT / "notebooks"

app = typer.Typer(add_completion=False)


def _load_spec(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Import a `<NN>_<slug>.py` spec module and return ``(title, cells)``."""
    spec = importlib.util.spec_from_file_location(f"_nb_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    title = getattr(mod, "TITLE", path.stem)
    cells = getattr(mod, "CELLS")
    return title, cells


def _build_notebook(title: str, cells: Iterable[tuple[str, str]]) -> Any:
    """Build an nbformat v4 notebook from ``(kind, source)`` pairs."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb_cells: list[Any] = []
    for kind, source in cells:
        if kind == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(source))
        elif kind == "code":
            nb_cells.append(nbf.v4.new_code_cell(source))
        else:
            raise ValueError(f"Unknown cell kind {kind!r} (expected 'markdown' or 'code').")
    nb["cells"] = nb_cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "open_geofm": {"title": title},
    }
    return nb


def _execute_notebook(nb: Any, work_dir: Path) -> None:
    """Run every cell in-process. Raises on the first cell error."""
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(work_dir)}},
    )
    client.execute()


def _write_notebook(nb: Any, path: Path) -> None:
    import nbformat as nbf

    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))


def _discover_specs(selectors: list[str] | None) -> list[Path]:
    """List spec paths, optionally filtered by stem-prefix (e.g. ``["01", "03"]``)."""
    specs = sorted(SPEC_DIR.glob("*.py"))
    if not selectors:
        return specs
    matches: list[Path] = []
    for sel in selectors:
        hit = [p for p in specs if p.stem.startswith(sel)]
        if not hit:
            raise typer.BadParameter(f"No spec matches {sel!r} under {SPEC_DIR}")
        matches.extend(hit)
    # Preserve sorted order, drop duplicates.
    return sorted(set(matches))


@app.command()
def main(
    names: list[str] | None = typer.Argument(
        None, help="Optional stem-prefix filters (e.g. 01 03). Default: all specs."
    ),
    execute: bool = typer.Option(True, "--execute/--no-execute", help="Run cells before saving."),
) -> None:
    """Build (and execute) every notebook under `scripts/notebook_specs/`."""
    if not SPEC_DIR.exists():
        raise typer.BadParameter(f"Spec dir not found: {SPEC_DIR}")

    paths = _discover_specs(names)
    if not paths:
        typer.echo(f"No specs found in {SPEC_DIR}.", err=True)
        raise typer.Exit(code=1)

    for spec_path in paths:
        title, cells = _load_spec(spec_path)
        nb = _build_notebook(title, cells)
        if execute:
            typer.echo(f"executing {spec_path.stem} ({len(cells)} cells)...")
            _execute_notebook(nb, work_dir=REPO_ROOT)
        out_path = OUT_DIR / f"{spec_path.stem}.ipynb"
        _write_notebook(nb, out_path)
        typer.echo(f"  wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    app()
