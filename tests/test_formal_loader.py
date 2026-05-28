"""Phase 1 FormalGeo loader tests.

Unit tests use a synthetic fixture dataset so CI doesn't need to download 521 MB.
Integration tests against the real FormalGeo7K data are gated by
`OPEN_GEOFM_DATA` env var (pointing to a directory containing `formalgeo7k_v2/`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from open_geofm.formal import loader
from open_geofm.formal.cdl import Problem


@pytest.fixture()
def fake_dataset(tmp_path: Path):
    """Build a minimal on-disk dataset that mirrors FormalGeo7K's layout."""
    root = tmp_path / "formalgeo7k_v2"
    (root / "problems").mkdir(parents=True)
    (root / "gdl").mkdir()
    info = {"problem_number": 2, "formalgeo_version": "0.0.5", "gdl_name": "GFS-Basic"}
    (root / "info.json").write_text(json.dumps(info))
    (root / "gdl" / "predicate_GDL.json").write_text("{}")
    (root / "gdl" / "theorem_GDL.json").write_text("{}")
    # formalgeo's `get_local_datasets` reads top-level `<name>.json` index files,
    # not subdirectories — mirror the layout `download_dataset` produces.
    (tmp_path / "formalgeo7k_v2.json").write_text(json.dumps(info))
    for pid, payload in [
        (
            1,
            {
                "problem_id": 1,
                "construction_cdl": ["Shape(AB,BC,CA)"],
                "text_cdl": ["Equal(LengthOfLine(AB),3)"],
                "image_cdl": ["MeasureOfAngle(ABC) = 90"],
                "goal_cdl": "Value(LengthOfLine(AC))",
                "problem_answer": "5",
                "theorem_seqs": ["pythagorean(1)"],
            },
        ),
        (
            2,
            {
                "problem_id": 2,
                "construction_cdl": ["Shape(XY,YZ,ZX)"],
                "text_cdl": [],
                "image_cdl": [],
                "goal_cdl": "Value(x)",
                "problem_answer": "7",
                "theorem_seqs": [],
            },
        ),
    ]:
        (root / "problems" / f"{pid}.json").write_text(json.dumps(payload))
    # Reset both module-level caches so the new on-disk dataset is picked up
    # *and* doesn't poison later tests that use the real dataset.
    from open_geofm.formal import solver as _solver_mod

    loader._get_loader.cache_clear()
    _solver_mod._searcher.cache_clear()
    yield tmp_path
    loader._get_loader.cache_clear()
    _solver_mod._searcher.cache_clear()


def test_load_problem_maps_cdl_blocks(fake_dataset: Path) -> None:
    p = loader.load_problem(1, root=fake_dataset)
    assert isinstance(p, Problem)
    assert p.pid == 1
    assert p.construction_cdl == ("Shape(AB,BC,CA)",)
    assert p.text_cdl == ("Equal(LengthOfLine(AB),3)",)
    assert p.image_cdl == ("MeasureOfAngle(ABC) = 90",)
    assert p.goal_cdl == "Value(LengthOfLine(AC))"
    assert p.answer == "5"
    assert p.theorem_seqs == ("pythagorean(1)",)


def test_iter_problems_yields_all_in_order(fake_dataset: Path) -> None:
    pids = [p.pid for p in loader.iter_problems(root=fake_dataset)]
    assert pids == [1, 2]


def test_resolve_root_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPEN_GEOFM_DATA", str(tmp_path))
    assert loader._resolve_root(None) == tmp_path
    explicit = tmp_path / "elsewhere"
    assert loader._resolve_root(explicit) == explicit


# Integration tests (require the real ~521 MB dataset). Skip when missing so CI
# stays portable; run locally after `bash scripts/01_download_formalgeo7k.sh`.
_REAL_ROOT = Path(os.environ.get("OPEN_GEOFM_DATA", "data"))
_HAS_REAL = (_REAL_ROOT / "formalgeo7k_v2" / "info.json").exists() or (
    Path(__file__).resolve().parents[1] / "data" / "formalgeo7k_v2" / "info.json"
).exists()


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_REAL, reason="FormalGeo7K not downloaded.")
def test_real_dataset_pid_1_has_expected_answer() -> None:
    p = loader.load_problem(1)
    assert p.answer == "15"  # Known label for FormalGeo7K v2 pid=1.
    assert p.goal_cdl == "Value(y)"
