"""Phase 6 dataset builder tests (JSONL fallback path — `datasets` is GPU-extras)."""

from __future__ import annotations

import json
from pathlib import Path

from open_geofm.dataset.builder import build
from open_geofm.sampling.algorithm1 import SyntheticSample


def _sample(pid: int = 1) -> SyntheticSample:
    return SyntheticSample(
        source_pid=pid,
        new_metrics=("Equal(LengthOfLine(AB),3)", "Equal(MeasureOfAngle(ABC),90)"),
        deleted_metrics=("Equal(LengthOfLine(BC),4)",),
        added_metrics=("Equal(MeasureOfAngle(ABC),90)",),
        goal="Equal(LengthOfLine(AC),5)",
        answer="5",
        theorem_seqs=("axiom -> done",),
        construction_cdl=("Triangle(A,B,C)",),
        nl_problem="In △ABC, AB = 3 and ∠ABC = 90°. Find AC.",
        nl_solution="AC = 5 by the Pythagorean theorem.",
    )


def test_build_emits_jsonl_with_bundled_cdls(tmp_path: Path) -> None:
    out = build([_sample(), _sample(pid=2)], image_dir=tmp_path / "img", out_dir=tmp_path / "out")
    # No `datasets` installed in host venv → returns the JSONL sidecar dict.
    assert isinstance(out, dict)
    assert out["n_records"] == 2

    records = [
        json.loads(line) for line in (tmp_path / "out" / "records.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    r = records[0]
    assert r["source_seed_pid"] == 1
    assert r["construction_cdl"] == ["Triangle(A,B,C)"]
    assert r["goal_cdl"] == "Equal(LengthOfLine(AC),5)"
    assert r["theorem_seq"] == ["axiom -> done"]
    # The image path points into image_dir even though we never wrote a PNG.
    assert r["image"].endswith(f"{r['id']}.png")
    # Added metric ends up on the image side.
    assert "Equal(MeasureOfAngle(ABC),90)" in r["image_cdl"]


def test_build_dedupes_repeated_cdls(tmp_path: Path) -> None:
    """FormalGeo7K seeds sometimes list the same metric in text+image; the
    builder must dedupe so JSONL records don't carry redundant copies."""
    sample = SyntheticSample(
        source_pid=7,
        new_metrics=(
            "Equal(LengthOfLine(AB),3)",
            "Equal(LengthOfLine(AB),3)",  # dup
            "Equal(MeasureOfAngle(ABC),90)",
        ),
        deleted_metrics=(),
        added_metrics=("Equal(MeasureOfAngle(ABC),90)",),
        goal="Equal(LengthOfLine(AC),5)",
        answer="5",
        theorem_seqs=(),
        construction_cdl=("Triangle(A,B,C)", "Triangle(A,B,C)"),
    )
    build([sample], image_dir=tmp_path / "img", out_dir=tmp_path / "out")
    rec = json.loads((tmp_path / "out" / "records.jsonl").read_text())
    assert rec["text_cdl"].count("Equal(LengthOfLine(AB),3)") == 1
    assert rec["construction_cdl"] == ["Triangle(A,B,C)"]
