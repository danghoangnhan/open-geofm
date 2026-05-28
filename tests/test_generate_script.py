"""Phase 2-6 driver script tests — `scripts/02_generate_dataset.py`.

We import the script as a module (it's stamped with `__main__` guard) and
exercise its helpers without launching the CLI. Heavy parts (renderer,
Algorithm 1) are stubbed so the test runs in <1s.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from open_geofm.sampling.algorithm1 import SyntheticSample

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_generate_dataset.py"


def _load_script_module():
    import sys

    spec = importlib.util.spec_from_file_location("gen_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses look up `cls.__module__` in `sys.modules`; register the module
    # so `_WorkerConfig` can be defined without an AttributeError.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_make_rewriter_template_is_deterministic() -> None:
    mod = _load_script_module()
    rewriter = mod._make_rewriter("template")
    from open_geofm.formal.cdl import Problem

    p = Problem(
        pid=1,
        construction_cdl=("Triangle(A,B,C)",),
        text_cdl=("Equal(LengthOfLine(AB),3)",),
        image_cdl=("Equal(MeasureOfAngle(ABC),90)",),
        goal_cdl="Value(LengthOfLine(AC))",
        answer="5",
    )
    nl_problem_a, nl_solution_a = rewriter(p, "5")
    nl_problem_b, nl_solution_b = rewriter(p, "5")
    # Template draft is RNG-seeded internally; just assert structure + repeatability.
    assert "5" in nl_solution_a
    assert nl_problem_a and nl_solution_a
    # Same input → same output (deterministic).
    assert (nl_problem_a, nl_solution_a) == (nl_problem_b, nl_solution_b)


def test_make_rewriter_rejects_unknown_mode() -> None:
    mod = _load_script_module()
    with pytest.raises(Exception, match="rewriter"):
        mod._make_rewriter("bogus")


def test_worker_config_is_picklable() -> None:
    """`_WorkerConfig` must round-trip through pickle for multiprocess dispatch."""
    import pickle

    mod = _load_script_module()
    cfg = mod._WorkerConfig(
        image_dir=Path("/tmp/x"),
        renderer="matplotlib",
        rewriter="template",
        m_per_seed=2,
        seed=42,
        gather_timeout=10.0,
        solve_timeout=5.0,
        bfs_depth=1,
        verify_with_sympy=False,
        max_attempts_factor=10,
    )
    roundtripped = pickle.loads(pickle.dumps(cfg))
    assert roundtripped == cfg


def test_drain_stops_at_target_n() -> None:
    """`_drain` should stop pulling once `target_n` is reached, leaving the
    remainder of the iterator untouched (so `pool.terminate()` short-circuits)."""
    mod = _load_script_module()

    samples = [
        SyntheticSample(
            source_pid=i,
            new_metrics=(),
            deleted_metrics=(),
            added_metrics=(),
            goal="g",
            answer="0",
            theorem_seqs=(),
        )
        for i in range(10)
    ]

    pulled: list[int] = []

    def _stream():
        for s in samples:
            pulled.append(s.source_pid)
            yield [s]

    out = mod._drain(_stream(), target_n=3, t_start=0.0, log_every=100)
    assert len(out) == 3
    # We stopped at the 3rd sample; the iterator should not have been fully drained.
    assert pulled[-1] == 2
    assert len(pulled) == 3


def test_render_sample_writes_png(tmp_path: Path) -> None:
    """`_render_sample` should produce a PNG at `<out_dir>/<id>.png`."""
    mod = _load_script_module()
    from open_geofm.render import matplotlib_renderer

    sample = SyntheticSample(
        source_pid=42,
        new_metrics=("Equal(LengthOfLine(AB),3)", "Equal(MeasureOfAngle(ABC),90)"),
        deleted_metrics=(),
        added_metrics=("Equal(MeasureOfAngle(ABC),90)",),
        goal="Equal(LengthOfLine(AC),5)",
        answer="5",
        theorem_seqs=(),
        construction_cdl=("Triangle(A,B,C)",),
    )
    out_dir = tmp_path / "imgs"
    out_dir.mkdir()
    path = mod._render_sample(sample, matplotlib_renderer, out_dir, seed=0)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
