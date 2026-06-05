"""Regression tests for the clean-OOP refactor: the confirmed bug fixes, the
config/registry/pipeline foundation, and the GPU-stack import isolation.

These are CPU-only (no torch/transformers/vllm/datasets) and run in the host venv.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from open_geofm.config import AppConfig, FGPSConfig, RenderConfig
from open_geofm.dataset.builder import HFDatasetBuilder, make_sample_id
from open_geofm.eval.compare import _parse_score_data
from open_geofm.eval.extract_answer import extract
from open_geofm.nlg.templates import draft_nl
from open_geofm.nlg.verify import verify
from open_geofm.pipeline import BuildStage, Pipeline, PipelineContext, RenderStage, SampleStage
from open_geofm.render.cdl_to_gmbl import ConstraintKind, translate
from open_geofm.render.matplotlib_renderer import MatplotlibRenderer
from open_geofm.sampling.algorithm1 import SyntheticSample
from open_geofm.train.collator import Qwen2VLDataCollator

# ---------------------------------------------------------------------------
# Import isolation: the GPU stack must stay out of the base import graph.
# ---------------------------------------------------------------------------


def test_base_graph_is_free_of_gpu_stack() -> None:
    for mod in ("open_geofm", "open_geofm.config", "open_geofm.pipeline", "open_geofm.factory"):
        importlib.import_module(mod)
    gpu = [m for m in ("torch", "vllm", "datasets", "transformers", "trl", "peft", "bitsandbytes") if m in sys.modules]
    assert gpu == [], f"GPU stack leaked into the base import graph: {gpu}"


def test_registry_resolves_every_backend() -> None:
    from open_geofm import registry

    # resolve() imports the backend module + returns the class (no instantiation).
    assert registry.RENDERERS.resolve("mpl").__name__ == "MatplotlibRenderer"
    assert registry.RENDERERS.resolve("gmbl").__name__ == "GmblRenderer"
    assert registry.VERIFIERS.resolve("sympy").__name__ == "SympyVerifier"
    assert registry.LLM_CLIENTS.resolve("hf").__name__ == "HFRewriter"
    with pytest.raises(ValueError, match="Unknown"):
        registry.RENDERERS.resolve("nope")


# ---------------------------------------------------------------------------
# No hardcoding: the config tree centralizes the literals.
# ---------------------------------------------------------------------------


def test_config_defaults_centralize_search_and_render_literals() -> None:
    cfg = AppConfig.default()
    assert (cfg.fgps.max_depth, cfg.fgps.beam_size) == (15, 20)
    assert cfg.render.length_layout_scale == 5.0
    assert cfg.formalgeo.dataset_name == "formalgeo7k_v2"
    # Frozen + typo-proof.
    with pytest.raises(ValidationError):
        cfg.fgps.max_depth = 99  # type: ignore[misc]
    with pytest.raises(ValidationError):
        FGPSConfig(unknown_field=1)  # extra='forbid'


# ---------------------------------------------------------------------------
# Bug #10/#16: FGPSConfig threads search depth/beam (no hardcoded 15/20).
# ---------------------------------------------------------------------------


def test_fgps_config_carries_search_hyperparameters() -> None:
    cfg = FGPSConfig(max_depth=30, beam_size=50)
    assert (cfg.max_depth, cfg.beam_size) == (30, 50)


# ---------------------------------------------------------------------------
# Bug #2/#3: nested predicates + line-pair token split.
# ---------------------------------------------------------------------------


def test_gmbl_parser_handles_nested_and_line_pairs() -> None:
    cs = translate(
        ("PerpendicularBetweenLine(AB,BC)", "Parallel(AB,CD)"),
        ("Equal(LengthOfLine(AB),LengthOfLine(CD))",),
    )
    kinds = {c.kind for c in cs}
    assert ConstraintKind.PERPENDICULAR in kinds and ConstraintKind.PARALLEL in kinds
    perp = next(c for c in cs if c.kind is ConstraintKind.PERPENDICULAR)
    assert perp.args == ("A", "B", "B", "C")  # split into 4 single points
    assert any(c.kind is ConstraintKind.EQUAL_LENGTH and c.args == ("A", "B", "C", "D") for c in cs)


# ---------------------------------------------------------------------------
# Bug #6: answer extractor returns the LAST standalone letter, not the first.
# ---------------------------------------------------------------------------


def test_extract_answer_last_letter_on_single_line() -> None:
    assert extract("We can rule out A and B. The correct choice is D.") == "D"


# ---------------------------------------------------------------------------
# Bug #24: nested-leaf mean ignores top-level metadata scalars.
# ---------------------------------------------------------------------------


def test_parse_score_json_ignores_top_level_metadata() -> None:
    score, metric, _ = _parse_score_data({"per_cat": {"a": 40.0, "b": 60.0}, "count": 100})
    assert (score, metric) == (50.0, "mean(leaves)")


# ---------------------------------------------------------------------------
# Bug #12/#13: no brace-leak on line-pairs; variadic predicates keep all points.
# ---------------------------------------------------------------------------


def test_templates_no_brace_leak_and_keep_all_points() -> None:
    text = draft_nl(("PerpendicularBetweenLine(AB,BC)", "Polygon(A,B,C,D,E)"), "LengthOfLine(AC)")
    assert "{" not in text and "}" not in text
    for pt in "ABCDE":
        assert pt in text


# ---------------------------------------------------------------------------
# Bug #22: thousands separators don't corrupt the extracted number.
# ---------------------------------------------------------------------------


def test_verify_handles_thousands_separator() -> None:
    assert verify("The area is 1,000", "1000").accepted


# ---------------------------------------------------------------------------
# Bug #4/#7/#9/#14: one deterministic, salt-free id shared by render + build.
# ---------------------------------------------------------------------------


def test_make_sample_id_is_deterministic_and_salt_free() -> None:
    s = SyntheticSample(
        source_pid=42, new_metrics=("Equal(LengthOfLine(AB),3)",), deleted_metrics=(),
        added_metrics=("Equal(MeasureOfAngle(ABC),90)",), goal="Equal(LengthOfLine(AC),5)",
        answer="5", theorem_seqs=(),
    )
    a, b = make_sample_id(s), make_sample_id(s)
    assert a == b  # deterministic within process
    assert "hash" not in a and a.startswith("openfm-00042-")


# ---------------------------------------------------------------------------
# Bug #5: collator masks the prompt (completion-only loss).
# ---------------------------------------------------------------------------


class _MaskTok:
    pad_token_id = 0
    unk_token_id = 1

    def convert_tokens_to_ids(self, name: str):
        return {"<|image_pad|>": 151655, "<|video_pad|>": 151656}.get(name)

    def encode(self, text: str, add_special_tokens: bool = False):
        return [10, 11, 12]  # stand-in assistant-header token ids


class _MaskProc:
    tokenizer = _MaskTok()


def test_collator_masks_prompt_up_to_assistant_header() -> None:
    c = Qwen2VLDataCollator(_MaskProc())
    assert c._response_template_ids == (10, 11, 12)
    input_ids = np.array([[5, 5, 10, 11, 12, 7, 8], [10, 11, 12, 9, 0, 0, 0]])
    labels = input_ids.copy()
    c._apply_completion_mask(labels, input_ids)
    # Row 0: header ends at index 5 → everything before is masked, response kept.
    assert (labels[0, :5] == -100).all() and labels[0, 5] == 7 and labels[0, 6] == 8
    # Row 1: header ends at index 3.
    assert (labels[1, :3] == -100).all() and labels[1, 3] == 9


# ---------------------------------------------------------------------------
# The critical end-to-end fix: every dataset record's image exists on disk.
# ---------------------------------------------------------------------------


class _FakeRunner:
    def run_for_seed(self, problem):
        return [
            SyntheticSample(
                source_pid=problem, new_metrics=("Equal(LengthOfLine(AB),3)",),
                deleted_metrics=(), added_metrics=("Equal(MeasureOfAngle(ABC),90)",),
                goal="Equal(LengthOfLine(AC),5)", answer="5", theorem_seqs=("t",),
                construction_cdl=("Triangle(A,B,C)",), nl_problem="Find AC.", nl_solution="AC=5.",
            )
        ]


def test_pipeline_end_to_end_record_images_exist(tmp_path: Path) -> None:
    cfg = AppConfig.default()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    pipe = Pipeline(
        [
            SampleStage(_FakeRunner(), seeds=[1, 2], target_n=2),
            RenderStage(MatplotlibRenderer(RenderConfig.matplotlib_baseline()), cfg.dataset),
            BuildStage(HFDatasetBuilder(cfg.dataset)),
        ]
    )
    ctx = pipe.run(PipelineContext(image_dir=image_dir, out_dir=tmp_path / "out"))
    assert len(ctx.rendered) == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "out" / "records.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    for r in records:
        assert Path(r["image"]).exists(), f"record image missing: {r['image']}"
