"""CPU tests for the Phase 7 train module (`open_geofm.train`).

What we can cover on the host venv (no torch / transformers / trl / peft /
datasets installed):
  * Collator helper functions (`_extract_images`, `_gather_image_token_ids`,
    `__init__`) — pure Python, no torch needed.
  * `_record_to_messages` — uses PIL only (base dep).
  * `load_config` for non-quantized configs, and the QLoRA env-var gate.
  * The Typer CLI's `--help` introspection.
  * The `--dry-run` path that writes a manifest without importing torch.

GPU paths (`SFTTrainer.train()`, full collator `__call__`, dataset loading)
are exercised by running `scripts/03_train.sh ... --max-steps 5` inside the
Blackwell Docker image and are not in pytest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from open_geofm.train.collator import Qwen2VLDataCollator

# ---------------------------------------------------------------------------
# Stubs for the collator (real Qwen2-VL tokenizer needs transformers + weights)
# ---------------------------------------------------------------------------


class _StubTokenizer:
    pad_token_id = 0
    unk_token_id = 1

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self._vocab = vocab or {"<|image_pad|>": 151655, "<|video_pad|>": 151656}

    def convert_tokens_to_ids(self, name: str) -> int | None:
        return self._vocab.get(name)


class _StubProcessor:
    def __init__(self, tokenizer: _StubTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or _StubTokenizer()


# ---------------------------------------------------------------------------
# Collator: pure-Python helpers
# ---------------------------------------------------------------------------


def test_collator_extract_images_pulls_image_parts() -> None:
    img = Image.new("RGB", (4, 4))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "what is this?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "a square"}]},
    ]
    out = Qwen2VLDataCollator._extract_images(messages)
    assert out == [img]


def test_collator_extract_images_accepts_url_and_path_keys() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://example.com/a.png"},
                {"type": "image", "path": "/tmp/b.png"},
                {"type": "text", "text": "x"},
            ],
        }
    ]
    assert Qwen2VLDataCollator._extract_images(messages) == [
        "https://example.com/a.png",
        "/tmp/b.png",
    ]


def test_collator_extract_images_text_only_returns_empty() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert Qwen2VLDataCollator._extract_images(messages) == []


def test_collator_extract_images_tolerates_string_content() -> None:
    """Some chat templates flatten `content` to a string; we just skip those."""
    messages = [{"role": "user", "content": "hi"}]
    assert Qwen2VLDataCollator._extract_images(messages) == []


def test_collator_gather_image_token_ids_returns_resolved_ids() -> None:
    ids = Qwen2VLDataCollator._gather_image_token_ids(_StubProcessor())
    assert ids == (151655, 151656)


def test_collator_gather_image_token_ids_filters_unk() -> None:
    """A tokenizer that doesn't know `<|video_pad|>` returns `unk_token_id` —
    that must NOT end up in the mask list (would zero out real tokens)."""
    tok = _StubTokenizer(vocab={"<|image_pad|>": 151655, "<|video_pad|>": _StubTokenizer.unk_token_id})
    ids = Qwen2VLDataCollator._gather_image_token_ids(_StubProcessor(tok))
    assert ids == (151655,)


def test_collator_construction_with_masking_off_yields_empty_ids() -> None:
    c = Qwen2VLDataCollator(_StubProcessor(), mask_image_tokens=False)
    assert c._image_token_ids == ()


def test_collator_construction_with_masking_on_default() -> None:
    c = Qwen2VLDataCollator(_StubProcessor())
    assert 151655 in c._image_token_ids and 151656 in c._image_token_ids


def test_collator_call_rejects_empty_batch() -> None:
    c = Qwen2VLDataCollator(_StubProcessor())
    with pytest.raises(ValueError, match="empty batch"):
        c([])


# ---------------------------------------------------------------------------
# `_record_to_messages` — exercises PIL loading
# ---------------------------------------------------------------------------


def _load_train_module():
    """Importing the train module via importlib so the test doesn't need
    `tests/__init__.py` tricks (it's already importable normally — this is
    defensive against future refactors)."""
    spec = importlib.util.find_spec("open_geofm.train.sft_vlm")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_record_to_messages_loads_pil_from_path(tmp_path: Path) -> None:
    mod = _load_train_module()
    img_path = tmp_path / "x.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path)

    out = mod._record_to_messages(
        {"image": str(img_path), "problem": "Find x.", "solution": "x = 1."}
    )
    msgs = out["messages"]
    assert msgs[0]["role"] == "user"
    user_parts = msgs[0]["content"]
    assert user_parts[0]["type"] == "image"
    assert isinstance(user_parts[0]["image"], Image.Image)
    assert user_parts[1] == {"type": "text", "text": "Find x."}
    assert msgs[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "x = 1."}],
    }


def test_record_to_messages_passes_through_pil_object() -> None:
    mod = _load_train_module()
    img = Image.new("RGB", (4, 4))
    out = mod._record_to_messages({"image": img, "problem": "p", "solution": "s"})
    assert out["messages"][0]["content"][0]["image"] is img


def test_record_to_messages_handles_missing_text_fields() -> None:
    mod = _load_train_module()
    img = Image.new("RGB", (4, 4))
    out = mod._record_to_messages({"image": img})
    assert out["messages"][0]["content"][1] == {"type": "text", "text": ""}
    assert out["messages"][1]["content"][0] == {"type": "text", "text": ""}


# ---------------------------------------------------------------------------
# `load_config` + QLoRA gate
# ---------------------------------------------------------------------------


def test_load_config_known_returns_dict() -> None:
    mod = _load_train_module()
    cfg = mod.load_config("qwen2vl_2b_lora")
    assert cfg["model_name_or_path"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert cfg["sft_config"]["max_length"] is None  # critical VLM gotcha
    assert cfg["lora_config"]["target_modules"] == "all-linear"


def test_load_config_unknown_raises_systemexit() -> None:
    mod = _load_train_module()
    with pytest.raises(SystemExit, match="qwen2vl_2b_lora"):
        mod.load_config("bogus_config")


def test_load_config_7b_lora_inherits_2b_recipe() -> None:
    mod = _load_train_module()
    cfg = mod.load_config("qwen2vl_7b_lora")
    assert cfg["model_name_or_path"] == "Qwen/Qwen2-VL-7B-Instruct"
    # Effective batch size held constant across configs (16).
    sft = cfg["sft_config"]
    assert sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"] == 16


def test_load_config_qlora_blocked_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_GEOFM_ENABLE_QLORA", raising=False)
    mod = _load_train_module()
    with pytest.raises(RuntimeError, match="OPEN_GEOFM_ENABLE_QLORA"):
        mod.load_config("qwen2vl_7b_qlora")


def test_load_config_qlora_loads_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_GEOFM_ENABLE_QLORA", "1")
    mod = _load_train_module()
    cfg = mod.load_config("qwen2vl_7b_qlora")
    assert cfg["quantization"]["load_in_4bit"] is True
    assert cfg["quantization"]["bnb_4bit_quant_type"] == "nf4"


# ---------------------------------------------------------------------------
# CLI `--help` + `--dry-run`
# ---------------------------------------------------------------------------


def test_cli_help_does_not_import_torch() -> None:
    """The host venv has no torch — `--help` must work anyway."""
    from typer.testing import CliRunner

    mod = _load_train_module()
    runner = CliRunner()
    result = runner.invoke(mod.app, ["--help"])
    assert result.exit_code == 0
    assert "config" in result.output.lower()


def test_cli_dry_run_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`train --dry-run` resolves the config and writes a JSON manifest with
    no torch / transformers / trl import."""
    from typer.testing import CliRunner

    mod = _load_train_module()
    monkeypatch.chdir(tmp_path)
    # Override the output dir to land under tmp_path so the test doesn't leak.
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda name: {
            "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "sft_config": {"output_dir": str(tmp_path / "out"), "max_length": None},
            "lora_config": {"r": 16, "target_modules": "all-linear"},
            "processor_kwargs": {"min_pixels": 200_704},
            "attn_implementation": "flash_attention_2",
            "vision_tower_lr": 1e-6,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        mod.app,
        ["--config", "qwen2vl_2b_lora", "--dataset", "data/smoke", "--dry-run"],
    )
    assert result.exit_code == 0, result.output

    manifest_path = tmp_path / "out" / "dry_run.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["config"] == "qwen2vl_2b_lora"
    assert manifest["dataset"] == "data/smoke"
    assert manifest["model_name_or_path"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert manifest["vision_tower_lr"] == 1e-6
    assert manifest["sft_config"]["max_length"] is None


def test_cli_dry_run_propagates_max_steps_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    mod = _load_train_module()
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda name: {
            "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "sft_config": {"output_dir": str(tmp_path / "out"), "max_length": None},
            "lora_config": {"r": 16},
            "processor_kwargs": {},
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        mod.app,
        ["--config", "qwen2vl_2b_lora", "--dataset", "data/smoke", "--dry-run", "--max-steps", "7"],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "out" / "dry_run.json").read_text())
    assert manifest["sft_config"]["max_steps"] == 7


def test_cli_dry_run_does_not_import_torch_or_trl() -> None:
    """Defensive: the dry-run path must not transitively pull torch / trl."""
    import sys

    from typer.testing import CliRunner

    mod = _load_train_module()
    # If torch / trl were already imported by something else in the test session,
    # we can't make a claim — just skip then.
    pre_torch = "torch" in sys.modules
    pre_trl = "trl" in sys.modules
    if pre_torch or pre_trl:
        pytest.skip("torch / trl already imported in this session; cannot assert")

    runner = CliRunner()
    result = runner.invoke(
        mod.app,
        [
            "--config", "qwen2vl_2b_lora",
            "--dataset", "data/smoke",
            "--dry-run",
        ],
    )
    # Allow non-zero if the real config writes to a path we don't have access
    # to; the assertion we actually care about is the import-graph check.
    assert "torch" not in sys.modules, "dry-run unexpectedly imported torch"
    assert "trl" not in sys.modules, "dry-run unexpectedly imported trl"
    assert "peft" not in sys.modules, "dry-run unexpectedly imported peft"
    # If exit was non-zero, surface the reason so future debug isn't blind.
    if result.exit_code != 0:
        pytest.fail(f"dry-run exited {result.exit_code}: {result.output}")


# ---------------------------------------------------------------------------
# Dataset loading — gated on `datasets` being installed (Docker only)
# ---------------------------------------------------------------------------


def test_resolve_dataset_local_jsonl(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    mod = _load_train_module()

    img_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(img_path)
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps({"image": str(img_path), "problem": "p", "solution": "s"}) + "\n"
    )

    ds = mod._resolve_dataset(str(tmp_path))
    assert len(ds) == 1
    assert ds[0]["image"] == str(img_path)


def test_load_train_dataset_yields_messages(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    mod = _load_train_module()

    img_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(img_path)
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps({"image": str(img_path), "problem": "p", "solution": "s"}) + "\n"
    )

    ds = mod._load_train_dataset(str(tmp_path))
    row = ds[0]
    assert "messages" in row
    msgs = row["messages"]
    assert msgs[0]["role"] == "user"
    assert isinstance(msgs[0]["content"][0]["image"], Image.Image)
    assert msgs[1]["content"][0]["text"] == "s"


# ---------------------------------------------------------------------------
# Defensive: make sure the existing scaffolding for non-essential parts
# (like `make_vision_tower_lr_trainer_cls`) doesn't blow up at import time.
# ---------------------------------------------------------------------------


def test_vision_lr_trainer_factory_requires_trl() -> None:
    """The factory imports trl lazily; without trl it must raise ImportError
    (not silently return a broken class)."""
    pytest.importorskip("typer")  # ensure host venv is configured
    mod = _load_train_module()
    try:
        import trl  # noqa: F401

        cls = mod._make_vision_tower_lr_trainer_cls(1e-6)
        assert cls.__name__ == "VisionTowerLRTrainer"
    except ImportError:
        # Host venv has no trl — factory must raise on call.
        with pytest.raises(ImportError):
            mod._make_vision_tower_lr_trainer_cls(1e-6)
