"""TRL SFTTrainer driver for Qwen2-VL / Qwen2.5-VL LoRA fine-tuning.

Blueprint §2 Phase 7. Adapted from `huggingface/trl/examples/scripts/sft_vlm.py`
(Apache-2.0). Config selection via `--config <name>` picks one of:
    qwen2vl_2b_lora | qwen2vl_7b_lora | qwen25vl_7b_lora | qwen2vl_7b_qlora

Critical VLM gotcha (TRL docs): `SFTConfig(max_length=None)` — truncation
breaks image-token spans. Already set in every config.

Heavy torch / transformers / trl / peft imports happen *inside* `train()` so
this module is importable from the host (CPU) venv for `--help` and config
validation.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = logging.getLogger("open_geofm.train")

_CONFIG_MAP = {
    "qwen2vl_2b_lora": "open_geofm.train.configs.qwen2vl_2b_lora",
    "qwen2vl_7b_lora": "open_geofm.train.configs.qwen2vl_7b_lora",
    "qwen25vl_7b_lora": "open_geofm.train.configs.qwen25vl_7b_lora",
    "qwen2vl_7b_qlora": "open_geofm.train.configs.qwen2vl_7b_qlora",
}


def load_config(name: str) -> dict[str, Any]:
    """Resolve a config name to its kwargs dict. Raises SystemExit on miss."""
    if name not in _CONFIG_MAP:
        raise SystemExit(f"Unknown --config {name!r}. Known: {sorted(_CONFIG_MAP)}")
    return importlib.import_module(_CONFIG_MAP[name]).get()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _record_to_messages(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one open-geofm record into Qwen2-VL ChatML.

    * `record["image"]` is either a string path (raw JSONL) or a `PIL.Image`
      (when loaded via `Dataset.load_from_disk` with the `Image()` feature).
      Path strings are converted lazily — PIL.Image.open is itself lazy until
      pixel data is needed.
    * `record["problem"]` / `record["solution"]` are NL strings.
    """
    from PIL import Image as PILImage  # PIL is in base deps

    img = record.get("image")
    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": record.get("problem", "")},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record.get("solution", "")}],
            },
        ]
    }


def _resolve_dataset(spec: str, *, split: str = "train"):
    """Load a `datasets.Dataset` from a hub id or local directory.

    Recognised local layouts:
      * dir with `dataset_info.json` (or `state.json`) -> ``load_from_disk``.
      * dir with `records.jsonl` (what `scripts/02_generate_dataset.py` writes
        on the host venv when `datasets` isn't installed) -> ``from_json``.
    Anything else is treated as a Hugging Face Hub repo id.
    """
    from datasets import Dataset, load_dataset, load_from_disk

    p = Path(spec)
    if p.is_dir() and (p / "dataset_info.json").exists():
        ds = load_from_disk(str(p))
        if not isinstance(ds, Dataset):
            # DatasetDict — pick the requested split, falling back to the first key.
            ds = ds[split] if split in ds else ds[next(iter(ds))]
        return ds
    if p.is_dir() and (p / "records.jsonl").exists():
        return Dataset.from_json(str(p / "records.jsonl"))
    return load_dataset(spec, split=split)


def _load_train_dataset(spec: str, *, split: str = "train"):
    """Resolve `spec` and apply a lazy transform to ChatML.

    Uses `with_transform` (not `map`) so we don't bake decoded PIL bytes into
    the arrow cache — the trainer holds tens of thousands of images and that
    would balloon RAM unnecessarily.
    """
    ds = _resolve_dataset(spec, split=split)

    def _batched(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(next(iter(batch.values())))
        out_messages: list[Any] = []
        for i in range(n):
            rec = {k: batch[k][i] for k in batch}
            out_messages.append(_record_to_messages(rec)["messages"])
        return {"messages": out_messages}

    return ds.with_transform(_batched)


# ---------------------------------------------------------------------------
# Model + trainer construction
# ---------------------------------------------------------------------------


def _make_quantization_config(quant: dict[str, Any]):
    """Translate our `cfg["quantization"]` dict into a `BitsAndBytesConfig`."""
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype_name = quant.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=quant.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


def _load_model(cfg: dict[str, Any]):
    """Load a Qwen2-VL / Qwen2.5-VL VLM with optional 4-bit quantization.

    Falls back from FA2 to SDPA when the FA2 wheel is unavailable on the host
    (sm_120 wheels are built from source inside the Docker image — see
    wiki/07-Blackwell-Setup-Log.md). The fallback is logged loudly so anyone
    chasing throughput notices.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": cfg.get("attn_implementation", "sdpa"),
    }
    if "quantization" in cfg:
        model_kwargs["quantization_config"] = _make_quantization_config(cfg["quantization"])
        model_kwargs["device_map"] = {"": 0}

    name = cfg["model_name_or_path"]
    try:
        return AutoModelForImageTextToText.from_pretrained(name, **model_kwargs)
    except (ImportError, RuntimeError, ValueError) as e:
        msg = str(e)
        if "flash" in msg.lower() and model_kwargs["attn_implementation"] != "sdpa":
            log.warning(
                "flash_attention_2 unavailable (%s); retrying with attn_implementation='sdpa'.",
                msg.splitlines()[0],
            )
            model_kwargs["attn_implementation"] = "sdpa"
            return AutoModelForImageTextToText.from_pretrained(name, **model_kwargs)
        raise


def _make_vision_tower_lr_trainer_cls(vision_lr: float):
    """Build an SFTTrainer subclass that puts vision-tower params in their own
    parameter group with `vision_lr`.

    Why a subclass and not a custom optimizer? `Trainer.create_optimizer` is
    invoked lazily at `trainer.train()` time; overriding it is the
    least-intrusive hook that survives FSDP / Accelerate wrapping.
    """
    from trl import SFTTrainer

    _NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    _VISION_KEYWORDS = ("visual.", "vision_tower", "vision_model")

    class VisionTowerLRTrainer(SFTTrainer):
        """SFTTrainer with a per-group LR for the visual encoder.

        Qwen2-VL names its ViT params under ``visual.*``; we also catch the
        legacy ``vision_tower`` and ``vision_model`` prefixes for safety.
        """

        def create_optimizer(self):  # type: ignore[override]
            if self.optimizer is not None:
                return self.optimizer

            decay: list = []
            no_decay: list = []
            vis_decay: list = []
            vis_no_decay: list = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                is_vis = any(k in name for k in _VISION_KEYWORDS)
                is_nd = any(k in name for k in _NO_DECAY_KEYWORDS)
                bucket = (vis_no_decay if is_nd else vis_decay) if is_vis else (
                    no_decay if is_nd else decay
                )
                bucket.append(p)

            wd = float(getattr(self.args, "weight_decay", 0.0))
            groups: list[dict[str, Any]] = []
            if decay:
                groups.append({"params": decay, "weight_decay": wd})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            if vis_decay:
                groups.append({"params": vis_decay, "weight_decay": wd, "lr": vision_lr})
            if vis_no_decay:
                groups.append({"params": vis_no_decay, "weight_decay": 0.0, "lr": vision_lr})

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.setdefault("lr", self.args.learning_rate)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            return self.optimizer

    return VisionTowerLRTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: str = typer.Option(..., help="Config name (see _CONFIG_MAP)."),
    dataset: str = typer.Option("open-geofm-mini-10k", help="HF dataset id or local dir."),
    dataset_split: str = typer.Option("train", help="Split name for HF-hub datasets."),
    max_steps: int = typer.Option(
        -1, help="Override SFTConfig.max_steps; -1 = use config / epoch-based."
    ),
    push_to_hub: bool = typer.Option(False, help="Push the adapter after training."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve config + dataset + write a manifest, do not load the model.",
    ),
) -> None:
    """Run SFT with the selected config."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(config)
    if max_steps and max_steps > 0:
        cfg["sft_config"]["max_steps"] = max_steps

    if dry_run:
        _write_dry_run_manifest(config, dataset, cfg)
        return

    # Heavy imports deferred so `--help` and `--dry-run` work on the CPU host venv.
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoProcessor
    from trl import SFTConfig, SFTTrainer

    from .collator import Qwen2VLDataCollator

    log.info("loading processor + model: %s", cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"], **cfg.get("processor_kwargs", {})
    )
    model = _load_model(cfg)
    if "quantization" in cfg:
        # PEFT helper: enable input-grad propagation through frozen 4-bit base.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg["sft_config"].get("gradient_checkpointing", True)
        )

    peft_config = LoraConfig(**cfg["lora_config"])

    sft_kwargs = dict(cfg["sft_config"])
    sft_kwargs.setdefault("push_to_hub", push_to_hub)
    sft_config = SFTConfig(**sft_kwargs)

    log.info("resolving dataset: %s (split=%s)", dataset, dataset_split)
    train_ds = _load_train_dataset(dataset, split=dataset_split)

    vision_lr = cfg.get("vision_tower_lr")
    trainer_cls = (
        _make_vision_tower_lr_trainer_cls(float(vision_lr)) if vision_lr is not None else SFTTrainer
    )
    log.info(
        "trainer: %s (vision_tower_lr=%s, push_to_hub=%s, max_steps=%s)",
        trainer_cls.__name__,
        vision_lr,
        push_to_hub,
        cfg["sft_config"].get("max_steps", -1),
    )

    trainer = trainer_cls(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        data_collator=Qwen2VLDataCollator(processor),
        peft_config=peft_config,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(sft_config.output_dir)
    if push_to_hub:
        trainer.push_to_hub()


def _write_dry_run_manifest(config: str, dataset: str, cfg: dict[str, Any]) -> None:
    """Write a JSON manifest summarising the resolved config (no GPU work).

    Useful as a CI gate: ``train ... --dry-run`` runs on the host venv and
    confirms the config + dataset spec resolve without importing torch.
    """
    out_dir = Path(cfg["sft_config"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "dataset": dataset,
        "model_name_or_path": cfg["model_name_or_path"],
        "sft_config": cfg["sft_config"],
        "lora_config": cfg["lora_config"],
        "processor_kwargs": cfg.get("processor_kwargs", {}),
        "quantization": cfg.get("quantization"),
        "vision_tower_lr": cfg.get("vision_tower_lr"),
        "attn_implementation": cfg.get("attn_implementation"),
    }
    path = out_dir / "dry_run.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("dry-run manifest written to %s", path)


if __name__ == "__main__":
    app()
