"""Eager TRL/transformers/peft training backend for Qwen2-VL / Qwen2.5-VL.

All heavy GPU imports live at module top; this module is imported only via
`importlib` from `sft_vlm` (the non-dry-run path) or the registry, so `--help` /
`--dry-run` / the CPU test suite never load it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .collator import Qwen2VLDataCollator
from .sft_vlm import _load_train_dataset

log = logging.getLogger("open_geofm.train")

_NO_DECAY_KEYWORDS = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
_VISION_KEYWORDS = ("visual.", "vision_tower", "vision_model")


def _make_quantization_config(quant: dict[str, Any]) -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=quant.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


def _load_model(cfg: dict[str, Any]):
    """Load a Qwen2-VL / Qwen2.5-VL VLM, falling back FA2 -> SDPA when the
    flash-attn wheel is unavailable (sm_120 builds it from source)."""
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
                "flash_attention_2 unavailable (%s); retrying with sdpa.", msg.splitlines()[0]
            )
            model_kwargs["attn_implementation"] = "sdpa"
            return AutoModelForImageTextToText.from_pretrained(name, **model_kwargs)
        raise


def _make_vision_tower_lr_trainer_cls(vision_lr: float):
    """SFTTrainer subclass that puts vision-tower params in their own LR group."""

    class VisionTowerLRTrainer(SFTTrainer):
        def create_optimizer(self):  # type: ignore[override]
            if self.optimizer is not None:
                return self.optimizer
            buckets: dict[str, list] = {"d": [], "nd": [], "vd": [], "vnd": []}
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                is_vis = any(k in name for k in _VISION_KEYWORDS)
                is_nd = any(k in name for k in _NO_DECAY_KEYWORDS)
                key = ("v" if is_vis else "") + ("nd" if is_nd else "d")
                buckets[key].append(p)
            wd = float(getattr(self.args, "weight_decay", 0.0))
            groups: list[dict[str, Any]] = []
            if buckets["d"]:
                groups.append({"params": buckets["d"], "weight_decay": wd})
            if buckets["nd"]:
                groups.append({"params": buckets["nd"], "weight_decay": 0.0})
            if buckets["vd"]:
                groups.append({"params": buckets["vd"], "weight_decay": wd, "lr": vision_lr})
            if buckets["vnd"]:
                groups.append({"params": buckets["vnd"], "weight_decay": 0.0, "lr": vision_lr})
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.setdefault("lr", self.args.learning_rate)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            return self.optimizer

    return VisionTowerLRTrainer


def run_training(
    cfg: dict[str, Any], *, dataset: str, dataset_split: str = "train", push_to_hub: bool = False
) -> None:
    """Build model + trainer from a resolved config dict and run SFT."""
    log.info("loading processor + model: %s", cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(
        cfg["model_name_or_path"], **cfg.get("processor_kwargs", {})
    )
    model = _load_model(cfg)
    if "quantization" in cfg:
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


class QwenVLTrainer:
    """Concrete `Trainer` for Qwen2-VL/2.5-VL LoRA/QLoRA (the registry backend)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def train(
        self, *, dataset: str, dataset_split: str = "train", push_to_hub: bool = False
    ) -> None:
        run_training(
            self.cfg, dataset=dataset, dataset_split=dataset_split, push_to_hub=push_to_hub
        )
