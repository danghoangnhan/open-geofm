"""Qwen2-VL-7B LoRA training config (headline run, Phase 7).

Blueprint §7 budget: ~10-18 h on 10K, 2 epochs. ~20 GB VRAM with grad checkpointing.
"""

from __future__ import annotations

from typing import Any

from .qwen2vl_2b_lora import get as _get_2b


def get() -> dict[str, Any]:
    cfg = _get_2b()
    cfg["model_name_or_path"] = "Qwen/Qwen2-VL-7B-Instruct"
    cfg["sft_config"]["output_dir"] = "outputs/qwen2vl-7b-lora"
    cfg["sft_config"]["per_device_train_batch_size"] = 1
    cfg["sft_config"]["gradient_accumulation_steps"] = 16
    return cfg
