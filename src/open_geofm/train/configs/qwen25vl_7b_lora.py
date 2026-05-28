"""Qwen2.5-VL-7B LoRA training config (stronger baseline, Phase 7).

Blueprint §9.5: drop-in upgrade vs Qwen2-VL, same ChatML template.
"""

from __future__ import annotations

from typing import Any

from .qwen2vl_7b_lora import get as _get_qwen2_7b


def get() -> dict[str, Any]:
    cfg = _get_qwen2_7b()
    cfg["model_name_or_path"] = "Qwen/Qwen2.5-VL-7B-Instruct"
    cfg["sft_config"]["output_dir"] = "outputs/qwen25vl-7b-lora"
    return cfg
