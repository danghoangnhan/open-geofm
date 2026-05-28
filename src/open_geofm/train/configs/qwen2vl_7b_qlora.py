"""Qwen2-VL-7B QLoRA training config — EXPERIMENTAL.

Blueprint §8 risk #1: bitsandbytes on sm_120 may produce garbage outputs
(informatico-madrid Blackwell-Linux-Infra-Optimizer, bnb #1642). This config is
gated behind `OPEN_GEOFM_ENABLE_QLORA=1`; the loader refuses to instantiate it
otherwise so users don't silently get corrupt outputs.
"""

from __future__ import annotations

import os
from typing import Any

from .qwen2vl_7b_lora import get as _get_qwen2_7b


def get() -> dict[str, Any]:
    if os.environ.get("OPEN_GEOFM_ENABLE_QLORA") != "1":
        raise RuntimeError(
            "QLoRA on Blackwell (sm_120) is gated. Set OPEN_GEOFM_ENABLE_QLORA=1 to opt in; "
            "see wiki/07-Blackwell-Setup-Log.md for the bitsandbytes status."
        )
    cfg = _get_qwen2_7b()
    cfg["sft_config"]["output_dir"] = "outputs/qwen2vl-7b-qlora"
    cfg["quantization"] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": True,
    }
    cfg["sft_config"]["per_device_train_batch_size"] = 2
    cfg["sft_config"]["gradient_accumulation_steps"] = 8
    return cfg
