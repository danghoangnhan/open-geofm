"""Qwen2-VL-2B LoRA training config (Phase 7 ablation workhorse).

Blueprint §7 budget: ~3-5 h on 10K samples, 2 epochs. ~10 GB VRAM headroom.
"""

from __future__ import annotations

from typing import Any


def get() -> dict[str, Any]:
    """Return kwargs to construct (SFTConfig, LoraConfig, processor_kwargs).

    Kept as a plain dict so we don't import `trl`/`peft` at module load time —
    the host (CPU-only) uv venv shouldn't need to import them.
    """
    return {
        "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
        "sft_config": {
            "output_dir": "outputs/qwen2vl-2b-lora",
            "num_train_epochs": 2,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": True,
            "gradient_checkpointing": True,
            "max_length": None,  # VLM: do not truncate image-token spans
            "logging_steps": 10,
            "save_strategy": "epoch",
            "report_to": "none",
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "processor_kwargs": {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        # FA2 if available; SFTTrainer falls back to SDPA when FA2 wheel is absent on sm_120.
        "attn_implementation": "flash_attention_2",
        # Vision tower trained but at a much smaller LR (Blueprint §7 pitfall).
        "vision_tower_lr": 1e-6,
    }
