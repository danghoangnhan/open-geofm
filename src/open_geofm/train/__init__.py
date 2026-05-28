"""TRL SFTTrainer-based fine-tuning for Qwen2-VL / Qwen2.5-VL.

Blueprint §2 Phase 7. Adapted from `huggingface/trl/examples/scripts/sft_vlm.py`.
Configs are Python modules under `train/configs/` returning (SFTConfig, LoraConfig, processor_kwargs).
"""
