"""Phase 0 Blackwell environment smoke test.

Blueprint §0 + §4. Checks:
    1. PyTorch reports `sm_120` in its compiled arch list.
    2. A small CUDA matmul succeeds (catches missing kernels / driver mismatch).
    3. (Optional) bitsandbytes imports cleanly.
    4. (Optional) Qwen2-VL-2B + a single LoRA step on a dummy batch.

Run inside the Docker image::

    docker compose run --rm train

Or from the host (CPU-only checks only)::

    uv run python scripts/00_verify_env.py --cpu-only
"""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

GREEN, RED, YELLOW, RESET = "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[ok]{RESET}    {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[warn]{RESET}  {msg}")


def fail(msg: str) -> NoReturn:
    print(f"{RED}[fail]{RESET}  {msg}", file=sys.stderr)
    sys.exit(1)


def check_torch_sm120() -> None:
    import torch

    arch_list = torch.cuda.get_arch_list()
    ok(f"torch {torch.__version__}, arch_list={arch_list}")
    if not torch.cuda.is_available():
        fail("CUDA not available - check NVIDIA driver and Docker runtime.")
    if not any("sm_120" in a or "compute_120" in a for a in arch_list):
        fail(
            "PyTorch wheel lacks sm_120 kernels. Reinstall with the cu128 index "
            "(pyproject.toml [[tool.uv.index]] name=pytorch-cu128)."
        )
    ok(f"GPU 0: {torch.cuda.get_device_name(0)}")


def check_matmul() -> None:
    import torch

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    ok(f"bf16 matmul on cuda:0 succeeded (norm={y.norm().item():.4g})")


def check_bnb() -> None:
    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except ImportError:
        warn("bitsandbytes not installed (QLoRA path will be unavailable).")
        return
    ok(f"bitsandbytes {bnb.__version__} imports cleanly")


def check_qwen2vl_lora_step() -> None:
    """One LoRA training step on a dummy batch."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForVision2Seq,
            AutoProcessor,
        )
    except ImportError as e:
        warn(f"Skipping Qwen2-VL LoRA step ({e}); install [torch]+[train] extras.")
        return
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
    except Exception as e:  # network / disk / OOM
        warn(f"Skipping Qwen2-VL load: {e}")
        return
    peft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    model.train()
    # Dummy batch: a tiny black image + a text prompt.
    from PIL import Image

    img = Image.new("RGB", (224, 224), color="black")
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    inputs["labels"] = inputs["input_ids"].clone()
    out = model(**inputs)
    out.loss.backward()
    torch.cuda.synchronize()
    ok(f"Qwen2-VL-2B LoRA backward step OK (loss={out.loss.item():.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip all CUDA checks (use from the host uv venv).",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip the Qwen2-VL-2B LoRA step (still requires the model download otherwise).",
    )
    args = parser.parse_args()

    if args.cpu_only:
        ok("CPU-only mode: skipping CUDA / Blackwell checks.")
        return

    check_torch_sm120()
    check_matmul()
    check_bnb()
    if not args.skip_qwen:
        check_qwen2vl_lora_step()
    ok("env verify complete")


if __name__ == "__main__":
    main()
