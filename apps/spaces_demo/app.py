"""Hugging Face Spaces demo for `open-geofm`.

Loads Qwen2-VL-2B-Instruct (optionally with an `open-geofm` LoRA adapter merged)
and exposes a 2-input Gradio chat: a geometry diagram (PNG) and a problem
statement. Runs on the free A10G tier.

Environment variables (set in Space → Settings → Variables):
* ``OPEN_GEOFM_BASE_MODEL``  — base model id (default ``Qwen/Qwen2-VL-2B-Instruct``).
* ``OPEN_GEOFM_LORA_ID``     — optional Hub LoRA adapter id (e.g.
  ``CallMeDaniel/Qwen2-VL-2B-OpenGeoFM-LoRA``). If unset the base model runs
  with a banner explaining the LoRA delta hasn't been wired yet.
* ``OPEN_GEOFM_MAX_NEW_TOKENS`` — default 512.

The app degrades gracefully when neither GPU nor transformers is available
(useful for previewing the UI locally on a CPU box).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

BASE_MODEL = os.environ.get("OPEN_GEOFM_BASE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
LORA_ID = os.environ.get("OPEN_GEOFM_LORA_ID")
MAX_NEW = int(os.environ.get("OPEN_GEOFM_MAX_NEW_TOKENS", "512"))

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def _load_pipeline():
    """Return a (model, processor) pair, or ``None`` if the GPU stack is unavailable.

    Loads lazily so this module imports cleanly on a CPU-only checkout (useful
    for `gr.Interface.launch(share=False, server_port=None)` smoke tests).
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=dtype, device_map="cuda:0"
    )
    if LORA_ID:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, LORA_ID)
    model.eval()
    return model, processor


_PIPELINE = _load_pipeline()


def _model_banner() -> str:
    """User-visible banner describing what the Space is actually serving."""
    if _PIPELINE is None:
        return (
            "🛈 **Preview mode** — no CUDA-capable transformers stack detected. "
            "This UI is running but the model isn't loaded. Deploy to HF Spaces "
            "with an A10G to enable real inference."
        )
    if LORA_ID:
        return f"✅ Loaded **{BASE_MODEL}** + LoRA adapter **{LORA_ID}**."
    return (
        f"⚠️ Loaded **{BASE_MODEL}** (no LoRA). "
        "Set `OPEN_GEOFM_LORA_ID` to attach an open-geofm LoRA adapter."
    )


def infer(image, problem: str) -> str:
    """Run a single forward pass and return the assistant text."""
    if _PIPELINE is None:
        return (
            "(preview mode — model not loaded)\n\n"
            "On the deployed Space this would return the model's response to:\n"
            f"  problem = {problem!r}\n"
            f"  image   = <PIL.Image>"
        )
    model, processor = _PIPELINE
    if image is None or not problem:
        return "Please provide both an image and a problem statement."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

    import torch

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
        )
    # Strip the prompt prefix so we only return the assistant text.
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[:, prompt_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _gallery_examples() -> list[list]:
    """Examples for the Gradio component. Falls back to empty if no PNGs are bundled."""
    out: list[list] = []
    if not EXAMPLES_DIR.exists():
        return out
    prompts = {
        "right_triangle.png": "In the figure, AB = 3 and BC = 4 with ∠ABC = 90°. Find AC.",
        "tangent_circle.png": "A tangent line CD touches circle E at point D. Arc EFD = 40°. Find ∠FCD.",
        "parallel_lines.png": "Lines AB and CD are parallel; transversal EF makes ∠AEF = 65°. Find ∠CFE.",
    }
    for png, prompt in prompts.items():
        path = EXAMPLES_DIR / png
        if path.exists():
            out.append([str(path), prompt])
    return out


with gr.Blocks(title="open-geofm — geometry-MLLM demo") as demo:
    gr.Markdown(
        """
        # open-geofm — single-RTX-5090 geometry-MLLM reproduction

        Educational reproduction of *GeoFM* (Zhang et al., 2025, arXiv:2510.27448).
        Upload a geometric figure (or pick an example) and ask the model a question.

        **Source:** [github.com/CallMeDaniel/open-geofm](https://github.com/CallMeDaniel/open-geofm)
        **Method:** see the [wiki](https://github.com/CallMeDaniel/open-geofm/wiki).
        """
    )
    gr.Markdown(_model_banner())

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Geometric figure")
            text_in = gr.Textbox(
                label="Problem statement",
                placeholder="e.g. In triangle ABC, AB = 3 and BC = 4 with ∠ABC = 90°. Find AC.",
                lines=3,
            )
            with gr.Row():
                run = gr.Button("Solve", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column():
            out = gr.Textbox(label="Model response", lines=12)

    examples = _gallery_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_in, text_in],
            outputs=out,
            fn=infer,
            cache_examples=False,
        )

    run.click(fn=infer, inputs=[image_in, text_in], outputs=out)
    clear.click(fn=lambda: (None, "", ""), outputs=[image_in, text_in, out])


if __name__ == "__main__":
    demo.launch()
