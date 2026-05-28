"""Vision data collator for Qwen2-VL / Qwen2.5-VL SFT.

Blueprint §2 Phase 7. TRL's default text collator can't handle the image-pad
token spans; this collator wraps `processor(text=..., images=..., ...)` and
masks pad + image/video special tokens so the model doesn't learn to predict
them.

No top-level torch / transformers import — the host (CPU-only) venv must be
able to introspect this module without pulling the GPU stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class Qwen2VLDataCollator:
    """Vision collator paired with `SFTTrainer(data_collator=...)`.

    Each `example` is a dict with a ``"messages"`` list in Qwen2-VL ChatML
    shape (see `open_geofm.dataset.qwen_vl_format.to_conversation`). Image
    parts may carry either a ``PIL.Image`` object or a path string; the
    processor accepts both.

    Args:
        processor: a Qwen2-VL `AutoProcessor` (or compatible).
        mask_image_tokens: if True (default), set ``labels == image_pad`` to
            -100 so the model doesn't train to *predict* image padding.
    """

    def __init__(self, processor, *, mask_image_tokens: bool = True) -> None:
        self.processor = processor
        self.mask_image_tokens = mask_image_tokens
        self._image_token_ids: tuple[int, ...] = (
            self._gather_image_token_ids(processor) if mask_image_tokens else ()
        )

    @staticmethod
    def _gather_image_token_ids(processor) -> tuple[int, ...]:
        """Resolve special-token ids for image / video padding (Qwen2-VL family).

        Qwen2-VL: ``<|image_pad|> = 151655``, ``<|video_pad|> = 151656``.
        We look them up via the tokenizer instead of hard-coding so the
        collator stays correct if Qwen ships a tokenizer update.
        """
        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            return ()
        unk = getattr(tok, "unk_token_id", None)
        ids: list[int] = []
        for name in ("<|image_pad|>", "<|video_pad|>"):
            try:
                i = tok.convert_tokens_to_ids(name)
            except Exception:
                continue
            if i is None or i == unk:
                continue
            ids.append(int(i))
        return tuple(ids)

    @staticmethod
    def _extract_images(messages: Iterable[dict[str, Any]]) -> list:
        """Pull image objects (PIL or path string) out of ChatML message content."""
        out: list = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                img = part.get("image") or part.get("url") or part.get("path")
                if img is not None:
                    out.append(img)
        return out

    def __call__(self, examples: list[dict[str, Any]]):
        if not examples:
            raise ValueError("Qwen2VLDataCollator received an empty batch.")

        texts = [
            self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            for ex in examples
        ]
        images_per_example = [self._extract_images(ex["messages"]) for ex in examples]
        # `processor(images=...)` accepts None when no images are present, but
        # mixing some-with / some-without in a single batch is undefined for
        # Qwen2-VL — every example in our pipeline has exactly one image.
        any_images = any(images_per_example)
        kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if any_images:
            kwargs["images"] = images_per_example

        batch = self.processor(**kwargs)
        labels = batch["input_ids"].clone()

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for tid in self._image_token_ids:
            labels[labels == tid] = -100

        batch["labels"] = labels
        return batch
