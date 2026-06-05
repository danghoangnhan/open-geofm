"""Vision data collator for Qwen2-VL / Qwen2.5-VL SFT.

Wraps `processor(text=..., images=..., ...)` and builds `labels` that train ONLY
on the assistant response: pad + image/video pad tokens are masked, AND every
token up to and including the assistant generation header is masked
(completion-only loss). Without that prompt mask the model would be trained to
generate the geometry *problem statement* too, not just the solution (bug #5).

No top-level torch / transformers import — the host (CPU-only) venv must be able
to introspect this module without the GPU stack.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class Qwen2VLDataCollator:
    """Vision collator paired with `SFTTrainer(data_collator=...)`.

    Args:
        processor: a Qwen2-VL `AutoProcessor` (or compatible).
        mask_image_tokens: mask image/video pad tokens in the labels.
        mask_prompt: mask everything up to and including the assistant header so
            loss is computed on the response only (completion-only SFT).
        response_template: the assistant generation header to anchor the prompt
            mask on (Qwen2-VL ChatML).
    """

    def __init__(
        self,
        processor,
        *,
        mask_image_tokens: bool = True,
        mask_prompt: bool = True,
        response_template: str = "<|im_start|>assistant\n",
    ) -> None:
        self.processor = processor
        self.mask_image_tokens = mask_image_tokens
        self.mask_prompt = mask_prompt
        self._image_token_ids: tuple[int, ...] = (
            self._gather_image_token_ids(processor) if mask_image_tokens else ()
        )
        self._response_template_ids: tuple[int, ...] = (
            self._encode_response_template(processor, response_template) if mask_prompt else ()
        )

    @staticmethod
    def _gather_image_token_ids(processor) -> tuple[int, ...]:
        """Resolve special-token ids for image / video padding via the tokenizer
        (never hardcode 151655/151656, in case Qwen ships a tokenizer update)."""
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
    def _encode_response_template(processor, template: str) -> tuple[int, ...]:
        """Token ids of the assistant header. Returns () when the tokenizer can't
        encode (e.g. the lightweight test stub) so prompt masking no-ops there."""
        tok = getattr(processor, "tokenizer", None)
        if tok is None or not hasattr(tok, "encode"):
            return ()
        try:
            return tuple(tok.encode(template, add_special_tokens=False))
        except Exception:
            return ()

    @staticmethod
    def _extract_images(messages: Iterable[dict[str, Any]]) -> list:
        """Pull image objects (PIL or path/url string) out of ChatML content."""
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

    def _apply_completion_mask(self, labels, input_ids) -> None:
        """Set labels to -100 up to and including the LAST assistant header in
        each row. Rows with no header are fully masked (never trained on)."""
        tpl = list(self._response_template_ids)
        span = len(tpl)
        for i in range(input_ids.shape[0]):
            row = input_ids[i].tolist()
            end = None
            for j in range(len(row) - span + 1):
                if row[j : j + span] == tpl:
                    end = j + span
            if end is None:
                labels[i, :] = -100
            else:
                labels[i, :end] = -100

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
        kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
        if any(images_per_example):
            kwargs["images"] = images_per_example

        batch = self.processor(**kwargs)
        labels = batch["input_ids"].clone()

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for tid in self._image_token_ids:
            labels[labels == tid] = -100
        if self._response_template_ids:
            self._apply_completion_mask(labels, batch["input_ids"])

        batch["labels"] = labels
        return batch
