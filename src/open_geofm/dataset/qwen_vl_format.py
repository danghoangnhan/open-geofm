"""Conversion to Qwen2-VL ChatML conversation format.

Blueprint §2 Phase 6. Vision tokens (hardcoded from Qwen2-VL tokenizer):
    <|vision_start|> = 151652, <|image_pad|> = 151655, <|vision_end|> = 151653.

LLaMA-Factory / TRL's `template: qwen2_vl` applies these automatically; we keep
the helper here so the dataset builder can write JSONL that downstream tools can
introspect without loading the tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class _ContentPart(TypedDict, total=False):
    type: str
    image: str
    text: str


class _Message(TypedDict):
    role: str
    content: list[_ContentPart]


class Conversation(TypedDict):
    messages: list[_Message]


def to_conversation(image_path: Path | str, problem_nl: str, solution_nl: str) -> Conversation:
    """Emit the (user[image+text] → assistant[text]) ChatML structure."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": problem_nl},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": solution_nl}]},
        ]
    }
