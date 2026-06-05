"""Backend selector for the NLG rewriter.

Env var `OPEN_GEOFM_REWRITER` ∈ {"local", "vllm", "hf", "openai"} (default
"local") chooses between vLLM-served Qwen2.5-7B (local), a `transformers`
pipeline (hf — the vLLM-free / Blackwell-robust fallback), and OpenAI
gpt-4o-mini. All expose the same `rewrite(prompt) -> str`.

The three rewriter modules import cleanly without their heavy deps (vllm /
transformers / openai are reached via importlib on first use), so importing them
eagerly here is safe in the CPU venv.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from .rewrite_hf import HFRewriter
from .rewrite_local import LocalRewriter
from .rewrite_openai import OpenAIRewriter

_REWRITERS: dict[str, type] = {
    "local": LocalRewriter,
    "vllm": LocalRewriter,
    "hf": HFRewriter,
    "openai": OpenAIRewriter,
}


@runtime_checkable
class Rewriter(Protocol):
    def rewrite(self, prompt: str) -> str: ...


def get_rewriter(choice: str | None = None) -> Rewriter:
    """Construct the rewriter selected by `choice` / `OPEN_GEOFM_REWRITER`."""
    name = (choice or os.environ.get("OPEN_GEOFM_REWRITER", "local")).lower()
    try:
        cls = _REWRITERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown OPEN_GEOFM_REWRITER={name!r}; expected one of {sorted(_REWRITERS)}."
        ) from None
    return cls()
