"""Backend selector for the NLG rewriter.

Blueprint §2 Phase 5: env var `OPEN_GEOFM_REWRITER` ∈ {"local", "openai"} chooses
between vLLM-served Qwen2.5-7B-Instruct (local, free, ~16 GB VRAM) and OpenAI
gpt-4o-mini (~$3 / 10K samples). Both expose the same `rewrite(prompt) -> str`.
"""

from __future__ import annotations

import os
from typing import Protocol


class Rewriter(Protocol):
    def rewrite(self, prompt: str) -> str: ...


def get_rewriter() -> Rewriter:
    """Construct the rewriter selected by `OPEN_GEOFM_REWRITER` (default: `local`)."""
    choice = os.environ.get("OPEN_GEOFM_REWRITER", "local").lower()
    if choice == "local":
        from .rewrite_local import LocalRewriter

        return LocalRewriter()
    if choice == "openai":
        from .rewrite_openai import OpenAIRewriter

        return OpenAIRewriter()
    raise ValueError(f"Unknown OPEN_GEOFM_REWRITER={choice!r}; expected 'local' or 'openai'.")
