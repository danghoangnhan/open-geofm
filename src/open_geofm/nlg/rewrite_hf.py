"""Local NLG rewriter backed by HuggingFace `transformers` (no vLLM).

The Blackwell-robust / CI fallback recommended by the transformers analysis:
slower than vLLM for bulk rewriting, but it needs no vLLM and runs anywhere
`transformers` + `torch` are installed. Uses a `text-generation` pipeline that
applies Qwen2.5-Instruct's ChatML template automatically.

`transformers` / `torch` are optional GPU deps reached via `importlib` (no bare
in-function import), so this module imports cleanly in the CPU venv.
"""

from __future__ import annotations

import importlib
import logging

from ..config import (
    DEFAULT_ATTN_IMPL,
    DEFAULT_LOCAL_NLG_MODEL,
    DEFAULT_NLG_DEVICE_MAP,
    DEFAULT_NLG_DTYPE,
    DEFAULT_NLG_MAX_TOKENS,
    DEFAULT_NLG_TEMPERATURE,
)
from .rewrite_openai import SYSTEM_PROMPT

log = logging.getLogger(__name__)


class HFRewriter:
    def __init__(
        self,
        model: str = DEFAULT_LOCAL_NLG_MODEL,
        temperature: float = DEFAULT_NLG_TEMPERATURE,
        max_new_tokens: int = DEFAULT_NLG_MAX_TOKENS,
        *,
        device_map: str = DEFAULT_NLG_DEVICE_MAP,
        dtype: str = DEFAULT_NLG_DTYPE,
        attn_implementation: str = DEFAULT_ATTN_IMPL,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.system_prompt = system_prompt
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ImportError as e:
            raise RuntimeError(
                "transformers/torch not installed. Install the GPU image or "
                "`uv sync --extra nlg`."
            ) from e
        torch_dtype = getattr(torch, self.dtype)
        kwargs = dict(
            task="text-generation",
            model=self.model,
            dtype=torch_dtype,
            device_map=self.device_map,
            model_kwargs={"attn_implementation": self.attn_implementation},
        )
        try:
            self._pipe = transformers.pipeline(**kwargs)
        except (ImportError, ValueError, RuntimeError) as e:
            # sm_120 has no prebuilt flash-attn wheel; fall back to SDPA.
            if "flash" not in str(e).lower() or self.attn_implementation == "sdpa":
                raise
            log.warning(
                "flash_attention_2 unavailable (%s); retrying with sdpa.", str(e).splitlines()[0]
            )
            kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}
            self._pipe = transformers.pipeline(**kwargs)

    def rewrite(self, prompt: str) -> str:
        self._ensure_loaded()
        assert self._pipe is not None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        out = self._pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        if not out:
            return ""
        text = out[0]["generated_text"]
        # Some versions return the chat list; normalise to the assistant string.
        if isinstance(text, list):
            text = text[-1].get("content", "") if text else ""
        return str(text).strip()
