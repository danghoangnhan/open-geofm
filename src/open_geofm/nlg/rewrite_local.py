"""Local NLG rewriter backed by vLLM-served Qwen2.5-7B-Instruct.

Blueprint §2 Phase 5. fp16 fits in ~16 GB on the 5090; ~100 samples/min. `vllm`
is an optional GPU dependency reached via `importlib` (no bare in-function
import) so this module imports cleanly in the CPU venv.
"""

from __future__ import annotations

import importlib

from ..config import DEFAULT_LOCAL_NLG_MODEL, DEFAULT_NLG_MAX_TOKENS, DEFAULT_NLG_TEMPERATURE
from .rewrite_openai import SYSTEM_PROMPT


class LocalRewriter:
    def __init__(
        self,
        model: str = DEFAULT_LOCAL_NLG_MODEL,
        temperature: float = DEFAULT_NLG_TEMPERATURE,
        max_tokens: int = DEFAULT_NLG_MAX_TOKENS,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        **vllm_kwargs,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.vllm_kwargs = vllm_kwargs
        self._llm = None
        self._sampling_params = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as e:
            raise RuntimeError(
                "vLLM not installed. Install the GPU image: `uv sync --extra eval`."
            ) from e
        self._llm = vllm.LLM(model=self.model, **self.vllm_kwargs)
        self._sampling_params = vllm.SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens
        )

    def rewrite(self, prompt: str) -> str:
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self._llm.chat(messages=[messages], sampling_params=self._sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()
