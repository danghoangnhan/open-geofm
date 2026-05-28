"""Local NLG rewriter backed by vLLM-served Qwen2.5-7B-Instruct.

Blueprint §2 Phase 5. fp16 fits in ~16 GB on the 5090; throughput ~100 samples/min.
Uses Qwen2.5-Instruct's ChatML format — vLLM applies it via `chat()`.
"""

from __future__ import annotations

from .rewrite_openai import SYSTEM_PROMPT


class LocalRewriter:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        **vllm_kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vllm_kwargs = vllm_kwargs
        self._llm = None
        self._sampling_params = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "vLLM not installed. Install the GPU image: `uv sync --extra eval`."
            ) from e
        self._llm = LLM(model=self.model, **self.vllm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def rewrite(self, prompt: str) -> str:
        self._ensure_loaded()
        assert self._llm is not None and self._sampling_params is not None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        outputs = self._llm.chat(messages=[messages], sampling_params=self._sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()
