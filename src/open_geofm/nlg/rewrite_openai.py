"""OpenAI gpt-4o-mini NLG rewriter.

Blueprint §2 Phase 5. The `openai` package is an optional dependency (the `nlg`
extra) that may be absent in the CPU/CI venv, so it's reached via
`importlib.import_module` *after* the API-key check rather than imported at module
top — there is no bare in-function `import openai` statement.
"""

from __future__ import annotations

import importlib
import os

from ..config import (
    DEFAULT_NLG_MAX_TOKENS,
    DEFAULT_NLG_SYSTEM_PROMPT,
    DEFAULT_NLG_TEMPERATURE,
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_NLG_MODEL,
)

SYSTEM_PROMPT = DEFAULT_NLG_SYSTEM_PROMPT


class OpenAIRewriter:
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_NLG_MODEL,
        temperature: float = DEFAULT_NLG_TEMPERATURE,
        max_tokens: int = DEFAULT_NLG_MAX_TOKENS,
        *,
        api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key_env = api_key_env
        self.system_prompt = system_prompt
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        # Cheap key check first so we fail fast in CI without needing openai installed.
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(f"{self.api_key_env} env var not set.")
        try:
            openai = importlib.import_module("openai")
        except ImportError as e:
            raise RuntimeError(
                "Install with `uv sync --extra nlg` to use the OpenAI backend."
            ) from e
        self._client = openai.OpenAI()
        return self._client

    def rewrite(self, prompt: str) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
