"""OpenAI gpt-4o-mini NLG rewriter.

Blueprint §2 Phase 5. ~$3 for 10K samples (500 in + 300 out @ $0.15 / $0.60 per M tok).
Requires `OPENAI_API_KEY`.

The system prompt is the paper's Appendix C verbatim:
    "Given a geometry problem and its answer hint, write a answer to the problem.
     Ensure the answer is correct, concise, easy to understand, and written with
     clarity and natural flow."
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = (
    "Given a geometry problem and its answer hint, write a answer to the problem. "
    "Ensure the answer is correct, concise, easy to understand, and written with "
    "clarity and natural flow."
)


class OpenAIRewriter:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        # Cheap env-var check first so we fail fast in CI (no openai install needed).
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY env var not set.")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("Install with `uv sync --extra nlg` to use the OpenAI backend.") from e
        self._client = OpenAI()
        return self._client

    def rewrite(self, prompt: str) -> str:
        """Smooth a Phase-5 step-1 template draft into natural prose.

        `prompt` is expected to contain both the problem statement and the answer
        hint (the caller stitches them with `templates.draft_nl` + the FGPS answer).
        """
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
