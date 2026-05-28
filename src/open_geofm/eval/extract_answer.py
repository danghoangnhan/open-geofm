"""Regex-based answer extractor for MathVista-GPS (cost-free fallback for judge LLM).

Blueprint §2 Phase 8 + Pitfalls: MathVista-GPS answers are almost always a
single letter (A-E) or a clean number.
"""

from __future__ import annotations

import re

# The trailing `(?=\W|$)` is load-bearing: without it, `[A-E]` (under
# IGNORECASE = `[A-Ea-e]`) matches the *first* letter of any word starting
# with a, b, c, d, or e — including "answer" / "equilateral" / "five",
# wrongly returning that letter as the MCQ answer. The lookahead pins the
# match to a *standalone* letter (one not part of a longer word).
_LETTER_RE = re.compile(
    r"(?:^|\b)(?:answer|final|option)?\s*[:=]?\s*\(?([A-E])\)?(?=\W|$)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract(response: str) -> str | None:
    """Pull the model's final answer out of a free-form reply.

    Strategy (same as MathVista's `extract_answer`):
        1. Search the *last line* first for a standalone MCQ letter A-E
           (case-insensitive; standalone = not part of a longer word). This
           lets models that think aloud and conclude on the last line
           ("…so it's not D; the answer is C") still be parsed correctly.
        2. If the last line has no letter, scan the full response.
        3. If still no letter, fall back to the last *clean number* in the
           response.
        4. Return None only if neither rule matches.
    """
    last_line = response.strip().splitlines()[-1] if response.strip() else ""
    m = _LETTER_RE.search(last_line) or _LETTER_RE.search(response)
    if m:
        return m.group(1).upper()
    nums = _NUMBER_RE.findall(response)
    return nums[-1] if nums else None
