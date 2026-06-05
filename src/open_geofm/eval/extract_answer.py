"""Regex-based answer extractor for MathVista-GPS (the $0 judge-LLM fallback).

MathVista-GPS answers are almost always a single MCQ letter (A-E) or a clean
number.
"""

from __future__ import annotations

import re

# The trailing `(?=\W|$)` pins the match to a *standalone* A-E (not the first
# letter of a word like "answer" / "equilateral").
_LETTER_RE = re.compile(
    r"(?:^|\b)(?:answer|final|option)?\s*[:=]?\s*\(?([A-E])\)?(?=\W|$)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _last_letter(text: str) -> str | None:
    """The LAST standalone MCQ letter in `text`, upper-cased, or None.

    Last (not first) so a reasoning trace that eliminates options before the
    conclusion — "rule out A and B; the answer is D" — yields D, not A (bug #6).
    """
    matches = _LETTER_RE.findall(text)
    return matches[-1].upper() if matches else None


def extract(response: str) -> str | None:
    """Pull the model's final answer out of a free-form reply.

    1. Scan the last line for a standalone MCQ letter (last match wins).
    2. Else scan the whole response (last match wins).
    3. Else fall back to the last clean number.
    4. Else None.
    """
    stripped = response.strip()
    if not stripped:
        return None
    last_line = stripped.splitlines()[-1]
    letter = _last_letter(last_line) or _last_letter(response)
    if letter:
        return letter
    nums = _NUMBER_RE.findall(response)
    return nums[-1] if nums else None


class AnswerExtractor:
    """OOP wrapper over `extract` (the `AnswerExtractor` seam)."""

    def extract(self, response: str) -> str | None:
        return extract(response)
