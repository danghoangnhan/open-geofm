"""Symbolic answer verification: compare the NL answer to the FGPS answer.

Blueprint §2 Phase 3. Numeric tolerance after `sympy.nsimplify` (handles
symbolic FGPS answers like `sqrt(3)/2`).

Note: when the *template* rewriter is used, the NL solution is generated from the
verified FGPS answer, so verification is necessarily a tautology — it adds real
signal only for the LLM rewriter, which can hallucinate. Use the LLM rewriter
(or a held-out check) when you need verification to catch errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy

from ..config import DEFAULT_VERIFY_TOL, VerifyConfig

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class VerifyResult:
    accepted: bool
    extracted_answer: str | None
    expected_answer: str
    reason: str = ""


def extract_number(text: str) -> str | None:
    """Pull the last numeric answer out of a free-form NL solution string.

    Thousands separators are stripped first (``1,000`` -> ``1000``) so a grouped
    number doesn't parse as the bogus trailing group ``000`` (bug #22).
    """
    cleaned = text.replace(",", "")
    matches = _NUMBER_RE.findall(cleaned)
    return matches[-1] if matches else None


def _to_float(token: str) -> float | None:
    """Evaluate a numeric or symbolic token to a float, or None if non-numeric."""
    for attempt in (
        lambda: float(sympy.nsimplify(token)),
        lambda: float(sympy.sympify(token).evalf()),
    ):
        try:
            return attempt()
        except (sympy.SympifyError, TypeError, ValueError):
            continue
    return None


def verify(nl_answer: str, fgps_answer: str, *, tol: float = DEFAULT_VERIFY_TOL) -> VerifyResult:
    """Accept `nl_answer` iff its extracted number agrees with `fgps_answer`
    (possibly symbolic, e.g. `sqrt(3)/2`) to within `tol`."""
    extracted = extract_number(nl_answer)
    if extracted is None:
        return VerifyResult(False, None, fgps_answer, "no number found in NL answer")
    lhs = _to_float(extracted)
    rhs = _to_float(fgps_answer)
    if lhs is None or rhs is None:
        return VerifyResult(False, extracted, fgps_answer, "could not evaluate to a number")
    return VerifyResult(abs(lhs - rhs) <= tol, extracted, fgps_answer)


class SympyVerifier:
    """Concrete `AnswerVerifier`: the `verify()` logic with a configured tol."""

    def __init__(self, config: VerifyConfig | None = None) -> None:
        self.config = config or VerifyConfig()

    def verify_result(self, nl_answer: str, expected: str) -> VerifyResult:
        return verify(nl_answer, expected, tol=self.config.tol)

    def verify(self, nl_answer: str, expected: str) -> bool:
        return self.verify_result(nl_answer, expected).accepted
