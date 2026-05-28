"""Symbolic answer verification: re-run FGPS on `P_new`, compare to the NL answer.

Blueprint §2 Phase 3. Numeric tolerance 1e-3 after `sympy.nsimplify`. Tracks the
reject rate per batch — expect 20-40% rejects in the first generation pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy


@dataclass(frozen=True, slots=True)
class VerifyResult:
    accepted: bool
    extracted_answer: str | None
    expected_answer: str
    reason: str = ""


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    """Pull the last numeric answer out of a free-form NL solution string."""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def verify(nl_answer: str, fgps_answer: str, *, tol: float = 1e-3) -> VerifyResult:
    """Accept `nl_answer` iff its extracted number agrees with `fgps_answer`
    (which may be a sympy expression like `sqrt(3)/2`) to within `tol`."""
    extracted = extract_number(nl_answer)
    if extracted is None:
        return VerifyResult(False, None, fgps_answer, "no number found in NL answer")
    try:
        lhs = float(sympy.nsimplify(extracted))
        rhs = float(sympy.nsimplify(fgps_answer))
    except (sympy.SympifyError, TypeError, ValueError) as e:
        return VerifyResult(False, extracted, fgps_answer, f"sympify error: {e}")
    return VerifyResult(abs(lhs - rhs) <= tol, extracted, fgps_answer)
