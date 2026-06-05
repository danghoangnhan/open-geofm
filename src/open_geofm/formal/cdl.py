"""Conditional Declaration Language (CDL) value objects.

A FormalGeo problem is four CDL blocks plus a goal:
    - construction_cdl : how the figure is built (points, lines, shapes)
    - text_cdl         : algebraic / numeric conditions stated in the text
    - image_cdl        : conditions readable only from the diagram (tick marks…)
    - goal_cdl         : the metric the solver must find
The solver also returns `theorem_seqs` (the proof trace) and `answer`.

`MetricCdl` is the canonical value object for a single metric condition. It owns
the `Equal(...)` / `Value(...)` duality in one place, replacing the three
duplicated regex helpers (`value_of`, `goal_metric_for`, `_to_equal_form`) that
used to live in `sampling.gather_metrics`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# `Equal(X,v)` and `Value(X,v)` both carry a value on the RHS; `Value(X)` is the
# goal form (no value). The leading group is greedy so it spans nested predicate
# parens like `LengthOfLine(AB)`.
_EQUAL_RE = re.compile(r"^Equal\((.+),([^,]+)\)$")
_VALUE_RE = re.compile(r"^Value\((.+),([^,]+)\)$")


@dataclass(frozen=True, slots=True)
class MetricCdl:
    """A parsed metric condition: ``head(inner[, value])``.

    Examples:
        ``Equal(LengthOfLine(AB),5)`` -> head='Equal', inner='LengthOfLine(AB)', value='5'
        ``Value(LengthOfLine(AB))``   -> head='Value', inner='LengthOfLine(AB)', value=None
    """

    head: str
    inner: str
    value: str | None = None

    @classmethod
    def from_str(cls, cdl: str) -> MetricCdl | None:
        """Parse a value-bearing metric (`Equal(X,v)` / `Value(X,v)`). Returns
        None when the string isn't a value-bearing metric condition."""
        for rx, head in ((_EQUAL_RE, "Equal"), (_VALUE_RE, "Value")):
            m = rx.match(cdl)
            if m:
                return cls(head=head, inner=m.group(1).strip(), value=m.group(2).strip())
        return None

    def to_equal_form(self) -> str:
        """Canonical `Equal(inner,value)` string (FGPS emits `Value(...)`, seeds
        use `Equal(...)`; we canonicalise so swaps can compare strings)."""
        return f"Equal({self.inner},{self.value})"

    def to_goal_cdl(self) -> str:
        """The goal-CDL form FGPS expects: ``Value(inner)`` (no value)."""
        return f"Value({self.inner})"


def to_equal_form(cdl: str) -> str:
    """`Value(LengthOfLine(AB),5)` -> `Equal(LengthOfLine(AB),5)`. Identity otherwise."""
    parsed = MetricCdl.from_str(cdl)
    return parsed.to_equal_form() if parsed is not None else cdl


def value_of(cdl: str) -> str | None:
    """Pull the RHS value out of an `Equal(X,v)` / `Value(X,v)` metric, or None."""
    parsed = MetricCdl.from_str(cdl)
    return parsed.value if parsed is not None else None


def goal_metric_for(cdl: str) -> str:
    """`Equal(LengthOfLine(AB),5)` -> `Value(LengthOfLine(AB))`. Identity otherwise."""
    parsed = MetricCdl.from_str(cdl)
    return parsed.to_goal_cdl() if parsed is not None else cdl


@dataclass(frozen=True, slots=True)
class Problem:
    """A FormalGeo problem in CDL form, with the symbolic answer once solved."""

    pid: int
    construction_cdl: tuple[str, ...]
    text_cdl: tuple[str, ...]
    image_cdl: tuple[str, ...]
    goal_cdl: str
    theorem_seqs: tuple[str, ...] = field(default_factory=tuple)
    answer: str | None = None

    @property
    def all_metric_conditions(self) -> tuple[str, ...]:
        """`M_p` in Algorithm 1: the metric statements composing the problem
        (text_cdl ∪ image_cdl; the geometric construction is held fixed)."""
        return self.text_cdl + self.image_cdl

    def to_fgps_cdl(
        self,
        answer: str | None = None,
        *,
        require_answer: bool = True,
        unused_goal_cdl: str = "Value(unused)",
        stub_answer: str = "0",
    ) -> dict:
        """Single canonical `Problem -> FGPS dict` mapping (used by both the
        solver verifier and the gather BFS, which used to duplicate it).

        `answer` overrides `self.answer`. When neither is set: the solver path
        passes `require_answer=True` and we raise; the gather path passes
        `require_answer=False` and we substitute `stub_answer` (the parser needs
        a value but forward expansion never reads it).
        """
        resolved = answer if answer is not None else self.answer
        if resolved is None:
            if require_answer:
                raise ValueError(
                    "FGPS requires a candidate answer to verify; pass `answer=...` "
                    "or set `Problem.answer`."
                )
            resolved = stub_answer
        return {
            "problem_id": self.pid,
            "construction_cdl": list(self.construction_cdl),
            "text_cdl": list(self.text_cdl),
            "image_cdl": list(self.image_cdl),
            "goal_cdl": self.goal_cdl or unused_goal_cdl,
            "problem_answer": str(resolved),
        }


def parse_cdl(raw: str, *, comment_prefix: str = "#") -> tuple[str, ...]:
    """Parse a CDL block (one statement per line, ignoring blanks and comments)."""
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.split(comment_prefix, 1)[0].strip()
        if stripped:
            out.append(stripped)
    return tuple(out)
