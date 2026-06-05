"""Pick the goal metric for a synthesised problem.

Blueprint §2 Phase 2: "randomly choose one metric condition different from the
new problem statement as the goal." If unsolvable, fall back to "the last valid
inference from the symbolic engine's reasoning path."
"""

from __future__ import annotations

import random
import re

from ..config import DEFAULT_TRACE_ARROW
from ..formal.cdl import MetricCdl

# `Pred(args) = value` infix form (how some traces print a derived metric).
_INFIX_RE = re.compile(r"^(.+?)\s*=\s*(.+)$")


def canonicalize_metric(metric: str) -> str:
    """Normalise a metric to canonical `Equal(inner,value)` form so the
    `value_of` / `goal_metric_for` parsers accept it (fix for the fallback
    format mismatch). Handles `Equal(...)`, `Value(X,v)`, and `Pred(args) = v`.
    """
    parsed = MetricCdl.from_str(metric)
    if parsed is not None:
        return parsed.to_equal_form()
    m = _INFIX_RE.match(metric.strip())
    if m:
        return f"Equal({m.group(1).strip()},{m.group(2).strip()})"
    return metric


def pick_goal(
    new_problem_metrics: tuple[str, ...],
    m_all: tuple[str, ...],
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Pick a goal from `M_all \\ new_problem_metrics`."""
    candidates = tuple(m for m in m_all if m not in new_problem_metrics)
    if not candidates:
        raise ValueError("No candidate goal: M_all is a subset of the new problem statement.")
    rng = rng if rng is not None else random.Random(seed)
    return rng.choice(candidates)


def fallback_goal_from_trace(
    theorem_seqs: tuple[str, ...], *, arrow: str = DEFAULT_TRACE_ARROW
) -> str:
    """Recover the last derived metric from a solver trace when the picked goal
    was unsolvable, in canonical `Equal(...)` form.

    Handles the human-readable ``"<theorem>(<ids>) -> <metric>"`` trace form.
    Raises ValueError on an empty trace or a step that carries no derived metric
    (e.g. the real FGPS ``(name, premise, args)`` tuple form, which has no
    arrow) — the caller then drops the sample instead of emitting a malformed
    goal. For real solves, prefer `SolverResult.derived_metrics`, which is a
    structured source that does not depend on trace string format.
    """
    if not theorem_seqs:
        raise ValueError("Cannot pick fallback goal from empty theorem trace.")
    last = theorem_seqs[-1]
    if arrow not in last:
        raise ValueError(f"Final theorem step carries no derived metric: {last!r}")
    return canonicalize_metric(last.rsplit(arrow, 1)[1].strip())


class GoalPicker:
    """OOP wrapper over goal selection + trace fallback (the `GoalPicker` seam)."""

    def __init__(self, *, arrow: str = DEFAULT_TRACE_ARROW) -> None:
        self.arrow = arrow

    def pick(
        self, new_problem_metrics: tuple[str, ...], m_all: tuple[str, ...], rng: random.Random
    ) -> str:
        return pick_goal(new_problem_metrics, m_all, rng=rng)

    def fallback_from_trace(self, theorem_seqs: tuple[str, ...]) -> str:
        return fallback_goal_from_trace(theorem_seqs, arrow=self.arrow)
