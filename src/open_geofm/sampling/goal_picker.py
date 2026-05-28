"""Pick the goal metric for a synthesised problem.

Blueprint §2 Phase 2: "randomly choose one metric condition different from the
new problem statement as the goal." If unsolvable, fall back to "the last valid
inference from the symbolic engine's reasoning path."
"""

from __future__ import annotations

import random


def pick_goal(
    new_problem_metrics: tuple[str, ...],
    m_all: tuple[str, ...],
    seed: int | None = None,
) -> str:
    """Pick a goal from `M_all \\ new_problem_metrics`."""
    candidates = tuple(m for m in m_all if m not in new_problem_metrics)
    if not candidates:
        raise ValueError("No candidate goal: M_all is a subset of the new problem statement.")
    rng = random.Random(seed)
    return rng.choice(candidates)


def fallback_goal_from_trace(theorem_seqs: tuple[str, ...]) -> str:
    """When the picked goal turns out unsolvable, recover the last derived metric
    from the solver's trace.

    FGPS step format (BitSecret/FGPS): ``"<theorem_name>(<premise_ids>) -> <metric>"``.
    We split on ``"->"`` and take the right-hand side of the final step; if the
    trace is empty or malformed, raise ValueError so the caller can drop this
    sample rather than emit something unverifiable.
    """
    if not theorem_seqs:
        raise ValueError("Cannot pick fallback goal from empty theorem trace.")
    last = theorem_seqs[-1]
    if "->" not in last:
        raise ValueError(f"Final theorem step lacks '->' arrow: {last!r}")
    return last.rsplit("->", 1)[1].strip()
