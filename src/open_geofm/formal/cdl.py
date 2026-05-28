"""Conditional Declaration Language (CDL) types.

A FormalGeo problem is represented by four CDL blocks plus a goal:
    - construction_cdl : how the figure is built (points, lines, shapes)
    - text_cdl         : algebraic / numeric conditions stated in the text
    - image_cdl        : conditions readable only from the diagram (e.g. tick marks)
    - goal_cdl         : the metric the solver must find
The solver also returns `theorem_seqs` (the proof trace) and `answer`.

Blueprint §2 Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        """`M_p` in Algorithm 1: the metric statements composing the problem.

        Per the paper, this is text_cdl ∪ image_cdl (geometric construction is held fixed).
        """
        return self.text_cdl + self.image_cdl


def parse_cdl(raw: str) -> tuple[str, ...]:
    """Parse a CDL block (one statement per line, ignoring blanks and `#` comments).

    FormalGeo7K stores each block as a list of one-statement-per-line. We accept
    either the raw block as a single string (what notebooks paste) or a single
    pre-split iterable (handled by the dataset loader). Comments use `#` to EOL.
    """
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            out.append(stripped)
    return tuple(out)
