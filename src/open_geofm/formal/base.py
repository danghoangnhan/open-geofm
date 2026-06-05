"""Stage interfaces for the formal subsystem (CPU-only, formalgeo-free).

`ProblemSource` and `Solver` are the seams the sampling pipeline depends on.
Concrete implementations (`FormalGeo7KSource` in loader.py, `FGPSSolver` in
solver.py) eagerly import `formalgeo`; this module does not, so anything that
only needs the *interface* (Algorithm 1, the pipeline, tests with fakes) imports
from here and stays import-light.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .cdl import Problem


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Outcome of a single symbolic-solver call.

    `derived_metrics` are the value-bearing conditions FGPS proved along the way,
    in canonical `Equal(...)` form. They give Algorithm 1's goal-fallback a
    *structured* source (the paper's "last valid inference from the reasoning
    path") instead of string-parsing the opaque `theorem_seqs` trace.
    """

    solved: bool
    answer: str | None
    theorem_seqs: tuple[str, ...]
    timed_out: bool = False
    error: str | None = None
    derived_metrics: tuple[str, ...] = ()


@runtime_checkable
class ProblemSource(Protocol):
    """Loads FormalGeo problems by PID and iterates the corpus."""

    def load(self, pid: int) -> Problem: ...
    def iter_problems(self) -> Iterator[Problem]: ...


@runtime_checkable
class Solver(Protocol):
    """Runs symbolic search/verification on a `Problem`.

    The whole per-call surface is `(problem, candidate_answer)`; all search
    hyperparameters are fixed at construction from config.
    """

    def solve(self, problem: Problem, candidate_answer: str | None = None) -> SolverResult: ...
