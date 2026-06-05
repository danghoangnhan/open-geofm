"""Two-step NLG: template draft + optional LLM smoothing.

`ProblemRewriter` is the `(problem, answer) -> (nl_problem, nl_solution)` seam
Algorithm 1 calls. `TemplateProblemRewriter` builds the draft from the predicate
templates and, in `mode='llm'`, delegates smoothing to an injected `Rewriter`
(vLLM / transformers / OpenAI) — consolidating what used to be the loose
`_make_rewriter` closure in the generation script.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import NlgConfig
from ..formal.cdl import Problem
from .templates import draft_nl


@runtime_checkable
class ProblemRewriter(Protocol):
    def rewrite(self, problem: Problem, answer: str) -> tuple[str, str]: ...


class TemplateProblemRewriter:
    """Concrete `ProblemRewriter`.

    `mode='template'` (default): deterministic, free, reproducible — the draft is
    the problem and the solution states the verified answer. `mode='llm'`: the
    draft + answer hint are smoothed by the injected `llm` rewriter.
    """

    def __init__(
        self, config: NlgConfig | None = None, *, llm=None, seed: int | None = None
    ) -> None:
        self.config = config or NlgConfig()
        self.llm = llm
        self.seed = seed

    def _draft(self, problem: Problem) -> str:
        return draft_nl(problem.text_cdl + problem.image_cdl, problem.goal_cdl, seed=self.seed)

    def rewrite(self, problem: Problem, answer: str) -> tuple[str, str]:
        draft = self._draft(problem)
        if self.config.mode == "template" or self.llm is None:
            solution = (
                "From the given conditions and the symbolic engine's derivation, "
                f"the answer is {answer}."
            )
            return draft, solution
        prompt = (
            f"Geometry problem (draft):\n{draft}\n\n"
            f"Hint: the correct answer is {answer}.\n\n"
            f"Rewrite the problem and provide a clear, concise solution that "
            f"reaches the answer."
        )
        text = self.llm.rewrite(prompt)
        if "\n\n" in text:
            head, tail = text.split("\n\n", 1)
            return head.strip(), tail.strip()
        return draft, text.strip()
