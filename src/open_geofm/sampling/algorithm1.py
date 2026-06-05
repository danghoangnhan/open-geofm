"""Algorithm 1 from the GeoFM paper (arXiv:2510.27448).

Blueprint §2 Phase 2 (the novel part)::

    for P in FS:
        M_p   = MetricInfoOfProblemStatement(P)
        M_all = GatheringMetricInfo(P)        # BFS over theorems
        while m_p > 1:
            n = Random(1, min(|M_p|, |M_all| − |M_p|))
            M_del = RandomSelect(M_p, n);  M_add = RandomSelect(M_all \\ M_p, n)
            P_new = (P \\ M_del) ∪ M_add
            A_new = FormalGeoSolver(P_new)
            P_syn, A_syn = Template_and_LLM(P_new, A_new)
            if AnswerVerify(A_syn, A_new): S.add((P_syn, A_syn))

`run_algorithm1` is the backward-compatible functional driver (callbacks
injectable for tests); `Algorithm1Runner` is the OOP form the pipeline uses.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

from ..config import DEFAULT_RNG_SEED, SamplingConfig
from ..formal.base import Solver, SolverResult
from ..formal.cdl import Problem, goal_metric_for, value_of
from ..formal.solver import solve as _default_solve
from ..nlg.verify import verify as _default_nlg_verify
from .gather_metrics import gather_metric_info as _default_gather
from .goal_picker import GoalPicker, fallback_goal_from_trace, pick_goal

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    """One verified (problem, answer, trace) triple from Algorithm 1.

    `index` is the stable position assigned at accept time; the renderer and the
    dataset builder both derive the sample id from it, so the on-disk PNG name
    and the dataset record's image path always agree.
    """

    source_pid: int
    new_metrics: tuple[str, ...]
    deleted_metrics: tuple[str, ...]
    added_metrics: tuple[str, ...]
    goal: str
    answer: str
    theorem_seqs: tuple[str, ...]
    construction_cdl: tuple[str, ...] = field(default_factory=tuple)
    nl_problem: str | None = None
    nl_solution: str | None = None
    index: int | None = None

    def drawable_image_cdl(self) -> tuple[str, ...]:
        """The metrics the renderer can draw (value-bearing image conditions)."""
        return tuple(m for m in self.new_metrics if "Equal(" in m or "Value(" in m)


def sample_new_problem(
    problem: Problem,
    m_all: tuple[str, ...],
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """One swap iteration: pick n, draw M_del / M_add, return `(P_new, M_del, M_add)`.

    M_p is deduplicated first so the swap honours the paper's *set* semantics —
    FormalGeo7K seeds occasionally list a metric in both text_cdl and image_cdl,
    and without dedup a 'deleted' condition could survive via its other copy.
    """
    m_p = tuple(dict.fromkeys(problem.all_metric_conditions))  # dedup, order-preserving
    n_max = min(len(m_p), len(m_all) - len(m_p))
    if n_max < 1:
        raise ValueError("Cannot swap: |M_p| or |M_all \\ M_p| is empty.")

    n = rng.randint(1, n_max)
    m_del = tuple(rng.sample(list(m_p), n))
    pool = [m for m in m_all if m not in m_p]
    m_add = tuple(rng.sample(pool, n))

    remaining = tuple(m for m in m_p if m not in m_del)
    return remaining + m_add, m_del, m_add


def _split_text_image(
    problem: Problem, p_new: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Re-partition `p_new` into (text_cdl, image_cdl) by membership in the seed's
    original blocks; metrics not in either default to text_cdl."""
    image_set = set(problem.image_cdl)
    text_out: list[str] = []
    image_out: list[str] = []
    for m in p_new:
        if m in image_set:
            image_out.append(m)
        else:
            text_out.append(m)  # original text_cdl or novel from M_add
    return tuple(text_out), tuple(image_out)


class ConditionSampler:
    """OOP `ConditionSampler`: the metric-swap + text/image re-partition."""

    def sample(self, problem: Problem, m_all: tuple[str, ...], rng: random.Random):
        return sample_new_problem(problem, m_all, rng)

    def split_text_image(self, problem: Problem, p_new: tuple[str, ...]):
        return _split_text_image(problem, p_new)


def _fallback_metric(result: SolverResult, p_new: tuple[str, ...]) -> str | None:
    """Recover a value-bearing fallback goal when the picked goal was unsolvable.

    Prefer the solver's structured `derived_metrics` (real solves); fall back to
    parsing the trace (stub/legacy form). Returns None if nothing usable.
    """
    for m in result.derived_metrics:
        if m not in p_new and value_of(m) is not None:
            return m
    if not result.theorem_seqs:
        return None
    try:
        metric = fallback_goal_from_trace(result.theorem_seqs)
    except ValueError:
        return None
    return metric if value_of(metric) is not None else None


def _attempt_once(
    problem: Problem,
    m_all: tuple[str, ...],
    value_pool: tuple[str, ...],
    rng: random.Random,
    *,
    solve,
    rewrite,
    verify,
) -> SyntheticSample | None:
    """One Algorithm 1 inner iteration. Returns an accepted sample or None."""
    try:
        p_new, m_del, m_add = sample_new_problem(problem, m_all, rng)
    except ValueError:
        return None
    try:
        # Pick the goal from VALUE-BEARING metrics only, so a relational draw
        # (e.g. PerpendicularBetweenLine) doesn't waste a whole attempt.
        goal_metric = pick_goal(p_new, value_pool, rng=rng)
    except ValueError:
        return None
    candidate_answer = value_of(goal_metric)
    if candidate_answer is None:
        return None

    text_cdl, image_cdl = _split_text_image(problem, p_new)
    candidate = replace(
        problem,
        text_cdl=text_cdl,
        image_cdl=image_cdl,
        goal_cdl=goal_metric_for(goal_metric),
        theorem_seqs=(),
        answer=candidate_answer,
    )
    result = solve(candidate)
    if not result.solved:
        fallback = _fallback_metric(result, p_new)
        if fallback is None:
            return None
        candidate = replace(
            candidate, goal_cdl=goal_metric_for(fallback), answer=value_of(fallback)
        )
        result = solve(candidate)
        if not result.solved or result.answer is None:
            return None
        goal_metric = fallback

    assert result.answer is not None
    try:
        nl_problem, nl_solution = rewrite(candidate, result.answer)
    except Exception as e:
        log.warning("rewrite raised for pid=%s: %s", problem.pid, e)
        return None
    if not verify(nl_solution, result.answer):
        return None

    return SyntheticSample(
        source_pid=problem.pid,
        new_metrics=p_new,
        deleted_metrics=m_del,
        added_metrics=m_add,
        goal=goal_metric,
        answer=result.answer,
        theorem_seqs=result.theorem_seqs,
        construction_cdl=problem.construction_cdl,
        nl_problem=nl_problem,
        nl_solution=nl_solution,
    )


def _run_seed(
    problem: Problem,
    m_per_seed: int,
    rng: random.Random,
    *,
    solve,
    gather,
    verify,
    rewrite,
    max_attempts_factor: int,
) -> list[SyntheticSample]:
    try:
        m_all = gather(problem)
    except Exception as e:
        log.warning("gather failed for pid=%s: %s", problem.pid, e)
        return []
    m_p = tuple(dict.fromkeys(problem.all_metric_conditions))
    if len(m_all) <= len(m_p):
        log.info("pid=%s: |M_all| <= |M_p|, no swap pool; skipping.", problem.pid)
        return []
    value_pool = tuple(m for m in m_all if value_of(m) is not None)

    accepted: list[SyntheticSample] = []
    max_attempts = max(1, m_per_seed * max_attempts_factor)
    attempts = 0
    while len(accepted) < m_per_seed and attempts < max_attempts:
        attempts += 1
        sample = _attempt_once(
            problem, m_all, value_pool, rng, solve=solve, rewrite=rewrite, verify=verify
        )
        if sample is not None:
            accepted.append(sample)
    return accepted


def _default_verify(nl_answer: str, fgps_answer: str) -> bool:
    return _default_nlg_verify(nl_answer, fgps_answer).accepted


def run_algorithm1(
    seeds: Iterable[Problem],
    m_per_seed: int,
    *,
    seed: int = DEFAULT_RNG_SEED,
    solve_fn=None,
    gather_fn=None,
    verify_fn=None,
    rewrite_fn=None,
    max_attempts_factor: int = 5,
) -> list[SyntheticSample]:
    """Functional Algorithm 1 driver. Callbacks injectable for tests; defaults
    bind to the real solver / gatherer / verifier. `rewrite_fn` is required."""
    rng = random.Random(seed)
    solve = solve_fn or _default_solve
    gather = gather_fn or _default_gather
    verify = verify_fn or _default_verify
    if rewrite_fn is None:
        raise ValueError("rewrite_fn is required; pass an NLG callback (Phase 5).")

    out: list[SyntheticSample] = []
    for problem in seeds:
        out.extend(
            _run_seed(
                problem,
                m_per_seed,
                rng,
                solve=solve,
                gather=gather,
                verify=verify,
                rewrite=rewrite_fn,
                max_attempts_factor=max_attempts_factor,
            )
        )
    return out


class Algorithm1Runner:
    """OOP Algorithm 1: holds the stage collaborators (gatherer, solver,
    rewriter, verifier) and yields verified samples per seed."""

    def __init__(
        self,
        *,
        gatherer,
        solver: Solver,
        rewriter,
        verifier,
        sampler: ConditionSampler | None = None,
        goal_picker: GoalPicker | None = None,
        config: SamplingConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.gatherer = gatherer
        self.solver = solver
        self.rewriter = rewriter
        self.verifier = verifier
        self.sampler = sampler or ConditionSampler()
        self.goal_picker = goal_picker or GoalPicker()
        self.config = config or SamplingConfig()
        self.rng = rng or random.Random(self.config.rng_seed)

    def run_for_seed(self, problem: Problem) -> list[SyntheticSample]:
        return _run_seed(
            problem,
            self.config.m_per_seed,
            self.rng,
            solve=lambda p: self.solver.solve(p),
            gather=self.gatherer.gather,
            verify=self.verifier.verify,
            rewrite=self.rewriter.rewrite,
            max_attempts_factor=self.config.max_attempts_factor,
        )

    def run(self, seeds: Iterable[Problem]) -> Iterator[SyntheticSample]:
        for problem in seeds:
            yield from self.run_for_seed(problem)
