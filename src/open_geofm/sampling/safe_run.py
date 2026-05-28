"""Notebook-safe wrapper around `run_algorithm1`.

FGPS occasionally deadlocks inside C-extension code (notably the sympy
`solve_equations` path when the metric set is underdetermined). The
`func_timeout` decorators inside `formal.solver.solve` and
`sampling.gather_metrics.gather_metric_info` catch most cases, but a
small minority sneak past and hang the Python kernel. That's fine for
the production multiprocess driver (each worker can be SIGKILL'd) but
breaks notebook execution — `nbclient` kills the *whole kernel* if a
single cell exceeds its timeout, leaving the rest of the notebook
unexecuted.

`run_algorithm1_in_subprocess(...)` solves that by running Algorithm 1
in a spawn-context child process with a hard wall-clock budget. If the
child overruns we SIGKILL it and return an empty list. The parent
kernel survives.

Callbacks are not user-injectable — that would require pickling
lambdas through `multiprocessing.Process`, which doesn't work. Instead
the wrapper accepts a fixed `rewriter` enum and named timeout knobs;
the child reconstitutes the callbacks from the production
implementations. Notebooks that need full callback control should keep
using `run_algorithm1` directly — they accept the kernel-death risk.

Blueprint §2 Phase 2 + Phase 5 (the deadlocks live at this seam).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Literal

from ..formal.cdl import Problem
from .algorithm1 import SyntheticSample

log = logging.getLogger(__name__)

Rewriter = Literal["template", "stub"]


def _worker(
    seeds: list[Problem],
    *,
    m_per_seed: int,
    gather_timeout_s: float,
    solve_timeout_s: float,
    bfs_depth: int,
    rewriter: Rewriter,
    rng_seed: int,
    max_attempts_factor: int,
    queue: mp.Queue,
) -> None:
    """Subprocess body. Catches every exception so the parent gets a clean signal."""
    try:
        from ..formal.solver import solve
        from ..nlg.templates import draft_nl
        from ..nlg.verify import verify
        from .algorithm1 import run_algorithm1
        from .gather_metrics import gather_metric_info

        def rewrite_template(problem: Problem, fgps_answer: str) -> tuple[str, str]:
            nl_problem = draft_nl(
                problem.text_cdl + problem.image_cdl, problem.goal_cdl
            )
            nl_solution = (
                f"From the given conditions, the symbolic engine derives that "
                f"the answer is {fgps_answer}."
            )
            return nl_problem, nl_solution

        def rewrite_stub(problem: Problem, fgps_answer: str) -> tuple[str, str]:
            return (
                f"(synthetic) From the given conditions, find {problem.goal_cdl}.",
                f"By FGPS derivation, the answer is {fgps_answer}.",
            )

        rewrite_fn = {"template": rewrite_template, "stub": rewrite_stub}[rewriter]

        result = run_algorithm1(
            seeds,
            m_per_seed=m_per_seed,
            seed=rng_seed,
            solve_fn=lambda p: solve(p, timeout_s=solve_timeout_s),
            gather_fn=lambda p: gather_metric_info(
                p, max_depth=bfs_depth, timeout_s=gather_timeout_s
            ),
            verify_fn=lambda nl, exp: verify(nl, exp).accepted,
            rewrite_fn=rewrite_fn,
            max_attempts_factor=max_attempts_factor,
        )
        queue.put(("ok", result))
    except BaseException as e:  # surface ANY error to parent
        # `repr(e)` keeps the type + message but loses traceback. That's
        # fine for the kernel-survival use case; the user can rerun with
        # `run_algorithm1` directly to get the full traceback.
        queue.put(("err", repr(e)))


def run_algorithm1_in_subprocess(
    seeds: list[Problem],
    *,
    m_per_seed: int,
    gather_timeout_s: float = 30.0,
    solve_timeout_s: float = 15.0,
    bfs_depth: int = 1,
    rewriter: Rewriter = "template",
    rng_seed: int = 42,
    max_attempts_factor: int = 10,
    process_timeout_s: float = 180.0,
) -> list[SyntheticSample]:
    """Run Algorithm 1 in a spawn-context subprocess with a hard wall-clock budget.

    A stuck FGPS call kills the subprocess (SIGKILL) without bringing down
    the parent Python kernel — the use case is notebook execution where
    nbclient's per-cell timeout would otherwise terminate the kernel.

    Args:
        seeds: list of seed `Problem`s.
        m_per_seed: target accepted samples per seed (Algorithm 1's `m`).
        gather_timeout_s, solve_timeout_s: inner per-call timeouts.
        bfs_depth: `gather_metric_info` `max_depth`.
        rewriter: "template" stitches `templates.draft_nl` with a brief
            solution; "stub" emits a fixed NL pair (test/demo path).
        rng_seed: RNG seed forwarded to `run_algorithm1`.
        max_attempts_factor: forwarded to `run_algorithm1`.
        process_timeout_s: hard wall-clock budget for the subprocess.
            Default 180 s; pick lower (~30 s) for one-seed demos.

    Returns:
        List of `SyntheticSample`. Empty list iff the subprocess
        overran `process_timeout_s` (logged) or died without producing
        output (also logged).

    Raises:
        RuntimeError: the subprocess raised before completing.
            ``str(exc)`` carries ``repr()`` of the child-side error.
    """
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker,
        args=(list(seeds),),
        kwargs={
            "m_per_seed": m_per_seed,
            "gather_timeout_s": gather_timeout_s,
            "solve_timeout_s": solve_timeout_s,
            "bfs_depth": bfs_depth,
            "rewriter": rewriter,
            "rng_seed": rng_seed,
            "max_attempts_factor": max_attempts_factor,
            "queue": queue,
        },
    )
    proc.start()
    proc.join(timeout=process_timeout_s)

    if proc.is_alive():
        log.warning(
            "Algorithm 1 subprocess exceeded %.1f s wall-clock; SIGKILL.",
            process_timeout_s,
        )
        proc.kill()
        proc.join(timeout=5.0)
        return []

    try:
        status, payload = queue.get_nowait()
    except Exception:
        # Subprocess died before posting a result (e.g. SIGSEGV inside a
        # C extension). Return [] so the notebook can continue.
        log.warning(
            "Algorithm 1 subprocess produced no output (exit=%s).", proc.exitcode
        )
        return []

    if status == "err":
        raise RuntimeError(f"Algorithm 1 subprocess raised: {payload}")
    return payload  # type: ignore[no-any-return]
