"""Generate the synthetic open-geofm dataset (host CPU venv).

Blueprint §2 Phases 2-6 driver. Runs on the host uv venv — no CUDA.

Pipeline per seed:
    load_problem → gather_metric_info (BFS) → run_algorithm1 (swap + FGPS verify)
    → render (matplotlib | gmbl) → template-draft NLG (+ optional LLM smooth)
    → JSONL + HF Dataset.save_to_disk

Usage::

    # Single-process baseline:
    uv run python scripts/02_generate_dataset.py \\
        --n 100 --renderer matplotlib --out data/open-geofm-mini-100

    # 8-worker multiprocess (blueprint's target throughput ~5K samples/hr):
    uv run python scripts/02_generate_dataset.py \\
        --n 5000 --n-workers 8 --out data/open-geofm-mini-5k

    # With local vLLM rewriter (force single worker — only one process can host
    # the vLLM model):
    OPEN_GEOFM_REWRITER=local \\
    uv run python scripts/02_generate_dataset.py --n 5000 --rewriter llm \\
        --n-workers 1 --out data/open-geofm-mini-5k-llm
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import typer

from open_geofm.dataset.builder import build, make_sample_id
from open_geofm.sampling.algorithm1 import SyntheticSample

log = logging.getLogger("open_geofm.generate")

_RENDERER_NAMES = ("matplotlib", "gmbl")

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ---------------------------------------------------------------------------
# Worker-side config + helpers (top-level so they pickle cleanly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerConfig:
    """Picklable config bundle for `_process_seed`. Keep field types primitive
    (str / int / float / bool / Path) so any start method works."""

    image_dir: Path
    renderer: str
    rewriter: str
    m_per_seed: int
    seed: int
    gather_timeout: float
    solve_timeout: float
    bfs_depth: int
    verify_with_sympy: bool
    max_attempts_factor: int


def _make_rewriter(mode: str):
    """Return a `(Problem, fgps_answer) -> (nl_problem, nl_solution)` callback.

    Modes:
      * ``template``: deterministic Phase-5 step-1 template draft only (no LLM).
        Fast, free, reproducible — the right default for smoke tests + CI.
      * ``llm``: template draft → LLM smooth via `OPEN_GEOFM_REWRITER` (local
        vLLM or OpenAI gpt-4o-mini). Requires the `nlg` or `eval` extras.
    """
    from open_geofm.nlg.templates import draft_nl

    # Memoised so the (possibly ~16 GB vLLM) rewriter is built ONCE per process,
    # not once per seed (fix for the per-seed model-reload OOM, bug #8).
    return _build_rewriter(mode, draft_nl)


@cache
def _build_rewriter(mode: str, draft_nl):
    if mode == "template":

        def rewrite(p, answer: str) -> tuple[str, str]:
            nl_problem = draft_nl(p.text_cdl + p.image_cdl, p.goal_cdl)
            nl_solution = (
                f"From the given conditions and the symbolic engine's derivation, "
                f"the answer is {answer}."
            )
            return nl_problem, nl_solution

        return rewrite

    if mode == "llm":
        from open_geofm.nlg.backend import get_rewriter

        backend = get_rewriter()

        def rewrite(p, answer: str) -> tuple[str, str]:
            draft = draft_nl(p.text_cdl + p.image_cdl, p.goal_cdl)
            prompt = (
                f"Geometry problem (draft):\n{draft}\n\n"
                f"Hint: the correct answer is {answer}.\n\n"
                f"Rewrite the problem and provide a clear, concise solution "
                f"that reaches the answer."
            )
            text = backend.rewrite(prompt)
            if "\n\n" in text:
                head, tail = text.split("\n\n", 1)
                return head.strip(), tail.strip()
            return draft, text.strip()

        return rewrite

    raise typer.BadParameter(f"--rewriter must be 'template' or 'llm', got {mode!r}")


def _render_sample(sample: SyntheticSample, renderer, out_dir: Path, *, seed: int) -> Path:
    """Render `sample` to `<out_dir>/<id>.png`. Returns the file path.

    The id comes from the SHARED `make_sample_id`, so the PNG filename matches
    the dataset record's image path exactly (renderer/builder id-mismatch fix).
    """
    sample_id = make_sample_id(sample)
    path = out_dir / f"{sample_id}.png"
    img = renderer.render(sample.construction_cdl, sample.drawable_image_cdl(), seed=seed)
    img.save(path)
    return path


def _render_seed_for(sample: SyntheticSample, base_seed: int) -> int:
    """A distinct, deterministic render seed per sample so each gets its own
    ±rotation / jitter augmentation (bug #26: all samples of one seed shared
    the same augmentation)."""
    return base_seed + int(make_sample_id(sample).rsplit("-", 1)[-1], 16)


def _worker_init() -> None:
    """Pool-init: cap BLAS / OpenMP threads so K workers don't oversubscribe.

    Each FGPS solve and sympy ``solve_equations`` calls numpy/scipy, which by
    default fans out to all cores. With multiple workers that's catastrophic —
    we want one BLAS thread per worker.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    # Re-init logging in the worker (Linux fork inherits it but Spawn doesn't).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _process_seed(args: tuple[int, _WorkerConfig]) -> list[SyntheticSample]:
    """Worker: run Algorithm 1 for one seed pid, render accepted samples to
    disk, return the in-memory `SyntheticSample` list for the parent to collect.

    Wrapped in a single ``try`` so one bad pid never crashes the pool.
    """
    pid, cfg = args
    # Imports happen *inside* the worker so the parent process doesn't need to
    # eagerly load formalgeo / matplotlib / etc. when --n-workers=1 is requested.
    from open_geofm.formal.loader import load_problem
    from open_geofm.formal.solver import solve as fgps_solve
    from open_geofm.nlg.verify import verify as nlg_verify
    from open_geofm.render import gmbl_renderer, matplotlib_renderer
    from open_geofm.sampling.algorithm1 import run_algorithm1
    from open_geofm.sampling.gather_metrics import gather_metric_info

    renderers = {"matplotlib": matplotlib_renderer, "gmbl": gmbl_renderer}
    render_mod = renderers[cfg.renderer]
    rewrite_fn = _make_rewriter(cfg.rewriter)

    try:
        problem = load_problem(pid)
    except Exception as e:
        log.debug("skip pid=%d (load failed: %s)", pid, e)
        return []

    try:
        batch = run_algorithm1(
            [problem],
            m_per_seed=cfg.m_per_seed,
            seed=cfg.seed + pid,  # decorrelate per-seed RNG
            solve_fn=lambda p: fgps_solve(p, timeout_s=cfg.solve_timeout),
            gather_fn=lambda p: gather_metric_info(
                p, max_depth=cfg.bfs_depth, timeout_s=cfg.gather_timeout
            ),
            verify_fn=(
                (lambda nl, ans: nlg_verify(nl, ans).accepted)
                if cfg.verify_with_sympy
                else (lambda *_: True)
            ),
            rewrite_fn=rewrite_fn,
            max_attempts_factor=cfg.max_attempts_factor,
        )
    except Exception as e:
        log.warning("Algorithm 1 raised for pid=%d: %s", pid, e)
        return []

    out: list[SyntheticSample] = []
    for sample in batch:
        try:
            _render_sample(
                sample, render_mod, cfg.image_dir, seed=_render_seed_for(sample, cfg.seed)
            )
        except Exception as e:
            log.warning("renderer failed for pid=%d: %s", sample.source_pid, e)
            continue
        out.append(sample)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _drain(
    results: Iterator[list[SyntheticSample]],
    target_n: int,
    *,
    t_start: float,
    log_every: int,
) -> list[SyntheticSample]:
    """Pull from `results` until we have `target_n` samples; log progress."""
    accepted: list[SyntheticSample] = []
    for batch in results:
        for sample in batch:
            accepted.append(sample)
            if len(accepted) % log_every == 0:
                rate = len(accepted) / max(time.time() - t_start, 1e-6) * 3600
                log.info("accepted=%d/%d (~%.0f samples/hr)", len(accepted), target_n, rate)
            if len(accepted) >= target_n:
                return accepted
    return accepted


@app.command()
def main(
    n: int = typer.Option(100, help="Total synthetic samples to generate."),
    m_per_seed: int = typer.Option(3, help="Samples per seed (Algorithm 1's `m`)."),
    n_workers: int = typer.Option(
        1, help="Worker processes (1 = single-process, no Pool overhead)."
    ),
    renderer: str = typer.Option("matplotlib", help="matplotlib | gmbl."),
    rewriter: str = typer.Option("template", help="template | llm."),
    out: str = typer.Option("data/open-geofm-mini", help="Output dataset directory."),
    seed: int = typer.Option(42, help="RNG seed."),
    seed_start: int = typer.Option(1, help="First FormalGeo7K PID to use as a seed."),
    seed_end: int = typer.Option(7000, help="Last PID (inclusive)."),
    gather_timeout: float = typer.Option(30.0, help="Per-seed BFS timeout (seconds)."),
    solve_timeout: float = typer.Option(15.0, help="Per-attempt FGPS verify timeout (seconds)."),
    bfs_depth: int = typer.Option(1, help="`gather_metric_info` max_depth."),
    verify_with_sympy: bool = typer.Option(
        True, help="Run nlg.verify on the NL solution (rejects mismatched answers)."
    ),
    max_attempts_factor: int = typer.Option(
        20,
        help=(
            "Per-seed swap attempts = `m_per_seed * this`. Higher = more chances "
            "to find a swap FGPS can verify, at the cost of wall-clock per seed."
        ),
    ),
    chunksize: int = typer.Option(
        1, help="`Pool.imap_unordered` chunk size. Larger = less IPC, less granular cancel."
    ),
    log_every: int = typer.Option(10, help="Log progress every N accepted samples."),
) -> None:
    """Phase 2-6 dataset generator. Multi-process; blueprint target ~5K samples/hr."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if renderer not in _RENDERER_NAMES:
        raise typer.BadParameter(f"--renderer must be one of {list(_RENDERER_NAMES)}.")
    if rewriter == "llm" and n_workers > 1:
        # vLLM holds the model in a single process; forking it would either OOM
        # or duplicate it. OpenAI rewriter is fine to multi-process but the
        # request-per-sample fan-out is fast enough that it's not the bottleneck.
        raise typer.BadParameter(
            "--rewriter llm requires --n-workers 1 (only one process can host the LLM)."
        )

    out_dir = Path(out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    cfg = _WorkerConfig(
        image_dir=image_dir,
        renderer=renderer,
        rewriter=rewriter,
        m_per_seed=m_per_seed,
        seed=seed,
        gather_timeout=gather_timeout,
        solve_timeout=solve_timeout,
        bfs_depth=bfs_depth,
        verify_with_sympy=verify_with_sympy,
        max_attempts_factor=max_attempts_factor,
    )

    pids = range(seed_start, seed_end + 1)
    t_start = time.time()

    if n_workers <= 1:
        # Inline path: avoids the Pool start-up overhead for single-process runs
        # (Docker-image NLG smoke tests, CI integration runs).
        def _inline():
            for pid in pids:
                if pid % log_every == 1:
                    log.info("pid=%d (running)", pid)
                yield _process_seed((pid, cfg))

        accepted = _drain(_inline(), n, t_start=t_start, log_every=log_every)
    else:
        log.info("starting Pool(%d) over %d pids", n_workers, len(pids))
        ctx = mp.get_context("spawn")  # spawn avoids inheriting matplotlib + caches
        with ctx.Pool(n_workers, initializer=_worker_init) as pool:
            results = pool.imap_unordered(
                _process_seed, ((pid, cfg) for pid in pids), chunksize=chunksize
            )
            try:
                accepted = _drain(results, n, t_start=t_start, log_every=log_every)
            finally:
                pool.terminate()
                pool.join()

    log.info(
        "DONE: %d samples accepted in %.1fs (target %d). Writing to %s",
        len(accepted),
        time.time() - t_start,
        n,
        out_dir,
    )
    artifact = build(accepted, image_dir=image_dir, out_dir=out_dir)
    log.info("artifact: %s", artifact if isinstance(artifact, dict) else type(artifact).__name__)


if __name__ == "__main__":
    app()
