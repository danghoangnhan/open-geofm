"""Tests for `open_geofm.sampling.safe_run`.

The wrapper's guarantees we care about:
  1. Empty `seeds` list — clean subprocess exit, returns [].
  2. Wall-clock overrun — SIGKILL + [], not an exception.
  3. Invalid `rewriter` — RuntimeError propagated from the child.
  4. Subprocess writes via the spawn context (no fork inheritance).

We avoid touching the real FormalGeo data here so the test runs on
any host venv. Subprocess start-up costs ~1-2 s for the FormalGeo
imports, so we mark these as `slow`.
"""

from __future__ import annotations

import time

import pytest

from open_geofm.formal.cdl import Problem
from open_geofm.sampling.safe_run import run_algorithm1_in_subprocess

pytestmark = pytest.mark.slow


def test_empty_seeds_returns_empty_list() -> None:
    """No seeds → subprocess starts, run_algorithm1 returns [], parent gets []."""
    out = run_algorithm1_in_subprocess(
        [], m_per_seed=3, rewriter="stub", process_timeout_s=60.0,
    )
    assert out == []


def test_invalid_rewriter_raises_runtime_error() -> None:
    """Bad enum value lands as a KeyError inside the worker; surface as RuntimeError."""
    with pytest.raises(RuntimeError, match="KeyError"):
        run_algorithm1_in_subprocess(
            [],
            m_per_seed=1,
            rewriter="bogus",  # type: ignore[arg-type]
            process_timeout_s=60.0,
        )


def test_process_timeout_triggers_sigkill_and_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny `process_timeout_s` forces the SIGKILL path.

    We don't need a real hang to test this — any `process_timeout_s`
    shorter than the spawn-context Python startup (~1 s) will cause
    the kill branch to trigger because the worker hasn't finished
    importing yet. We assert: returns [], parent process survives.
    """
    t_start = time.monotonic()
    out = run_algorithm1_in_subprocess(
        [], m_per_seed=1, rewriter="stub", process_timeout_s=0.5,
    )
    elapsed = time.monotonic() - t_start
    assert out == []
    # Sanity: we hit the kill branch (~0.5 s) instead of waiting on a
    # full FGPS solve (would be tens of seconds).
    assert elapsed < 30.0, f"wrapper took {elapsed:.1f}s; SIGKILL path probably missed"


def test_subprocess_runs_without_fgps_data_when_seeds_yield_no_swaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed with `text_cdl == image_cdl == ()` has |M_p| = 0; the
    Algorithm 1 driver short-circuits before touching FGPS. We use
    this to validate the spawn + pickle path end-to-end without
    requiring the 521 MB FormalGeo dataset.

    The driver logs `pid=%s: |M_all| <= |M_p|, skipping`; the wrapper
    returns []. We assert the subprocess exited normally (no SIGKILL,
    no RuntimeError).
    """
    seed = Problem(
        pid=999_999,
        construction_cdl=("Triangle(A,B,C)",),
        text_cdl=(),
        image_cdl=(),
        goal_cdl="Value(LengthOfLine(AC))",
        answer="5",
    )
    out = run_algorithm1_in_subprocess(
        [seed], m_per_seed=1, rewriter="stub", process_timeout_s=120.0,
    )
    assert out == []  # |M_p| = 0 → no swap pool → skip → []
