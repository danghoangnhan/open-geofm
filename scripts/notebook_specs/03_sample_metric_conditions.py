"""Spec for `notebooks/03_sample_metric_conditions.ipynb`.

The headline of the paper, executable. Take PID=200, compute M_all via BFS,
run Algorithm 1's swap once to inspect the components, then drive the full
`run_algorithm1` with `m_per_seed=3` to show end-to-end behaviour.

CPU-only. Run from the repo root. ~30-90 seconds wall-clock (FGPS solves
dominate the run-time).
"""

from __future__ import annotations

TITLE = "03 — Condition Sampling (Algorithm 1)"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 03 — Condition Sampling (Algorithm 1)

> **Run-time:** ~30–90 seconds on CPU (FGPS solves dominate).
> **Prerequisites:** Notebook 01 (loads the dataset). Same `uv sync
> --extra formal --extra dev`.

This notebook executes the **novel part of the paper**: Algorithm 1, which
takes a seed problem and swaps `n` of its metric conditions for `n` new
ones that the symbolic engine can derive from the same figure. Goal stays
solvable, figure stays valid, but the problem is *new*.

We use PID=200 (the congruent-triangles problem from notebook 01) as the
seed, then walk through one swap iteration, then drive the full
`run_algorithm1` for `m_per_seed=3` accepted samples.

Long-form: [wiki/02-Condition-Sampling.md](../wiki/02-Condition-Sampling.md).
""",
    ),
    (
        "markdown",
        """\
## 1. Load the seed problem

PID=200 is the integration-tested fast-BFS seed (an `Value(x)` algebraic
problem with `answer = sqrt(161)`). Its eight metrics across text + image
expand into 30+ derived metrics in a single round of theorem application
— plenty of swap pool for Algorithm 1.
""",
    ),
    (
        "code",
        """\
import warnings
warnings.filterwarnings('ignore')   # silence FGPS's EE-check UserWarnings

from open_geofm.formal.loader import load_problem

p = load_problem(200)
print(f"pid={p.pid}, goal_cdl={p.goal_cdl}, answer={p.answer}")
print(f"|M_p| = |text_cdl| + |image_cdl| = {len(p.text_cdl)} + {len(p.image_cdl)} = {len(p.all_metric_conditions)}")
""",
    ),
    (
        "markdown",
        """\
## 2. Compute `M_all` via BFS over the theorem library

`gather_metric_info(problem, max_depth=1)` runs one round of every theorem
in the GDL via `formalgeo.solver.interactive.Interactor.apply_theorem_by_name`,
snapshots the resulting metric set, and canonicalises to `Equal(...)` form
so the Algorithm 1 swap can do set arithmetic.

Defaults: `max_depth=1, timeout_s=30`. Higher depths multiply theorem-application
time without proportionally increasing accepted-sample yield, so v1 ships
single-round BFS. We bump the timeout to 90 s here so the notebook is
deterministic across machines — the production driver leaves the 30 s default
and just drops the few seeds that don't finish in time.
""",
    ),
    (
        "code",
        """\
from open_geofm.sampling.gather_metrics import gather_metric_info

m_all = gather_metric_info(p, max_depth=1, timeout_s=90.0)
print(f"|M_all| = {len(m_all)}")
print(f"|M_all \\ M_p| = {len(set(m_all) - set(p.all_metric_conditions))}  (the swap-in pool)")
""",
    ),
    (
        "markdown",
        """\
For PID=200 we expect `|M_all|` ≈ 15–30 — the theorem-application BFS expands
the congruence into per-vertex angle / side equalities, derived sums and
ratios, etc. The exact count varies with the GDL version.

Let's peek at the new pool (`M_all \\ M_p`) — these are the facts that the
*figure* implies that the textbook chose not to state explicitly. They're
the swap-in candidates for Algorithm 1.
""",
    ),
    (
        "code",
        """\
swap_pool = sorted(set(m_all) - set(p.all_metric_conditions))
for s in swap_pool[:8]:
    print(f"  {s}")
print(f"...({len(swap_pool)} total)")
""",
    ),
    (
        "markdown",
        """\
## 3. One iteration of the swap (`sample_new_problem`)

`sample_new_problem(p, m_all, rng)` is the pure-RNG kernel of Algorithm 1:

```text
n        = Random(1, min(|M_p|, |M_all| − |M_p|))
M_del    = RandomSelect(M_p, n)
M_add    = RandomSelect(M_all \\ M_p, n)
P_new    = (P \\ M_del) ∪ M_add
```

The seven set-theoretic invariants asserted by
`tests/test_algorithm1_invariants.py` must hold — size-preserving swap,
disjoint M_del / M_add, etc.
""",
    ),
    (
        "code",
        """\
import random
from open_geofm.sampling.algorithm1 import sample_new_problem

rng = random.Random(42)
p_new, m_del, m_add = sample_new_problem(p, m_all, rng)

print(f"n = |M_del| = |M_add| = {len(m_del)}")
print()
print("M_del (removed from M_p):")
for s in m_del:
    print(f"  - {s}")
print()
print("M_add (drawn from M_all \\ M_p):")
for s in m_add:
    print(f"  + {s}")
print()
print(f"|P_new| = {len(p_new)} (must equal |M_p| = {len(p.all_metric_conditions)})")
""",
    ),
    (
        "markdown",
        """\
## 4. Pick a new goal and verify with FGPS

The new goal comes from `M_all \\ P_new`. If the random pick yields an
unsolvable goal, Algorithm 1's driver falls back to `fallback_goal_from_trace`
— it pulls the last derived metric from FGPS's trace. We're not doing that
fallback here for clarity; in production the `run_algorithm1` driver wires
it automatically.
""",
    ),
    (
        "code",
        """\
from dataclasses import replace
from open_geofm.formal.solver import solve
from open_geofm.sampling.gather_metrics import goal_metric_for, value_of
from open_geofm.sampling.goal_picker import pick_goal

# `pick_goal` can return a *non-value-bearing* metric (e.g. a parallel-line
# relation). Algorithm 1 wants a numeric goal, so we restrict the pool to
# value-bearing metrics here. The production driver `run_algorithm1` falls
# back to `fallback_goal_from_trace` when the picked goal isn't usable.
value_pool = tuple(m for m in m_all if value_of(m) is not None)
goal_metric = pick_goal(p_new, value_pool, seed=42)
goal_cdl = goal_metric_for(goal_metric)
candidate_answer = value_of(goal_metric)
print(f"goal_metric    = {goal_metric}")
print(f"goal_cdl       = {goal_cdl}")
print(f"candidate_answer = {candidate_answer}")

# Build the candidate problem: same construction, new metric set, new goal.
# The Algorithm 1 driver also re-splits P_new into text_cdl / image_cdl — we
# elide that here for clarity (FGPS doesn't care which block a metric lives in).
candidate = replace(
    p,
    text_cdl=tuple(p_new),
    image_cdl=(),
    goal_cdl=goal_cdl,
    theorem_seqs=(),
    answer=candidate_answer,
)

result = solve(candidate, timeout_s=60.0)
print(f"\\nFGPS solved? {result.solved}  answer={result.answer!r}")
""",
    ),
    (
        "markdown",
        """\
If `solved=True` we have a verified synthetic sample. If `solved=False`
the driver retries with `fallback_goal_from_trace(result.theorem_seqs)` or
moves on to a different swap.

## 5. End-to-end with `run_algorithm1`

The driver wires everything: gather → sample → goal → solve → rewrite →
verify → append. Pass a stub `rewrite_fn` for now (Phase 5 supplies the
real NL template + LLM rewriter).
""",
    ),
    (
        "code",
        """\
from open_geofm.sampling.algorithm1 import run_algorithm1

def stub_rewrite(problem, fgps_answer: str):
    nl_problem = f"(synthetic) From the given conditions, find {problem.goal_cdl}."
    nl_solution = f"By FGPS derivation, the answer is {fgps_answer}."
    return nl_problem, nl_solution


# The driver's default `gather_fn` uses the production 30 s BFS timeout; we
# bump it to 90 s here for the same reason as in section 2 above (the
# notebook must run deterministically across machines).
samples = run_algorithm1(
    [p],
    m_per_seed=3,
    seed=42,
    rewrite_fn=stub_rewrite,
    gather_fn=lambda problem: gather_metric_info(problem, max_depth=1, timeout_s=90.0),
    solve_fn=lambda problem: solve(problem, timeout_s=60.0),
    max_attempts_factor=10,
)
print(f"accepted samples: {len(samples)}")
for i, s in enumerate(samples, 1):
    print(f"  [{i}] goal={s.goal!r}  answer={s.answer!r}  +{len(s.added_metrics)}/-{len(s.deleted_metrics)} swap")
""",
    ),
    (
        "markdown",
        """\
## What we just did

* Loaded a seed (PID=200, |M_p|=8).
* Computed `M_all` via 1-round BFS over the 234-theorem GDL.
* Ran one swap iteration and inspected the components.
* Picked a goal from `M_all \\ P_new` and confirmed FGPS verifies it.
* Drove `run_algorithm1` for `m_per_seed=3` end-to-end (with a stub
  rewriter; Phase 5 supplies the real NL pipeline).

## What's next

* **Notebook 04 — Render Diagram (matplotlib):** turn the candidate's
  construction + metric set into a PNG.
* **Phase 5:** plug `open_geofm.nlg.templates.draft_nl` in as the
  `rewrite_fn` instead of the stub above.
* **`scripts/02_generate_dataset.py`:** the production driver that does
  all of the above across all 7000 seeds with a multiprocessing pool.
""",
    ),
]
