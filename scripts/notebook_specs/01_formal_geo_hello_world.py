"""Spec for `notebooks/01_formal_geo_hello_world.ipynb`.

Pedagogical entry-point: load FormalGeo7K v2 PID=4 (a short
angle-chase problem with `answer=68`) via our wrapper, inspect its
four CDL blocks, then run FGPS to verify the dataset's recorded
answer. Mirrors wiki/01-Formal-Language.md but in executable form.

CPU-only. Run from the repo root. ~15 seconds wall-clock.
"""

from __future__ import annotations

TITLE = "01 — FormalGeo Hello World"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 01 — FormalGeo Hello World

> **Run-time:** ~30 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal` (installs the `formalgeo`
> PyPI package) and the FormalGeo7K v2 data unpacked under
> `data/formalgeo7k_v2/` (run `bash scripts/01_download_formalgeo7k.sh`
> once if you haven't).

The goal of this notebook is to get a *Conditional Declaration Language*
problem from the dataset, look at its four CDL blocks, and run the FGPS
symbolic solver to **verify** the dataset's recorded answer.

This is the foundation every other phase builds on — the rest of the
pipeline (Algorithm 1, the renderer, the NL templater, the trainer) reads
and emits CDL strings.

The metaphor: *FormalGeo is to geometry what a typed AST is to a programming
language.* CDL is the AST; FGPS is the type-checker / evaluator.
""",
    ),
    (
        "markdown",
        """\
## 1. Load problem PID=4

`open_geofm.formal.loader.load_problem` wraps the `formalgeo` `DatasetLoader`
and returns a frozen, slot-bearing `Problem` dataclass with five fields.
We use PID=4 — a textbook angle-chase whose dataset answer is `68` — as the
hello-world seed because FGPS solves it in a single theorem step. (PID=1
is the dataset's first problem but its backward-search runs into a slow
spot in FGPS; notebook 02 uses PID=10 for a richer 2-step solve.)

| Field | Type | What it is |
|---|---|---|
| `pid` | `int` | The FormalGeo7K problem id (1..7000) |
| `construction_cdl` | `tuple[str, ...]` | How the figure is built (lines, shapes) |
| `text_cdl` | `tuple[str, ...]` | Conditions stated in the **problem text** |
| `image_cdl` | `tuple[str, ...]` | Conditions only readable from the **diagram** |
| `goal_cdl` | `str` | The metric the solver must find |
| `theorem_seqs` | `tuple[str, ...]` | The proof trace (when available) |
| `answer` | `str \\| None` | The verified answer (a string, possibly numeric) |
""",
    ),
    (
        "code",
        """\
from open_geofm.formal.loader import load_problem

p = load_problem(4)
print(f"pid={p.pid}")
print(f"goal_cdl={p.goal_cdl!r}")
print(f"answer={p.answer!r}")
""",
    ),
    (
        "markdown",
        """\
## 2. Inspect the four CDL blocks

Each block answers a different question. Algorithm 1 only swaps the
*metric* conditions (`text_cdl` and `image_cdl`); construction is held
fixed because changing it would change the figure.
""",
    ),
    (
        "code",
        """\
def show(label: str, lines):
    print(f"--- {label} ({len(lines)} statements) ---")
    for s in lines:
        print(f"  {s}")
    print()

show("construction_cdl", p.construction_cdl)
show("text_cdl", p.text_cdl)
show("image_cdl", p.image_cdl)
print(f"goal_cdl: {p.goal_cdl}")
print(f"theorem_seqs: {list(p.theorem_seqs)}")
""",
    ),
    (
        "markdown",
        """\
## 3. `M_p` — the metric pool Algorithm 1 will work on

The convenience property `all_metric_conditions` returns `text_cdl + image_cdl`.
That's the `M_p` set in the paper's Algorithm 1: the conditions Phase 2 can
delete from, paired with the BFS-derived `M_all` it can add from.

Note the *duplication*: `image_cdl` is often a subset of `text_cdl` because
textbook problems print the same value both in the prose and on the diagram.
That's expected; the dataset builder dedupes downstream.
""",
    ),
    (
        "code",
        """\
mp = p.all_metric_conditions
print(f"|M_p| = {len(mp)} (text {len(p.text_cdl)} + image {len(p.image_cdl)})")
for m in mp:
    print(f"  {m}")
""",
    ),
    (
        "markdown",
        """\
## 4. Run FGPS to verify the dataset's answer

`open_geofm.formal.solver.solve` wraps the BitSecret/FGPS searcher with a
15-second timeout. FGPS is an **answer-verifier**, not an answer-finder: it
takes a candidate answer (the dataset's `problem_answer`) and proves that
`goal.item == candidate_answer` using the GDL theorem library.

For seed problems we don't have to pass anything — the wrapper falls back
to `problem.answer`. For *synthetic* problems (Algorithm 1, Phase 2) the
driver passes a candidate derived from `gather_metric_info`'s BFS.

This call typically takes 1–5 seconds for an easy seed like PID=4.
""",
    ),
    (
        "code",
        """\
from open_geofm.formal.solver import solve

result = solve(p)
print(f"solved        = {result.solved}")
print(f"answer        = {result.answer!r}")
print(f"theorem_seqs  = {list(result.theorem_seqs)}")
print(f"timed_out     = {result.timed_out}")
""",
    ),
    (
        "markdown",
        """\
## 5. Sanity check

The dataset says `answer=68` (the measure of angle YZW in degrees). FGPS
should agree.
""",
    ),
    (
        "code",
        """\
assert result.solved, "FGPS failed to verify PID=4 — something is wrong with the data or the GDL."
assert result.answer == p.answer, (
    f"FGPS reported {result.answer!r} but the dataset has {p.answer!r}; "
    "this usually means the GDL has been updated and the dataset hasn't been re-derived."
)
print("✅ PID=4 verifies. ∠YZW =", result.answer, "°")
""",
    ),
    (
        "markdown",
        """\
## What's next

* **Notebook 02 — Parse CDL & Solve:** deeper FGPS walkthrough (PID=10,
  a 2-step circle-and-tangent proof).
* **Notebook 03 — Sample Metric Conditions:** the headline of the paper.
  Take a seed, run Algorithm 1's swap, get a *new* (verified) problem.
* **Notebook 04 — Render Diagram (matplotlib):** turn the construction +
  image_cdl into a 224-pixel PNG.
* **Wiki page 02 — Condition Sampling:** the long-form companion to
  notebook 03.

If you got a `ModuleNotFoundError: formalgeo`, run `uv sync --extra formal`.
If the FGPS call timed out, the wrapper returns `SolverResult(timed_out=True)`
and the rest of the pipeline carries on — that's the design.
""",
    ),
]
