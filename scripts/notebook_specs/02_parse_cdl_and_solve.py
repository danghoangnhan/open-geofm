"""Spec for `notebooks/02_parse_cdl_and_solve.ipynb`.

Deeper Phase-1 walkthrough than notebook 01: parse a raw CDL block with
`parse_cdl`, then drive the FGPS backward searcher on PID=10 (a
circle-and-tangent problem with `Value(x)` algebraic goal and a 2-step
theorem trace).

CPU-only. Run from the repo root. ~30 seconds wall-clock (FGPS
backward-search dominates).
"""

from __future__ import annotations

TITLE = "02 — Parse CDL & Solve"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 02 — Parse CDL & Solve

> **Run-time:** ~30 seconds on CPU.
> **Prerequisites:** Notebook 01 (loads the dataset). Same
> `uv sync --extra formal --extra dev`.

Notebook 01 loaded a problem and called the solver. This notebook goes
one level deeper:

1. Parse a raw CDL block (as a textbook author might paste it) with
   `open_geofm.formal.cdl.parse_cdl`.
2. Load a harder FormalGeo7K problem (PID=10 — a circle-and-tangent
   problem whose goal is the algebraic unknown `x = 5`) and dissect its
   four CDL blocks.
3. Run FGPS's backward searcher and walk through the theorem trace.

Long-form: [wiki/01-Formal-Language.md](../wiki/01-Formal-Language.md).
""",
    ),
    (
        "markdown",
        """\
## 1. `parse_cdl` — ingest a textbook block

`parse_cdl(raw: str) -> tuple[str, ...]` is the input adapter. It
accepts a multi-line string, strips `#`-to-EOL comments and blank
lines, and returns one statement per element.

This is the function notebooks paste into when demoing problems by
hand; the FormalGeo7K dataset loader uses its own list-based shape
(see `loader._problem_from_raw`).
""",
    ),
    (
        "code",
        """\
from open_geofm.formal.cdl import parse_cdl

raw_construction = '''
# Construction block for a right triangle on legs 3 and 4.
Triangle(A,B,C)
'''

raw_text = '''
Equal(LengthOfLine(AB),3)   # AB = 3
Equal(LengthOfLine(BC),4)   # BC = 4
'''

raw_image = '''
PerpendicularBetweenLine(AB,BC)   # the right-angle tick on the diagram
'''

print('construction_cdl ->', parse_cdl(raw_construction))
print('text_cdl         ->', parse_cdl(raw_text))
print('image_cdl        ->', parse_cdl(raw_image))
""",
    ),
    (
        "markdown",
        """\
The four blocks together form a `Problem` dataclass. The trio above
plus `goal_cdl='Value(LengthOfLine(AC))'` and `answer='5'` is the
3-4-5 right triangle used as the unit-test fixture
(`tests/conftest.py::toy_problem`).

## 2. A harder real problem: PID=10

The hello-world (PID=1) is a one-step congruence. PID=10 is a
**circle-and-tangent** problem: a tangent line from outside the
circle, an arc, and an angle that depends on the unknown `x`. The
goal is `Value(x)` — i.e. the dataset asks for the algebraic value
of `x` itself, not the length of a segment. The answer is `5`.
""",
    ),
    (
        "code",
        """\
from open_geofm.formal.loader import load_problem

p = load_problem(10)
print(f'pid={p.pid}')
print(f'goal_cdl = {p.goal_cdl!r}   # find the value of `x`')
print(f'answer   = {p.answer!r}')
print(f'|construction_cdl| = {len(p.construction_cdl)}, |text_cdl| = {len(p.text_cdl)}, |image_cdl| = {len(p.image_cdl)}')
print(f'|theorem_seqs|     = {len(p.theorem_seqs)}   # what the dataset records as the proof')
""",
    ),
    (
        "markdown",
        """\
## 3. The four CDL blocks side-by-side

The four blocks answer different questions:

| Block | Question it answers | Example statement |
|---|---|---|
| `construction_cdl` | How is the figure built? | `Shape(EDB,BF,EFD)` |
| `text_cdl` | What does the *text* claim? | `IsTangentOfCircle(CD,E)` |
| `image_cdl` | What's visible *only on the diagram*? | `Equal(MeasureOfArc(EFD),40)` |
| `goal_cdl` | What metric do we want? | `Value(x)` |

Algorithm 1 (Phase 2 / notebook 03) only swaps text + image — the
construction stays fixed because changing it would change the figure.
""",
    ),
    (
        "code",
        """\
def show(label, lines):
    if isinstance(lines, str):
        print(f'{label:18s} -> {lines!r}')
        return
    print(f'{label:18s} ({len(lines)} statements)')
    for s in lines:
        print(f'    {s}')

show('construction_cdl', p.construction_cdl)
show('text_cdl', p.text_cdl)
show('image_cdl', p.image_cdl)
show('goal_cdl', p.goal_cdl)
""",
    ),
    (
        "markdown",
        """\
Note how `text_cdl ⊃ image_cdl` here: every numeric label visible on
the diagram is also restated in the text. FormalGeo7K mirrors the
textbook convention of printing the same value twice. Algorithm 1's
`_split_text_image` re-partitions metrics by membership in the seed's
original blocks — `new_metrics` from the swap default to `text_cdl`.

## 4. Run FGPS (backward search) and inspect the result

`open_geofm.formal.solver.solve(problem)` defaults to the **backward**
strategy — start from the goal, search the GDL for theorems whose
conclusions match, and recurse. The alternative is **forward**, which
applies every theorem from the premises until a fixed point is reached
(that's the engine `gather_metric_info` uses to enumerate `M_all` in
notebook 03).

A 15-second timeout guards against pathological seeds; PID=10 returns
in under a second.
""",
    ),
    (
        "code",
        """\
import warnings
warnings.filterwarnings('ignore')   # FGPS emits noisy EE-check UserWarnings on some seeds

from open_geofm.formal.solver import solve

result = solve(p, strategy='backward', timeout_s=60.0)
print(f'solved      = {result.solved}')
print(f'answer      = {result.answer!r}')
print(f'timed_out   = {result.timed_out}')
print(f'#trace steps = {len(result.theorem_seqs)}')
""",
    ),
    (
        "markdown",
        """\
## 5. Pretty-print the theorem sequence

FGPS's `BackwardSearcher` emits each step as the stringified tuple
`(theorem_name, premise_id, (args...))`. The first element is the
theorem from the 234-theorem GDL; the third is the binding of theorem
variables to figure points. Our solver wrapper stringifies each step
before returning so the rest of the pipeline never depends on the
internal FGPS types.

The same trace is what `goal_picker.fallback_goal_from_trace` (Phase
2) reads when Algorithm 1's randomly-picked goal turns out unsolvable.
""",
    ),
    (
        "code",
        """\
import ast

def pretty_step(step: str) -> str:
    # The wrapper stores each step as `str(tuple)`; literal_eval is the safe
    # way to recover the structure for display. Falls back to the raw string
    # if the format changes in a future FGPS release.
    try:
        name, premise, args = ast.literal_eval(step)
    except (SyntaxError, ValueError):
        return step
    return f'theorem={name!r:55s} premise_id={premise}  args={args}'

for i, step in enumerate(result.theorem_seqs, 1):
    print(f'step {i}: {pretty_step(step)}')
""",
    ),
    (
        "markdown",
        """\
## 6. Sanity check

The dataset records `problem_answer = '5'`. FGPS should agree exactly
— `x = 5` makes the tangent-secant identity balance.
""",
    ),
    (
        "code",
        """\
assert result.solved, 'FGPS failed on PID=10 — see wiki/07 Blackwell setup log if running inside Docker.'
assert result.answer == p.answer, f'FGPS={result.answer!r}, dataset={p.answer!r} — GDL/dataset out of sync.'
print('✅ PID=10 verifies. x =', result.answer)
""",
    ),
    (
        "markdown",
        """\
## What we just did

* Parsed a raw CDL block with `parse_cdl`, the input adapter notebooks
  reach for when demoing by hand.
* Loaded a non-trivial seed (PID=10, a tangent / arc problem with
  `goal_cdl = Value(x)` and `answer = 5`) and inspected each of its
  four CDL blocks.
* Drove the FGPS **backward** searcher to the verified answer.
* Pretty-printed the theorem trace — the same format
  `fallback_goal_from_trace` (Phase 2) consumes.

## What's next

* **Notebook 03 — Condition Sampling:** plug this seed into Algorithm
  1's `gather_metric_info` BFS + swap.
* **Notebook 04 — Render Diagram:** turn the construction + image_cdl
  into a 224-px PNG.
* **Wiki page 03 — Symbolic Verification:** how Phase 3 uses FGPS to
  reject hallucinated NL answers.
""",
    ),
]
