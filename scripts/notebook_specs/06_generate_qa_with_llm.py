"""Spec for `notebooks/06_generate_qa_with_llm.ipynb`.

Phase 5 NLG: predicate templates -> draft NL -> (optional) LLM smooth
-> Phase 3 answer-verify with sympy. CPU-only (no real LLM call); a
stub rewriter stands in for the local Qwen2.5-7B-Instruct / OpenAI
gpt-4o-mini backends.

Run from the repo root. ~30 seconds wall-clock (one Algorithm 1 driver
pass dominates).
"""

from __future__ import annotations

TITLE = "06 — NL Templates + LLM Rewriter"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 06 — NL Templates + LLM Rewriter

> **Run-time:** ~30 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal --extra dev`. *No GPU
> needed* — the notebook uses a stub rewriter; the real local /
> OpenAI backends are wired by the production driver
> (`scripts/02_generate_dataset.py --rewriter llm`).

Phase 5 of the paper (the two-step NLG) is::

    step 1  draft = templates.draft_nl(metric_conditions, goal)
    step 2  prose = LLM.rewrite(draft + answer_hint)

The templates carry the predicate semantics; the LLM smooths them
into readable prose. This notebook walks through both steps, then
plugs the result back into Algorithm 1 with the Phase-3 sympy verifier
in the loop.

Long-form: blueprint §2 Phase 5; rewriter prompt = paper Appendix C
verbatim.
""",
    ),
    (
        "markdown",
        """\
## 1. Step 1 — Predicate templates

`open_geofm.nlg.templates.TEMPLATES` is a `dict[predicate -> list[str]]`
of 3-6 English variants per predicate. The rewriter picks one at
random per condition; slot names `{a}, {b}, {c}, {d}, {value}` match
the order points appear in the CDL arg list.

`GOAL_TEMPLATES` is the parallel dict for the *question* form
(`Value(LengthOfLine(AB))` becomes “Find the length of AB.”).
""",
    ),
    (
        "code",
        """\
from open_geofm.nlg.templates import GOAL_TEMPLATES, TEMPLATES

print('---TEMPLATES (sample)---')
for pred in ('Triangle', 'PerpendicularBetweenLine', 'LengthOfLine', 'MeasureOfAngle'):
    print(f'  {pred}:')
    for tmpl in TEMPLATES[pred]:
        print(f'    {tmpl}')

print('\\n---GOAL_TEMPLATES---')
for pred, tmpls in GOAL_TEMPLATES.items():
    print(f'  {pred}:')
    for tmpl in tmpls:
        print(f'    {tmpl}')
""",
    ),
    (
        "markdown",
        """\
## 2. `draft_nl(...)` end-to-end

`draft_nl(metric_conditions, goal, *, seed=None)` stitches one sentence
per condition, then the goal sentence. Unknown predicates pass through
verbatim — the rewriter cleans them up downstream.

Let's use the toy 3-4-5 right triangle from
`tests/conftest.py::toy_problem` so the input is fully transparent.
""",
    ),
    (
        "code",
        """\
from open_geofm.nlg.templates import draft_nl

text_cdl = ('Equal(LengthOfLine(AB),3)', 'Equal(LengthOfLine(BC),4)')
image_cdl = ('PerpendicularBetweenLine(AB,BC)',)
construction_cdl = ('Triangle(A,B,C)',)
goal_cdl = 'Value(LengthOfLine(AC))'

draft = draft_nl(construction_cdl + text_cdl + image_cdl, goal_cdl, seed=0)
print(draft)
""",
    ),
    (
        "markdown",
        """\
Re-running with a different `seed=` picks different template variants:
""",
    ),
    (
        "code",
        """\
for s in range(3):
    print(f'seed={s}: ', draft_nl(construction_cdl + text_cdl + image_cdl, goal_cdl, seed=s))
""",
    ),
    (
        "markdown",
        """\
## 3. Step 2 — LLM rewrite (two backends)

The paper rewrites with Qwen2.5-72B-Instruct. We can't run 72B on a
single 5090, so `open_geofm.nlg.backend.get_rewriter()` selects between
two practical alternatives via the `OPEN_GEOFM_REWRITER` env var:

| `OPEN_GEOFM_REWRITER=` | Backend | Cost | VRAM | Throughput |
|---|---|---|---|---|
| `local` (default) | vLLM-served Qwen2.5-7B-Instruct fp16 | free | ~16 GB | ~100 samples/min |
| `openai` | gpt-4o-mini via the OpenAI API | ~$3 / 10K | 0 | API-limited |

The system prompt is the paper's Appendix C verbatim
(`open_geofm.nlg.rewrite_openai.SYSTEM_PROMPT`):

> *Given a geometry problem and its answer hint, write a answer to the
> problem. Ensure the answer is correct, concise, easy to understand,
> and written with clarity and natural flow.*

In this notebook we use a **stub rewriter** instead — it stitches
draft + answer into a deterministic prose answer. That's also what the
production driver does when called with `--rewriter template` (no LLM).
""",
    ),
    (
        "code",
        """\
def stub_rewriter(draft: str, answer: str) -> tuple[str, str]:
    nl_problem = draft
    nl_solution = (
        f'From the given conditions, the symbolic engine derives the answer. '
        f'Therefore, the answer is {answer}.'
    )
    return nl_problem, nl_solution


prob, sol = stub_rewriter(draft, '5')
print('--- nl_problem ---')
print(prob)
print('\\n--- nl_solution ---')
print(sol)
""",
    ),
    (
        "markdown",
        """\
## 4. Phase 3 — answer verification

`open_geofm.nlg.verify.verify(nl_answer, fgps_answer)` extracts the
last numeric token from the NL answer, sympifies both sides, and
accepts iff they agree to within `tol=1e-3`. This is the gate that
rejects a rewriter hallucination from ever entering the dataset.
""",
    ),
    (
        "code",
        """\
from open_geofm.nlg.verify import verify

cases = [
    ('Therefore the answer is 5.',     '5'),
    ('After computation, AC = 5.0',    '5'),
    ('I think AC is 6.',               '5'),     # wrong number → reject
    ('The hypotenuse is sqrt(25).',    '5'),     # sympify-friendly → accept
    ('cannot be determined',           '5'),     # no number → reject
    ('h = 2*sqrt(21) ≈ 9.165',         '2*sqrt(21)'),
]
for nl, fgps in cases:
    r = verify(nl, fgps)
    print(f'  accepted={r.accepted}  extracted={r.extracted_answer!r:14s}  fgps={fgps!r:14s}  reason={r.reason!r}')
""",
    ),
    (
        "markdown",
        """\
## 5. Algorithm 1 + the verify-reject loop

`run_algorithm1(..., rewrite_fn=..., verify_fn=...)` wires everything:
gather → swap → goal → solve → rewrite → verify → append. Each rejected
swap costs one attempt; `max_attempts_factor` bounds wall-clock per
seed.

Here we drive it with the stub rewriter (returns the right answer
every time, so all accepted) and the real `nlg.verify` so we can
inspect the trace of acceptance.
""",
    ),
    (
        "code",
        """\
import warnings
warnings.filterwarnings('ignore')   # silence FGPS's EE-check UserWarnings

from open_geofm.formal.loader import load_problem
from open_geofm.formal.solver import solve as fgps_solve
from open_geofm.nlg.verify import verify as nlg_verify
from open_geofm.sampling.algorithm1 import run_algorithm1
from open_geofm.sampling.gather_metrics import gather_metric_info


def rewrite_fn(problem, fgps_answer: str) -> tuple[str, str]:
    draft = draft_nl(problem.text_cdl + problem.image_cdl, problem.goal_cdl)
    return stub_rewriter(draft, fgps_answer)


seed = load_problem(200)   # PID=200 (Value(x), answer=sqrt(161)), |M_p|=8, fast BFS
samples = run_algorithm1(
    [seed],
    m_per_seed=2,
    seed=42,
    rewrite_fn=rewrite_fn,
    verify_fn=lambda nl, exp: nlg_verify(nl, exp).accepted,
    gather_fn=lambda problem: gather_metric_info(problem, max_depth=1, timeout_s=90.0),
    solve_fn=lambda problem: fgps_solve(problem, timeout_s=60.0),
    max_attempts_factor=20,
)
print(f'accepted = {len(samples)}')
for i, s in enumerate(samples, 1):
    print()
    print(f'--- sample {i} ---')
    print(f'goal   = {s.goal!r}')
    print(f'answer = {s.answer!r}')
    print(f'nl_problem  : {s.nl_problem}')
    print(f'nl_solution : {s.nl_solution}')
""",
    ),
    (
        "markdown",
        """\
## 6. Where the *real* LLM plugs in

The production driver (`scripts/02_generate_dataset.py --rewriter llm`)
swaps `stub_rewriter` for `nlg.backend.get_rewriter().rewrite(prompt)`.
The prompt template (from `02_generate_dataset.py::_make_rewriter`) is::

    Geometry problem (draft):
    <draft from templates.draft_nl>

    Hint: the correct answer is <FGPS answer>.

    Rewrite the problem and provide a clear, concise solution that
    reaches the answer.

After the LLM responds, the driver splits on the first blank line —
head becomes `nl_problem`, tail becomes `nl_solution`. The Phase-3
verify call then accepts or rejects.

## What we just did

* Walked the predicate `TEMPLATES` + `GOAL_TEMPLATES` dicts.
* Drafted NL from a metric-condition tuple with `draft_nl`.
* Showed both rewriter backends (local vLLM Qwen2.5-7B-Instruct vs
  OpenAI gpt-4o-mini) and their cost / VRAM trade-off.
* Exercised the Phase-3 sympy verifier on a mix of accept and reject
  inputs.
* Plugged a stub rewriter into the Algorithm 1 driver end-to-end.

## What's next

* **Notebook 07 — Build the HF dataset:** zip accepted samples +
  rendered PNGs into a `datasets.Dataset`.
* **Notebook 08 — LoRA training dry-run:** what the SFTTrainer config
  looks like.
* **Production driver:** `OPEN_GEOFM_REWRITER=local uv run python
  scripts/02_generate_dataset.py --n 5000 --rewriter llm`.
""",
    ),
]
