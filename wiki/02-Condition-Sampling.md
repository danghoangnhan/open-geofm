# 02 — Condition Sampling (Algorithm 1)

> **The novel part of the paper, and the educational headline of this repo.**
> Everything upstream (Phase 1 FormalGeo) is plumbing; everything downstream
> (Phases 3–8) is commodity ML. Phase 2 is where 7,000 seed problems become
> 5K–20K verified synthetic ones — without an LLM hallucinating answers.

## What Algorithm 1 actually does, in one sentence

For each seed problem, randomly **swap n of its metric conditions for n new
ones that the symbolic engine can derive from the same figure**, pick a new
goal from what's left over, then re-verify with FGPS. Repeat until you have
`m` accepted samples per seed.

That's it. The cleverness is that the *figure stays fixed* (so you don't have
to re-render and the construction_cdl stays valid), the *swap is
size-preserving* (so problem difficulty roughly tracks the seed's), and the
*goal is always derivable* (because both `M_p` and `M_all \ M_p` were produced
by the same theorem-BFS).

## Algorithm 1, verbatim from the paper

```text
Input:  formalised seed problem set FS, target number of synthetic problems m
Output: set S of (P_syn, A_syn) NL pairs

for P in FS:
    M_p   = MetricInfoOfProblemStatement(P)        # text_cdl ∪ image_cdl
    M_all = GatheringMetricInfo(P)                  # BFS over 196 theorems
    m_p = m
    while m_p > 1:
        n      = Random(1, min(|M_p|, |M_all| − |M_p|))
        M_del  = RandomSelect(M_p, n)
        M_add  = RandomSelect(M_all \ M_p, n)
        P_new  = (P \ M_del) ∪ M_add
        A_new  = FormalGeoSolver(P_new)            # verifier, not finder
        P_syn, A_syn = Template_and_LLM(P_new, A_new)
        if AnswerVerify(A_syn, A_new):
            S.add((P_syn, A_syn)); m_p −= 1
return S
```

Implementation: [`src/open_geofm/sampling/algorithm1.py`](../src/open_geofm/sampling/algorithm1.py).
The driver `run_algorithm1(...)` accepts callable hooks for `solve_fn`,
`gather_fn`, `verify_fn`, `rewrite_fn` so each phase can be exercised in
isolation — that's how the 9 `test_algorithm1_invariants.py` tests run on CPU
with no FGPS at all.

## What `M_p` and `M_all` mean concretely

Take seed PID=1 from [01 — Formal Language](./01-Formal-Language) (congruent
triangles RST/XYZ, find `y`):

```
construction_cdl = (Shape(RS,ST,TR), Shape(XY,YZ,ZX))      # held fixed
text_cdl         = (CongruentBetweenTriangle(RST,XYZ),
                    Equal(LengthOfLine(TR), x+21),
                    Equal(LengthOfLine(ZX), 2*x-14),
                    Equal(MeasureOfAngle(TRS), 4*y-10),
                    Equal(MeasureOfAngle(ZXY), 3*y+5))
image_cdl        = (… 4 of the above 5, also on the diagram …)
```

- **`M_p = text_cdl + image_cdl`** — the *metric conditions* visible to the
  problem statement. `Problem.all_metric_conditions` returns this. For PID=1
  that's 5 + 4 = 9 strings (with duplicates kept — the duplication is real:
  values that appear in both text and diagram).
- **`M_all = gather_metric_info(problem)`** — every metric derivable from the
  *construction* by one round of theorem application. For a triangle-congruence
  seed this typically expands to 30–50 metrics: the picked angles equal their
  partners, the corresponding sides are equal, the perimeters are equal, the
  areas are equal, derived angle sums equal 180°, etc.
- **`M_all \ M_p`** is the "swap pool" — facts the engine *proved* from the
  figure but the textbook *didn't bother to state*. These are the new
  conditions a synthetic problem can hand the model.

Toy example (`tests/conftest.py::toy_m_all`) for a 3-4-5 right triangle:
```text
M_p:        Equal(LengthOfLine(AB),3), Equal(LengthOfLine(BC),4),
            PerpendicularBetweenLine(AB,BC)
M_all \ M_p: Equal(LengthOfLine(AC),5), Equal(MeasureOfAngle(ABC),90),
             Equal(MeasureOfAngle(BAC),53.13),
             Equal(MeasureOfAngle(BCA),36.87),
             Equal(AreaOfTriangle(A,B,C),6)
```
Picking `n=2` and swapping might delete `Equal(LengthOfLine(BC),4)` and
`PerpendicularBetweenLine(AB,BC)`, add `Equal(MeasureOfAngle(ABC),90)` and
`Equal(AreaOfTriangle(A,B,C),6)`, then pick `Equal(LengthOfLine(AC),5)` as the
goal — *new* problem (area-then-hypotenuse), *same* figure, *FGPS-verifiable*.

## Why the swap is size-preserving (a paper detail often missed)

The line `n = Random(1, min(|M_p|, |M_all| − |M_p|))` is the keystone. It
ensures:
1. **`|P_new| == |M_p|`** — the new problem has exactly as many conditions as
   the seed, so difficulty distribution is preserved. A 3-condition geometry
   seed never balloons into a 12-condition synthetic.
2. **`M_del ⊆ M_p`** and **`M_add ⊆ M_all \ M_p`** by construction.
3. **`M_del ∩ M_add == ∅`** — the swap is *strict*; you never delete-then-add
   the same fact.

These three invariants are tested in `tests/test_algorithm1_invariants.py`
across 20 RNG seeds (lines 26–47):

```python
@pytest.mark.parametrize("rng_seed", list(range(20)))
def test_swap_invariants(toy_problem, toy_m_all, rng_seed):
    rng = random.Random(rng_seed)
    p_new, m_del, m_add = sample_new_problem(toy_problem, toy_m_all, rng)
    # equal-size, bounded, subsets, disjoint, size-preserving, set-correct
    assert len(m_add) == len(m_del)
    assert 1 <= len(m_del) <= min(len(m_p), len(m_all) - len(m_p))
    assert set(m_del).issubset(m_p)
    assert set(m_add).issubset(set(m_all) - set(m_p))
    assert set(m_del).isdisjoint(m_add)
    assert len(p_new) == len(m_p)
    assert set(p_new) == (set(m_p) - set(m_del)) | set(m_add)
```

If you change `sample_new_problem`, these invariants must still hold — that's
the contract Algorithm 1 imposes.

## The text_cdl / image_cdl split — why it forces the model to *read* the diagram

After the swap, `P_new` is a flat list of metrics; FGPS doesn't care which
block they came from. But Qwen2-VL does: a condition in `text_cdl` is
*rendered as a sentence* in the NL prompt ("∠TRS = 4y − 10°"), while a
condition in `image_cdl` is *rendered onto the diagram* as a tick mark or
angle-arc label ("the angle marked ⌒ at vertex R").

`_split_text_image` (lines 89–104 of `algorithm1.py`) decides the routing:

```python
def _split_text_image(problem, p_new):
    text_set = set(problem.text_cdl)
    image_set = set(problem.image_cdl)
    text_out, image_out = [], []
    for m in p_new:
        if m in image_set:    image_out.append(m)
        elif m in text_set:   text_out.append(m)
        else:                 text_out.append(m)   # default: text
    return tuple(text_out), tuple(image_out)
```

A swapped-in metric (`m ∈ M_add`) appears in neither set, so it defaults to
`text_cdl`. A future ablation (Phase 9) randomises the split — when more
conditions land in `image_cdl`, the model has to *look* to solve. The paper
attributes a non-trivial chunk of the headline gain to this forcing function;
[09 — Ablations](./09-ablations) will quantify it on our 10K dataset.

## Goal picking and the trace-fallback

The paper says: *"randomly choose one metric condition different from the new
problem statement as the goal,"* and *"if unsolvable, we select the last
valid inference from the symbolic engine's reasoning path as the new goal."*
Both halves are in [`goal_picker.py`](../src/open_geofm/sampling/goal_picker.py):

```python
def pick_goal(new_problem_metrics, m_all, seed=None):
    candidates = tuple(m for m in m_all if m not in new_problem_metrics)
    if not candidates: raise ValueError(...)
    return random.Random(seed).choice(candidates)

def fallback_goal_from_trace(theorem_seqs):
    # FGPS step format: "<theorem_name>(<premises>) -> <metric>"
    last = theorem_seqs[-1]
    return last.rsplit("->", 1)[1].strip()
```

The fallback is non-cosmetic: when the BFS-derived `M_all` is generous, many
goal picks end up *too easy* for the swapped problem (the answer is already
implied). FGPS reports `solved=False` with a populated trace; the driver
extracts the last derived metric and retries. Tested at
`tests/test_algorithm1_invariants.py::test_run_algorithm1_falls_back_on_unsolvable_goal`.

## `gather_metric_info` — the BFS that yields `M_all`

Implementation: [`gather_metrics.py`](../src/open_geofm/sampling/gather_metrics.py).

```python
def gather_metric_info(problem, max_depth=1, timeout_s=30.0, ...) -> tuple[str, ...]:
    interactor = _interactor(root, dataset_name)     # cached FGPS Interactor
    interactor.load_problem(problem_cdl)
    for _ in range(max_depth):
        updated = False
        for t_name in theorem_names:                  # ~234 in v2 GDL
            if interactor.apply_theorem_by_name(t_name):
                updated = True
        if not updated: break
    snapshot = inverse_parse_logic_to_cdl(interactor.problem)
    return tuple(_to_equal_form(c) for c in snapshot ...)   # deduped
```

Two details that bit us early:
1. **Default `max_depth=1`.** One round applies every theorem once, which is
   enough for >90% of FormalGeo7K seeds to get `|M_all| > |M_p|`. Higher
   depths multiply theorem-application time without proportionally increasing
   accepted-sample yield.
2. **`_to_equal_form`.** FGPS emits derived metrics as
   `Value(LengthOfLine(AB),5)` but the seeds use `Equal(LengthOfLine(AB),5)`.
   Without normalisation, the set difference `M_all \ M_p` is wrong and the
   swap deletes from one form / adds the other → spurious duplicates. The
   helper rewrites every CDL string to the `Equal(...)` canonical form before
   the set ops.

> **Note on the "196 theorems".** The paper quotes 196 (FormalGeo7K v1's
> theorem count); the v2 GDL we use ships **~234** in
> `theorem_GDL.json`. Both work identically — the BFS just iterates over
> whatever theorem set the loader returns. Don't hard-code 196 anywhere.

## End-to-end flow inside `run_algorithm1`

```mermaid
flowchart TD
  S[for each seed P] --> G[gather_metric_info P<br/>→ M_all]
  G --> A{|M_all| > |M_p|?}
  A -- no --> Skip[skip seed]
  A -- yes --> W[while accepted &lt; m]
  W --> SP[sample_new_problem<br/>P_new, M_del, M_add]
  SP --> PG[pick_goal from<br/>M_all \ P_new]
  PG --> SPL[_split_text_image<br/>route to text vs image]
  SPL --> SV[solve_fn candidate]
  SV -- solved --> R[rewrite_fn NL]
  SV -- failed --> FB[fallback_goal_from_trace<br/>retry once]
  FB -- still failed --> Drop[drop, attempts++]
  FB -- solved --> R
  R --> V[verify_fn NL vs FGPS]
  V -- accept --> OUT[append SyntheticSample<br/>accepted++]
  V -- reject --> Drop2[drop, attempts++]
  Drop --> W
  Drop2 --> W
  OUT --> W
  W -- attempts > 5·m --> S
```

`max_attempts_factor=5` keeps a single bad seed from monopolising the worker:
if 5 × `m_per_seed` rejects accumulate, the driver moves to the next seed.

## Pitfalls

1. **`|M_all| ≤ |M_p|` (under-determined seed).** Some FormalGeo7K problems
   have so many premises (e.g. heavy-conditioned circle problems) that
   one-round BFS expands to fewer new metrics than `M_p` already contains.
   The driver logs and skips. Bumping `max_depth` to 2 helps these at the
   cost of ~3× wall-clock per seed.
2. **Solver hangs (FGPS `auto_run` on pathological seeds).** `func_timeout`
   wraps both `gather_metric_info` (30s default) and `solve` (15s default).
   On timeout you get an empty tuple / `SolverResult(timed_out=True)` — the
   driver continues. Expect ~5–10% timeout rate at default knobs.
3. **Goals that aren't value-bearing.** `pick_goal` may return e.g.
   `PerpendicularBetweenLine(AB,BC)` — a *relation*, not a numeric metric.
   `value_of(...) is None` for these; the driver `continue`s and re-rolls.
   Numeric goals dominate the dataset by design (matches the paper).
4. **Empty `M_p` after construction-only seeds.** `sample_new_problem` raises
   `ValueError("Cannot swap: |M_p| or |M_all \\ M_p| is empty.")`; tested at
   `test_swap_raises_when_pool_empty`.
5. **Rewriter hallucinations.** Even with a verified FGPS answer, the LLM
   rewriter can drop a condition or invent one. That's why
   [03 — Symbolic Verification](./03-Symbolic-Verification) is non-negotiable:
   the verifier re-extracts a numeric answer from the rewriter's NL solution
   and compares to `result.answer` before accepting.
6. **Determinism.** `run_algorithm1(seed=42)` is reproducible only if
   `solve_fn` / `gather_fn` are deterministic. Real FGPS is — same problem +
   same GDL → same trace. Stubbed test runs use `random.Random(seed)`
   exclusively.

## Throughput target

On a modern 8-core CPU, the per-sample wall-clock breaks down roughly as:

| Stage                      | Time / sample | Bottleneck                  |
|---------------------------|---------------|------------------------------|
| `gather_metric_info` (cached after first call) | ~10–50 ms | FGPS Interactor             |
| `sample_new_problem`       | <1 ms         | `random.sample`              |
| `pick_goal`                | <1 ms         | list comprehension           |
| `solve_fn`                 | ~50–500 ms    | FGPS BackwardSearcher        |
| `rewrite_fn` (local Qwen)  | ~50–200 ms    | vLLM batch on GPU            |
| `verify_fn` (sympy)        | ~5–20 ms      | sympy.nsimplify              |

Multiprocessing across 8 cores gives the blueprint-quoted ~5K samples/h. The
10K target dataset (`geofm-mini-10k`) thus generates in **~2 hours** on the
host venv, with the rewriter offloaded to a vLLM server on the GPU.

## Tests at a glance

```
tests/test_algorithm1_invariants.py  ── 27 tests, all green on CPU
  ├─ test_swap_invariants[0..19]               ← the 7 set-theoretic invariants
  ├─ test_swap_raises_when_pool_empty
  ├─ test_goal_picker_excludes_problem_statement
  ├─ test_goal_picker_raises_when_no_candidates
  ├─ test_fallback_goal_from_trace_extracts_rhs
  ├─ test_fallback_goal_from_trace_rejects_empty
  ├─ test_run_algorithm1_driver_smoke           ← end-to-end with stubs
  └─ test_run_algorithm1_falls_back_on_unsolvable_goal

tests/test_gather_metrics.py        ── 8 tests
  ├─ canonical form / value extraction         ← regex normalisation
  ├─ BFS termination / depth bounds            ← stops when no updates
  ├─ timeout returns empty                     ← graceful degradation
  ├─ real PID=200 end-to-end                   ← needs formalgeo7k_v2 data
  └─ test_algorithm1_end_to_end_real_seed      ← full stack, real FGPS
```

`test_algorithm1_end_to_end_real_seed` is the integration test you want when
modifying anything in `sampling/` — it actually loads PID=200, runs
`gather_metric_info`, runs the swap, runs FGPS to verify, and asserts at
least one synthetic sample drops out the back. ~10s wall-clock.

## Where this fits in the pipeline

```mermaid
flowchart LR
  L[loader.load_problem<br/>seed Problem] --> GM[gather_metric_info<br/>BFS → M_all]
  L --> A[sample_new_problem<br/>swap M_del ↔ M_add]
  GM --> A
  A --> GP[pick_goal from M_all\P_new]
  GP --> SP[_split_text_image<br/>route to text/image_cdl]
  SP --> S[formal.solver.solve<br/>verify candidate]
  S -- failed --> FB[fallback_goal_from_trace]
  S -- ok --> R[Phase 5 rewrite_fn<br/>NL templates + LLM]
  FB --> R
  R --> V[Phase 3 verify_fn<br/>nsimplify check]
  V --> O[SyntheticSample appended]
```

Once the `SyntheticSample` list is in hand, [Phase 6 — Dataset
Assembly](./05-Qwen2VL-Finetuning) bundles them as Qwen2-VL ChatML, the
renderer ([Phase 4](./04-Diagram-Rendering)) emits the PNG, and the training
loop ([Phase 7](./05-Qwen2VL-Finetuning)) consumes the result.

## Further reading

- GeoFM paper §2.3.2, "Algorithm 1: Formal-Language Condition Sampling".
- Source:
  [`algorithm1.py`](../src/open_geofm/sampling/algorithm1.py),
  [`gather_metrics.py`](../src/open_geofm/sampling/gather_metrics.py),
  [`goal_picker.py`](../src/open_geofm/sampling/goal_picker.py)
  — ~400 LOC total.
- Tests: `tests/test_algorithm1_invariants.py`,
  `tests/test_gather_metrics.py` — 35 tests, all green.
