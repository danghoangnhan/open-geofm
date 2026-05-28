"""Spec for `notebooks/09_run_vlmevalkit.ipynb`.

Phase 8 evaluation: regex answer extraction (the $0 fallback for the
MCQ-and-number subset of MathVista-GPS) + the `eval.compare` work-dir
walker that produces the headline delta table. No real VLMEvalKit
run — that requires the Docker image and a couple of GBs of weights.

CPU-only. Run from the repo root. ~3 seconds wall-clock.
"""

from __future__ import annotations

TITLE = "09 — Eval Extraction & Judging"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 09 — Eval Extraction & Judging

> **Run-time:** ~3 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal --extra dev`. The real
> VLMEvalKit run needs the Docker image and the LoRA adapter — that's
> the `bash scripts/04_eval.sh ...` command at the bottom of this
> notebook. The pieces shown here (extractor + work-dir parser) are
> the host-side bookkeeping around it.

Two host-runnable pieces:

1. `open_geofm.eval.extract_answer.extract(response)` — regex-based
   pull of an MCQ letter / clean number from a free-form model reply.
   This is the **$0 judge** for MathVista-GPS (no gpt-4o-mini calls).
2. `open_geofm.eval.compare` — walks a VLMEvalKit `--work-dir`,
   pivots `*_score.json` files into a Markdown table (with optional
   baseline-relative deltas).

Long-form: [wiki/06-Evaluation.md](../wiki/06-Evaluation.md).
""",
    ),
    (
        "markdown",
        """\
## 1. The regex extractor

`extract(response)` runs a two-rule pipeline:

* **MCQ letter** — `[A-E]` standalone, on the **last line** first
  (models tend to write \"...so the answer is C\"), then anywhere in
  the response. Case-insensitive. The trailing `(?=\\W|$)` is
  load-bearing: without it `[A-E]` matches the first letter of
  *answer* / *equilateral* / *five* and silently returns the wrong
  letter. `F` is deliberately excluded — MathVista-GPS is A-E only,
  and `F` shows up in real geometry text (`F = ma`, point label `F`).
* **Number fallback** — if no MCQ letter, return the **last** clean
  number (handles \"step 1: 4+5=9; …; answer is 2\" by returning `2`).

Returns `None` only when neither rule matches.
""",
    ),
    (
        "code",
        """\
from open_geofm.eval.extract_answer import extract

samples = [
    ('A',                                       'standalone letter'),
    ('(C)',                                     'parenthesised letter'),
    ('Answer: D',                               'MathVista-style answer prefix'),
    ('4 + 5 = 9, but the answer choice is (D)', 'MCQ wins over numbers'),
    ('Option A is X. Option B is Y. \\nanswer: C', 'last-line rule wins over earlier mentions'),
    ('The answer is 5',                         'clean number'),
    ('step 1: 4+5=9; …; answer is 2',           'last number wins'),
    ('y = -0.5',                                'negative decimal'),
    ('triangle is equilateral, no number',      'neither rule matches'),
    ('The answer is F.',                        'F excluded (not A-E)'),
]
for text, note in samples:
    print(f'  extract={extract(text)!r:6s}  note={note}')
    print(f'    input  ={text!r}')
""",
    ),
    (
        "markdown",
        """\
## 2. The work-dir layout

A VLMEvalKit run writes::

    <work_dir>/
        <model_name_1>/
            MathVista_MINI_<model_name_1>_score.json
            GeoQA_<model_name_1>_score.json
            ...
        <model_name_2>/
            ...

Each score file is a JSON object with one of these shapes:

* `{ \"Overall\": 35.0, \"GPS\": 38.0, ... }` — flat scalar metrics
  (the canonical MathVista shape; `Overall` wins).
* `{ \"per_task\": { \"task_a\": 10.0, ... } }` — averaged across leaves.
* `42.5` — a bare number.

`parse_score_json(path)` handles all three. Below we synthesise a
small work-dir in `tmp_path` so the rest of the notebook runs without
a real VLMEvalKit invocation.
""",
    ),
    (
        "code",
        """\
import json
import tempfile
from pathlib import Path

from open_geofm.eval.compare import scan_work_dir, to_delta_table, to_markdown_table

work = Path(tempfile.mkdtemp(prefix='openfm-nb09-'))

def write_score(model: str, benchmark: str, score: float, extras=None):
    payload = {'Overall': score}
    if extras:
        payload.update(extras)
    p = work / model / f'{benchmark}_{model}_score.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))

# Synthetic numbers in the ballpark of the blueprint's expected-outcome
# table (Qwen2-VL-2B base on MathVista-GPS ~20%, +12 pp after open-geofm).
write_score('qwen2vl_2b_base',  'MathVista_MINI', 20.0, extras={'GPS': 22.0})
write_score('qwen2vl_2b_base',  'GeoQA',          40.0)
write_score('qwen2vl_2b_geofm', 'MathVista_MINI', 32.0, extras={'GPS': 35.5})
write_score('qwen2vl_2b_geofm', 'GeoQA',          48.0)
write_score('qwen2vl_7b_geofm', 'MathVista_MINI', 50.5)
write_score('qwen2vl_7b_geofm', 'GeoQA',          55.0)

scores = scan_work_dir(work)
print(f'#score files = {len(scores)}')
for s in scores:
    print(f'  {s.model:18s}  {s.benchmark:16s}  {s.score}  (extras={s.extras})')
""",
    ),
    (
        "markdown",
        """\
## 3. The raw pivot table

`to_markdown_table(scores)` produces a model-x-benchmark pivot.
Useful for the README headline.
""",
    ),
    (
        "code",
        """\
from IPython.display import Markdown

Markdown(to_markdown_table(scores))
""",
    ),
    (
        "markdown",
        """\
## 4. The delta table — pick `qwen2vl_2b_base` as the baseline

`to_delta_table(scores, baseline_model=...)` shows every other row as
`score (+Δ)` against the baseline. This is the format the wiki and the
final blog post use.
""",
    ),
    (
        "code",
        """\
Markdown(to_delta_table(scores, baseline_model='qwen2vl_2b_base'))
""",
    ),
    (
        "markdown",
        """\
## 5. From this notebook to a real run

Inside the Docker image (CUDA 12.8 + vLLM with `VLLM_FLASH_ATTN_VERSION=2`):

```bash
# 1. Base + finetuned VLMEvalKit runs (one per model name).
docker compose -f docker/docker-compose.yml run --rm train \\
    bash scripts/04_eval.sh qwen2vl_2b_base   outputs/qwen2vl-2b-lora
docker compose -f docker/docker-compose.yml run --rm train \\
    bash scripts/04_eval.sh qwen2vl_2b_geofm  outputs/qwen2vl-2b-lora

# 2. Headline table (host venv, no GPU needed).
uv run python -m open_geofm.eval.compare outputs/eval \\
    --baseline qwen2vl_2b_base --out results.md
```

**Judge cost:** VLMEvalKit defaults to GPT-4 as the answer-extraction
judge (~$25/eval). Override to `gpt-4o-mini` in
`src/open_geofm/eval/vlmevalkit_run.sh` (~10x cheaper, ~$2-5 per full
MathVista+MathVerse+We-Math+GeoQA pass). The regex extractor above is
the **free** fallback for MathVista-GPS and works for ~95% of
responses.

## What we just did

* Exercised the `eval.extract_answer.extract` regex against the
  pedagogically interesting cases (MCQ wins over numbers, last-line
  rule, F-exclusion, none-of-the-above).
* Synthesised a VLMEvalKit work-dir in `tmp_path` and walked it with
  `scan_work_dir`.
* Rendered both the raw pivot table and the delta-vs-baseline table.

## What's next

* **Notebook 10 — Ablations & curves:** the data-scale curve +
  renderer ablation figures.
* **Wiki page 06 — Evaluation:** long-form companion + judge cost
  budgeting.
""",
    ),
]
