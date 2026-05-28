"""Spec for `notebooks/07_build_hf_dataset.ipynb`.

Phase 6 dataset assembly: SyntheticSample tuples + rendered PNGs ->
JSONL sidecar (and, when `datasets` is installed, an HF Dataset on
disk). Demonstrates the bundled-CDL schema that is the educational
killer feature, plus the Qwen2-VL ChatML conversation format.

CPU-only. Run from the repo root. ~5 seconds wall-clock.
"""

from __future__ import annotations

TITLE = "07 — Build the HF Dataset"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 07 — Build the HF Dataset

> **Run-time:** ~5 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal --extra dev`. The
> `datasets` package is GPU-extras only — this notebook runs the JSONL
> fallback path. Inside the Docker image the same call produces an HF
> `Dataset` on disk.

Phase 6 takes verified `SyntheticSample` tuples from Algorithm 1, pairs
them with PNGs from the renderer, and emits an introspectable record
set. The schema bundles **the four CDLs alongside the NL** — that's
the educational killer feature: readers can train alternative formats,
re-run symbolic checks, or ablate template-vs-LLM rewriting without
re-generating images.

Long-form: blueprint §2 Phase 6.
""",
    ),
    (
        "markdown",
        """\
## 1. The schema

`open_geofm.dataset.builder.build(samples, image_dir, out_dir)` writes:

* `out_dir/records.jsonl` — one JSON object per record, **always**.
* `out_dir/dataset_info.json` + arrow shards — only if `datasets` is
  installed (the Docker image; not the host venv).

Per record:

| Field | Type | Notes |
|---|---|---|
| `id` | str | `openfm-<source_pid:05d>-<idx:04d>` |
| `image` | str (path) | `<image_dir>/<id>.png` |
| `problem` | str | NL problem from `templates.draft_nl` + LLM smooth |
| `solution` | str | NL solution, verified by Phase-3 sympy |
| `answer` | str | FGPS-derived (possibly symbolic, e.g. `2*sqrt(21)`) |
| `construction_cdl` | list[str] | inherits from the seed |
| `image_cdl` | list[str] | metrics from `M_add` after the swap |
| `text_cdl` | list[str] | seed-text metrics that survived the swap |
| `goal_cdl` | str | the new goal Algorithm 1 picked |
| `theorem_seq` | list[str] | FGPS proof trace |
| `source_seed_pid` | int | provenance back to FormalGeo7K |

The split between `image_cdl` and `text_cdl` in the *output* uses
`added_metrics` as the discriminator — anything newly introduced by
the swap lands on the image side. This mirrors blueprint §2 Phase 2's
*separable text/image allocator* requirement.
""",
    ),
    (
        "markdown",
        """\
## 2. Hand-craft two synthetic samples

We bypass Algorithm 1 here and build two `SyntheticSample` tuples
directly so we can read the schema without running the full pipeline.
""",
    ),
    (
        "code",
        """\
from open_geofm.sampling.algorithm1 import SyntheticSample

sample_a = SyntheticSample(
    source_pid=1,
    new_metrics=('Equal(LengthOfLine(AB),3)', 'Equal(MeasureOfAngle(ABC),90)'),
    deleted_metrics=('Equal(LengthOfLine(BC),4)',),
    added_metrics=('Equal(MeasureOfAngle(ABC),90)',),
    goal='Equal(LengthOfLine(AC),5)',
    answer='5',
    theorem_seqs=('right_triangle_property -> Equal(LengthOfLine(AC),5)',),
    construction_cdl=('Triangle(A,B,C)',),
    nl_problem='In triangle ABC, AB = 3 and angle ABC = 90 degrees. Find the length of AC.',
    nl_solution='By the Pythagorean theorem, AC = sqrt(3^2 + 4^2) = 5.',
)
sample_b = SyntheticSample(
    source_pid=2,
    new_metrics=('Equal(LengthOfLine(XY),6)', 'Equal(MeasureOfAngle(XYZ),60)'),
    deleted_metrics=('Equal(LengthOfLine(YZ),8)',),
    added_metrics=('Equal(MeasureOfAngle(XYZ),60)',),
    goal='Equal(LengthOfLine(XZ),6)',
    answer='6',
    theorem_seqs=('equilateral_property -> done',),
    construction_cdl=('Triangle(X,Y,Z)',),
    nl_problem='Triangle XYZ has XY = 6 and angle XYZ = 60 degrees. Find XZ.',
    nl_solution='Since the triangle is isoceles with a 60° apex, XZ = 6.',
)
print(f'sample_a.source_pid = {sample_a.source_pid}')
print(f'sample_b.source_pid = {sample_b.source_pid}')
""",
    ),
    (
        "markdown",
        """\
## 3. `build(...)` -> JSONL sidecar

`build(samples, image_dir, out_dir)` runs unconditionally, writing
the JSONL. On the host venv (no `datasets` installed) it returns a
small `dict` with the path and record count; inside Docker it returns
a `Dataset` object and also writes the arrow shards.

Run in a tmp directory so the demo cleans up after itself.
""",
    ),
    (
        "code",
        """\
import json
import tempfile
from pathlib import Path

from open_geofm.dataset.builder import build

work = Path(tempfile.mkdtemp(prefix='openfm-nb07-'))
out_dir = work / 'out'
image_dir = work / 'images'
image_dir.mkdir(parents=True)
# In the real pipeline the renderer writes the PNGs; here we touch empty
# files so the builder's image-path resolution sees a real filesystem entry.
for sid in ('openfm-00001-0000', 'openfm-00002-0001'):
    (image_dir / f'{sid}.png').write_bytes(b'')

result = build([sample_a, sample_b], image_dir=image_dir, out_dir=out_dir)
print(f'build() returned: {result!r}')

records = [json.loads(line) for line in (out_dir / 'records.jsonl').read_text().splitlines()]
print(f'\\n#records in JSONL = {len(records)}')
print('\\n--- record 0 ---')
print(json.dumps(records[0], indent=2))
""",
    ),
    (
        "markdown",
        """\
Note three behaviours of `build`:

* **Dedup**: if a metric appears in both `text_cdl` and `image_cdl` of
  the seed, the swap propagates the duplicate — the builder dedupes
  per record (covered by
  `tests/test_dataset_builder.py::test_build_dedupes_repeated_cdls`).
* **Path resolution**: `image` is a string path, not an inlined PIL
  image — keeps JSONL small. The `datasets.Features` schema (inside
  Docker) maps this to the `Image()` feature for lazy decoding.
* **Added-metric side**: anything from `sample.added_metrics` ends up
  in `image_cdl` of the output, mirroring the swap's "force the model
  to read the figure" intent.

## 4. Qwen2-VL ChatML conversation

`open_geofm.dataset.qwen_vl_format.to_conversation(image, problem, solution)`
produces the ChatML dict the trainer expects. LLaMA-Factory and
TRL's `template: qwen2_vl` apply the `<|vision_start|><|image_pad|><|vision_end|>`
markers automatically — we keep this helper here so JSONL consumers can
introspect the structure without loading the tokenizer.

Vision token IDs (hardcoded from the Qwen2-VL tokenizer):

* `<|vision_start|>` = 151652
* `<|image_pad|>`    = 151655
* `<|vision_end|>`   = 151653
""",
    ),
    (
        "code",
        """\
from open_geofm.dataset.qwen_vl_format import to_conversation

conv = to_conversation(
    image_path=records[0]['image'],
    problem_nl=records[0]['problem'],
    solution_nl=records[0]['solution'],
)
print(json.dumps(conv, indent=2))
""",
    ),
    (
        "markdown",
        """\
## 5. The production driver

`scripts/02_generate_dataset.py` chains Phases 2–6 end to end. Common
invocations from the README quickstart:

```bash
# 100-sample smoke test (single process, matplotlib renderer, template-only NLG).
uv run python scripts/02_generate_dataset.py --n 100 --renderer matplotlib --out data/smoke

# 5K release: 8 workers, GMBL renderer, template NLG. Blueprint target ~5K/hr.
uv run python scripts/02_generate_dataset.py --n 5000 --n-workers 8 --renderer gmbl \\
    --out data/open-geofm-mini-5k

# 5K release with local vLLM rewriter (single process — only one can host vLLM).
OPEN_GEOFM_REWRITER=local uv run python scripts/02_generate_dataset.py \\
    --n 5000 --rewriter llm --n-workers 1 --out data/open-geofm-mini-5k-llm
```

Output layout::

    data/open-geofm-mini-5k/
        records.jsonl                       (always)
        dataset_info.json + arrow shards/    (only when `datasets` is installed)
        images/
            openfm-<pid>-<idx>.png

Three published versions: **`open-geofm-mini-{5k,10k,20k}`**, all
CC-BY-4.0.
""",
    ),
    (
        "markdown",
        """\
## What we just did

* Walked the record schema — the four CDLs bundled with NL is the
  educational killer feature.
* Ran `dataset.builder.build` on two hand-crafted `SyntheticSample`
  tuples; saw the JSONL sidecar.
* Generated the Qwen2-VL ChatML conversation dict via
  `dataset.qwen_vl_format.to_conversation`.

## What's next

* **Notebook 08 — LoRA training dry-run:** how the records above plug
  into the TRL `SFTTrainer`.
* **Wiki page 05 — Qwen2-VL Fine-tuning:** long-form companion +
  VRAM budget.
""",
    ),
]
