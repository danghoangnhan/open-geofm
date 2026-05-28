"""Spec for `notebooks/10_ablations_and_curves.ipynb`.

Phase 9 ablation harness in pedagogical form: scan a synthetic
VLMEvalKit work-dir, slice it along each documented axis (data-scale
curve, renderer ablation, LoRA-rank sweep), and emit both Markdown
tables and matplotlib line plots. Uses fabricated numbers (in the
ballpark of the blueprint's expected-outcome table) because no real
eval has run yet — the same pipeline reads real numbers once
`bash scripts/04_eval.sh ...` has been invoked.

CPU-only. Run from the repo root. ~5 seconds wall-clock.
"""

from __future__ import annotations

TITLE = "10 — Ablations & Curves"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 10 — Ablations & Curves

> **Run-time:** ~5 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal --extra dev`. *No real
> VLMEvalKit run required* — this notebook fabricates a small work-dir
> in `tmp_path` so the harness can be exercised end-to-end. Once a
> real eval lands at `outputs/eval/`, the same calls produce the real
> figures.

Phase 9 ablations from the blueprint:

1. **Data-scale curve** (5K / 10K / 20K → MathVista-GPS plot) —
   headline figure.
2. **Renderer ablation** (matplotlib vs GMBL-style) — quantifies how
   much diagram fidelity matters.
3. **LoRA-rank sweep** (r=8 / r=16 / r=32 / r=64).

The harness slices a `VLMEvalKit --work-dir` along one of the
suffix tokens on the model name (e.g. `qwen2vl_2b_lora_10k`) and
emits a Markdown table + a matplotlib line plot. Same code path,
different axis token.

Long-form: blueprint §2 Phase 9, §9.
""",
    ),
    (
        "markdown",
        """\
## 1. Fabricate a tiny work-dir

Each ablation arm writes one `<benchmark>_<model>_score.json` file
under `<work_dir>/<model>/`. We synthesise a few here so the rest of
the notebook runs without a real VLMEvalKit invocation. The fabricated
deltas track the blueprint's expected-outcome table:

* Qwen2-VL-2B base ≈ 20% on MathVista-GPS, +10–15 pp after open-geofm.
* Qwen2-VL-7B base ≈ 40%, +8–15 pp.
* GMBL-style renderer beats matplotlib by ~3–5 pp.
""",
    ),
    (
        "code",
        """\
import json
import tempfile
from pathlib import Path

work = Path(tempfile.mkdtemp(prefix='openfm-nb10-'))

def write_score(model: str, benchmark: str, score: float) -> None:
    p = work / model
    p.mkdir(parents=True, exist_ok=True)
    (p / f'{benchmark}_{model}_score.json').write_text(json.dumps({'Overall': score}))

# Data-scale arm: same renderer (GMBL), three data scales, two base configs.
for n, score_2b, score_7b in ((5, 28.0, 47.0), (10, 32.0, 51.0), (20, 34.5, 53.5)):
    write_score(f'qwen2vl_2b_lora_{n}k', 'MathVista_MINI', score_2b)
    write_score(f'qwen2vl_7b_lora_{n}k', 'MathVista_MINI', score_7b)

# Renderer ablation arm: same scale (10k), two renderers, on GeoQA.
write_score('qwen2vl_2b_lora_mpl',  'GeoQA', 44.0)
write_score('qwen2vl_2b_lora_gmbl', 'GeoQA', 48.5)

# LoRA-rank sweep arm: r=8/16/32/64, fixed 10k, MathVista-MINI.
for r, score in ((8, 30.5), (16, 32.0), (32, 32.5), (64, 32.0)):
    write_score(f'qwen2vl_2b_lora_r{r}', 'MathVista_MINI', score)

print(f'work_dir = {work}')
print('models:')
for p in sorted(work.iterdir()):
    print(f'  {p.name}')
""",
    ),
    (
        "markdown",
        """\
## 2. Data-scale curve — the headline figure

`open_geofm.ablate.scale_curve(scores, benchmark='MathVista_MINI')`
parses the `_5k` / `_10k` / `_20k` suffix off each model name, groups
by the *base config* (the part before the suffix), and returns a list
of `SweepPoint(base_config, benchmark, axis_value, axis_label, score)`.
""",
    ),
    (
        "code",
        """\
from IPython.display import Markdown

from open_geofm.ablate import scale_curve, to_markdown_sweep
from open_geofm.eval.compare import scan_work_dir

scores = scan_work_dir(work)
points = scale_curve(scores, benchmark='MathVista_MINI')
for p in points:
    print(f'  base={p.base_config!r:20s} n={int(p.axis_value):6d} score={p.score:5.2f}')

print()
print('--- Markdown ---')
Markdown(to_markdown_sweep(points))
""",
    ),
    (
        "markdown",
        """\
And the headline plot — one line per base config, x = dataset size
(log scale because the 5K-10K-20K spacing is multiplicative), y =
benchmark score.
""",
    ),
    (
        "code",
        """\
from open_geofm.ablate import plot_sweep

fig = plot_sweep(
    points,
    title='Data-scale curve (open-geofm-mini, MathVista-MINI)',
    xlabel='dataset size (samples)',
    ylabel='MathVista-MINI score',
    log_x=True,
)
fig
""",
    ),
    (
        "markdown",
        """\
## 3. Renderer ablation — matplotlib vs GMBL-style

Same harness, different suffix parser. The Phase-9 blueprint expects
a +3–5 pp gap in favour of the GMBL-style renderer. The fabricated
numbers above show **+4.5 pp** on GeoQA.
""",
    ),
    (
        "code",
        """\
from open_geofm.ablate.sweep import renderer_split

points = renderer_split(scores, benchmark='GeoQA')
Markdown(to_markdown_sweep(points))
""",
    ),
    (
        "code",
        """\
fig = plot_sweep(
    points,
    title='Renderer ablation (GeoQA)',
    xlabel='renderer (0=matplotlib, 1=GMBL-style)',
    ylabel='GeoQA score',
)
fig
""",
    ),
    (
        "markdown",
        """\
## 4. LoRA-rank sweep

Last documented Phase-9 axis. The fabricated numbers show the typical
LoRA-rank curve shape — gains flatten past r=16, then dip slightly at
r=64 (extra capacity over-fits the 10K samples).
""",
    ),
    (
        "code",
        """\
from open_geofm.ablate.sweep import lora_rank_curve

points = lora_rank_curve(scores, benchmark='MathVista_MINI')
for p in points:
    print(f'  r={int(p.axis_value):3d}  score={p.score}')
print()
Markdown(to_markdown_sweep(points))
""",
    ),
    (
        "code",
        """\
fig = plot_sweep(
    points,
    title='LoRA rank sweep (Qwen2-VL-2B, 10K, MathVista-MINI)',
    xlabel='LoRA rank r',
    ylabel='MathVista-MINI score',
    log_x=True,
)
fig
""",
    ),
    (
        "markdown",
        """\
## 5. From this notebook to real numbers

Once a real VLMEvalKit run has populated `outputs/eval/`:

```bash
# Data-scale curve, MathVista-MINI.
uv run python scripts/05_ablate.py scale outputs/eval \\
    --benchmark MathVista_MINI \\
    --out-md outputs/ablate/scale_mathvista.md \\
    --out-png outputs/ablate/scale_mathvista.png

# Renderer ablation, GeoQA.
uv run python scripts/05_ablate.py renderer outputs/eval \\
    --benchmark GeoQA --out-md outputs/ablate/renderer_geoqa.md

# LoRA rank sweep, MathVista-MINI.
uv run python scripts/05_ablate.py rank outputs/eval \\
    --benchmark MathVista_MINI --out-png outputs/ablate/rank_curve.png
```

For each ablation arm the corresponding training run is::

    bash scripts/03_train.sh qwen2vl_2b_lora --dataset data/open-geofm-mini-{5,10,20}k \\
        --output-dir outputs/qwen2vl_2b_lora_{5,10,20}k

(The model-name suffix `_10k` is what the sweep parser keys off.)

## What we just did

* Synthesised a tiny VLMEvalKit work-dir in `tmp_path` and walked it
  with the Phase-8 `scan_work_dir`.
* Sliced it along three axes (data scale, renderer, LoRA rank) and
  emitted both a Markdown table and a matplotlib plot per axis.
* Showed the production CLI surface (`scripts/05_ablate.py`) that
  runs the same harness against a real eval run.

## What's next

* **Phase 7 + 8 in earnest:** generate `open-geofm-mini-{5k,10k,20k}`,
  train the 2B + 7B LoRAs, run VLMEvalKit, re-run this notebook.
* **Wiki page 06 — Evaluation:** judge-cost budgeting + the headline
  delta table.
* **Blog post:** *Reproducing GeoFM on a single RTX 5090* — these
  three plots are the visual spine.
""",
    ),
]
