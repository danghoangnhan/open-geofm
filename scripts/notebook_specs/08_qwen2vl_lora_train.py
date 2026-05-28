"""Spec for `notebooks/08_qwen2vl_lora_train.ipynb`.

Phase 7 LoRA dry-run: load each of the four training configs from the
host venv (no torch / trl / peft imports), write a `dry_run.json`
manifest, then walk the VRAM + wall-clock budget. Actual training
happens inside the Blackwell Docker image — that step is *not* in this
notebook by design.

CPU-only. Run from the repo root. ~3 seconds wall-clock.
"""

from __future__ import annotations

TITLE = "08 — Qwen2-VL LoRA (Dry-Run)"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """\
# 08 — Qwen2-VL LoRA (Dry-Run)

> **Run-time:** ~3 seconds on CPU.
> **Prerequisites:** `uv sync --extra formal --extra dev`. The host
> venv has no torch / trl / peft — by design. This notebook exercises
> the **dry-run** path that resolves the config + dataset spec and
> writes a manifest without importing any GPU dependencies.

The real training step runs inside the Docker image
(`docker compose run --rm train bash scripts/03_train.sh ...`). This
notebook documents what that command produces and why.

Long-form: [wiki/05-Qwen2VL-Finetuning.md](../wiki/05-Qwen2VL-Finetuning.md).
""",
    ),
    (
        "markdown",
        """\
## 1. The four configs

`open_geofm.train.sft_vlm.load_config(name)` resolves one of:

| Name | Base model | Method | VRAM (5090, 32 GB) | Wall-clock (10K, 2 ep) |
|---|---|---|---|---|
| `qwen2vl_2b_lora` | `Qwen/Qwen2-VL-2B-Instruct` | LoRA r=16 bf16 | ~10 GB | 3-5 h |
| `qwen2vl_7b_lora` | `Qwen/Qwen2-VL-7B-Instruct` | LoRA r=16 bf16 | ~20 GB | 10-18 h |
| `qwen25vl_7b_lora` | `Qwen/Qwen2.5-VL-7B-Instruct` | LoRA r=16 bf16 | ~20 GB | 10-18 h |
| `qwen2vl_7b_qlora` | `Qwen/Qwen2-VL-7B-Instruct` | QLoRA nf4 r=16 | ~12 GB | 8-14 h |

QLoRA is **gated** behind `OPEN_GEOFM_ENABLE_QLORA=1` because
bitsandbytes' sm_120 path can produce garbage outputs (blueprint §8
risk #1; bnb upstream issue #1642). Wiki 07 has the running status.

Effective batch size is held constant at **16** across configs (so
loss curves are comparable):

* 2B: `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`.
* 7B: `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`.
* 7B QLoRA: `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`.
""",
    ),
    (
        "code",
        """\
from open_geofm.train.sft_vlm import load_config

for name in ('qwen2vl_2b_lora', 'qwen2vl_7b_lora', 'qwen25vl_7b_lora'):
    cfg = load_config(name)
    sft = cfg['sft_config']
    print(f'{name:20s} base={cfg[\"model_name_or_path\"]:34s} bs={sft[\"per_device_train_batch_size\"]} x acc={sft[\"gradient_accumulation_steps\"]} = eff={sft[\"per_device_train_batch_size\"] * sft[\"gradient_accumulation_steps\"]}')
""",
    ),
    (
        "markdown",
        """\
## 2. The QLoRA env-var gate

Loading the QLoRA config without `OPEN_GEOFM_ENABLE_QLORA=1` raises so
nobody silently launches a run that may produce garbage tokens.
""",
    ),
    (
        "code",
        """\
import os

# Pretend the env var isn't set — should raise.
os.environ.pop('OPEN_GEOFM_ENABLE_QLORA', None)
try:
    load_config('qwen2vl_7b_qlora')
except RuntimeError as e:
    print(f'(expected) RuntimeError: {e}')

# Opt in and try again.
os.environ['OPEN_GEOFM_ENABLE_QLORA'] = '1'
cfg = load_config('qwen2vl_7b_qlora')
print(f'\\nQLoRA opt-in OK. quantization = {cfg[\"quantization\"]}')
del os.environ['OPEN_GEOFM_ENABLE_QLORA']
""",
    ),
    (
        "markdown",
        """\
## 3. The critical VLM gotcha — `max_length=None`

Every config sets `sft_config[\"max_length\"] = None`. Truncation
*would* slice through image-token spans (the `<|vision_start|>...<|vision_end|>`
markers around each image's 200K-1M patch tokens), corrupting the
batch. This is documented in the TRL VLM examples; we lock it in at
the config level so it's hard to override accidentally.
""",
    ),
    (
        "code",
        """\
for name in ('qwen2vl_2b_lora', 'qwen2vl_7b_lora', 'qwen25vl_7b_lora'):
    cfg = load_config(name)
    assert cfg['sft_config']['max_length'] is None, f'{name} must not truncate VLM batches'
    print(f'{name:20s}  max_length=None  ✓')
""",
    ),
    (
        "markdown",
        """\
## 4. LoRA targets — `all-linear`, not `q_proj+v_proj`

The default LLaMA recipe targets only the attention projections, which
under-fits Qwen2-VL's MLPs. Blueprint §7 pitfalls says use
`target_modules=\"all-linear\"` — `peft.LoraConfig` then matches every
`nn.Linear` in the model (attention + MLP + the cross-attention pooler
on the vision side).
""",
    ),
    (
        "code",
        """\
cfg = load_config('qwen2vl_2b_lora')
print('lora_config:')
for k, v in cfg['lora_config'].items():
    print(f'  {k:18s} = {v!r}')
""",
    ),
    (
        "markdown",
        """\
## 5. Vision-tower learning rate

Qwen2-VL's ViT was pre-trained on natural images. Our synthetic
diagrams are a domain shift; freezing the ViT entirely loses signal,
but a full LR (1e-4) destabilises it. The `vision_tower_lr` field
(consumed by `_make_vision_tower_lr_trainer_cls`) puts every parameter
under `visual.*` / `vision_tower` / `vision_model` into its own group
at **1e-6** — two orders of magnitude under the LLM-side LR. The
factory wraps `SFTTrainer.create_optimizer` to split the parameter
groups.
""",
    ),
    (
        "code",
        """\
for name in ('qwen2vl_2b_lora', 'qwen2vl_7b_lora', 'qwen25vl_7b_lora'):
    cfg = load_config(name)
    print(f'{name:20s}  learning_rate={cfg[\"sft_config\"][\"learning_rate\"]:.0e}  vision_tower_lr={cfg[\"vision_tower_lr\"]:.0e}')
""",
    ),
    (
        "markdown",
        """\
## 6. The dry-run manifest

`scripts/03_train.sh ... --dry-run` (or `uv run python -m
open_geofm.train.sft_vlm --config qwen2vl_2b_lora --dataset data/smoke
--dry-run`) writes `outputs/<config>/dry_run.json` summarising the
resolved kwargs. Useful as a CI gate — no torch needed, no GPU needed,
and the resulting JSON is exactly what `train()` will see at runtime.

We call it inline here via Typer's `CliRunner` so the notebook works
on a CPU-only kernel.
""",
    ),
    (
        "code",
        """\
import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from open_geofm.train import sft_vlm

work = Path(tempfile.mkdtemp(prefix='openfm-nb08-'))
# Re-point the output dir into the tmp dir so the demo cleans up.
orig_load = sft_vlm.load_config

def patched_load(name: str):
    cfg = orig_load(name)
    cfg['sft_config']['output_dir'] = str(work / name)
    return cfg

sft_vlm.load_config = patched_load
runner = CliRunner()
result = runner.invoke(
    sft_vlm.app,
    ['--config', 'qwen2vl_2b_lora', '--dataset', 'data/smoke', '--dry-run'],
)
sft_vlm.load_config = orig_load   # restore

print(f'exit_code = {result.exit_code}')
manifest = json.loads((work / 'qwen2vl_2b_lora' / 'dry_run.json').read_text())
print(json.dumps(manifest, indent=2)[:1200])
print('...')
""",
    ),
    (
        "markdown",
        """\
## 7. From dry-run to real training

After the dry-run JSON looks right, the actual training command
(inside the Docker image — see wiki/07 for the Blackwell setup) is:

```bash
docker compose -f docker/docker-compose.yml run --rm train \\
    bash scripts/03_train.sh qwen2vl_2b_lora \\
        --dataset data/open-geofm-mini-10k --max-steps 50
```

`scripts/03_train.sh` wraps the Typer CLI shown above. The Docker
image carries torch cu128, FA2 built from source, peft, trl, datasets,
and bitsandbytes built from source. The wall-clock budget per
`scripts/03_train.sh` invocation is in the table at the top of this
notebook.

The output adapter lands at `outputs/<config>/` (LoRA weights, ~50-200
MB) and can be pushed to the Hub with `--push-to-hub`.

## What we just did

* Loaded each of the four configs through the host (CPU) venv — no
  torch, no trl.
* Saw the QLoRA env-var gate that protects against silent bnb
  sm_120 failures.
* Verified `max_length=None` (the load-bearing VLM gotcha), the
  `target_modules=\"all-linear\"` LoRA target, and the
  `vision_tower_lr=1e-6` parameter group.
* Wrote a dry-run manifest the CI gate can compare against.

## What's next

* **Notebook 09 — Eval:** answer extraction + VLMEvalKit judging.
* **Notebook 10 — Ablations & curves:** the headline data-scale +
  renderer ablation plot.
* **Wiki page 07 — Blackwell Setup Log:** every \"wheel didn't build\" /
  \"garbage output\" landmine, documented.
""",
    ),
]
