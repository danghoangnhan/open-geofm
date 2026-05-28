# 07 — Blackwell setup log

> **Living document.** Every Blackwell (sm_120) error message + the fix that
> made it go away + the upstream issue link. As of May 2026, the Blackwell
> software stack still has sharp edges; this page is the highest-leverage thing
> in the wiki for anyone trying to reproduce this work on their own RTX 5090.

## Status snapshot (May 2026)

| Component | Status on sm_120 | Action |
|---|---|---|
| PyTorch | Stable ≥ 2.7 cu128; 2.12 nightly recommended | NGC `pytorch:25.02-py3` is the safe base |
| CUDA toolkit | 12.8 min, 12.9 recommended | **Not** Ubuntu's apt 12.0 |
| Driver | 570+ Linux, 576+ Windows | Latest Game Ready / Studio |
| flash-attn v2 | Builds from source on sm_120 | `FLASH_ATTENTION_FORCE_BUILD=TRUE` |
| flash-attn v4 | **Cannot run on sm_120** (TMEM absent) | Pin FA2 forever |
| bitsandbytes | No official wheel; INT8 may produce garbage | Build from source; QLoRA gated |
| Unsloth | Blackwell instability (#1679) | Not used in this repo |
| vLLM | ≥ 0.8.0 has Blackwell fp8 GEMM | `VLLM_FLASH_ATTN_VERSION=2` |
| Triton | ≥ 3.2 Linux | Stick to Linux |

## Smoke commands

```bash
nvidia-smi                                       # driver + memory
docker compose -f docker/docker-compose.yml run --rm train \
    python -c "import torch; print(torch.cuda.get_arch_list())"   # expect 'sm_120'
docker compose -f docker/docker-compose.yml run --rm train \
    python scripts/00_verify_env.py
```

## Known landmines

### 1. Ubuntu apt `nvidia-cuda-toolkit` pins to CUDA 12.0
Symptom: no sm_120 in `torch.cuda.get_arch_list()`, runtime errors like
`no kernel image is available for execution on the device`.
Fix: install CUDA 12.8 toolkit from NVIDIA directly, or use the NGC container.

### 2. `TORCH_CUDA_ARCH_LIST="8.0"` silently drops sm_120
Symptom: building FA2 / bnb / xformers, the build "succeeds" but the resulting
extension is unusable on the 5090.
Fix: set `TORCH_CUDA_ARCH_LIST="12.0;12.0+PTX"` (already done in the Dockerfile).

### 3. `bitsandbytes` INT8 produces garbage characters
Symptom: model trains, no error, generated text is `▁▁▁▁▁` / Unicode soup.
Root cause: bnb kernels not yet compiled for SM_120 instruction set
(upstream issue: `bitsandbytes-foundation/bitsandbytes#1642`).
Mitigation: this repo gates QLoRA behind `OPEN_GEOFM_ENABLE_QLORA=1` and ships
bf16 LoRA as the primary path. Re-check on bnb releases.

### 4. `flash-attention` PyPI wheel lacks sm_120
Symptom: `RuntimeError: FlashAttention only supports SM 80, ...`.
Fix: `FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation flash-attn`.
The Dockerfile does this; on the host, just don't install flash-attn.

### 5. vLLM's FA3 backend doesn't work on Blackwell
Symptom: vLLM serve crashes at startup, FA3 import error.
Fix: `export VLLM_FLASH_ATTN_VERSION=2` (set in `docker-compose.yml`).

### 6. Mixed CUDA 12.6 + 12.8 toolkits → NCCL skew
Symptom: distributed init hangs.
Fix: only have one CUDA toolkit on PATH; the NGC container has 12.8 only.

## When to fall back

Per the project plan's decision threshold, if Blackwell rebuild takes **more
than 2 weeks**, rent a RunPod A100 hour ($1–2) to validate the rest of the
pipeline. Document the workaround here. The CPU phases (1–4, 6) run fine on
the host venv without touching CUDA — only Phases 5 / 7 / 8 actually need a GPU.
