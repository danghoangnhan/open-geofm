# 00 — Overview

`open-geofm` reproduces the **data-generation methodology** of *GeoFM*
(Zhang et al., 2025, arXiv:2510.27448) on a single RTX 5090. The fine-tuning
step is commodity — the engineering budget goes into the FormalGeo + sampler +
renderer + verifier pipeline.

## Pipeline

![open-geofm pipeline diagram](img/pipeline.svg)

(The same flow, in Mermaid for editors that don't render SVG inline:)

```mermaid
flowchart TD
  subgraph CPU [CPU phases — host uv venv]
    A[FormalGeo7K<br/>seed problem] --> B[Algorithm 1<br/>metric swap]
    B --> C[FGPS symbolic<br/>solver]
    C --> D[matplotlib /<br/>GMBL renderer]
    C --> E[NL templates]
    E --> F[LLM rewriter<br/>Qwen2.5-7B or gpt-4o-mini]
    F --> G[Answer verify<br/>vs FGPS]
    D --> H[HF dataset<br/>open-geofm-mini-{5,10,20}K]
    G --> H
  end
  subgraph GPU [GPU phases — Docker NGC 25.02]
    H --> I[TRL SFTTrainer<br/>LoRA r=16, all-linear]
    I --> J[Adapter on HF Hub<br/>Qwen2-VL-{2B,7B}-OpenGeoFM-LoRA]
    J --> K[VLMEvalKit<br/>MathVista-GPS · GeoQA · ...]
  end
```

## Why the split?

* **CPU phases** — FormalGeo, sampling, verification, rendering, NLG templates,
  dataset assembly — are pure Python / numpy / scipy / matplotlib and do *not*
  benefit from CUDA. Running them in the host uv venv keeps Blackwell rebuild
  pain confined to the GPU phases.
* **GPU phases** — vLLM-served local NLG rewriter (Phase 5), TRL training
  (Phase 7), VLMEvalKit eval (Phase 8) — happen inside the
  `nvcr.io/nvidia/pytorch:25.02-py3` container where CUDA 12.8, sm_120 PyTorch
  wheels, FA2-from-source and bnb-from-source are pre-baked.

## Repo tour

```
src/open_geofm/
├── formal/        # FormalGeo loader, CDL types, FGPS solver wrapper        (Phase 1)
├── sampling/      # Algorithm 1, M_all BFS, goal picker                     (Phase 2)
├── render/        # matplotlib + GMBL renderers + CDL→GMBL translator       (Phase 4)
├── nlg/           # NL templates, local/openai rewriter, answer verify      (Phases 3+5)
├── dataset/       # HF dataset builder, Qwen2-VL ChatML format               (Phase 6)
├── train/         # TRL SFTTrainer driver, 4 model configs                  (Phase 7)
└── eval/          # VLMEvalKit wrapper + regex answer extractor             (Phase 8)
```

## Where to start

* **Run the smoke tests**:
  ```bash
  uv sync --extra formal --extra dev
  uv run pytest -q
  ```
  The host-side suite currently runs **167 tests in ~20 s** (2 skipped require
  the `datasets` GPU extra).
* **Read [02 — Condition Sampling](./02-Condition-Sampling)** — that's where
  the paper's novel contribution is.
* **Watch [07 — Blackwell Setup Log](./07-Blackwell-Setup-Log)** — it's the
  most useful page for anyone else trying this on a 5090.
