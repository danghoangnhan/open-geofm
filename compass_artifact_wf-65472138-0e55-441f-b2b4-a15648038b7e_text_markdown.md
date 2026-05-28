# A Day-1 Blueprint for `geofm-edu`: An Educational Reproduction of GeoFM on a Single RTX 5090

## TL;DR
- **Build a 5090-friendly walkthrough repo around three open-source pillars** — FormalGeo (formal language + symbolic solver), a custom GMBL-inspired diagram renderer, and LLaMA-Factory + ms-swift for Qwen2-VL LoRA fine-tuning — using a *scaled-down* GeoFM-Mini-10K dataset (vs the paper's 80K) and accepting that you cannot match the paper's full-parameter 8B SOTA on a single 32 GB card.
- **Realistic expected outcome:** +8 to +15 points over base Qwen2-VL-2B on MathVista-GPS and GeoQA, +3 to +6 points over base Qwen2-VL-7B with LoRA. The paper's headline +18.7 percentage points over GPT-4o on MathVista GPS (and +16.5 pp on GeoQA, per GeoFM arXiv:2510.27448, Zhang et al., Tencent Hunyuan Team: "The model trained with our data surpass the proprietary GPT-4o model by 18.7% on geometry problem-solving tasks in MathVista and by 16.5% on GeoQA") is *not* a target for a solo project — it depends on InternVL2-8B-MPO + full fine-tuning + 80K data + an H20 96 GB GPU.
- **Hard Blackwell (sm_120) gotchas as of May 2026** dominate the first two weeks of work: you must build on PyTorch ≥ 2.7 / cu128 (or nightly), bitsandbytes from source against CUDA 12.8/12.9, flash-attention v2 only (FA4 is silicon-locked-out of sm_120), and avoid Unsloth's Triton path until they ship official Blackwell wheels. Plan ~3 weeks of pure infra before any real ML.

---

## Key Findings

1. **The original GeoFM is NOT a Qwen2-VL paper.** The authors fine-tune **LLaVA-NeXT-8B** and **InternVL2-8B-MPO** with full-parameter SFT on 80K synthetic samples ("GeoFM80K"), seeded from FormalGeo7K and PGPS9K, on Nvidia H20 96 GB GPUs (Appendix A, Table 5). "GeoFM-8B" specifically means InternVL2-8B-MPO + GeoFM data. Your repo is therefore a *re-targeted* educational reproduction, not a 1:1 reproduction. State this prominently.
2. **The "novel" part you need to faithfully reproduce is the data-generation pipeline** (Algorithm 1 in §2.3.2). The fine-tuning step is commodity. Spend your engineering budget on FormalGeo integration + diagram rendering + symbolic verification, not on chasing optimizer tricks.
3. **The diagram renderer in the paper is NOT matplotlib.** It is a custom engine built on top of **GMBL (Geometry Model Building Language, Krueger et al. 2021b)** with a hand-coded mapping table: "Geometry Model Building Language (GMBL) uses a formal language and computational geometry to approximate target images through numerical optimization … we developed a new engine capable of automatically synthesizing large-scale geometric images based on GMBL … This conversion requires the prior construction of a mapping table from the FormalGeo language to the GMBL language." This is the single biggest "easy to under-estimate" component. A naive matplotlib renderer will leak distribution and tank evaluation.
4. **Bitsandbytes 4-bit / QLoRA is risky on sm_120** as of May 2026. `bitsandbytes-foundation/bitsandbytes` Issue #1642 ("Cuda 12.9 support"), filed specifically for RTX 5090 / Windows / CUDA 12.9 and requesting sm_120 kernel support, remains open. There are reports (informatico-madrid Blackwell-Linux-Infra-Optimizer) of garbage-character output on Blackwell with INT8 paths: "Investigation revealed that bitsandbytes kernels are currently incompatible with the SM_120 instruction set." Plan around LoRA (bf16) + gradient checkpointing as the primary recipe; treat QLoRA as an experimental optional path.
5. **LLaMA-Factory is the right framework** for Qwen2-VL on a single 5090 in May 2026 (officially maintained `examples/train_lora/qwen2vl_lora_sft.yaml` recipe + Qwen2-VL DPO support added Sept 2024). ms-swift is the runner-up; Unsloth supports Qwen2-VL but Blackwell stability is still maturing (Unsloth issue #1679). Avoid G-LLaVA's frozen `torch==2.0.1` codebase.
6. **vLLM works on Blackwell from v0.8.0** (per its March 2025 release notes: "Add cutlass support for blackwell fp8 gemm (#13798)" and "Support nvfp4 cutlass gemm (#13571)"). The vLLM v0.16/v0.17 line is what's stable today on RTX 5090. VLMEvalKit is the right evaluation harness because it natively supports MathVista (GPS subset), MathVerse, We-Math, and GeoQA in one config.

---

## Details

### 1. Repository Structure & Blueprint

**Recommended repo name (in order of preference):**
- `geofm-edu` — short, signals "educational"
- `geofm-walkthrough`
- `geofm-from-scratch`
- `geofm-mini-qwen2vl`
- `open-geofm`

**License:** **Apache-2.0.** FormalGeo / FGPS, MAVIS, Qwen2-VL, LLaMA-Factory, and PEFT are all Apache-2.0 (or MIT for ms-swift), so Apache-2.0 is the maximally compatible choice. MIT is also fine but Apache-2.0's explicit patent grant protects you and downstream users. **Do not use GPL.** G-LLaVA inherits LLaVA's Apache-2.0; FormalGeo7K dataset is released under its own non-restrictive academic-use license — check before re-bundling.

**Directory layout:**
```
geofm-edu/
├── README.md
├── LICENSE                       # Apache-2.0
├── CITATION.cff                  # for Zenodo DOI
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md            # Contributor Covenant v2.1
├── pyproject.toml                # uv / hatch managed
├── requirements/
│   ├── base.txt
│   ├── blackwell.txt             # torch cu128/nightly, FA2 from source
│   ├── train.txt                 # llamafactory, peft, trl
│   └── eval.txt                  # vlmevalkit, openai, vllm
├── docker/
│   ├── Dockerfile.blackwell      # nvcr.io/nvidia/pytorch:25.02-py3 base
│   └── docker-compose.yml
├── docs/                         # MkDocs Material
│   ├── index.md
│   ├── 00-overview.md
│   ├── 01-formal-language.md
│   ├── 02-condition-sampling.md
│   ├── 03-symbolic-verification.md
│   ├── 04-diagram-rendering.md
│   ├── 05-qwen2vl-finetuning.md
│   ├── 06-evaluation.md
│   └── img/                      # pipeline diagrams (excalidraw/draw.io)
├── notebooks/                    # numbered, pedagogical
│   ├── 01_formal_geo_hello_world.ipynb
│   ├── 02_parse_cdl_and_solve.ipynb
│   ├── 03_sample_metric_conditions.ipynb
│   ├── 04_render_diagram_matplotlib.ipynb
│   ├── 05_render_diagram_gmbl_style.ipynb
│   ├── 06_generate_qa_with_llm.ipynb
│   ├── 07_build_hf_dataset.ipynb
│   ├── 08_qwen2vl_lora_train.ipynb
│   ├── 09_run_vlmevalkit.ipynb
│   └── 10_ablations_and_curves.ipynb
├── src/geofm_edu/
│   ├── formal/                   # thin wrapper over BitSecret/FGPS
│   │   ├── loader.py, cdl.py, solver.py
│   ├── sampling/
│   │   ├── algorithm1.py         # the metric-combination sampler
│   │   ├── gather_metrics.py     # BFS over theorems to enumerate M_all
│   │   └── goal_picker.py
│   ├── render/
│   │   ├── matplotlib_renderer.py
│   │   ├── gmbl_renderer.py
│   │   └── cdl_to_gmbl.py
│   ├── nlg/
│   │   ├── templates.py
│   │   ├── rewrite_llm.py
│   │   └── verify.py
│   ├── dataset/
│   │   ├── builder.py
│   │   └── qwen_vl_format.py
│   ├── train/
│   │   ├── llamafactory_configs/
│   │   │   ├── qwen2vl_2b_lora.yaml
│   │   │   ├── qwen2vl_7b_lora.yaml
│   │   │   └── qwen2vl_7b_qlora.yaml
│   │   └── train_cli.py
│   └── eval/
│       ├── vlmevalkit_run.sh
│       └── extract_answer.py
├── scripts/
│   ├── 00_setup_env.sh
│   ├── 01_download_formalgeo7k.sh
│   ├── 02_generate_dataset.py
│   ├── 03_train.sh
│   ├── 04_eval.sh
│   └── 05_upload_to_hf.sh
├── data/                         # gitignored
├── outputs/                      # gitignored
├── tests/
│   ├── test_cdl_parse.py
│   ├── test_sampling_invariants.py
│   └── test_render_smoke.py
└── .github/
    ├── workflows/{ci.yml, docs.yml}
    └── ISSUE_TEMPLATE/
```

**README.md sections (in order):** (1) Banner + arXiv/license/DOI badges + a bold "Educational Reproduction — not the official code" disclaimer; (2) one-paragraph summary + pipeline diagram; (3) explicit "What this IS / IS NOT"; (4) 5-command quickstart with a 1-minute inference demo; (5) hardware + Blackwell notes; (6) the 9-phase pipeline TOC; (7) **two-citation block** (your repo via Zenodo DOI + original paper BibTeX), explicitly: "If you use this implementation, please cite both the original GeoFM paper and this repository"; (8) acknowledgements (FormalGeo, DFE-GPS, MAVIS, G-LLaVA, LLaMA-Factory, VLMEvalKit, Qwen); (9) license + data license notes.

**Citable via Zenodo:** `CITATION.cff` + GitHub-Zenodo integration auto-mints a DOI on each release. Add the DOI badge to README.

**CONTRIBUTING.md:** Black + ruff, pre-commit hooks, DCO sign-off, no CLA. Issue templates: bug / pedagogical-improvement / new-notebook.

**CODE_OF_CONDUCT.md:** Contributor Covenant v2.1 verbatim.

---

### 2. Technical Pipeline (10 Phases)

#### Phase 0 — Environment Setup (Blackwell, ~3–5 days)

**Goal:** A reproducible Docker-or-conda env where `torch.cuda.get_arch_list()` includes `sm_120` and a small matmul on `cuda:0` returns without error.

**Hard facts about sm_120 as of May 2026:**
- Stable PyTorch ≥ 2.7 cu128 ships sm_120 wheels; PyTorch 2.12.x nightlies are the safe bet for cutting-edge work.
- bitsandbytes does not yet have an official Blackwell wheel; build from source against CUDA 12.8/12.9 (Issue #1642 open).
- flash-attention v2 builds from source for sm_120 with `FLASH_ATTENTION_FORCE_BUILD=TRUE`.
- **flash-attention v4 cannot run on sm_120** (silicon-level absence of TMEM — Dao-AILab/flash-attention Issue #1665; the gau-nernst blog and solatticus's deep investigation document this: "SM120 uses HMMA — the same register-to-register MMA approach … There are zero UTC* or UTMA* opcodes in any SM120a binary").
- Unsloth has Blackwell issues (Issue #1679: "Compatibility with Blackwell"); supports Qwen2-VL on Hopper but requires manual recompilation of triton/bitsandbytes on Blackwell.
- vLLM ≥ v0.8.0 (March 2025) ships Blackwell fp8 GEMM. The vLLM Blackwell docs explicitly say: "Flash Attention 3 backend doesn't work with Blackwell yet, please use `VLLM_FLASH_ATTN_VERSION=2`."

**Recommended base:** `nvcr.io/nvidia/pytorch:25.02-py3` (CUDA 12.8, PyTorch 2.6.0a0+ with sm_120). Alternative: WSL2 Ubuntu 24.04 with CUDA 12.9 toolkit installed manually (NOT Ubuntu's `nvidia-cuda-toolkit` apt package which pins to 12.0).

**Deliverables:** `docker/Dockerfile.blackwell`, `scripts/00_setup_env.sh`, and a `verify_env.py` notebook that prints `torch.cuda.get_arch_list()`, runs a matmul, and runs a 1-step Qwen2-VL LoRA on a dummy batch.

**Common pitfalls:**
- Installing `nvidia-cuda-toolkit` from Ubuntu apt → pins to CUDA 12.0, no sm_120.
- Installing `bitsandbytes` via pip → wheel may lack sm_120 kernels.
- Setting `TORCH_CUDA_ARCH_LIST="8.0"` → silently drops sm_120.
- Mixing CUDA 12.6 and 12.8 toolkits → NCCL version skew.

**Pedagogical content:** `notebooks/00_setup_smoke_test.ipynb` shows the exact warning messages users will see, then the fix. A docs page logs every error message + fix + upstream-issue link.

---

#### Phase 1 — Understand & Install FormalGeo (~3 days)

**Goal:** Load a problem from `formalgeo7k_v1` by PID, print its CDL, run FGPS forward-search, get a verified answer.

**Components:**
- `pip install formalgeo` (PyPI package by the FormalGeo team) or `git clone https://github.com/BitSecret/formalgeo7k && pip install -e .`
- `git clone https://github.com/BitSecret/FGPS` — the solver.
- 88 geometric predicates and 196 theorems documented in Zhang et al. 2024 "FormalGeo: An Extensible Formalized Framework" (arXiv:2310.18021).

**Concrete API:** `src/geofm_edu/formal/loader.py::load_problem(pid: int) -> Problem` with fields `construction_cdl, text_cdl, image_cdl, goal_cdl, theorem_seqs, answer`. `solver.py::solve(problem, strategy='backward')` returns the symbolic engine's verified answer.

**Compute:** CPU-only.

**Pitfalls:** FormalGeo's predicate library has Chinese comments in some files. The predicate *names* themselves are English. FGPS's `auto_run` is slow (~30s/problem on harder examples) — use timeouts.

**Pedagogical content:** `notebooks/01_formal_geo_hello_world.ipynb` — load PID=1, print CDL, naive matplotlib render, call FGPS, walk through theorem-by-theorem trace. `docs/01-formal-language.md` — 10-minute intro to "what is a Conditional Declaration Language" with worked examples.

---

#### Phase 2 — Formal-Language Condition Sampling (~1 week — the novel part)

**Goal:** Implement Algorithm 1 from the paper verbatim:
```
Input: formalized seed problem set FS, number of synthetic problems m
for P in FS:
    M_p   = MetricInfoOfProblemStatement(P)
    M_all = GatheringMetricInfo(P)        # BFS over theorems
    m_p = m
    while m_p > 1:
        n = Random(1, min(|M_p|, |M_all| − |M_p|))
        M_del = RandomSelect(M_p, n)
        M_add = RandomSelect(M_all \ M_p, n)
        P_new = (P \ M_del) ∪ M_add
        A_new = FormalGeoSolver(P_new)
        P_syn, A_syn = Template_and_LLM(P_new, A_new)
        if AnswerVerify(A_syn, A_new): S.add((P_syn, A_syn)); m_p -= 1
return S
```

**Implementation:**
- `gather_metrics.py` — BFS applying the 196 theorems to the seed CDL until solved or timeout; returns `M_all`. Per the paper: "We sample a random number n (where n ≤ min(|M_p|, |M_all| − |M_p|)). Next, we replace n metric conditions from M_p with n new conditions sampled from the remaining metric set M_all − M_p and randomly choose one metric condition different from the new problem statement as the goal."
- `algorithm1.py` — main loop with unit tests asserting `M_del ∩ M_add = ∅`.
- `goal_picker.py` — random pick from `M_all \ M_p_new`; if unsolvable, "we select the last valid inference from the symbolic engine's reasoning path as the new goal."
- Metric allocation: random split between text_cdl and image_cdl ("forcing the model to interpret the problem by reading the images") — implement as a separable, ablatable function.

**Compute:** CPU-bound. With multiprocessing on 8 cores: ~5K samples/hour; 10K in 2h, 20K in 4h.

**Pitfalls:** Solver hangs on some seeds (15s timeout per call). Empty `M_p` → skip. Under-determined problems "solve" with degenerate answers → filter.

**Pedagogical content:** `notebooks/03_sample_metric_conditions.ipynb` — for one seed problem, visualize `M_p`, `M_all`, sample 5 new problems, render side-by-side.

---

#### Phase 3 — Symbolic Verification (~2 days)

**Goal:** Every synthetic `(P_new, A_new)` pair is verified by FGPS before being added to the dataset.

**Approach:** `verify.py` re-runs FGPS on `P_new` and compares to the LLM-rewritten answer using MathVista-style answer extraction (regex + numeric tolerance 1e-3). Track reject rate (expect 20–40% early).

**Pitfalls:** FGPS answers can be expressions (`sqrt(3)/2`) not decimals — use `sympy.nsimplify`. The paper rewrites with Qwen2.5-72B-Instruct; you cannot run 72B locally. Use Qwen2.5-7B-Instruct or Qwen3-4B locally, or fall back to OpenAI gpt-4o-mini API.

---

#### Phase 4 — Diagram Rendering (~1.5–2 weeks — highest-risk phase)

**Goal:** Convert (construction_cdl + image_cdl) → high-fidelity PNG resembling a real textbook diagram.

**Two renderers, two notebooks:**
- **Baseline (Notebook 04):** clean matplotlib renderer. Points solved via `scipy.optimize.minimize` over sum-of-squares of CDL constraints. Output: clean but recognizable as synthetic.
- **High-fidelity (Notebook 05):** a GMBL-inspired renderer. Paper §2.4 builds on Krueger et al. 2021b's GMBL with a "formal language converter that automatically transforms construction CDL and image CDL statements … into GMBL formal language" via "a mapping table from the FormalGeo language to the GMBL language." For the educational repo, write a minimal version: numerical-optimization point placement + PIL stroke renderer with hand-tuned aesthetics.

**Why this matters:** Paper §1 explicitly identifies low-fidelity images as the main limitation of MAVIS-style rule engines: "the low fidelity of the synthesized images … resulting in a significant disparity from real geometric problems." Qwen2-VL's vision encoder is trained on natural images; cartoonish diagrams compound the domain shift.

**Baseline recipe:**
- White background, anti-aliased.
- 0.8 line width, RGBA black.
- 1–2 px jitter on label positions.
- Serif font (CMU-Serif) at variable 10–14 pt.
- Random ±5° rotation augmentation.
- Auxiliary lines: dashed, lighter gray.
- Resolution: 4:3 aspect, shorter edge in {112, 224, 336} px (matches paper §3.1).

**Pitfalls:** matplotlib's default Arial-sans-serif + thick blue/orange lines is a dead giveaway — override `rcParams` aggressively. Don't forget image_cdl annotations (length labels, angle marks). Detect/reject self-intersecting polygons. Don't oversample regular shapes.

---

#### Phase 5 — Natural Language Generation (~3–5 days)

**Two-step (matches paper §2.3.3):**
1. **Template draft:** For each FormalGeo predicate, 5–20 English NL templates. The paper uses "for each formal language expression in FormalGeo, we use GPT-4o to generate 20 corresponding natural language templates, which are then manually reviewed and corrected." Store as a Python dict in `templates.py`.
2. **LLM rewrite:** Pass draft through Qwen2.5-7B-Instruct (local) or gpt-4o-mini (API). Verbatim prompt from Appendix C: "Given a geometry problem and its answer hint, write a answer to the problem. Ensure the answer is correct, concise, easy to understand, and written with clarity and natural flow."

**Local rewriter on 5090:** Qwen2.5-7B-Instruct fp16 fits in ~16 GB; with vLLM ~100 samples/min → 10K in 2h, 20K in 4h.

**API cost:** gpt-4o-mini is $0.150/1M input tokens and $0.600/1M output tokens, per OpenAI's launch announcement (July 18, 2024): "Developers pay 15 cents per 1M input tokens and 60 cents per 1M output tokens" (unchanged through May 2026). Each sample ~500 in + 300 out tokens → ~$0.0003/sample → $3 for 10K, $6 for 20K. Trivial.

**Pitfalls:** Rewriter may hallucinate conditions → answer-verification (Phase 3) is non-negotiable. Pick one convention for ambiguous predicates ("∥" vs "is parallel to") and document.

---

#### Phase 6 — Dataset Assembly for Qwen2-VL (~2 days)

**Conversation format:**
```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "images/synth_00001.png"},
    {"type": "text", "text": "<NL problem statement>"}
  ]},
  {"role": "assistant", "content": "<verified NL solution>"}
]}
```

**Qwen2-VL specifics:** ChatML with `<|im_start|>` / `<|im_end|>` and `<|vision_start|><|image_pad|><|vision_end|>` for image slots. LLaMA-Factory's `template: qwen2_vl` handles this automatically. The vision token id is `151655`, vision_start is `151652`, vision_end is `151653`.

**Image preprocessing:** `min_pixels=256*28*28` (≈200K px), `max_pixels=1280*28*28` (≈1M px). For 336-px shorter edge, raise `max_pixels` to `1536*28*28`. Note the 28 = patch_size(14) × merge_size(2).

**Versioning:**
- `geofm-mini-5k` — smoke test (~1h gen).
- `geofm-mini-10k` — first real fine-tune (~2–3h gen).
- `geofm-mini-20k` — base educational reproduction (~4–6h gen).
- `geofm-mini-80k` — paper-scale, stretch goal.

**Synthetic-data license:** CC-BY-4.0 (or CC-BY-SA-4.0). Document seed provenance; consider not redistributing seeds.

---

#### Phase 7 — LoRA / QLoRA Fine-Tuning (~3–7 days of training time)

**Framework:** LLaMA-Factory's `examples/train_lora/qwen2vl_lora_sft.yaml`. Has Qwen2-VL DPO support added Sept 2024.

**VRAM budget on 32 GB 5090 (extrapolated from RTX 4090 24 GB published reports):**

| Model | Method | Approx VRAM | Per-device BS | Grad accum | Effective BS |
|---|---|---|---|---|---|
| Qwen2-VL-2B | full SFT (bf16) | ~22 GB | 2 | 8 | 16 |
| Qwen2-VL-2B | LoRA r=16 (bf16) | ~10 GB | 4 | 4 | 16 |
| Qwen2-VL-2B | QLoRA nf4 r=16 | ~7 GB | 8 | 2 | 16 |
| Qwen2-VL-7B | LoRA r=16 (bf16) | ~20 GB | 1 | 16 | 16 |
| Qwen2-VL-7B | QLoRA nf4 r=16 | ~12 GB | 2 | 8 | 16 |
| Qwen2-VL-7B | full SFT (bf16) | ~50 GB | — | — | OOM |

The Qwen2-VL-7B LoRA at ~20 GB is documented on RTX 4090 24 GB (Bhavya Joshi Medium, 2024).

**Start-point YAML:**
```yaml
model_name_or_path: Qwen/Qwen2-VL-2B-Instruct
template: qwen2_vl
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
dataset: geofm_mini_10k
cutoff_len: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 2
bf16: true
gradient_checkpointing: true
flash_attn: fa2
```

**Wall-clock estimates on single 5090** (the 5090's 1,792 GB/s bandwidth gives roughly 1.78× the 4090's 1,008 GB/s — a 78% improvement, per NVIDIA's RTX 5090 spec sheet and RTX 4090 datasheet, 384-bit GDDR6X bus):
- Qwen2-VL-2B LoRA, 10K samples, 2 epochs ≈ **3–5 hours**.
- Qwen2-VL-2B LoRA, 20K samples, 2 epochs ≈ **6–10 hours**.
- Qwen2-VL-7B LoRA, 10K samples, 2 epochs ≈ **10–18 hours**.
- Qwen2-VL-7B LoRA, 20K samples, 2 epochs ≈ **24–36 hours** (overnight × 2).

**Pitfalls:** Use `lora_target: all`, not the LLaMA q_proj+v_proj default. Don't freeze the vision tower entirely — small LR (1e-6) on the ViT helps the diagram domain shift. bf16 not fp16 on Blackwell (Blackwell tensor cores are bf16/fp8-first). Avoid Unsloth on Blackwell as the primary path.

---

#### Phase 8 — Evaluation (~1–2 days)

**Tool:** `open-compass/VLMEvalKit`. Supports MathVista (GPS subset), MathVerse, We-Math, GeoQA, native Qwen2-VL.

```bash
git clone https://github.com/open-compass/VLMEvalKit && cd VLMEvalKit && pip install -e .
python run.py --model Qwen2VL_2B_GeoFM \
              --data MathVista_MINI MathVerse_MINI_Vision_Only WeMath_MINI GeoQA \
              --judge gpt-4o-mini
```
Use `gpt-4o-mini` as the answer-extraction judge (~10× cheaper than `gpt-4`). Budget $2–5 per full eval run.

**Cost-free fallback:** regex-based answer extractor for MathVista-GPS (it's almost always A/B/C/D/E or a clean number).

**Reporting:** Always report `(base 2B, base 7B, GeoFM-edu-2B, GeoFM-edu-7B)` on MathVista-GPS and GeoQA. Optionally MathVerse Vision-Only (the hardest subset).

**Pitfalls:** Default MathVista judge is GPT-4 (expensive) — override in VLMEvalKit config. Use the official `lupantech/MathVista` testmini GPS subset (208 problems).

---

#### Phase 9 — Comparison & Ablation (~3 days)

**Ablations:** (1) **data-scale curve** (1K, 5K, 10K, 20K → MathVista-GPS plot — headline figure); (2) **formal sampling vs template-only** (turn off Algorithm 1's swap, train on seeds alone); (3) **renderer ablation** (matplotlib vs GMBL-style — likely 3–5pp gap); (4) **vision LR ablation** (freeze vs unfreeze ViT); (5) **LoRA rank sweep** (r=8/16/32/64). Use 2B for ablations to keep runs short (3–10h each).

---

### 3. Realistic Compute & Timeline

| Phase | Calendar | GPU hours |
|---|---|---|
| 0 — Env setup | 1 week | <1 |
| 1 — FormalGeo install | 0.5 week | 0 |
| 2 — Sampling | 1 week | 0 |
| 3 — Symbolic verify | 0.5 week | 0 |
| 4 — Diagram rendering | 1.5 weeks | 0 |
| 5 — NLG | 0.5 week | 2–4 |
| 6 — Dataset assembly | 0.5 week | 0 |
| 7 — Fine-tuning (2B + 7B) | 1 week training | 25–50 |
| 8 — Evaluation | 0.5 week | 5 |
| 9 — Ablations | 1.5 weeks | 50–80 |
| **Total** | **8–10 weeks** | **~150 GPU-hours** |

That is **2–2.5 months** at 20 focused hours/week, plus 50% Blackwell-detour buffer → realistically plan **3 months end-to-end** with a viable v0.1 (no ablations) at ~7 weeks.

---

### 4. RTX 5090 / Blackwell-specific Considerations (May 2026)

| Component | Status on sm_120 | Action |
|---|---|---|
| PyTorch | Stable ≥2.7 cu128; 2.12+ nightly recommended | NGC `nvcr.io/nvidia/pytorch:25.02-py3` or build from source |
| CUDA Toolkit | 12.8 min, 12.9 recommended | NOT 12.0/12.6 |
| Driver | 570+ Linux, 576+ Windows | Latest Game Ready/Studio |
| flash-attn v2 | Builds from source on sm_120 | `FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation` |
| flash-attn v4 | **Cannot run on sm_120** (TMEM absent — Dao-AILab #1665) | Use FA2 |
| bitsandbytes | No official Blackwell wheel; INT8 may produce garbage chars | Build from source; QLoRA = experimental |
| Unsloth | Blackwell issues (#1679) | Optional/experimental |
| xFormers | Build from source | Skip — PyTorch SDPA is fine |
| vLLM | ≥ v0.8.0 has Blackwell fp8 GEMM (PR #13798) | Use for fast eval |
| Triton | ≥ 3.2 Linux; triton-windows 3.5.x | Stick to Linux |
| transformers | ≥ 4.45.0 | Pin in requirements |
| FA3 backend in vLLM | Doesn't work on Blackwell | `VLLM_FLASH_ATTN_VERSION=2` |

**Recommended Dockerfile:**
```dockerfile
FROM nvcr.io/nvidia/pytorch:25.02-py3
RUN pip install transformers>=4.46 datasets accelerate peft trl
RUN pip install --no-build-isolation "flash-attn>=2.7.0" --no-cache-dir
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes \
    && cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cuda -S . && make && pip install -e .
RUN pip install llamafactory ms-swift vllm vlmevalkit
```

**Smoke tests:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_arch_list())"   # expect 'sm_120'
python -c "import torch; x=torch.randn(2,2,device='cuda'); print(x@x.T)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

---

### 5. Pedagogical Features

**Notebooks** (each ≤30 min runtime, ≤500 lines): 01 hello world → 02 parse+solve → 03 sample conditions → 04 matplotlib render → 05 GMBL-style render → 06 NLG → 07 build HF dataset → 08 LoRA train demo → 09 VLMEvalKit eval → 10 ablations + plots.

**Diagrams to include** (Excalidraw/draw.io SVG, editable):
- Headline **pipeline diagram** (seed → CDL → sampler → symbolic verify → renderer → NLG → train → eval).
- **Formal language example** side-by-side (NL ↔ CDL ↔ Solved CDL).
- **Symbolic verification trace** as a networkx tree.
- **Renderer comparison** (matplotlib vs GMBL-style).
- **Data-scale curve** (accuracy vs dataset size, 1K/5K/10K/20K).

**Docs:** MkDocs Material on GitHub Pages (~30-min setup) gives free search, dark mode, Mermaid diagrams.

**HuggingFace Spaces demo:** YES, Gradio template, Qwen2-VL-2B-GeoFM-LoRA on free A10G tier. 10–20 pre-loaded one-click examples. High-impact, low-cost.

**Blog post:** *"Reproducing Tencent's GeoFM on a single RTX 5090"* — 2,500–4,000 words, include the Blackwell-setup pain story (Medium-readers love it), final eval table, repo + Spaces links.

**Bridging formal language ↔ MLLM researchers:** First docs page bridges vocabulary — define "predicate", "CDL", "theorem application", "metric condition" in MLLM-friendly terms. Use the metaphor "FormalGeo is to geometry what a typed AST is to a programming language."

---

### 6. Dataset Release Strategy

**Recommended:** ship **both** generate-on-demand scripts AND a pre-generated HF dataset upload.

**HF dataset features:**
```python
DatasetDict({"train": Dataset(features={
    "id": Value("string"),
    "image": Image(),
    "problem": Value("string"),
    "solution": Value("string"),
    "answer": Value("string"),
    "construction_cdl": Value("string"),
    "image_cdl": Value("string"),
    "text_cdl": Value("string"),
    "goal_cdl": Value("string"),
    "theorem_seq": Sequence(Value("string")),
    "source_seed_pid": Value("int32"),
})})
```
Bundling the CDLs alongside the NL is the **educational killer feature** — readers can train alternative formats, run their own symbolic checks, ablate template vs LLM rewriting.

**Versions:** `geofm-mini-5k` / `-10k` / `-20k` published as separate dataset repos. License: **CC-BY-4.0**.

**Model checkpoint release:** YES — push LoRA adapters to HF as `Qwen2-VL-2B-GeoFM-edu-lora` and `Qwen2-VL-7B-GeoFM-edu-lora`. Adapters are 50–200 MB, don't carry base license obligations, give 60-second try-it.

---

### 7. Positioning & Differentiation

State in README intro, bold:

> **This repository is an educational, single-GPU reproduction of the data-generation methodology described in the GeoFM paper (Zhang et al., 2025, arXiv:2510.27448). It is not affiliated with the authors. The original paper has no public code release. We fine-tune Qwen2-VL-2B/7B with LoRA on a reduced 10–20K dataset, not the 80K dataset used in the paper. Our goal is pedagogical clarity, not state-of-the-art performance.**

| Project | MLLM | Data scale | Engine | Key insight |
|---|---|---|---|---|
| **GeoFM (paper)** | LLaVA-NeXT-8B, InternVL2-8B-MPO | 80K | Formal-language metric-combination + symbolic verify + GMBL diagrams | Symbolic-engine-verified synthesis |
| **DFE-GPS** | SigLIP-0.4B + Yi/Qwen-0.5B | 228K (SynthGeo228K) | Geometry Model Builder + ConsCDL captions | Diagram formalizer + projection module |
| **MAVIS** | CLIP-Math + Mammoth2-7B | 558K caption + 834K instruct | Rule-based engine (no GPT) | Math-specific CLIP encoder |
| **GeoX** | custom GeoCLIP + small LM | (various) | architecture-focused | New geometric pretraining objectives |
| **G-LLaVA** | LLaVA-1.5-7B | 170K (Geo170K) | ChatGPT rephrase + scale | First MLLM to beat GPT-4V on geometry |
| **geofm-edu (this repo)** | Qwen2-VL-2B/7B | 10–20K | Re-impl of GeoFM Algorithm 1, simplified renderer | Pedagogical clarity, single-5090 |

**Phrasing rules:** "Inspired by" / "re-implements the methodology of" / "educational reproduction (unofficial)". Avoid bare "reproduces."

---

### 8. Risks, Gotchas, Realistic Outcomes

1. **FormalGeo's predicate library has Chinese strings in places** — not in the predicate names but in some comments. Don't translate the code; write English docstrings over your wrapper API.
2. **Diagram rendering quality is your accuracy bottleneck.** Budget 1.5 weeks; iterate with visual diffs vs MathVista-GPS examples.
3. **Qwen2-VL dynamic-resolution + image_pad tokens** — call `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. LLaMA-Factory's `template: qwen2_vl` does this; verify by printing tokenized inputs.
4. **The chat template requires `<|vision_start|><|image_pad|><|vision_end|>` markers** — Qwen2-VL's chat_template.json contains them; LLaMA-Factory applies them automatically. If you bypass with a custom collator, copy verbatim.
5. **MathVista answer extraction with GPT-4 is expensive.** Switch to gpt-4o-mini in VLMEvalKit config (~10× cheaper).
6. **Reproducibility:** seed=42 everywhere, pin versions. Expect ±1pp variance on MathVista-GPS (testmini has only 208 GPS problems).
7. **bitsandbytes QLoRA on Blackwell** can produce garbage outputs (informatico-madrid Blackwell-Linux-Infra-Optimizer documents this). Ship LoRA-only as the primary path; QLoRA as experimental.

**Realistic outcome targets:**

| Model | Method | Base | Expected GeoFM-edu | Delta |
|---|---|---|---|---|
| Qwen2-VL-2B | LoRA on 10K | ~20% MathVista-GPS | ~30–35% | +10–15 pp |
| Qwen2-VL-2B | LoRA on 20K | ~20% | ~33–38% | data-scale gain |
| Qwen2-VL-7B | LoRA on 10K | ~40% | ~48–55% | +8–15 pp |
| Qwen2-VL-7B | LoRA on 20K | ~40% | ~50–58% | competitive w/ MAVIS-7B class |

Calibrated against the GeoFM paper's own ablation in §3.3.2: "GeoFM significantly outperforms MAVIS-Geometry, with an average improvement of 8.2% on MathVista-GPS and 11.1% on GeoQA" across 10K–80K data scales using LLaVA-NeXT-8B — scaled down ~50% for the smaller setup (Qwen2-VL-7B base ≠ InternVL2-8B-MPO base, LoRA ≠ full SFT, 10–20K ≠ 80K).

**Be honest in the README:** "We do not reach the paper's GeoFM-8B numbers. The point of this repo is to teach the *method*, not chase SOTA."

---

### 9. Open Questions to Flag for the User

1. **Other base models (InternVL2-2B/4B, Phi-3.5-Vision-4.2B)** — both 5090-friendly, both have LLaMA-Factory recipes. Doubles eval matrix. **Recommendation:** defer to v1.1.
2. **DPO / RLHF on top of SFT** — LLaMA-Factory has Qwen2-VL DPO (Sept 2024). Symbolic-correctness as implicit reward is natural. **Recommendation:** v1.1, add ~2 weeks.
3. **Diagram captioning as pre-training** (MAVIS-Caption pattern, 558K caption-diagram pairs). Probably overkill for an educational repo. **Recommendation:** note as future direction only.
4. **vLLM for fast inference/eval** — YES. vLLM ≥ v0.8.0 has Blackwell fp8 GEMM. Document the Blackwell setup as a docs page.
5. **Qwen2.5-VL or Qwen3-VL as alternative bases** — drop-in upgrades with the same chat template, stronger baselines. **Recommendation:** ship both Qwen2-VL and Qwen2.5-VL configs from day one.

---

## Recommendations (staged action plan)

**Week 1 (env):** Get the Docker image to pass `verify_env.py`. Don't touch anything else until green. **If Blackwell rebuild hell exceeds 5 days, fall back to renting a RunPod A100 hour ($1–2) to validate the rest of the pipeline.**

**Weeks 2–3 (FormalGeo + sampler):** Notebooks 01–03 running. Generate 100 synthetic problems; visually inspect 20; socialize on issue tracker.

**Weeks 4–5 (rendering):** Iterate until synthetic diagrams pass the "looks like a textbook" eye test against MathVista-GPS samples. Notebook 05 to >80% subjective quality.

**Week 6 (data + training v0):** Generate `geofm-mini-5k`, fine-tune Qwen2-VL-2B LoRA, run VLMEvalKit. Any positive delta validates the pipeline.

**Weeks 7–8 (scale + 7B):** Generate `geofm-mini-10k` and `-20k`. Fine-tune 2B and 7B LoRA. Full eval.

**Weeks 9–10 (ablations + polish):** Run 5 ablations, write blog, push Gradio Spaces, mint Zenodo DOI, write dataset card.

**Decision thresholds:**
- **If 2B-LoRA-on-10K gives <5pp delta vs base on MathVista-GPS** → data-quality problem (rendering or NL templates). Stop scaling; debug data.
- **If Blackwell setup takes >2 weeks** → escape-hatch to RunPod A100.
- **If bitsandbytes Blackwell instability blocks QLoRA** → drop QLoRA from v1.0 scope, ship LoRA-only.

---

## Caveats

1. **The paper has no public code release as of May 2026.** All Algorithm 1 implementation details here are derived from the paper's pseudocode and prose. Edge cases (timeout policy, randomization seed, empty `M_p`) are your implementation choices — document them.
2. **The paper's diagram pipeline depends on GMBL** (Krueger et al. 2021b). GMBL's source code availability is uncertain — if not public, you re-implement from the GMBL paper. Plan a "good matplotlib" baseline as MVP; treat GMBL-style as a stretch.
3. **Blackwell software stack is still maturing as of May 2026.** Expect breaking changes in bitsandbytes, flash-attention, Unsloth wheels over the next 3 months. Pin versions in `requirements/blackwell.txt`; document snapshot date.
4. **Eval-judge cost is non-trivial.** Full VLMEvalKit run across MathVista + MathVerse + We-Math + GeoQA with gpt-4o-mini judge costs $2–5/run. Budget ~$50 across all eval runs.
5. **The "expected gain" numbers are calibrated estimates, not guarantees.** Solo-researcher reproductions of recent papers commonly see 50–70% of the headline gain.
6. **License compatibility audit needed** before publishing the synthetic dataset. FormalGeo7K and PGPS9K have their own licenses; check before re-bundling.
7. **No Qwen2-VL benchmark for GeoFM exists.** The paper's GeoFM-8B = InternVL2-8B-MPO + 80K + full SFT on an H20 96 GB GPU. Your Qwen2-VL-7B + 20K + LoRA result is not directly comparable; do not advertise it as such.