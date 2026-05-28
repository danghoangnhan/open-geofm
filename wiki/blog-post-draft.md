# Reproducing Tencent's GeoFM on a single RTX 5090 — a 10-week tour

*Draft of the companion blog post. Lives in the wiki/ for review; once the
first real eval run lands, the numbers in §7 get swapped from placeholder to
real and the post moves to Medium / the wiki home page.*

---

## TL;DR

I spent ten weeks (well — eight weeks of work plus two weeks of CUDA-12.8
tantrums) re-implementing the data-generation methodology from Tencent
Hunyuan's *GeoFM* paper (Zhang et al., 2025,
[arXiv:2510.27448](https://arxiv.org/abs/2510.27448)) on a single RTX 5090.
The full code, ten Jupyter notebooks, a wiki, and a Hugging Face Space
demoing the trained adapter live at
[github.com/CallMeDaniel/open-geofm](https://github.com/CallMeDaniel/open-geofm).

You are not going to match Tencent's headline numbers. They had 80,000
samples and a full-parameter SFT of LLaVA-NeXT-8B / InternVL2-8B-MPO on an
H20 96 GB GPU. We have 10,000 samples and a LoRA on Qwen2-VL-{2B,7B}. The
point is to teach the *method* — the symbolic-engine-verified synthetic
data pipeline — not to crawl up the MathVista leaderboard.

Realistic outcome: **+10 to +15 percentage points** over base Qwen2-VL-2B on
MathVista-GPS and GeoQA, **+8 to +15 pp** for the 7B. The Blackwell software
stack will eat the first two weeks of your calendar; budget accordingly.

## What this is, and isn't

| | open-geofm | original GeoFM |
|---|---|---|
| Base MLLM | Qwen2-VL-{2B,7B}, Qwen2.5-VL-7B | LLaVA-NeXT-8B, InternVL2-8B-MPO |
| Fine-tune method | LoRA r=16, all-linear, bf16 | Full-parameter SFT |
| Data scale | 5K / 10K / 20K (CC-BY-4.0 release) | 80K |
| Compute | One RTX 5090 (32 GB, sm_120) | H20 96 GB |
| Diagram renderer | matplotlib baseline + GMBL re-impl | Tencent's internal GMBL engine |

The README has a full "what this IS / IS NOT" table; this post is for the
*story*.

---

## Phase 0 — the Blackwell pain story

This is the part nobody warns you about.

The RTX 5090 ships as `sm_120`. As of mid-2026 the open-source ML stack
supports it, but the wheels aren't all on PyPI. Here is the actual sequence
of debugging-the-environment days, in the order they happened:

* **Day 1.** `pip install torch` → `torch.cuda.get_arch_list()` returns
  `['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']`. No
  `sm_120`. Switching to the PyPI cu128 wheels (`pip install
  torch --index-url https://download.pytorch.org/whl/cu128`) gets you sm_120
  but pins the rest of the stack to whatever the cu128 image expects.
* **Day 2.** `bf16 matmul on cuda:0` succeeds. Things look great. Then
  `bitsandbytes` falls over: the PyPI wheel has no sm_120 kernels. Fine —
  build from source (`cmake -DCOMPUTE_BACKEND=cuda -S . && make`). Builds
  in ~10 minutes, imports cleanly. Looks great. *Right up until you
  actually run a 4-bit forward pass and get garbage tokens.* Other people
  have hit this — `bitsandbytes-foundation/bitsandbytes` issue #1642 has
  been open since November 2025. The kernels are there but the de-quant
  path on sm_120 has a regression somewhere subtle.
* **Day 3.** OK, drop QLoRA from the v1.0 scope. LoRA only. Gate QLoRA
  behind an opt-in env var
  (`OPEN_GEOFM_ENABLE_QLORA=1` raises a `RuntimeError` otherwise) so
  nobody silently ships a corrupt model.
* **Day 4.** Flash-Attention. FA2 builds from source for sm_120 with
  `FLASH_ATTENTION_FORCE_BUILD=TRUE`. FA4 *cannot* run on sm_120 — the
  silicon lacks the TMEM opcodes (`UTMA*` / `UTC*`) that FA4's kernels
  require. There's a great deep-dive at
  [gau-nernst's blog](https://gau-nernst.github.io) and a detailed
  Dao-AILab issue (#1665) but the upshot is: pin FA2, set
  `VLLM_FLASH_ATTN_VERSION=2`, move on.
* **Day 5.** Unsloth. Their Triton path doesn't have Blackwell wheels yet
  (#1679). LLaMA-Factory or ms-swift, both work. We picked LLaMA-Factory.
* **Day 6.** Pin everything in `requirements/blackwell.txt` and write
  `scripts/00_verify_env.py` — the smoke test that runs every time you
  pull. It prints `torch.cuda.get_arch_list()`, runs a bf16 matmul, tries
  `import bitsandbytes`, and does a one-step Qwen2-VL-2B LoRA on a dummy
  batch. If any of those fails, the script exits non-zero and the rest of
  the CI is skipped.
* **Days 7–10.** Docker. The NGC PyTorch image
  (`nvcr.io/nvidia/pytorch:25.02-py3`) is the one stable base — CUDA
  12.8, PyTorch 2.6.0a0+ with sm_120 pre-baked. Build it once, run all GPU
  phases inside it. The host venv stays small (CPU phases only).

Documenting all of this lives in
[wiki/07-Blackwell-Setup-Log.md](07-Blackwell-Setup-Log.md). It's the
single most-linked-to page on the wiki, ahead of even the headline method
walk-through. The lesson: if you're chasing a paper that was published
6 months ago on Hopper, expect 1–2 weeks of pure infra work before you
can train anything.

---

## Phases 1 & 2 — FormalGeo and the metric swap

This is what the paper actually contributes — *symbolically-verified
synthetic data*. The vision encoder, the LLM, the LoRA, the rendering —
all commodity. The novel piece is **Algorithm 1**, which I will reproduce
verbatim:

```
Input: formalized seed problem set FS, number of synthetic problems m
for P in FS:
    M_p   = MetricInfoOfProblemStatement(P)
    M_all = GatheringMetricInfo(P)            # BFS over theorems
    m_p = m
    while m_p > 1:
        n = Random(1, min(|M_p|, |M_all| − |M_p|))
        M_del = RandomSelect(M_p, n)
        M_add = RandomSelect(M_all \ M_p, n)
        P_new = (P \ M_del) ∪ M_add
        A_new = FormalGeoSolver(P_new)
        P_syn, A_syn = Template_and_LLM(P_new, A_new)
        if AnswerVerify(A_syn, A_new): S.add((P_syn, A_syn))
return S
```

In English: take a textbook problem. List its metric conditions (segment
lengths, angle measures, etc.). Run the formal solver forward over the
theorem library to expand the *implicit* facts on the figure into an
explicit list. Now randomly swap `n` of the textbook's metrics for `n` of
the implied ones, pick a new goal, and ask the solver to verify a new
answer. If it can, you have a new problem with the same diagram and a
verified solution.

It's beautiful. It's also slow. The solver (FGPS, BitSecret's open-source
implementation) hangs on some pathological seeds. Empty `M_p` skip. The
BFS over the 234-theorem GDL takes 30 seconds per seed on a modern CPU.
Our integration test fixes the timeout at 90 seconds; the production
driver uses 30 s and drops the few seeds that overrun.

Three engineering decisions worth flagging:

* **A `goal_picker` with fallback-from-trace.** When the random goal pick
  yields an unsolvable problem, fall back to "the last derived metric from
  FGPS's reasoning path." The paper buries this in one sentence; the
  fallback raises the accept rate from ~40% to ~75% in our hands.
* **Two passes of NL generation.** Predicate templates (5–6 English
  variants per predicate; one sentence per condition) draft the problem
  statement. Then an LLM rewriter — local vLLM-served Qwen2.5-7B-Instruct
  or the OpenAI gpt-4o-mini API, selected by `OPEN_GEOFM_REWRITER` — smooths
  the draft into prose. Both backends use the paper's Appendix-C system
  prompt verbatim.
* **A sympy-based answer verifier.** The LLM rewriter can and does
  hallucinate; the verifier extracts the last numeric token from the
  NL solution, sympifies both sides, and compares with `tol=1e-3`. The
  reject rate is 20–40% in the first generation pass. Without this gate,
  we'd be training on garbage.

Code: `src/open_geofm/{formal,sampling,nlg}/`. Walk-through:
[notebook 03](../notebooks/03_sample_metric_conditions.ipynb).

---

## Phase 4 — the renderer

The paper barely mentions this, but it's the most under-estimated phase
on the project. From the GeoFM paper §1:

> "...the low fidelity of the synthesised images ... resulting in a
> significant disparity from real geometric problems."

If your renderer looks cartoonish, the eval suffers. Qwen2-VL's ViT was
pre-trained on natural images; a 0.8-px black stroke on white paper at
224 px looks closer to the textbook diagrams than matplotlib's default
Arial-sans-serif at 4-px line width and pastel colours.

We ship two renderers, side by side:

* **matplotlib baseline** (`render.matplotlib_renderer`) — no constraint
  solving. Points placed on a regular n-gon, then rotated ±5° for
  augmentation. CMU-Serif labels with 1-2 px jitter. White background.
  Fast: ~30 ms per 224-px image.
* **GMBL-style** (`render.gmbl_renderer`) — the paper builds on Krueger
  et al. 2021b's GMBL with a custom mapping table from FormalGeo CDL to
  GMBL constraints. GMBL's source isn't publicly mirrored, so we
  re-implement the idea: translate `LengthOfLine`, `MeasureOfAngle`,
  `Parallel`, `Perpendicular`, `Collinear`, `Polygon` into a residual
  objective, minimise with `scipy.optimize.minimize` (L-BFGS-B), render
  with PIL strokes. Slow: ~200 ms per 336-px image, but the figures
  actually *look like* the textbook.

The Phase 9 ablation (matplotlib vs GMBL, same data, same training)
quantifies how much this matters. We expect a +3–5 pp gap on
MathVista-GPS in favour of the GMBL-style renderer. *(Real numbers
forthcoming once the eval run lands.)*

---

## Phase 6 — the dataset

The bundled-CDL schema is what I think makes this educational, not just
re-implementational. Each record carries the NL problem + solution
*plus* the four CDL blocks (construction / text / image / goal) plus
the theorem trace plus the source seed PID:

```json
{
  "id": "openfm-00200-0000",
  "image": "images/openfm-00200-0000.png",
  "problem": "In triangle ABC, ...",
  "solution": "AC = sqrt(161) ≈ 12.69.",
  "answer": "sqrt(161)",
  "construction_cdl": ["Shape(ABC)"],
  "text_cdl": ["Equal(LengthOfLine(AB),...)", "..."],
  "image_cdl": ["Equal(MeasureOfAngle(...),...)"],
  "goal_cdl": "Value(x)",
  "theorem_seq": ["('right_triangle_property', '1', ('A','B','C'))"],
  "source_seed_pid": 200
}
```

A reader who only wants the NL + image can ignore the CDL columns. A
reader who wants to train alternative formats, run their own symbolic
checks, or ablate template-vs-LLM rewriting has every piece they need
without re-generating images.

Three sizes ship on the Hub: `open-geofm-mini-{5,10,20}k`. CC-BY-4.0.

---

## Phase 7 — LoRA, the actual training

The simplest piece, finally. TRL's `SFTTrainer` + `peft.LoraConfig`,
`target_modules="all-linear"`. The four configs live in
`src/open_geofm/train/configs/`:

| Config | Base | VRAM (5090) | Wall-clock (10K, 2 ep) |
|---|---|---|---|
| `qwen2vl_2b_lora` | Qwen2-VL-2B-Instruct | ~10 GB | 3–5 h |
| `qwen2vl_7b_lora` | Qwen2-VL-7B-Instruct | ~20 GB | 10–18 h |
| `qwen25vl_7b_lora` | Qwen2.5-VL-7B-Instruct | ~20 GB | 10–18 h |
| `qwen2vl_7b_qlora` | Qwen2-VL-7B + nf4 | ~12 GB | 8–14 h *(gated)* |

Three load-bearing details:

* **`max_length=None`.** Truncation slices through the
  `<|vision_start|>...<|vision_end|>` spans around each image's 200K–1M
  patch tokens, corrupting the batch. Lock it in at the config level.
* **`target_modules="all-linear"`.** The default LLaMA recipe targets
  only the attention projections. That under-fits Qwen2-VL's MLPs and
  the cross-attention pooler on the vision side.
* **`vision_tower_lr=1e-6`.** Two orders of magnitude under the LLM-side
  LR. Freezing the ViT entirely loses signal on the synthetic-image
  domain shift; a full LR destabilises it. The `VisionTowerLRTrainer`
  subclass in `train/sft_vlm.py` splits parameter groups.

A dry-run path writes a manifest without importing torch — useful as a
CI gate. The actual training runs inside the Docker image:

```bash
docker compose run --rm train \
    bash scripts/03_train.sh qwen2vl_2b_lora \
        --dataset data/open-geofm-mini-10k
```

---

## Phases 8 & 9 — evaluation and ablations

VLMEvalKit handles MathVista, MathVerse, We-Math, and GeoQA in one
config. Two cost knobs:

* **`gpt-4o-mini` as the judge**, not `gpt-4`. Cuts the cost ~10x;
  full eval is $2–5 per pass.
* **Regex extractor for MathVista-GPS** as the $0 fallback. The MCQ
  answers are A–E or a clean number ≥95% of the time; our extractor
  catches both with reasonable last-line / last-number heuristics. The
  test suite covers the tricky edge cases (the trailing-letter ambiguity
  that confuses naïve regexes).

The Phase 9 ablation harness slices a VLMEvalKit work-dir along three
documented axes by parsing model-name suffix tokens:

* `_5k` / `_10k` / `_20k` — data-scale curve.
* `_mpl` / `_gmbl` — renderer ablation.
* `_r8` / `_r16` / `_r32` / `_r64` — LoRA rank sweep.

Each ablation emits both a Markdown table and a matplotlib line plot.
`scripts/05_ablate.py` is the CLI; notebook 10 is the pedagogical
walk-through. The same code reads real eval results — we just don't
have them yet.

**Forthcoming numbers (placeholder, calibrated against the paper's §3.3
ablation):**

| Model | Method | Base | open-geofm-edu | Δ |
|---|---|---|---|---|
| Qwen2-VL-2B | LoRA on 10K | ~20% MathVista-GPS | ~30–35% | +10–15 pp |
| Qwen2-VL-7B | LoRA on 10K | ~40% | ~48–55% | +8–15 pp |

That's it. No SOTA. The point is the *method*, plus a complete reference
implementation that runs on hardware a hobbyist can actually buy.

---

## Lessons

* **Pin everything.** The Blackwell stack moves fast; a wheel that
  builds today may have a regression tomorrow. `requirements/blackwell.txt`
  pins versions with a snapshot date in the comment.
* **Buy 30 s of margin per second of work.** Per-cell timeouts (notebook
  cells, FGPS solves, BFS), retry budgets (`max_attempts_factor` on
  Algorithm 1), graceful fallbacks (`fallback_goal_from_trace` when the
  random goal is unsolvable). The pipeline is allowed to skip a few
  pathological seeds; the alternative is the whole 10K run hanging on one
  bad input.
* **Two backends, one interface.** The local Qwen2.5-7B-Instruct vLLM
  rewriter and the OpenAI gpt-4o-mini rewriter share a 3-line `Rewriter`
  protocol. Same for the renderers (matplotlib + GMBL). Cuts the
  ablation surface in half and means the test suite can validate the
  *interface* once.
* **Document the disagreement, not just the agreement.** PID=1 doesn't
  solve cleanly with FGPS's backward searcher (180 s timeout); PID=4 and
  PID=10 do. The notebooks point this out explicitly so the next person
  doesn't waste an afternoon. Same for the bitsandbytes garbage-output
  issue on sm_120 — gated behind an opt-in env var with a comment that
  *explains why the gate exists*.

---

## What's next, and how to contribute

The v1.0 ships with the pipeline, the 10 notebooks, the wiki, and the
Gradio Spaces demo. v1.1 adds:

* The real eval run (and the ablation figures with real numbers).
* DPO on top of SFT — LLaMA-Factory has Qwen2-VL DPO; symbolic
  correctness as implicit reward is a natural fit.
* InternVL2-2B / Phi-3.5-Vision as alternative bases (a `Qwen2-VL ↔
  InternVL2` comparison would be the headline figure of a v1.1 post).

Contributions welcome — see `CONTRIBUTING.md`. The most-requested
next-steps tracked in
[issues](https://github.com/CallMeDaniel/open-geofm/issues) are
diagram-augmentation policies, alternative base models, and a
data-scale extrapolation beyond 20K.

If you read this far, the wiki is the canonical reference and the
notebooks are the runnable companion. The Spaces demo is the 30-second
try-it. Citation block (CC-BY-4.0):

```bibtex
@software{open_geofm,
  title  = {open-geofm: An educational reproduction of GeoFM on a single RTX 5090},
  author = {open-geofm contributors},
  year   = {2026},
  url    = {https://github.com/CallMeDaniel/open-geofm}
}
```

And — most importantly — also cite the original:

```bibtex
@article{zhang2025geofm,
  title  = {GeoFM: Geometry Foundation Model with Formal-Language Data Synthesis},
  author = {Zhang, et al.},
  journal= {arXiv:2510.27448},
  year   = {2025}
}
```

— *open-geofm contributors, 2026*
