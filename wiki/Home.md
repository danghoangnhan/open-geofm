# open-geofm wiki

Pedagogical documentation for [`open-geofm`](https://github.com/CallMeDaniel/open-geofm),
an educational, single-RTX-5090 reproduction of *GeoFM* (Zhang et al., 2025, arXiv:2510.27448).

> **This is not the official GeoFM code.** See the [README](https://github.com/CallMeDaniel/open-geofm#readme) for the IS / IS NOT table.

## Contents

* [00 — Overview](./00-Overview) — Pipeline diagram, repo tour, where to start.
* [01 — Formal Language](./01-Formal-Language) — CDL (Conditional Declaration Language) for the geometry-curious.
* [02 — Condition Sampling](./02-Condition-Sampling) — Algorithm 1 in detail, the novel part.
* [03 — Symbolic Verification](./03-Symbolic-Verification) — FGPS, answer extraction, reject rates.
* [04 — Diagram Rendering](./04-Diagram-Rendering) — Matplotlib baseline vs. GMBL-style renderer.
* [05 — Qwen2-VL Fine-tuning](./05-Qwen2VL-Finetuning) — TRL SFTTrainer, LoRA configs, VRAM budget.
* [06 — Evaluation](./06-Evaluation) — VLMEvalKit, MathVista-GPS, GeoQA, judge cost.
* [07 — Blackwell Setup Log](./07-Blackwell-Setup-Log) — Running pain-log: error → fix → upstream issue.
* [Blog post draft](./blog-post-draft) — *Reproducing Tencent's GeoFM on a single RTX 5090* — narrative companion to the wiki.

## How to read this wiki

If you're here to **understand the method**, start at [00 — Overview](./00-Overview), then read 01 → 02 → 03 → 04 → 05 in order.

If you're here to **run the code on your own Blackwell GPU**, jump straight to [07 — Blackwell Setup Log](./07-Blackwell-Setup-Log) — that page is where every "wheel didn't build" / "garbage output" landmine is documented.

If you're here to **extend the work** (DPO, new base models, larger data), start at [05 — Qwen2-VL Fine-tuning](./05-Qwen2VL-Finetuning) — the TRL `SFTTrainer` driver is the simplest hook point.
