"""Phase 9 ablation harness.

Walks `eval.compare.scan_work_dir` output (the headline table) and slices it
along one of the documented axes — data-scale (5K/10K/20K), renderer
(matplotlib vs GMBL), LoRA rank, etc. Each ablation parses model names of
the shape ``<base_config>_<axis_token>`` (e.g. ``qwen2vl_2b_lora_10k``) and
emits both a Markdown table and a matplotlib line plot.

Blueprint §2 Phase 9. CPU-only; no torch / vllm imports.
"""

from .sweep import SweepPoint, plot_sweep, scale_curve, to_markdown_sweep

__all__ = ["SweepPoint", "scale_curve", "plot_sweep", "to_markdown_sweep"]
