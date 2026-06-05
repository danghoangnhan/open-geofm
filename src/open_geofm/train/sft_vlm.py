"""TRL SFTTrainer driver for Qwen2-VL / Qwen2.5-VL LoRA fine-tuning.

Blueprint §2 Phase 7. The CPU-importable surface lives here (config resolution,
ChatML conversion, dataset resolution, the Typer CLI, `--dry-run`); the heavy
model-load + training lives in `_trl_backend.py` (eager torch/transformers/trl/
peft) and is reached via `importlib` only on the non-dry-run path — so `--help`
and `--dry-run` never pull the GPU stack.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import typer
from PIL import Image as PILImage

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = logging.getLogger("open_geofm.train")

_TRL_BACKEND = "open_geofm.train._trl_backend"
_CONFIG_MAP = {
    "qwen2vl_2b_lora": "open_geofm.train.configs.qwen2vl_2b_lora",
    "qwen2vl_7b_lora": "open_geofm.train.configs.qwen2vl_7b_lora",
    "qwen25vl_7b_lora": "open_geofm.train.configs.qwen25vl_7b_lora",
    "qwen2vl_7b_qlora": "open_geofm.train.configs.qwen2vl_7b_qlora",
}


def load_config(name: str) -> dict[str, Any]:
    """Resolve a config name to its kwargs dict. Raises SystemExit on miss."""
    if name not in _CONFIG_MAP:
        raise SystemExit(f"Unknown --config {name!r}. Known: {sorted(_CONFIG_MAP)}")
    return importlib.import_module(_CONFIG_MAP[name]).get()


def _make_vision_tower_lr_trainer_cls(vision_lr: float):
    """Delegate to the eager TRL backend. Raises ImportError when trl/torch are
    absent (the host venv), as the existing test asserts."""
    return importlib.import_module(_TRL_BACKEND)._make_vision_tower_lr_trainer_cls(vision_lr)


# ---------------------------------------------------------------------------
# Dataset loading (CPU-importable; `datasets` reached via importlib)
# ---------------------------------------------------------------------------


def _record_to_messages(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one open-geofm record into Qwen2-VL ChatML."""
    img = record.get("image")
    if isinstance(img, str):
        img = PILImage.open(img).convert("RGB")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": record.get("problem", "")},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record.get("solution", "")}],
            },
        ]
    }


def _resolve_dataset(spec: str, *, split: str = "train"):
    """Load a `datasets.Dataset` from a hub id or local directory."""
    datasets = importlib.import_module("datasets")
    p = Path(spec)
    if p.is_dir() and (p / "dataset_info.json").exists():
        ds = datasets.load_from_disk(str(p))
        if not isinstance(ds, datasets.Dataset):
            ds = ds[split] if split in ds else ds[next(iter(ds))]
        return ds
    if p.is_dir() and (p / "records.jsonl").exists():
        return datasets.Dataset.from_json(str(p / "records.jsonl"))
    return datasets.load_dataset(spec, split=split)


def _load_train_dataset(spec: str, *, split: str = "train"):
    """Resolve `spec` and lazily transform records to ChatML (no arrow blow-up)."""
    ds = _resolve_dataset(spec, split=split)

    def _batched(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(next(iter(batch.values())))
        out_messages: list[Any] = []
        for i in range(n):
            rec = {k: batch[k][i] for k in batch}
            out_messages.append(_record_to_messages(rec)["messages"])
        return {"messages": out_messages}

    return ds.with_transform(_batched)


def _write_dry_run_manifest(config: str, dataset: str, cfg: dict[str, Any]) -> None:
    """Write a JSON manifest summarising the resolved config (no GPU work)."""
    out_dir = Path(cfg["sft_config"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "dataset": dataset,
        "model_name_or_path": cfg["model_name_or_path"],
        "sft_config": cfg["sft_config"],
        "lora_config": cfg["lora_config"],
        "processor_kwargs": cfg.get("processor_kwargs", {}),
        "quantization": cfg.get("quantization"),
        "vision_tower_lr": cfg.get("vision_tower_lr"),
        "attn_implementation": cfg.get("attn_implementation"),
    }
    (out_dir / "dry_run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("dry-run manifest written to %s", out_dir / "dry_run.json")


@app.command()
def train(
    config: str = typer.Option(..., help="Config name (see _CONFIG_MAP)."),
    dataset: str = typer.Option("open-geofm-mini-10k", help="HF dataset id or local dir."),
    dataset_split: str = typer.Option("train", help="Split name for HF-hub datasets."),
    max_steps: int = typer.Option(-1, help="Override SFTConfig.max_steps; -1 = config/epoch."),
    push_to_hub: bool = typer.Option(False, help="Push the adapter after training."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve config + write manifest only."),
) -> None:
    """Run SFT with the selected config."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(config)
    if max_steps and max_steps > 0:
        cfg["sft_config"]["max_steps"] = max_steps

    if dry_run:
        _write_dry_run_manifest(config, dataset, cfg)
        return

    # Heavy training is deferred to the eager TRL backend (imported only here).
    importlib.import_module(_TRL_BACKEND).run_training(
        cfg, dataset=dataset, dataset_split=dataset_split, push_to_hub=push_to_hub
    )


if __name__ == "__main__":
    app()
