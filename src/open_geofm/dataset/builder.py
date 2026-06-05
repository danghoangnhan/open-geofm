"""Build an HF dataset from rendered synthetic samples (+ a JSONL sidecar).

Blueprint §2 Phase 6. The sample id is computed in ONE place (`make_sample_id`)
and used by BOTH the renderer (PNG filename) and this builder (record image
path), so they always agree — fixing the renderer/builder id mismatch that left
every record's `image` pointing at a non-existent file. The id is a stable,
salt-free SHA-1 (not Python's `PYTHONHASHSEED`-salted `hash()`), so PNG names are
reproducible across processes and runs.

The heavy `datasets` dependency lives in `_hf_backend.py` (imported via importlib
only when present); without it, the JSONL sidecar is the artifact.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterable
from pathlib import Path

from ..config import DatasetConfig
from ..sampling.algorithm1 import SyntheticSample

_HF_BACKEND = "open_geofm.dataset._hf_backend"


def make_sample_id(sample: SyntheticSample, config: DatasetConfig | None = None) -> str:
    """Deterministic, salt-free sample id shared by renderer + builder.

    Derived from (source_pid, goal, added_metrics) — the tuple that uniquely
    identifies an accepted swap — so the same sample always maps to the same id
    regardless of process or run.
    """
    config = config or DatasetConfig()
    digest = hashlib.sha1(
        "|".join((str(sample.source_pid), sample.goal, *sample.added_metrics)).encode()
    ).hexdigest()
    return (
        f"{config.id_prefix}-{sample.source_pid:0{config.pid_pad}d}-{digest[: config.idx_pad + 4]}"
    )


def _dedup(seq: Iterable[str]) -> list[str]:
    """Preserve order, drop duplicates."""
    return list(dict.fromkeys(seq))


def _sample_to_record(sample: SyntheticSample, image_dir: Path, config: DatasetConfig) -> dict:
    sample_id = make_sample_id(sample, config)
    added = set(sample.added_metrics)
    return {
        "id": sample_id,
        "image": str(image_dir / f"{sample_id}{config.image_ext}"),
        "problem": sample.nl_problem or "",
        "solution": sample.nl_solution or "",
        "answer": sample.answer,
        "construction_cdl": _dedup(sample.construction_cdl),
        "image_cdl": _dedup(m for m in sample.new_metrics if m in added),
        "text_cdl": _dedup(m for m in sample.new_metrics if m not in added),
        "goal_cdl": sample.goal,
        "theorem_seq": list(sample.theorem_seqs),
        "source_seed_pid": sample.source_pid,
    }


def build(
    samples: Iterable[SyntheticSample],
    image_dir: Path,
    out_dir: Path,
    *,
    image_loader=None,
    config: DatasetConfig | None = None,
):
    """Materialize `samples` + rendered images into an HF dataset at `out_dir`.

    Always writes a JSONL sidecar. If `datasets` is installed, also builds and
    saves a `datasets.Dataset`; otherwise the JSONL dict is the return value.
    """
    config = config or DatasetConfig()
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [_sample_to_record(s, image_dir, config) for s in samples]
    jsonl_path = out_dir / config.records_filename
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    try:
        backend = importlib.import_module(_HF_BACKEND)
    except ImportError:
        return {"records_jsonl": str(jsonl_path), "n_records": len(records)}
    return backend.build_hf_dataset(records, out_dir, image_loader=image_loader)


class HFDatasetBuilder:
    """Concrete `DatasetBuilder` (HF dataset + JSONL sidecar)."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def build(self, samples, image_dir, out_dir, *, image_loader=None):
        return build(samples, image_dir, out_dir, image_loader=image_loader, config=self.config)
