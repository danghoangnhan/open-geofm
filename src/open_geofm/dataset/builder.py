"""Build an HF `datasets.Dataset` from rendered synthetic samples.

Blueprint §2 Phase 6. Features (CDLs bundled alongside NL — the educational
killer feature):

    {
      "id": str,
      "image": Image,
      "problem": str,
      "solution": str,
      "answer": str,
      "construction_cdl": list[str],
      "image_cdl": list[str],
      "text_cdl": list[str],
      "goal_cdl": str,
      "theorem_seq": list[str],
      "source_seed_pid": int,
    }

Ships as three HF dataset repos: `open-geofm-mini-5k`, `-10k`, `-20k`. CC-BY-4.0.

The `image_dir` argument resolves images by sample id (``f"{image_dir}/{id}.png"``).
The builder doesn't render — Phase 4 renderers write images first, then the
builder zips them with the verified text records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..sampling.algorithm1 import SyntheticSample


def _dedup(seq):
    """Preserve order, drop duplicates. FormalGeo7K seeds sometimes list the
    same metric in both text_cdl and image_cdl; the Algorithm 1 swap propagates
    that into `new_metrics`, and we dedupe before emitting the dataset record."""
    seen: set = set()
    out: list = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _sample_to_record(sample: SyntheticSample, image_dir: Path, idx: int) -> dict:
    sample_id = f"openfm-{sample.source_pid:05d}-{idx:04d}"
    added = set(sample.added_metrics)
    image_cdl = _dedup(m for m in sample.new_metrics if m in added)
    text_cdl = _dedup(m for m in sample.new_metrics if m not in added)
    return {
        "id": sample_id,
        "image": str(image_dir / f"{sample_id}.png"),
        "problem": sample.nl_problem or "",
        "solution": sample.nl_solution or "",
        "answer": sample.answer,
        "construction_cdl": _dedup(sample.construction_cdl),
        # The driver split P_new into text+image; we keep both here so downstream
        # filters can recover the split. `new_metrics` is the union, by design.
        "image_cdl": image_cdl,
        "text_cdl": text_cdl,
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
):
    """Materialize `samples` + rendered images at `image_dir` into an HF dataset
    saved to `out_dir`.

    Args:
        samples: verified Algorithm 1 outputs.
        image_dir: directory containing one PNG per sample (named by `id`).
        out_dir: where `datasets.Dataset.save_to_disk` writes the result.
        image_loader: optional ``str -> PIL.Image`` callback (defaults to PIL's
            lazy loader via `datasets.Image()`); useful for unit tests with no
            real PNGs on disk.
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [_sample_to_record(s, image_dir, i) for i, s in enumerate(samples)]
    # Always write a JSONL sidecar — it's the introspectable copy for users
    # without `datasets` installed (e.g. the host venv).
    jsonl_path = out_dir / "records.jsonl"
    with jsonl_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    try:
        from datasets import (  # type: ignore[import-not-found]
            Dataset,
            Features,
            Image,
            Sequence,
            Value,
        )
    except ImportError:
        # No `datasets` in the host venv → JSONL is the artifact.
        return {"records_jsonl": str(jsonl_path), "n_records": len(records)}

    features = Features(
        {
            "id": Value("string"),
            "image": Image(),
            "problem": Value("string"),
            "solution": Value("string"),
            "answer": Value("string"),
            "construction_cdl": Sequence(Value("string")),
            "image_cdl": Sequence(Value("string")),
            "text_cdl": Sequence(Value("string")),
            "goal_cdl": Value("string"),
            "theorem_seq": Sequence(Value("string")),
            "source_seed_pid": Value("int32"),
        }
    )
    ds = Dataset.from_list(records, features=features)
    if image_loader is not None:
        ds = ds.map(lambda r: {"image": image_loader(r["image"])})
    ds.save_to_disk(str(out_dir))
    return ds
