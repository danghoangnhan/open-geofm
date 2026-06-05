"""HuggingFace `datasets` backend for the dataset builder.

`datasets` is imported eagerly at module top. This module is imported only via
`importlib` from `builder.build` when `datasets` is present, so it never enters
the CPU/base import graph.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Image, Sequence, Value  # type: ignore[import-not-found]

_FEATURES = Features(
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


def build_hf_dataset(records: list[dict], out_dir: Path, *, image_loader=None) -> Dataset:
    """Build + save a `datasets.Dataset` from record dicts.

    When `image_loader` (a ``str -> PIL.Image``) is given, it is applied to the
    raw path string BEFORE the `Image()` feature decodes — the loader receives a
    path, matching its contract (fix for the image_loader-after-cast bug, #23).
    """
    if image_loader is not None:
        records = [{**r, "image": image_loader(r["image"])} for r in records]
    ds = Dataset.from_list(records, features=_FEATURES)
    ds.save_to_disk(str(out_dir))
    return ds
