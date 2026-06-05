"""Trainer interface (CPU-light; the concrete impl lives in _trl_backend.py)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Trainer(Protocol):
    """Run SFT for a resolved config against a dataset."""

    def train(
        self, *, dataset: str, dataset_split: str = "train", push_to_hub: bool = False
    ) -> None: ...
