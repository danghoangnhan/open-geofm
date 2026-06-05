"""VLMEvalKit-backed evaluator (eager `vlmeval` import).

Imported only via the registry when the `eval` extra is present; never part of
the CPU/base import graph. VLMEvalKit loads Qwen2-VL/2.5-VL through transformers
internally, so transformers is implicitly the eval backbone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import EvalConfig
from .compare import BenchmarkScore, scan_work_dir


class VLMEvalKitEvaluator:
    """Drive a VLMEvalKit run, then parse its work-dir into scores.

    `run_cmd` is the argv used to invoke VLMEvalKit (defaults to the project's
    `vlmevalkit_run.sh` wrapper). After the run, `evaluate()` scans `work_dir`.
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        *,
        work_dir: Path | str = "outputs/eval",
        run_cmd: list[str] | None = None,
    ) -> None:
        self.config = config or EvalConfig()
        self.work_dir = Path(work_dir)
        self.run_cmd = run_cmd

    def run(self) -> None:
        if self.run_cmd:
            subprocess.run(self.run_cmd, check=True)

    def evaluate(self) -> list[BenchmarkScore]:
        self.run()
        return scan_work_dir(self.work_dir)
