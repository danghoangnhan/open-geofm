"""Typed configuration tree for the whole open-geofm pipeline.

This module is the *single source of truth* for every tunable literal in the
codebase — timeouts, search depths, scaling factors, resolutions, model ids,
tolerances, paths, colours. Nothing downstream should hardcode a magic number;
it reads the relevant field off an `AppConfig` instance instead.

Design rules:
  * Every group is a frozen pydantic-v2 model (`frozen=True, extra='forbid'`),
    so configs are immutable and typo-proof.
  * Field defaults come from module-level `DEFAULT_*` named constants, so the
    literal has exactly one home and can be referenced symbolically.
  * The module imports only pydantic + stdlib + pyyaml — it stays CPU-clean and
    never pulls torch / formalgeo / vllm into the import graph.

`AppConfig.from_yaml(path)` loads + validates a config file; omitted groups fall
back to their defaults, so a minimal YAML only needs to override what differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Named constants — the one home for every default literal.
# ---------------------------------------------------------------------------

# paths / dataset
DEFAULT_DATA_ENV_VAR = "OPEN_GEOFM_DATA"
DEFAULT_DATA_DIRNAME = "data"
DEFAULT_DATASET_NAME = "formalgeo7k_v2"
DEFAULT_PROBLEM_COUNT_KEY = "problem_number"
DEFAULT_FIRST_PID = 1
DEFAULT_LAST_PID = 7000
DEFAULT_CDL_COMMENT_PREFIX = "#"

# FGPS solver
DEFAULT_SOLVE_STRATEGY = "backward"
DEFAULT_FGPS_SEARCH_STRATEGY = "bfs"
DEFAULT_MAX_DEPTH = 15
DEFAULT_BEAM_SIZE = 20
DEFAULT_SOLVE_TIMEOUT_S = 15.0
DEFAULT_UNUSED_GOAL_CDL = "Value(unused)"
DEFAULT_STUB_ANSWER = "0"
DEFAULT_SEARCHER_CACHE_SIZE = 4
DEFAULT_LOADER_CACHE_SIZE = 4
DEFAULT_INTERACTOR_CACHE_SIZE = 2

# sampling / Algorithm 1
DEFAULT_RNG_SEED = 42
DEFAULT_M_PER_SEED = 3
DEFAULT_MAX_ATTEMPTS_FACTOR = 20
DEFAULT_MIN_ATTEMPTS = 1
DEFAULT_BFS_MAX_DEPTH = 1
DEFAULT_BFS_TIMEOUT_S = 30.0
DEFAULT_TRACE_ARROW = "->"

# verify
DEFAULT_VERIFY_TOL = 1e-3

# NLG
DEFAULT_NLG_CLIENT = "vllm"
DEFAULT_NLG_MODE = "template"
DEFAULT_LOCAL_NLG_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPENAI_NLG_MODEL = "gpt-4o-mini"
DEFAULT_NLG_TEMPERATURE = 0.7
DEFAULT_NLG_MAX_TOKENS = 512
DEFAULT_NLG_DEVICE_MAP = "cuda"
DEFAULT_NLG_DTYPE = "bfloat16"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
# Appendix C of the paper, verbatim.
DEFAULT_NLG_SYSTEM_PROMPT = (
    "Given a geometry problem and its answer hint, write a answer to the problem. "
    "Ensure the answer is correct, concise, easy to understand, and written with "
    "clarity and natural flow."
)

# render — shared
DEFAULT_RESOLUTION_CHOICES = (112, 224, 336)
DEFAULT_INK_RGB = (20, 20, 20)
DEFAULT_BG_RGB = (252, 250, 245)
# render — matplotlib baseline
DEFAULT_MPL_RESOLUTION = 224
DEFAULT_MPL_DPI = 100
DEFAULT_AXIS_LIMIT = 1.5
DEFAULT_ROTATION_DEG = 5.0
DEFAULT_STROKE_WIDTH = 0.8
DEFAULT_LAYOUT_RADIUS = 1.0
DEFAULT_FONT_FAMILY = "serif"
DEFAULT_VERTEX_FONT_PT = 12
DEFAULT_LENGTH_FONT_PT = 10
DEFAULT_ANGLE_FONT_PT = 9
DEFAULT_LABEL_JITTER = 0.02
DEFAULT_VERTEX_LABEL_DY = 0.06
DEFAULT_EDGE_LABEL_OFFSET = 0.08
DEFAULT_ANGLE_LABEL_OFFSET = 0.12
DEFAULT_PAD_INCHES = 0.1
DEFAULT_DEGREE_SYMBOL = "°"
# render — GMBL textbook
DEFAULT_GMBL_RESOLUTION = 336
DEFAULT_FONT_FILE = "DejaVuSerif.ttf"
DEFAULT_FONT_PT_DIVISOR = 22
DEFAULT_FONT_PT_MIN = 11
DEFAULT_LENGTH_LAYOUT_SCALE = 5.0
DEFAULT_LAYOUT_FILL_FRACTION = 0.85
DEFAULT_INIT_JITTER = 0.05
DEFAULT_OPTIMIZER_METHOD = "L-BFGS-B"
DEFAULT_CANVAS_MARGIN = 0.45  # to_px: 0.5 ± margin*coord
DEFAULT_VERTEX_RADIUS_DIVISOR = 80
DEFAULT_GMBL_LABEL_OFFSET_PX = 8

# dataset
DEFAULT_ID_PREFIX = "openfm"
DEFAULT_PID_PAD = 5
DEFAULT_IDX_PAD = 4
DEFAULT_IMAGE_EXT = ".png"
DEFAULT_RECORDS_FILENAME = "records.jsonl"

# train
DEFAULT_ATTN_IMPL = "flash_attention_2"
DEFAULT_ENABLE_QLORA_ENV = "OPEN_GEOFM_ENABLE_QLORA"
DEFAULT_IMAGE_PAD_TOKENS = ("<|image_pad|>", "<|video_pad|>")

# eval
DEFAULT_PREFERRED_METRIC_KEYS = (
    "Overall",
    "overall",
    "Accuracy",
    "accuracy",
    "Average",
    "average",
    "score",
    "Score",
)
DEFAULT_SCORE_GLOB = "*_score.json"
DEFAULT_SCORE_SUFFIX = "_score"
DEFAULT_MISSING_CELL = "—"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_RENDERER_AXIS_VALUES = {"mpl": 0.0, "gmbl": 1.0}


class _Frozen(BaseModel):
    """Base for every config group: immutable + reject unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class PathsConfig(_Frozen):
    """Filesystem roots. Resolves the FormalGeo7K data root."""

    data_root: Path | None = None
    data_env_var: str = DEFAULT_DATA_ENV_VAR
    data_dirname: str = DEFAULT_DATA_DIRNAME

    def resolved_data_root(self, env: dict[str, str]) -> Path:
        """Explicit `data_root` → ``$OPEN_GEOFM_DATA`` → repo ``data/`` default.

        `env` is passed in (typically ``os.environ``) so this stays a pure
        function of its inputs and is trivially testable.
        """
        if self.data_root is not None:
            return Path(self.data_root)
        from_env = env.get(self.data_env_var)
        if from_env:
            return Path(from_env)
        # config.py lives at src/open_geofm/config.py → repo root is parents[2].
        return Path(__file__).resolve().parents[2] / self.data_dirname


class FormalGeoConfig(_Frozen):
    """FormalGeo7K corpus + CDL grammar settings."""

    dataset_name: str = DEFAULT_DATASET_NAME
    problem_count_key: str = DEFAULT_PROBLEM_COUNT_KEY
    first_pid: int = DEFAULT_FIRST_PID
    last_pid: int = DEFAULT_LAST_PID
    comment_prefix: str = DEFAULT_CDL_COMMENT_PREFIX


class FGPSConfig(_Frozen):
    """FGPS forward/backward symbolic-search hyperparameters."""

    strategy: Literal["forward", "backward"] = DEFAULT_SOLVE_STRATEGY
    search_strategy: str = DEFAULT_FGPS_SEARCH_STRATEGY
    max_depth: int = DEFAULT_MAX_DEPTH
    beam_size: int = DEFAULT_BEAM_SIZE
    timeout_s: float = DEFAULT_SOLVE_TIMEOUT_S
    debug: bool = False
    theorem_usage_stats: dict[str, Any] = Field(default_factory=dict)
    unused_goal_cdl: str = DEFAULT_UNUSED_GOAL_CDL
    stub_answer: str = DEFAULT_STUB_ANSWER
    searcher_cache_size: int = DEFAULT_SEARCHER_CACHE_SIZE


class SamplingConfig(_Frozen):
    """Algorithm 1 + M_all BFS knobs."""

    rng_seed: int = DEFAULT_RNG_SEED
    m_per_seed: int = DEFAULT_M_PER_SEED
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR
    min_attempts: int = DEFAULT_MIN_ATTEMPTS
    bfs_max_depth: int = DEFAULT_BFS_MAX_DEPTH
    bfs_timeout_s: float = DEFAULT_BFS_TIMEOUT_S
    trace_arrow: str = DEFAULT_TRACE_ARROW


class VerifyConfig(_Frozen):
    """Symbolic answer-verification tolerance."""

    tol: float = DEFAULT_VERIFY_TOL


class NlgConfig(_Frozen):
    """NL generation: template draft + optional LLM smoothing backend."""

    client: Literal["vllm", "hf", "openai"] = DEFAULT_NLG_CLIENT
    mode: Literal["template", "llm"] = DEFAULT_NLG_MODE
    model: str = DEFAULT_LOCAL_NLG_MODEL
    temperature: float = DEFAULT_NLG_TEMPERATURE
    max_tokens: int = DEFAULT_NLG_MAX_TOKENS
    device_map: str = DEFAULT_NLG_DEVICE_MAP
    dtype: str = DEFAULT_NLG_DTYPE
    attn_implementation: str = DEFAULT_ATTN_IMPL
    system_prompt: str = DEFAULT_NLG_SYSTEM_PROMPT
    openai_api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class RenderConfig(_Frozen):
    """Diagram renderer settings (covers both matplotlib + GMBL)."""

    renderer: Literal["mpl", "gmbl"] = "mpl"
    resolution: int = DEFAULT_MPL_RESOLUTION
    resolution_choices: tuple[int, ...] = DEFAULT_RESOLUTION_CHOICES
    ink_rgb: tuple[int, int, int] = DEFAULT_INK_RGB
    bg_rgb: tuple[int, int, int] = DEFAULT_BG_RGB
    # matplotlib baseline
    dpi: int = DEFAULT_MPL_DPI
    axis_limit: float = DEFAULT_AXIS_LIMIT
    rotation_deg: float = DEFAULT_ROTATION_DEG
    stroke_width: float = DEFAULT_STROKE_WIDTH
    layout_radius: float = DEFAULT_LAYOUT_RADIUS
    font_family: str = DEFAULT_FONT_FAMILY
    vertex_font_pt: int = DEFAULT_VERTEX_FONT_PT
    length_font_pt: int = DEFAULT_LENGTH_FONT_PT
    angle_font_pt: int = DEFAULT_ANGLE_FONT_PT
    label_jitter: float = DEFAULT_LABEL_JITTER
    vertex_label_dy: float = DEFAULT_VERTEX_LABEL_DY
    edge_label_offset: float = DEFAULT_EDGE_LABEL_OFFSET
    angle_label_offset: float = DEFAULT_ANGLE_LABEL_OFFSET
    pad_inches: float = DEFAULT_PAD_INCHES
    degree_symbol: str = DEFAULT_DEGREE_SYMBOL
    # GMBL textbook
    font_file: str = DEFAULT_FONT_FILE
    font_pt_divisor: int = DEFAULT_FONT_PT_DIVISOR
    font_pt_min: int = DEFAULT_FONT_PT_MIN
    length_layout_scale: float = DEFAULT_LENGTH_LAYOUT_SCALE
    layout_fill_fraction: float = DEFAULT_LAYOUT_FILL_FRACTION
    init_jitter: float = DEFAULT_INIT_JITTER
    optimizer_method: str = DEFAULT_OPTIMIZER_METHOD
    canvas_margin: float = DEFAULT_CANVAS_MARGIN
    vertex_radius_divisor: int = DEFAULT_VERTEX_RADIUS_DIVISOR
    gmbl_label_offset_px: int = DEFAULT_GMBL_LABEL_OFFSET_PX

    @classmethod
    def matplotlib_baseline(cls, **overrides: Any) -> RenderConfig:
        return cls(renderer="mpl", resolution=DEFAULT_MPL_RESOLUTION, **overrides)

    @classmethod
    def gmbl_textbook(cls, **overrides: Any) -> RenderConfig:
        return cls(renderer="gmbl", resolution=DEFAULT_GMBL_RESOLUTION, **overrides)


class DatasetConfig(_Frozen):
    """HF dataset assembly settings (also drives the sample-id scheme)."""

    id_prefix: str = DEFAULT_ID_PREFIX
    pid_pad: int = DEFAULT_PID_PAD
    idx_pad: int = DEFAULT_IDX_PAD
    image_ext: str = DEFAULT_IMAGE_EXT
    records_filename: str = DEFAULT_RECORDS_FILENAME


class SFTSubConfig(_Frozen):
    """Kwargs forwarded verbatim to `trl.SFTConfig` (extra keys allowed)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    output_dir: str
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_length: int | None = None  # VLM: do not truncate image-token spans
    logging_steps: int = 10
    save_strategy: str = "epoch"
    report_to: str = "none"


class LoraSubConfig(_Frozen):
    """Kwargs forwarded verbatim to `peft.LoraConfig`."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class QuantSubConfig(_Frozen):
    """Kwargs for `transformers.BitsAndBytesConfig` (QLoRA path)."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# Image-processor pixel bounds: 28 = patch_size(14) * merge_size(2).
_QWEN_PATCH = 28
DEFAULT_MIN_PIXELS = 256 * _QWEN_PATCH * _QWEN_PATCH
DEFAULT_MAX_PIXELS = 1280 * _QWEN_PATCH * _QWEN_PATCH


class TrainConfig(_Frozen):
    """One SFT run config (replaces the train/configs/*.py dict modules)."""

    model_name_or_path: str
    sft_config: SFTSubConfig
    lora_config: LoraSubConfig = Field(default_factory=LoraSubConfig)
    processor_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"min_pixels": DEFAULT_MIN_PIXELS, "max_pixels": DEFAULT_MAX_PIXELS}
    )
    attn_implementation: str = DEFAULT_ATTN_IMPL
    vision_tower_lr: float | None = 1e-6
    quantization: QuantSubConfig | None = None
    enable_qlora_env: str = DEFAULT_ENABLE_QLORA_ENV
    image_pad_tokens: tuple[str, ...] = DEFAULT_IMAGE_PAD_TOKENS

    def to_legacy_dict(self) -> dict[str, Any]:
        """Render to the plain-dict shape the legacy `configs/*.get()` returned,
        so the existing train driver + tests keep working unchanged."""
        out: dict[str, Any] = {
            "model_name_or_path": self.model_name_or_path,
            "sft_config": self.sft_config.model_dump(),
            "lora_config": self.lora_config.model_dump(),
            "processor_kwargs": dict(self.processor_kwargs),
            "attn_implementation": self.attn_implementation,
        }
        if self.vision_tower_lr is not None:
            out["vision_tower_lr"] = self.vision_tower_lr
        if self.quantization is not None:
            out["quantization"] = self.quantization.model_dump()
        return out


class EvalConfig(_Frozen):
    """VLMEvalKit score parsing + ablation-axis settings."""

    preferred_metric_keys: tuple[str, ...] = DEFAULT_PREFERRED_METRIC_KEYS
    score_glob: str = DEFAULT_SCORE_GLOB
    score_suffix: str = DEFAULT_SCORE_SUFFIX
    missing_cell: str = DEFAULT_MISSING_CELL
    judge_model: str = DEFAULT_JUDGE_MODEL
    renderer_axis_values: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_RENDERER_AXIS_VALUES)
    )


class PipelineConfig(_Frozen):
    """Which stages run, and which backend each selects."""

    stages: tuple[str, ...] = ("sample", "render", "build")
    source_backend: str = "formalgeo7k"
    solver_backend: str = "fgps"
    gatherer_backend: str = "formalgeo"
    renderer_backend: str = "mpl"
    llm_client_backend: str = "vllm"
    verifier_backend: str = "sympy"
    builder_backend: str = "hf"


class AppConfig(_Frozen):
    """Root config tree. Every group defaults, so a partial YAML is valid."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    formalgeo: FormalGeoConfig = Field(default_factory=FormalGeoConfig)
    fgps: FGPSConfig = Field(default_factory=FGPSConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    nlg: NlgConfig = Field(default_factory=NlgConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: Path | str, *, overrides: dict[str, Any] | None = None) -> AppConfig:
        """Load + validate a YAML config. Omitted groups use defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        if overrides:
            data = {**data, **overrides}
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> AppConfig:
        """An all-defaults config (the in-memory equivalent of an empty YAML)."""
        return cls()
