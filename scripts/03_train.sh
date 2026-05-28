#!/usr/bin/env bash
# Train one of the four supported LoRA configs (Phase 7).
# Wraps `python -m open_geofm.train.sft_vlm`. Run inside the Docker image.

set -euo pipefail

CONFIG="${1:?usage: $0 <config> [extra train args...]}"
shift || true

exec uv run python -m open_geofm.train.sft_vlm \
    --config "${CONFIG}" \
    "$@"
