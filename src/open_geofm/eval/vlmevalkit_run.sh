#!/usr/bin/env bash
# Eval an open-geofm LoRA checkpoint against MathVista-GPS, MathVerse, We-Math, GeoQA.
# Blueprint §2 Phase 8.
#
# Usage:
#   bash src/open_geofm/eval/vlmevalkit_run.sh <model-config> <ckpt-path>
#
# Judge defaults to gpt-4o-mini (~10x cheaper than gpt-4). Override with $JUDGE.

set -euo pipefail

MODEL_CONFIG="${1:?usage: $0 <model-config> <ckpt-path>}"
CKPT="${2:?usage: $0 <model-config> <ckpt-path>}"
JUDGE="${JUDGE:-gpt-4o-mini}"

export VLMEVALKIT_MODEL_PATH="${CKPT}"
export VLLM_FLASH_ATTN_VERSION=2  # Blueprint §4: FA3 doesn't work on Blackwell.

uv run python -m vlmeval.run \
  --model "${MODEL_CONFIG}" \
  --data MathVista_MINI MathVerse_MINI_Vision_Only WeMath_MINI GeoQA \
  --judge "${JUDGE}" \
  --work-dir "outputs/eval/${MODEL_CONFIG}"
