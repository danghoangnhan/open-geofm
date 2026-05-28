#!/usr/bin/env bash
# Run VLMEvalKit against an open-geofm checkpoint (Phase 8).
# Defers to src/open_geofm/eval/vlmevalkit_run.sh.

set -euo pipefail

exec bash "$(dirname "$0")/../src/open_geofm/eval/vlmevalkit_run.sh" "$@"
