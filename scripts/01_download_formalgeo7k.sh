#!/usr/bin/env bash
# Download FormalGeo7K v2 (~521 MB) into the repo's data/ directory.
#
# Idempotent: if `data/formalgeo7k_v2/` already exists, the script just prints a
# notice and exits. Pass `OPEN_GEOFM_DATA=/some/path` to install elsewhere.
#
# Blueprint §2 Phase 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${OPEN_GEOFM_DATA:-${REPO_ROOT}/data}"
DATASET="${1:-formalgeo7k_v2}"

if [ -d "${DATA_DIR}/${DATASET}" ]; then
  echo "[skip] ${DATA_DIR}/${DATASET} already present."
  exit 0
fi

mkdir -p "${DATA_DIR}"
echo "[download] ${DATASET} -> ${DATA_DIR}"
uv run python -c "
from formalgeo.data.data import download_dataset
download_dataset('${DATASET}', '${DATA_DIR}')
print('done')
"
