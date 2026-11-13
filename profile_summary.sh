#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${CUDA_PY:?Set CUDA_PY}" -B "$SCRIPT_DIR/tools/hetero/profiler/ci_summary.py" "$@"
