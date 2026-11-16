#!/usr/bin/env bash
# Source in each CI job; install once using --install in prepare-run.
PDD_TOOLS_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PDD_TOOLS_HASH=$(sha256sum "$PDD_TOOLS_REPO/tools/hetero/third_party/versions.env")
PDD_TOOLS_ROOT="$HOME/.cache/pdd-tools/${PDD_TOOLS_HASH%% *}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH="$PDD_TOOLS_REPO"
pdd_install() {
    "$CUDA_PY" - "$PDD_TOOLS_REPO" "$@" <<'PYTHON'
import os, sys
from tools.pdd.runner import SAVED_ENV, execution_env
runtime = {"REPO": sys.argv[1]}
runtime.update({"ENV_" + k: os.environ[k] for k in SAVED_ENV if k in os.environ})
os.execvpe("bash", ["bash", *sys.argv[2:]], execution_env(runtime))
PYTHON
}
if [[ "${1:-}" == --install && ! -f "$PDD_TOOLS_ROOT/installed" ]]; then
    (
        set -euo pipefail
        # Pinned wheels target Python 3.10; use the existing CUDA interpreter.
        export PATH="$(dirname -- "$CUDA_PY"):$PATH"
        pdd_install "$PDD_TOOLS_REPO/tools/hetero/third_party/make_offline_bundle.sh" --out "$PDD_TOOLS_ROOT/bundle"
        pdd_install "$PDD_TOOLS_REPO/tools/hetero/third_party/install_offline.sh" \
            --bundle "$PDD_TOOLS_ROOT/bundle" --venv "$PDD_TOOLS_ROOT/venv" --tools-dir "$PDD_TOOLS_ROOT/bin"
        touch "$PDD_TOOLS_ROOT/installed"
    ) || return 1
fi
export ALLSMI_BIN="$PDD_TOOLS_ROOT/bin/all-smi"
export OTELCOL_BIN="$PDD_TOOLS_ROOT/bin/otelcol-contrib"
export PERFETTO_TP_BIN="$PDD_TOOLS_ROOT/bin/trace_processor_shell"
export PDD_TOOLS_PY="$PDD_TOOLS_ROOT/venv/bin/python"
