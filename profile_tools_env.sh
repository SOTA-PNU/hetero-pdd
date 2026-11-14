#!/usr/bin/env bash
# Source in each CI job; install once using --install in prepare-run.
PDD_TOOLS_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PDD_TOOLS_HASH=$(sha256sum "$PDD_TOOLS_REPO/tools/hetero/third_party/versions.env")
PDD_TOOLS_ROOT="$HOME/.cache/pdd-tools/${PDD_TOOLS_HASH%% *}"
if [[ "${1:-}" == --install && ! -f "$PDD_TOOLS_ROOT/installed" ]]; then
    (
        set -euo pipefail
        # Pinned wheels target Python 3.10; use the existing CUDA interpreter.
        export PATH="$(dirname -- "$CUDA_PY"):$PATH"
        bash "$PDD_TOOLS_REPO/tools/hetero/third_party/make_offline_bundle.sh" --out "$PDD_TOOLS_ROOT/bundle"
        bash "$PDD_TOOLS_REPO/tools/hetero/third_party/install_offline.sh" \
            --bundle "$PDD_TOOLS_ROOT/bundle" --venv "$PDD_TOOLS_ROOT/venv" --tools-dir "$PDD_TOOLS_ROOT/bin"
        touch "$PDD_TOOLS_ROOT/installed"
    ) || return 1
fi
export ALLSMI_BIN="$PDD_TOOLS_ROOT/bin/all-smi"
export OTELCOL_BIN="$PDD_TOOLS_ROOT/bin/otelcol-contrib"
export PERFETTO_TP_BIN="$PDD_TOOLS_ROOT/bin/trace_processor_shell"
export PDD_TOOLS_PY="$PDD_TOOLS_ROOT/venv/bin/python"
