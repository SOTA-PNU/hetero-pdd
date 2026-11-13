#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(realpath "$here/../../..")
bundle= venv= tools_dir=
while (($#)); do
    case "$1" in
        --bundle) bundle=$2; shift 2 ;;
        --venv) venv=$2; shift 2 ;;
        --tools-dir) tools_dir=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
: "${bundle:?--bundle is required}" "${venv:?--venv is required}" "${tools_dir:?--tools-dir is required}"
bundle=$(realpath -m -- "$bundle")
venv=$(realpath -m -- "$venv")
tools_dir=$(realpath -m -- "$tools_dir")
for path in "$bundle" "$venv" "$tools_dir"; do
    case "$path/" in "$repo/"*) echo 'Paths must be outside repository' >&2; exit 1 ;; esac
done
(cd "$bundle" && sha256sum --check --quiet SHA256SUMS)
cmp "$here/versions.env" "$bundle/versions.env"
source "$here/versions.env"
while read -r digest filename url; do
    printf '%s  %s\n' "$digest" "$bundle/artifacts/$filename" | sha256sum --check --status
done <<< "$ARTIFACTS"
python_version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ "$python_version" == "$(cat "$bundle/python-version")" ]] || {
    echo 'Use a python3 matching the bundle target version on PATH' >&2; exit 1;
}
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --no-cache-dir --no-index --upgrade \
    --find-links "$bundle/wheels" "pip==$PIP_VERSION" "setuptools==$SETUPTOOLS_VERSION"
"$venv/bin/python" -m pip install --no-cache-dir --no-index \
    --find-links "$bundle/wheels" --requirement "$bundle/requirements.txt"
mkdir -p "$tools_dir"
tar -xzf "$bundle/artifacts/all-smi.tar.gz" -C "$tools_dir" ./all-smi ./liball_smi_amd.so
tar -xzf "$bundle/artifacts/otelcol.tar.gz" -C "$tools_dir" otelcol-contrib
unzip -p "$bundle/artifacts/perfetto.zip" linux-amd64/trace_processor_shell > "$tools_dir/trace_processor_shell"
chmod +x "$tools_dir/all-smi" "$tools_dir/otelcol-contrib" "$tools_dir/trace_processor_shell"
export ALLSMI_BIN="$tools_dir/all-smi" OTELCOL_BIN="$tools_dir/otelcol-contrib" PERFETTO_TP_BIN="$tools_dir/trace_processor_shell"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo" "$venv/bin/python" - <<'PY'
from tools.hetero.profiler.tools import check_tools
for tool, version in check_tools('deep').items():
    print(f'{tool}: {version}')
PY
printf 'export PDD_TOOLS_PY=%q\nexport ALLSMI_BIN=%q\nexport OTELCOL_BIN=%q\nexport PERFETTO_TP_BIN=%q\n' \
    "$venv/bin/python" "$ALLSMI_BIN" "$OTELCOL_BIN" "$PERFETTO_TP_BIN"
