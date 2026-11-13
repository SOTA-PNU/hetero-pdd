#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(realpath "$here/../../..")
out=
python_version=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
while (($#)); do
    case "$1" in
        --out) out=$2; shift 2 ;;
        --python-version) python_version=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done
: "${out:?Usage: make_offline_bundle.sh --out DIR [--python-version X.Y]}"
out=$(realpath -m -- "$out")
case "$out/" in "$repo/"*) echo 'Output must be outside repository' >&2; exit 1 ;; esac
source "$here/versions.env"
mkdir -p "$out/artifacts" "$out/wheels" "$out/licenses"
cp "$here/versions.env" "$out/versions.env"
printf '%s\n' "$PYTHON_REQUIREMENTS" > "$out/requirements.txt"
printf 'pip==%s\nsetuptools==%s\n' "$PIP_VERSION" "$SETUPTOOLS_VERSION" > "$out/bootstrap.txt"
printf '%s\n' "$python_version" > "$out/python-version"
while read -r digest filename url; do
    file="$out/artifacts/$filename"
    if [[ ! -f "$file" ]]; then
        curl --fail --location --retry 2 --output "$file" "$url"
    fi
    printf '%s  %s\n' "$digest" "$file" | sha256sum --check --status
done <<< "$ARTIFACTS"
python3 -m pip download --no-cache-dir --only-binary=:all: \
    --python-version "$python_version" --implementation cp \
    --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
    --dest "$out/wheels" --requirement "$out/requirements.txt" \
    --requirement "$out/bootstrap.txt"
export WHEEL_SHA256
python3 - "$out" <<'PY'
import hashlib
import os
import sys
import zipfile
from pathlib import Path

out = Path(sys.argv[1])
pins = dict(line.split()[::-1] for line in os.environ['WHEEL_SHA256'].splitlines())
for wheel in sorted((out / 'wheels').glob('*.whl')):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != pins.get(wheel.name):
        raise SystemExit(f'Unpinned wheel or SHA256 mismatch: {wheel.name}')
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.endswith('/') and any(word in name.upper() for word in ('LICENSE', 'LICENCE', 'COPYING', 'NOTICE')):
                dest = out / 'licenses' / wheel.stem / name.replace('/', '__')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(name))
licenses = list((out / 'artifacts').glob('LICENSE.*'))
(out / 'LICENSE').write_bytes(b'\n\n'.join(p.name.encode() + b'\n' + p.read_bytes() for p in licenses))
(out / 'NOTICE').write_text(
    'Pinned upstream release archives; see versions.env for versions, URLs and hashes.\n'
    'Binary licenses: LICENSE; Python dependency licenses/notices: licenses/.\n'
    'No separate top-level NOTICE is published by these pinned binary projects.\n'
)
with (out / 'SHA256SUMS').open('w') as manifest:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            manifest.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}\n')
PY
echo "Offline bundle ready: $out"
