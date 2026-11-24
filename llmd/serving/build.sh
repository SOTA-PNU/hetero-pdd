#!/usr/bin/env bash
# Build official runner and strict upstream PD sidecar. Outputs stay user-local.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${1:-${LLMD_BIN_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/hetero-pdd/llmd-v1/bin}}
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
exec 8>"$dest/.build.lock"
flock 8
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
router_rev=b9393d1a9f1d5a72ecd86f8d2b643ab54c2a9b92
case "$(uname -sm)" in
  'Linux x86_64') arch=amd64; envoy_arch=x86_64; crane_arch=x86_64
    go_sha=ecbadb99091a3f46e31f5f934b068b1864eafa7995211b39eaddf76996045fe5
    envoy_sha=bd2ff87c5efc95fd7880d710915f573c1aab4d9c55d8da886dcd91df43fad741
    crane_sha=36c67a932f489b3f2724b64af90b599a8ef2aa7b004872597373c0ad694dc059 ;;
  'Linux aarch64') arch=arm64; envoy_arch=aarch_64; crane_arch=arm64
    go_sha=94f3e30b8e374bc285e7dadc11e0865726b9bc6e85b841ccceaabc0214c6b7c8
    envoy_sha=81ec3689a82122eff0ca680c48176f4d351b6f9881a79cc9d9078a6fb4b0b6a8
    crane_sha=d2235f7779cd39c6e40f43701d2512c997409f629fb53e621ede0d57d3f995e2 ;;
  *) echo 'The released local bundle supports Linux amd64 and arm64.' >&2; exit 1 ;;
esac
fetch() {
  curl --fail --location --silent --show-error --retry 3 "$1" --output "$2"
  echo "$3  $2" | sha256sum --check --status
}
go_bin=${LLMD_GO:-$(command -v go || true)}
if [[ -z "$go_bin" ]] || [[ $(GOTOOLCHAIN=local "$go_bin" version) != *"go1.27.2 "* ]]; then
  fetch "https://go.dev/dl/go1.27.2.linux-$arch.tar.gz" "$tmp/go.tgz" "$go_sha"
  tar -xzf "$tmp/go.tgz" -C "$tmp"
  go_bin="$tmp/go/bin/go"
fi
export CGO_ENABLED=0 GOTOOLCHAIN=local
git clone --quiet --depth 1 --branch v1.0.0 https://github.com/llm-d/llm-d-router.git "$tmp/router"
test "$(git -C "$tmp/router" rev-parse HEAD)" = "$router_rev"
(
  cd "$tmp/router"
  git apply "$here/patches/router-v1.0.0-nixl.patch"
  "$go_bin" test -p "${LLMD_BUILD_JOBS:-4}" ./pkg/sidecar/proxy -run '^TestHetero'
  "$go_bin" build -p "${LLMD_BUILD_JOBS:-4}" -buildvcs=false -trimpath -ldflags='-s -w' -o "$tmp/pd-sidecar" ./cmd/pd-sidecar
)
(
  cd "$here/../epp"
  "$go_bin" build -p "${LLMD_BUILD_JOBS:-4}" -buildvcs=false -trimpath -ldflags='-s -w' -o "$tmp/epp" ./cmd/epp
)
fetch "https://github.com/envoyproxy/envoy/releases/download/v1.33.2/envoy-1.33.2-linux-$envoy_arch" "$tmp/envoy" "$envoy_sha"
for binary in pd-sidecar epp envoy; do
  install -m 0755 "$tmp/$binary" "$dest/.$binary.new"
  mv -f "$dest/.$binary.new" "$dest/$binary"
done
(cd "$dest" && sha256sum pd-sidecar epp envoy > binaries.sha256)
if [[ ${2:-} == --images ]]; then
  fetch "https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_${crane_arch}.tar.gz" "$tmp/crane.tgz" "$crane_sha"
  tar -xzf "$tmp/crane.tgz" -C "$tmp" crane
  "$here/images.sh" "$dest" "$tmp/crane" "$arch"
fi
