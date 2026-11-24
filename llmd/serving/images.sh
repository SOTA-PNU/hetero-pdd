#!/usr/bin/env bash
# Build local image archives using the upstream runtime base, without a daemon.
set -euo pipefail
dest=$(cd "$1" && pwd)
crane=$2
arch=$3
base=gcr.io/distroless/static@sha256:aafe993665bf6200db38a02c11a9351e782a84b6a45e7f33b6695638c88e3931
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for binary in epp pd-sidecar; do
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$dest" -cf "$tmp/layer.tar" "$binary"
  "$crane" mutate --platform "linux/$arch" "$base" --append "$tmp/layer.tar" \
    --entrypoint "/$binary" --user 65532:65532 -t "hetero-pdd/$binary:v1" -o "$tmp/$binary.tar"
  "$crane" validate --tarball "$tmp/$binary.tar"
  "$crane" digest --tarball "$tmp/$binary.tar" > "$tmp/$binary.digest"
  mv "$tmp/$binary.tar" "$tmp/$binary.digest" "$dest"
done
