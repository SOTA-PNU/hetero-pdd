# Official serving components

The deployment uses llm-d v1.0.0, llm-d-router v1.0.0 and the router's Go module dependency on GIE v1.6.2. The custom EPP registers Go plugins then starts the upstream runner. Envoy invokes its external processor for every inference request. The selected consumer's official PD sidecar makes the prefill request and forwards the validated lease to the local decoder. KV moves between engines through NIXL. [llm-d release](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) · [Router source](https://github.com/llm-d/llm-d-router/tree/b9393d1a9f1d5a72ecd86f8d2b643ab54c2a9b92)

## Binaries and images

Run these commands from the repository root on Linux amd64 or arm64. The build needs git, curl, tar, sha256sum and flock. It downloads Go 1.27.2 if the pinned toolchain is unavailable. Use `LLMD_GO` for the path to an existing Go 1.27.2 binary. Downloads of the toolchain, Envoy and the image tool use pinned SHA256 values. The build verifies the router commit before applying the sidecar patch.

```sh
llmd/serving/build.sh
python3 -m llmd.serving.protocol_test --work "$(mktemp -d)/official-2p2d"
```

The default binary directory is `${XDG_CACHE_HOME:-$HOME/.cache}/hetero-pdd/llmd-v1/bin`. `LLMD_BIN_DIR` overrides it for builds and the local launcher. The build records `binaries.sha256`. An explicit directory and `--images` additionally produce validated, loadable `epp.tar` and `pd-sidecar.tar` archives plus the image manifest digest for each build. This command requires no Docker socket. Each image uses static Go binaries over the release's distroless static runtime base pinned by digest. The Envoy cluster image is `docker.io/envoyproxy/envoy:distroless-v1.33.2@sha256:85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8`.

```sh
bin="${LLMD_BIN_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/hetero-pdd/llmd-v1/bin}"
llmd/serving/build.sh "$bin" --images
cat "$bin/epp.digest" "$bin/pd-sidecar.digest"
```

For a kind cluster, import both archives with `kind load image-archive "$bin/epp.tar"` and `kind load image-archive "$bin/pd-sidecar.tar"`. Their tags are `hetero-pdd/epp:v1` and `hetero-pdd/pd-sidecar:v1`. Multi-node installations can distribute the same image archives through their registry and use resolved image references in the site configuration. Rebuilding after plugin or patch changes produces new digests. Use the matching engine image containing the engine adapters and connector changes.

## Local processes and Kubernetes

The local renderer starts one official sidecar per consumer, the official runner with the file-discovery plugin, and Envoy 1.33.2 using the no-Kubernetes guide's `envoy.yaml`. The run directory contains the resolved `epp.yaml`, `endpoints.yaml`, and `envoy.yaml`. The listener strips incoming routing headers before ext_proc. The EPP rejects client-supplied KV transfer parameters. Envoy preserves request IDs, emits access logs, and returns the selected `x-hetero-target`, `x-hetero-prefill`, `x-hetero-path`, `x-hetero-policy` and `x-hetero-reason`. `x-hetero-preload: 1` is present on selected lookahead requests. Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT` when starting the local experiment to enable upstream OTLP tracing. EPP metrics use `epp_metrics`; `sidecar_metrics_port` enables a separate official sidecar Prometheus listener per consumer. The worker's own `/metrics` remains available through its data-plane port. [Release guide](https://github.com/llm-d/llm-d/tree/378b42d18dabe0b9c4ced4e6f3670d93b52db08e/guides/no-kubernetes-deployment)

Kubernetes uses the official standalone Helm chart's Deployment, service account, Role/RoleBinding, Service and ConfigMaps. The EPP watches Pods by selector and stable port 8000. Envoy uses its chart-managed in-Pod TLS connection to EPP. The chart's fail-open path is disabled. Containers with the consumer role run the engine on 8001 and the sidecar on 8000. Each producer advertises its Pod IP as the NIXL side-channel host. The Pod's endpoint labels identify its model, compatibility group, raw engine port, engine ID, KV listener, TP, and cache dimensions. Keep backend and sidecar listeners on the experiment's internal network. The sidecar trusts the selection headers it receives from the gateway. [Pinned standalone chart](https://github.com/llm-d/llm-d-router/tree/b9393d1a9f1d5a72ecd86f8d2b643ab54c2a9b92/config/charts/llm-d-router-standalone)

`upstream/standalone.yaml` is the output of Helm v3.17.3 using `upstream/values.yaml`. To reproduce it, set `project` to this checkout's absolute path. In a checkout of llm-d-router at `b9393d1a9f1d5a72ecd86f8d2b643ab54c2a9b92` run the following commands.

```sh
helm dependency build config/charts/llm-d-router-standalone
helm template pdd-serving config/charts/llm-d-router-standalone --namespace pdd-namespace \
  -f "$project/llmd/serving/upstream/values.yaml" | diff - "$project/llmd/serving/upstream/standalone.yaml"
```

The renderer specializes the native EPP profile, images, namespace, selector, Service port, gateway header handling and request logging for each experiment. The upstream chart and no-Kubernetes template use Apache-2.0, included in `LICENSE-upstream`.

## Strict NIXL compatibility patch

The v1.0.0 stock sidecar supports decoder-only fallback when the producer header is absent, and can forward a response missing transfer metadata. `patches/router-v1.0.0-nixl.patch` adds an opt-in `HETERO_PDD_STRICT=1` guard to the existing PD dispatch and serial NIXLv2 implementation. The guard checks the EPP path and endpoint identity before any local fallback. The prefill leg asks for one token, ignores early stopping, preserves the external request ID and requests prompt token IDs. The patch checks the producer response's request and engine IDs, direction, address, port, TP, prompt length and dense-attention block ranges. Tokenized requests also check the original prompt IDs against the producer echo. The sidecar relays the producer's lease only after successful validation. A response carries `x-kv-remote-{request-id,engine-id,blocks}` for correlation. The decode request carries the per-request `hetero_preload` flag only when selected by EPP.

Strict PDD currently serves one string or a single token array at `/v1/completions` with one output sequence, using the vLLM 0.22 dense-attention handle shape. Batched prompts, incomplete dense groups and stateful APIs require their own lease/identity contract. The full route retains the native engine's inference path. A well-shaped handle still requires a successful connector import and generation to establish end-to-end correctness. Upstream's producer lease timeout bounds lifetime when a rejected response or cancelled consumer cannot acknowledge the lease.

The build exercises the upstream Go regression tests added by the patch. The protocol test starts the real official processes and uses HTTP engine fixtures. It changes file discovery from the first P/D pair to the second, checks the exact decoder lease, verifies emitted routing headers, and exercises malformed handles, missing routing decisions and non-200 responses. It writes process logs, sidecar Prometheus output and `protocol-results.json` under the requested directory. Physical KV movement, host staging and exact generated token checks use the separate engine verification flow.
