# RBLN host-staged remote KV

The TikTok profile selects a normal full or P→D route using estimated queue, prefill, transfer and decode times. For a compatible busy decoder with measured host credit it can also authorize a preload. The official PD sidecar adds `hetero_preload: true` to the decoder's validated NIXL transfer metadata. Clients submit ordinary completion requests to the gateway.

The scheduler ports the bounded-preparation mechanism from the local research checkout `pdd_lookahead_20260926-202449/src`, commit `8cdcb24b`. This version requires the per-request authorization and adds independent native READ completion polling, host retention and explicit import permission. Predictions and the fixture's cost tables have no bearing on the engine's local allocation checks.

## Configuration

The RBLN endpoint's `engine.lookahead` object maps to `kv_connector_extra_config.rbln_lookahead`.

| Field | Meaning |
| --- | --- |
| `active_decode_limit` | Maximum executing requests, bounded by compiled `max_num_seqs`. |
| `lookahead_depth` | Maximum extra prepared requests when execution admission is full. |
| `prefetch_byte_budget` | Maximum reserved host bytes for tagged preparation. A prompt larger than the budget can receive normally when an execution slot is available. |
| `receive_timeout_s` | Optional deadline for a posted preload READ, measured from worker submission (default 30 seconds). The native handshake has its own timeout. |
| `events_path` | Launcher-provided scheduler trace path in the run directory. |

The tested adapter uses FCFS, TP/PP/DP 1, CPU host buffers, full self-attention, no prefix caching or speculative decoding, and KV-load failure policy `fail`. HND layout, dtype, block geometry and shard compatibility are checked when resolving the pair. Endpoint metrics `hetero_pdd_prepare_slots_free` and `hetero_pdd_prepare_bytes_free` publish current local headroom. Each engine needs its own clean `PROMETHEUS_MULTIPROC_DIR` before Python starts. The RBLN scheduler also publishes its running and waiting counts and physical KV utilization, including on a fresh engine before the first native vLLM statistics iteration. Metrics are sampled, so the allocator always rechecks availability.

## Buffer ownership and execution

1. The producer owns its original KV blocks until the native NIXL READ acknowledgement or its producer lease expires. Consumer-side heartbeats extend the lease while a valid remote request waits.
2. The RBLN scheduler can admit a tagged receive when all compiled execution slots are busy. It reserves physical block IDs for the transfer. Each ID addresses matching entries in the registered host buffer and device KV pool. The admission budget measures allocated blocks, including padding. The scheduler also reserves enough future device capacity for the active and prepared requests' requested output bounds. This reserve consumes no extra READ descriptors.
3. The worker posts the existing NIXL READ into host storage. A bounded poller checks the native handle independently of the model thread. At DONE the worker returns the handle to upstream vLLM for telemetry and release. A successful transfer stays in host storage while its consumer execution slot is busy.
4. The scheduler grants import to the oldest live prepared requests with free execution slots. A grant can precede READ completion. After both conditions hold, the native worker calls the RBLN copy callback on the original block table and reports `finished_recving`. Only then can the scheduler admit that request's first decode call.
5. Canceling an in-flight request keeps the host/device block reservation quarantined until the READ quiesces. A canceled host-ready transfer skips H2D. Native transfer failures invalidate the reserved blocks and fail the request. On a READ deadline, successful NIXL handle cancellation allows the normal failed-completion path to release the reservation. If NIXL refuses to cancel a still-live handle, the worker fails and process teardown revokes its memory registrations. A dead consumer stops refreshing the producer's lease. Worker shutdown joins its pollers before upstream deregistration.

## Hardware verification

Start the TikTok experiment using the managed launcher and wait for readiness. Run the following from the checkout with the launcher's Python environment.

```sh
python -m llmd.overlap --work-dir /absolute/path/to/run
python -m llmd.overlap --work-dir /absolute/path/to/run --cancel
```

The fixture uses the public gateway for two requests, reads the running metric during the first decode, and submits the second after the EPP's one-second collection interval. Its input sizes and output bounds target the shipped 1024-token Qwen3 example. The report requires per-request evidence from the selected producer and consumer, a matching local block table and byte count, an independently polled NIXL PROC or DONE inside the native decode step identified by its enclosed request-scoped model call (forward and sampler), host retention until the first request terminates, and H2D before the next model call. A missing route, event or timing condition produces a failed result. The timestamps are host monotonic observations at the model and connector boundaries. Device-internal kernel timestamps and performance gains require separate measurements.

`overlap-verification.json` contains the client and internal IDs, raw event subset and intersecting spans. Full event streams are in `events/<decoder>/events_*.jsonl` and `<decoder>-admission.jsonl`. RBLN event emission is enabled by `instrumentation.rbln_events`. The ordinary verifier handles exact-token comparisons against a separate matched GPU baseline.

The cancellation fixture closes the second gateway HTTP connection after the scheduler reserves its blocks. It requires the terminal abort, actual pending-credit release with no import grant, and a successful subsequent PD completion. It writes `cancel-verification.json`.

For exact-token verification of both requests, stop the TikTok run, start a separate GPU-only baseline of the same model, and run `python -m llmd.overlap --work-dir /absolute/path/to/tiktok-run --gpu-reference-run /absolute/path/to/ready-gpu-run`. This replays each saved request payload through the independent gateway and records the full token IDs, exact-match status and first differing index in `overlap-gpu-reference.json`.
