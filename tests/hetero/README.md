# WP13 test infrastructure

Run `python3 -m unittest discover -s tests/hetero` from the repository root.
Core analysis tests require prometheus_client; hardware and optional tool tests skip when their environment is unset. Tests use temporary directories and loopback ports.

- `with MockServer(...) as server` starts an OS-selected port and exposes `server.url`.
  Configure `first_token_delay`, `token_delay`, `fail_ids`, `metrics=False` and
  `role`. IDs follow the fixed rule `100 + index`, independently per request.
  `server.requests` records request ID, headers, body and monotonic reception time.
  Read records after requests finish, or hold `server.lock` for concurrent access.
- `with MockProxy(prefill_url, decoder_urls, ...) as proxy` exposes `proxy.url`.
  Round-robin is the default; `pattern=[1, 1, 0]` repeats that list and `pattern=[0]`
  selects decode0 for every request. Prefill uses one non-streaming token, then
  the selected decoder's stream is relayed with `x-decode-target: host:port`.
  Both upstream requests carry the incoming request ID, or one generated UUID.
- `fakes.install_fake_tools(temp_bin)` creates `all-smi` and `otelcol` scripts for
  a test's temporary PATH. `all-smi api --port 0 --interval 1` serves all-smi metrics.
  `otelcol --config CONFIG` checks the config exists, receives opaque trace POSTs
  on `OTLP_PORT`, and copies the JSONL file named by `FAKE_OTEL_SPANS` to
  `OTEL_OUT/traces.jsonl` on graceful termination. Both print the actual URL.
- Import `FakePopen` from `tests.hetero.fakes`. It records copied argv/env and launch
  kwargs in `.calls`; `.processes` exposes `args`, `pid`, `returncode` and `poll()`.
  Assign `process.returncode` to model an exited process.
  PIDs are synthetic and must never be passed to OS process-control functions.

`mock_protocol.py` centralizes the unverified SSE, KV metadata and metrics shapes.
These fixtures do not validate actual HW formats. Real and synthetic fixture
README files document provenance and the WP7a expectations. `assert_log_line`
in `test_smoke.py` checks UTC ISO8601 and the run/component/message structure;
WP5 will add production integration. Common HTTP lifecycle and POST helpers live
in `mock_server.py`; fake entry points and script generation live in `fakes/__init__.py`.

Only three smoke tests remain: streamed token count, proxy decoder response header,
and loading all real fixture files without absolute paths. Detailed mock/tool tests
are omitted under AGENTS.md's test policy: **생략(HW 확인으로 대체)**. Client metrics,
expected ID sets and analysis calculations will be tested in their owning WPs.

Unused facilities were removed: stream interruption, counter resets, per-request
prefill delays, a separate skew policy, configurable token-ID rules, standalone
mock CLI, proxy health/auth/chat-only options, and fake-process lifecycle methods.
Responses retain the fields consumed by loadgen (`choices`, text, token IDs,
finish reason, usage and `[DONE]`); unused OpenAI envelope metadata was removed.
Histograms retain sum/count and the required `+Inf` bucket; arbitrary finite buckets
were removed. Counter-reset analysis belongs to WP7b's synthetic metrics text.
