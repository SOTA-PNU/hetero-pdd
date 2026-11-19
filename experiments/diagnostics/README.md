# CA22 KV import diagnostics

Run from the repository root with the RBLN service Python environment active.
The supplied YAML identifies physical CA22 0 by ordinal and UUID. Each command
writes a new directory. Run one device measurement at a time on an idle card.

```bash
export RBLN_DEVICES=0 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD"
OMP_NUM_THREADS=1 python -m tools.hetero.profiler.kv_import_micro experiments/diagnostics/ca22-kv-import.yaml --result-dir results/ca22-factorial
python -m tools.hetero.profiler.kv_import_micro experiments/diagnostics/ca22-kv-import-threads.yaml --result-dir results/ca22-threads
python -m tools.hetero.profiler.kv_import_values experiments/diagnostics/ca22-kv-values.yaml --result-dir results/ca22-values
```

The first experiment runs the installed Rebellions bandwidth test and measures
flat and KV-shaped copies. The second reproduces the TP1 service worker's CPU
thread setup and varies the scoped import thread limit. The third compares
integer markers, seeded fractional FP16 samples and canonicalized fractional
samples on independent materialized CA22 allocations. Change the physical
device, UUID, input seed, layer shape, block range and repetition counts in YAML.

To reproduce the CPU converter comparison with the accelerators hidden, choose
an allowed CPU and run the following command. The archived control used CPU 47.

```bash
RBLN_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 47 python -m tools.hetero.profiler.kv_import_values \
  experiments/diagnostics/ca22-kv-values.yaml --cpu-only --result-dir results/ca22-values-cpu
```

The value experiment builds a diagnostic codec against the installed SDK and
enumerates all 65,536 input encodings. It checks candidate and original physical
bits against the SDK's scalar converter and its bulk converter. On the device
it checks the entire destination allocation against the independent decoded
reference, runs a compiled negate before readback and again after readback,
and compares original and candidate result hashes bit-for-bit. The CA22
negation of positive zero returned positive zero in this study. The checker
counts that exact discrepancy against IEEE unary minus separately. Direct
transfer readback retains strict signed-zero and NaN-payload comparisons.

In the value YAML, `canonical_implementation: diagnostic` prepares a bounded
aligned scratch buffer and calls the ordinary blocking logical `copy_`. To
exercise the guarded production candidate, build its shim with
`python -m vllm_rbln.v1.worker.build_kv_canonical`, change the value to
`worker` and rerun into a new directory. This trial requires the worker to
accept each requested view and physically push every required byte. Each
sample includes view construction, preparation, submission and a single final
device synchronization. Validation, readbacks and compilation run outside
the timing interval.

`encoding-proof.json` records the host encoding check. `checks.json` contains
full-allocation correctness and real H2D counts for each device case.
`geometry.json` records per-call views. `samples.jsonl` and `summary.json`
contain individual timings and condition medians, P95 and sample variance.
In worker mode, `copy_and_metadata_ms` includes the eligibility and runtime
counter checks. The original archived measurements are in
`micro/factorial-final/`, `micro/threads/`, `runtime/value-cpu.json` and
`runtime/value-device/` inside the CA22 hardware evidence archive.
