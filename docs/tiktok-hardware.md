# TikTok on the RTX PRO 6000 and CA22

The primary comparison is output tokens per second. Latency uses client send to first token or completion. Each client burst has a separate arrival timestamp, so these service times exclude the client-side wait for an available concurrency slot. Throughput covers the entire measured burst. Warmup, verification, compilation and process startup precede measurement.

At the representative load, TikTok delivered 675.32 tokens/s (median of three), compared with 243.04 for baseline-hetero and 610.24 for GPU-only. All three repeats used both devices for full requests. GPU-only had the lower P95 E2E (857.40 vs 1,216.70 ms). Preload-off delivered 676.12 tokens/s. No automatic request took the split route. This shape provides no evidence of a throughput gain from enabling preload.

Separately, on matched PDD assignments the staged policy delivered 92.77 vs 89.45 tokens/s with 15 verified overlaps. The 350 ms fixed-arrival probe on the final code reduced median two-request makespan from 1,638.41 to 1,595.43 ms and median decoder handoff from 54.06 to 23.05 ms. The anchor and following request of every final staged pair replayed exactly on the independent CUDA full gateway. Each matched ordinary pair used the same payloads and produced identical token IDs.

The follow-up study measures the FP16 import path, compares the optimized worker under fixed and automatic routing, and checks every measured output against GPU full. [CA22 KV import study](ca22-kv-import.md)

## Final automatic routing results

Each latency cell is the median of the per-run client statistics in milliseconds. The throughput column lists individual runs in execution order. Route counts cover all measured requests for the arm. All rows use the same engine limits.

| Shape (input / output / concurrency) | Arm | Output tokens/s | TTFT mean / P95 | TPOT mean | E2E mean / P95 | GPU full / NPU full / PDD | GPU ID mismatches |
|---|---|---|---:|---:|---:|---:|---:|
| 128/32/1 | baseline-hetero | 158.63 | 17.62 / 21.42 | 5.92 | 201.39 / 314.87 | 14 / 18 / 0 | 0 |
| 128/32/1 | TikTok | 473.13 | 19.63 / 23.81 | 1.52 | 67.17 / 72.79 | 32 / 0 / 0 | 0 |
| 128/128/4 | baseline-hetero | 218.71 / 251.21 / 243.04 | 1375.52 / 3645.50 | 4.97 | 2007.52 / 4841.11 | 106 / 86 / 0 | 0 |
| 128/128/4 | gpu-only | 613.38 / 605.65 / 610.24 | 621.05 / 657.91 | 1.56 | 819.00 / 857.40 | 192 / 0 / 0 | 0 |
| 128/128/4 | preload-off | 676.12 / 674.99 / 681.98 | 359.45 / 441.50 | 2.78 | 712.68 / 1214.69 | 162 / 30 / 0 | 0 |
| 128/128/4 | TikTok | 674.02 / 675.72 / 675.32 | 358.46 / 440.55 | 2.78 | 711.44 / 1216.70 | 162 / 30 / 0 | 0 |
| 512/128/8 | baseline-hetero | 245.12 | 3235.37 / 10223.07 | 4.98 | 3868.60 / 11624.26 | 82 / 46 / 0 | 2 |
| 512/128/8 | gpu-only | 545.01 | 1628.73 / 1719.86 | 1.56 | 1826.64 / 1917.65 | 128 / 0 / 0 | 0 |
| 512/128/8 | preload-off | 596.27 | 1232.75 / 2821.16 | 2.97 | 1609.84 / 4219.14 | 109 / 19 / 0 | 1 |
| 512/128/8 | TikTok | 567.20 | 1247.11 / 2727.88 | 3.04 | 1632.96 / 4145.49 | 108 / 20 / 0 | 1 |

On the 192 representative TikTok requests, predicted and actual mean TTFT were 431.09 and 421.81 ms on GPU full (16.97 ms mean absolute error), and 20.17 and 18.04 ms on NPU full (2.59 ms mean absolute error). Predicted and actual completion times were 620.82 and 619.05 ms on GPU full and 1,258.20 and 1,213.05 ms on NPU full.

Rows with token disagreements are raw observations and excluded from validated performance claims. Per-request EPP and worker IDs, selection and admission counts, proof of overlap, and per-route prediction errors are included in the evidence. [Machine-readable results](tiktok-hardware-results.json)

## Cost calibration

On November 20, 2026 (KST), one RTX PRO 6000 Blackwell Workstation Edition and one RBLN-CA22 served Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. The model, physical GPU/NPU UUIDs, driver and package versions, and original RBLN plugin hash matched the [original CI artifact](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44511339934/artifacts/14523918685). Both endpoints used FP16 weights and KV, HND, TP1, one active sequence, 32 cache blocks of 128 tokens, a 1024-token context and 128-token prefill chunks. The NPU had one host staging slot and 128 MiB of staging credit. Client concurrency never changes these limits.

The original 128-in/32-out, concurrency-one, 20-request CI result recorded 162.21 tokens/s for heterogeneous full (9 GPU and 11 NPU) and 83.16 for TikTok (20 split, zero selected preload). The checkout advanced from the analysis commit `582d345` to `b226979` through profiler reporting changes. The scheduling work used a new worktree based on the latter revision.

The calibration pinned each route through Envoy and native EPP on the same warm engine pair. Two warmups and six measured requests per route and length produced the following medians. Each request generated 32 tokens with `ignore_eos`. GPU and NPU prefill use deltas of the isolated engine's prefill-time histogram. TPOT uses the first and last streamed token. Receive spans worker READ start to completion. Import spans the subsequent host-to-device copy operation. A block represents 14,680,064 bytes. The import envelope is a host observation of the backend's import operation.

| Input | GPU prefill ms | NPU prefill ms | GPU/NPU full TTFT ms | GPU/NPU full TPOT ms | PDD TTFT ms | Receive ms | Import ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 14.19 | 13.41 | 22.02 / 20.17 | 1.50 / 9.50 | 87.26 | 28.12 | 22.46 |
| 256 | 24.36 | 23.71 | 30.23 / 30.69 | 1.49 / 10.15 | 148.48 | 55.81 | 42.67 |
| 512 | 40.73 | 46.52 | 48.17 / 54.10 | 1.45 / 11.17 | 269.97 | 110.79 | 84.02 |

The frozen table fits nonnegative intercepts and slopes to the length-cell medians. Full-route overhead is the median residual between client TTFT and fitted engine prefill. The GPU uses median TPOT. NPU TPOT increases with mean decode context. The PDD dispatch term is the median first-token residual after producer prefill, receive, import and first decode. Its residuals at the three lengths were 12.42, 16.30 and 22.97 ms. Scheduling uses the median, 16.30 ms. Results below were collected after freezing this table.

## Running the experiments

From this checkout, with the pinned CUDA and RBLN Python environments named by `CUDA_PY` and `RBLN_PY`, run the following. The Experiment Test container supplies both environments. Locally, set `LLMD_BIN_DIR` to a writable directory.

```bash
llmd/serving/build.sh "$LLMD_BIN_DIR"
"$CUDA_PY" -m llmd validate experiments/tiktok-parallel.yaml
"$CUDA_PY" -m tools.hetero.profiler.experiment_ci experiments/tiktok.yaml --result-dir results/low
"$CUDA_PY" -m tools.hetero.profiler.experiment_ci experiments/tiktok-parallel.yaml --result-dir results/parallel
"$CUDA_PY" -m tools.hetero.profiler.experiment_ci experiments/tiktok-long.yaml --result-dir results/long
"$CUDA_PY" -m tools.hetero.profiler.experiment_ci experiments/tiktok-overlap.yaml --result-dir results/fixed
```

Use a fresh results directory per execution. Change the native picker, cost coefficients, `objective` or `preload` in the referenced EPP profile. Change `benchmark` input and output lengths, concurrency, requests, seed, warmup waves, repeats and comparison profile references in the experiment. On pushes to `main` or `feat/pdd-scheduling-llmd`, CI selects enabled experiments that changed or reference a changed candidate or comparison profile. The low regression requires every measured candidate request to execute on CUDA full via `benchmark.expected_paths: [gpu_full]`. An unexpected or unverified route fails the comparison. The heterogeneous control is free to select either full endpoint. The larger YAMLs use manual selection. The comparisons keep the same endpoint and engine configuration. The GPU-only performance profile selects only CUDA full on that configuration. The separate GPU-only deployment supplies verification answers in CI. Each local replay through the resident GPU-only profile also checks the CUDA engine destination in the gateway access log. The fixed pair profiles exclude the full scheduling profile and use the same measured prices as automatic selection.

```bash
gh workflow run experiment_test.yml --repo SOTA-PNU/hetero-pdd \
  --ref exp/tiktok-hardware-20261120 -f experiment=tiktok-parallel.yaml
```

The fixed throughput command above generates `results/fixed/comparison-ordinary-pdd.yaml` by replacing the EPP profile in the candidate configuration. Both files retain the same `verify.overlap_probe` (128 input and 128 output for the anchor, 64 input and 32 output for the follower, 350 ms offset). The following Bash example starts a fresh service for each arm. It warms the ordinary PDD path with verification, runs one unmeasured pair at the target shape, then measures a second pair. Each measured run keeps its own event logs and JSON report. The original hardware series reused resident engines between arms.

```bash
(
  set -euo pipefail
  ordinary=results/fixed/comparison-ordinary-pdd.yaml
  test -f "$ordinary"
  probe_dir=$(mktemp -d "$PWD/results/fixed-pairs.XXXXXX")
  echo "Controlled pair results in $probe_dir"

  run_arm() (
    arm=$1
    rep=$2
    arm_run="$probe_dir/$rep-$arm"
    yaml=experiments/tiktok-overlap.yaml
    report=overlap-verification.json
    if [[ $arm == off ]]; then
      yaml=$ordinary
      report=ordinary-verification.json
    fi
    "$CUDA_PY" -m llmd start "$yaml" --work-dir "$arm_run"
    trap '"$CUDA_PY" -m llmd stop --work-dir "$arm_run"' EXIT
    "$CUDA_PY" -m llmd verify --work-dir "$arm_run"
    "$CUDA_PY" -m llmd.overlap --work-dir "$arm_run"
    mv "$arm_run/$report" "$arm_run/warmup-$report"
    "$CUDA_PY" -m llmd.overlap --work-dir "$arm_run"
    if [[ $arm == on && $rep == 1 ]]; then
      "$CUDA_PY" -m llmd.overlap --work-dir "$arm_run" --cancel
    fi
  )
  for rep in 1 2 3; do
    order=(off on)
    if [[ $rep == 2 ]]; then order=(on off); fi
    for arm in "${order[@]}"; do run_arm "$arm" "$rep"; done
  done

  gpu_run="$probe_dir/gpu-reference"
  "$CUDA_PY" -m llmd start experiments/qwen3-baseline.yaml --work-dir "$gpu_run"
  trap '"$CUDA_PY" -m llmd stop --work-dir "$gpu_run"' EXIT
  for rep in 1 2 3; do
    "$CUDA_PY" -m llmd.overlap --work-dir "$probe_dir/$rep-on" \
      --gpu-reference-run "$gpu_run"
  done
)
```

## How scheduling changed

The cost handler records the input length source and uses the explicit generation limit for fixed-length requests. With the measured table it prices full TTFT from local prefill and dispatch. Completion adds the remaining decode steps, including the RBLN context-length dependence. A split reservation occupies the GPU for producer prefill and the NPU through decode. Ordinary PDD receives KV after the producer and consumer become available. Staged PDD can receive into a credited host slot during an earlier decode. Both schedules pay import at admission.

The per-EPP reservation ledger serializes simultaneous placements, reconciles local work with the last aggregate scrape, and retires completed requests. For unmatched work observed only through aggregate gauges, remaining decode length comes from the YAML fallback. Multiple EPP replicas cannot identify each other's individual requests through those gauges. The consumer owns the physical block and byte accounting. The optional work-conserving objective uses a free primary endpoint when another locally assigned request is in flight, subject to any TTFT goal. At concurrency one, the comparison uses predicted completion. A request assigned to the slower NPU can increase the tail while freeing GPU capacity.

The optional RBLN import switch groups adjacent requested physical blocks for a single K or V tensor. It checks identity of source/destination block maps, uniqueness, bounds, native layout, dtype, shape and contiguous storage. Fragmented allocations yield multiple runs. It preserves the single-block and layout-reorder paths. The default remains per-block import. The same worker function records the copy count for both paths.

## Hardware observations

At 128 input / 128 output / concurrency 4 the initial three policy runs each chose 55 GPU full and nine NPU full requests. The median output throughput was 193.24 tokens/s for random heterogeneous full routing, 659.82 for CUDA full, 741.39 for cost-aware preload-off and 743.59 for TikTok. Every one of the 768 responses matched the corresponding CUDA token IDs. TikTok's P95 service E2E was 1,220 ms and GPU-only's was 803 ms. No request took a split route in this series.

The initial 512 / 128 / concurrency 8 expansion exposed two token disagreements. In the unfiltered four-arm measurement, the baseline routed prompt 71 to NPU full and preload-off routed prompt 119 there. CUDA full emitted token 576 at zero-based positions 2 and 3 respectively. NPU full emitted 785. Each of two isolated replays on each physical endpoint reproduced its own output. These were full requests with no KV import. The other runs matched their GPU reference. Raw throughput is retained for inspection. In the final expansion, the heterogeneous baseline differed on prompts 54 and 95. Preload-off and TikTok each differed on prompt 71. Every disagreement occurred on NPU full. All 512 measured requests completed. The table retains those raw timings. The three heterogeneous runs failed exact token agreement and do not establish a performance win for this shape.

The fixed-arrival probes release the following request by the same client clock on both arms. A worker READ and an identified, sampler-completed decode step must intersect. The report checks the lease, native request identity, imported block map, prior terminal, import grant and first subsequent decode. With the original 896-output anchor and 2 s offset, the decoder handoff medians were 55.06 ms ordinary and 20.84 ms staged. Whole-pair medians were 10.016 s and 10.056 s. Before the streaming scheduler correction, the 128-output anchor and 350 ms offset yielded ordinary makespans of 1,661.5 / 1,670.3 / 1,665.4 ms and staged makespans of 1,634.9 / 1,640.4 / 1,636.6 ms. Repeating those assignments and arrivals on the final scheduler yielded 1,638.41 / 1,638.35 / 1,639.05 ms ordinary and 1,615.05 / 1,585.21 / 1,595.43 ms staged. Each series excluded one initial warmup pair per arm.

Closing a real client connection while it held a staging credit quarantined its 14,680,064-byte block until the posted READ completed. The worker discarded the host buffer without granting import, the allocator released the block, and the next ordinary PDD request completed all 32 tokens.

## Four-block import

The import-only microbenchmark on the second CA22 used 28 layers and the real RBLN FP16 tensor views, checked allocated device memory, round-tripped data, and checked untouched guards. Across 11 alternating pairs, four contiguous blocks took a median 79.67 ms with 224 per-block copies and 67.39 ms with 56 range copies. One block took 20.03 and 19.97 ms with 56 calls in both arms. Synchronization for this diagnostic was outside the production import path.

On the evaluation CA22, a fixed ordinary-PDD workload (512 input, 32 output, concurrency 1) measured the real allocator's block mappings. Each of three trials per arm issued 12 requests. The individual median import times were 86.61, 88.71 and 87.80 ms with ordinary copies, and 62.02, 78.04 and 60.96 ms with coalescing. Fragmentation changed the number of ranges. Some requests still needed 224 calls. Mean TTFT per trial was 267.4 / 267.4 / 268.8 ms and 235.4 / 251.7 / 238.5 ms respectively. Output throughput was 52.16 / 52.00 / 51.72 and 55.30 / 53.50 / 54.70 tokens/s. All 72 measured outputs matched the first 32 tokens for their corresponding 512-token GPU-only prompts. Use `experiments/tiktok-import.yaml` and toggle `endpoints[1].engine.coalesce_kv_import` to reproduce the two arms. The automatic policy comparison uses the default copy path in every arm. Across the initial and final fixed-PDD throughput trials and the four-block import trials, the independent CUDA parity check matched 216/216 measured request instances and all 6,912/6,912 output token positions. It compared 32-token prefixes after checking the prompt matrix, model, tokenizer and observed CUDA full route. The archive records the checks in `fixed-pdd-gpu-parity.json`.

## Local execution and audit

The calibration and main performance trials used the same physical devices as the linked historical run. To avoid recompilation between native profiles, the local harness reused each resident engine pair and started isolated EPP and Envoy processes for each arm. It verified the endpoint roles, device ordinals, engine settings and listener owners. It used the repository load generator, deterministic prompt builder, timing metrics and per-request evidence join. CI launches each arm separately. The per-arm logs, configuration, command scripts and client token arrays are in the accompanying hardware evidence archive.

For a mixed physical pool serving GPU-only replay, use `--collect-gpu-reference-run "$GPU_RUN"`, close the reference gateway and use `--audit-gpu-reference-run "$GPU_RUN"` with the same probe work directory. The second command joins saved outputs to finalized access logs. The final report records the first differing output index on a mismatch.

## Queued streams and fixed PDD throughput

The first C2 burst exposed an error that elapsed-time reservations alone missed. The sidecar could submit the first pair to the NPU in a different order than EPP selected them. For ten later arrivals, the predecessor was still waiting for its first streamed output. EPP treated that predecessor as nearly finished and predicted about 91 ms to first token. Actual service TTFT was 421–433 ms. Preload initially yielded no proven overlap over the three bursts.

The corrected policy retains the unsent output tail until it observes real streaming progress. If a split route already wins, it can briefly wait for that signal, then reprice. The wait releases the policy mutex, respects cancellation, has a YAML cap of 90 ms, and is further limited by the time available to hide producer prefill and receipt. Full winners dispatch immediately. The cost coefficients remain frozen. The C2 control keeps the same coefficients and engine limits with preload disabled.

With 128 input / 32 output / concurrency 2 / 12 requests, each of the three fixed-PDD repeats used the same finite-burst load generator. Ordinary throughput was 89.17 / 89.51 / 89.45 tokens/s. Staged throughput was 92.46 / 94.00 / 92.77 tokens/s. The medians differ by 3.71%. The staged arm selected and admitted 16 receives across 36 requests. Fifteen had a qualifying previous-decode witness (5 / 6 / 4). All final host reservations were released and all outputs agreed between arms. In the first staged repeat, the predicted vs measured mean service TTFT was 367.17 vs 368.27 ms. Mean absolute error was 2.38 ms.

The frozen coefficients were checked in at `e1f6b2e`. The streaming scheduler fix is `7bc8aa6`, and `6f09ea1` sets the matched on/off profiles' first-output wait cap. All comparison arms use the pinned model. Use the standalone `experiments/tiktok-off.yaml` for a single preload-disabled run. The other YAMLs include comparison profiles so their arms run in one invocation.

The following historical commands describe the local calibration and evaluation. The original environment defined `EVIDENCE`, `CUDA_PY`, `RBLN_PY` and `LLMD_BIN_DIR` for that workstation. Calibration required the pinned model in the local cache, the built binaries and a ready native deployment under `$EVIDENCE/calibration`. The scripts create run directories and write raw measurements. To rerun them, copy the scripts and YAMLs to a fresh evidence directory and update their workstation-specific paths. The archive is the record of the completed runs.

```bash
"$CUDA_PY" -m llmd start "$EVIDENCE/calibration.yaml" --work-dir "$EVIDENCE/calibration"
"$CUDA_PY" "$EVIDENCE/calibrate.py"
"$CUDA_PY" "$EVIDENCE/fit.py"
"$CUDA_PY" -m llmd stop --work-dir "$EVIDENCE/calibration"
"$CUDA_PY" "$EVIDENCE/import_ab.py"
bash "$EVIDENCE/final-policies.sh"
```

The archive contains the raw evidence from those commands. For new workloads, use the repository commands above with fresh result directories. To run the current 128-output controlled probe, use the shipped `verify.overlap_probe` in `tiktok-overlap.yaml`. The historical long-anchor fixture uses the defaults when that mapping is omitted.

## Hardware CI replays

Both self-hosted runs used the default low workload. A separate CUDA answer deployment checked the short verify requests. Each benchmark compared all 32 complete output sequences, joined by prompt index, to the run with verified GPU full execution.

- At `d396260`, TikTok ran all 32 requests on GPU full and delivered 499.34 tokens/s. The heterogeneous baseline ran 13 GPU full and 19 NPU full and delivered 151.88 tokens/s. [Workflow](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44589192513) · [Raw artifact](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44589192513/artifacts/14563051845)
- At `4e2ca93`, TikTok ran all 32 requests on GPU full and delivered 498.76 tokens/s. The heterogeneous baseline ran 15 GPU full and 17 NPU full and delivered 163.63 tokens/s. [Workflow](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44590188059) · [Raw artifact](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44590188059/artifacts/14564405743)

The second run enabled the candidate route check on every measured request. Local unit verification ran 69 hetero tests with four environment-dependent skips, plus the Go plugin race suite and both native routing fixtures. [Unit CI](https://github.com/SOTA-PNU/hetero-pdd/actions/runs/44590188296)
