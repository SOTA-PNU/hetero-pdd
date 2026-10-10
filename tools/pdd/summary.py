# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize recorded script steps and observed requests without a live server."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import statistics
from pathlib import Path

from runner import atomic_text

STEPS = (
    "prepare_run", "start_prefill", "start_decode", "start_proxy",
    "check_correctness", "warmup", "run_workload", "stop_pdd",
    "summarize_results",
)
REQUIRED_STEPS = STEPS[:5]
CONFIG_KEYS = (
    "REPO", "BRANCH", "COMMIT", "REQUIRED_LAYOUT_COMMIT", "CUDA_PY", "RBLN_PY",
    "CUDA_VLLM_VERSION", "RBLN_VLLM_VERSION", "COMPILER_VERSION", "MODEL_PATH",
    "MODEL_NAME", "MODEL_DTYPE", "KV_CACHE_DTYPE", "OFFLINE", "CUDA_DEVICE",
    "CUDA_GPU_UUID", "CUDA_GPU_LINE", "RBLN_DEVICE", "PREFILL_PORT", "DECODE_PORT",
    "PROXY_PORT", "PREFILL_SIDE_PORT", "DECODE_SIDE_PORT", "PREFILL_ENGINE_ID",
    "DECODE_ENGINE_ID", "PRODUCER_CONFIG", "CONSUMER_CONFIG",
    "VLLM_RBLN_PDD_LAYOUT_REORDER", "VLLM_KV_CACHE_LAYOUT", "UCX_NET_DEVICES",
    "PRODUCER_KV_BUFFER_DEVICE", "CONSUMER_KV_BUFFER_DEVICE",
    "PRODUCER_LOCAL_MEMORY_TYPE", "CONSUMER_LOCAL_MEMORY_TYPE",
    "CONSUMER_REMOTE_MEMORY_TYPE", "TRANSFER_PATH", "PRODUCER_USE_HOST_BUFFER",
    "PRODUCER_APPLICATION_D2H_EXPECTED", "PYTHON_BIT_TRANSFORM_EXPECTED",
    "DTYPE_CAST_EXPECTED", "BLOCK_SIZE", "MAX_NUM_BATCHED_TOKENS",
    "WARMUP_REQUESTS", "MEASURE_REQUESTS", "CONCURRENCY", "MAX_TOKENS", "PROMPT_MODE",
    "MAX_MODEL_LEN", "MAX_NUM_SEQS", "TENSOR_PARALLEL_SIZE",
    "NUM_GPU_BLOCKS_OVERRIDE", "SEED", "ENABLE_CHUNKED_PREFILL",
    "ENABLE_PREFIX_CACHING",
)
UNAVAILABLE_METRICS = (
    "ttft_seconds", "itl_seconds", "pure_prefill_seconds", "pure_decode_seconds",
    "kv_transfer_seconds", "layout_seconds", "operator_seconds",
)
TIMING_DEFINITION = (
    "Step elapsed_seconds measures script execution using a monotonic clock, "
    "not internal PDD operations. Phase elapsed_seconds covers request submission "
    "through completion of all requests and incremental result writes. Measurement "
    "excludes warmup, server startup, the legacy profiling delay, and summarization."
)
THROUGHPUT_DEFINITION = (
    "successful_generated_token_total / completed measurement elapsed_seconds. "
    "Tokens from failed requests are excluded from the numerator; their time "
    "remains in the interval. No throughput is reported for an incomplete phase. "
    "generated_token_total separately counts all observed generated tokens, "
    "including any recorded for failed requests."
)
LATENCY_DEFINITION = (
    "End-to-end non-streaming HTTP latency of successful requests only; "
    "p50 and p95 use nearest rank. No TTFT or ITL is collected."
)


def number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_rows(path: Path, phase: str, run_id: str, diagnostics: list) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    seen = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("request record is not an object")
                    if row.get("phase") != phase:
                        if path.name == "workload_results.jsonl":
                            continue
                        raise ValueError("request phase mismatch")
                    if row.get("run_id") != run_id:
                        raise ValueError("request run ID mismatch")
                    index = row.get("request_index")
                    if not integer(index) or index in seen:
                        raise ValueError("invalid or duplicate request_index")
                    if not isinstance(row.get("success"), bool):
                        raise ValueError("request success is not a boolean")
                    if not number(row.get("latency_seconds")):
                        raise ValueError("request latency is unavailable or invalid")
                    tokens = row.get("generated_token_count")
                    if tokens is not None and not integer(tokens):
                        raise ValueError("generated token count is invalid")
                    if row["success"] and (tokens is None or tokens == 0):
                        raise ValueError("successful request has no generated tokens")
                    seen.add(index)
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    diagnostics.append(f"{path}:{line_number}: {exc}")
    except (OSError, UnicodeError) as exc:
        diagnostics.append(f"Cannot read {path}: {exc}")
    return rows


def phase_summary(
    result_dir: Path, name: str, runtime: dict, record: dict,
    diagnostics: list,
) -> dict:
    initial_errors = len(diagnostics)
    path = result_dir / f"{name}_results.jsonl"
    if not path.exists() and (result_dir / "workload_results.jsonl").exists():
        path = result_dir / "workload_results.jsonl"
    rows = read_rows(path, name, runtime.get("RUN_ID", ""), diagnostics)
    config = record.get("config", {})
    if not isinstance(config, dict):
        diagnostics.append(f"{name}: invalid recorded request configuration")
        config = {}
    requested = record.get("requested")
    if not integer(requested):
        count_key = "WARMUP_REQUESTS" if name == "warmup" else "MEASURE_REQUESTS"
        try:
            requested = int(config.get(count_key, runtime.get(count_key)))
            if requested < 0:
                raise ValueError("negative request count")
        except (KeyError, ValueError, TypeError):
            requested = None
    passed = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    elapsed = record.get("elapsed_seconds")
    if elapsed is not None and not number(elapsed):
        diagnostics.append(f"{name}: invalid phase elapsed_seconds")
        elapsed = None
    if record.get("status") == "FAILED" or failed:
        status = "FAILED"
    elif not rows and not record:
        status = "NOT_RUN"
    elif (
        record.get("status") == "SUCCESS"
        and requested is not None
        and len(rows) == requested
        and len(diagnostics) == initial_errors
        and elapsed is not None
        and (name != "measure" or elapsed > 0)
    ):
        status = "SUCCESS"
    else:
        status = "INCOMPLETE"
    if record and requested is not None and len(rows) != requested:
        diagnostics.append(
            f"{name}: collected {len(rows)} of {requested} requested results"
        )
    if record.get("status") == "RUNNING":
        diagnostics.append(f"{name}: no completion record; the phase was interrupted")
    known_tokens = [row["generated_token_count"] for row in rows
                    if integer(row.get("generated_token_count"))]
    total_tokens = sum(known_tokens) if known_tokens or status == "SUCCESS" else None
    successful_tokens = sum(row["generated_token_count"] for row in passed)
    latencies = sorted(row["latency_seconds"] for row in passed)
    complete_interval = (
        record.get("status") in ("SUCCESS", "FAILED")
        and requested is not None and len(rows) == requested
        and elapsed is not None and elapsed > 0
        and len(diagnostics) == initial_errors
    )
    return {
        "status": status,
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "elapsed_seconds": elapsed,
        "requested": requested,
        "completed": len(rows),
        "succeeded": len(passed),
        "failed": len(failed),
        "missing": max(0, requested - len(rows)) if requested is not None else None,
        "generated_token_total": total_tokens,
        "successful_generated_token_total": successful_tokens,
        "tokens_per_second": (
            successful_tokens / elapsed
            if name == "measure" and complete_interval else None
        ),
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_p50_seconds": (
            latencies[max(0, math.ceil(0.50 * len(latencies)) - 1)]
            if latencies else None
        ),
        "latency_p95_seconds": (
            latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
            if latencies else None
        ),
        "reason": record.get("reason"),
        "request_errors": [
            {"request_index": row["request_index"], "error": row.get("error")}
            for row in failed
        ],
        "log_path": str(path),
        "config": config,
    }


def display(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict) -> str:
    lines = [
        "# PDD execution summary", "",
        f"Experiment: **{summary['experiment_status']}**; "
        f"cleanup: **{summary['cleanup_status']}**.",
        "Summary generation: SUCCESS. This does not imply experiment success.", "",
        f"- Run ID: `{display(summary['run_id'])}`",
        f"- Commit: `{display(summary['commit'])}`",
        f"- Result directory: `{display(summary['result_dir'])}`", "",
        "## Script execution steps", "",
        "| Step | Status | Seconds | Exit code | Failure reason | Log |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for name, step in summary["steps"].items():
        lines.append("| " + " | ".join(display(value) for value in (
            name, step["status"], step.get("elapsed_seconds"),
            step.get("exit_code"), step.get("reason"), step.get("log_path"),
        )) + " |")
    lines += ["", TIMING_DEFINITION, "", "## Observed requests", "",
              "| Phase | Status | Requested | Completed | Succeeded | Failed | "
              "Seconds | Observed tokens | Successful tokens | Tokens/s |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, phase in summary["phases"].items():
        lines.append("| " + " | ".join(display(value) for value in (
            name, phase["status"], phase["requested"], phase["completed"],
            phase["succeeded"], phase["failed"], phase["elapsed_seconds"],
            phase["generated_token_total"], phase["successful_generated_token_total"],
            phase["tokens_per_second"],
        )) + " |")
    lines += ["", THROUGHPUT_DEFINITION, "", LATENCY_DEFINITION, "",
              "| Phase | Mean latency (s) | p50 (s) | p95 (s) |",
              "| --- | ---: | ---: | ---: |"]
    for name, phase in summary["phases"].items():
        lines.append("| " + " | ".join(display(value) for value in (
            name, phase["latency_mean_seconds"], phase["latency_p50_seconds"],
            phase["latency_p95_seconds"],
        )) + " |")
    lines += ["", "## Saved configuration", "", "| Setting | Value |",
              "| --- | --- |"]
    for key, value in summary["configuration"].items():
        lines.append(f"| {display(key)} | {display(value)} |")
    for name, phase in summary["phases"].items():
        if phase["config"]:
            lines += ["", f"Actual {name} request settings:", "", "```json",
                      json.dumps(phase["config"], indent=2, sort_keys=True), "```"]
    lines += ["", "## Uncollected metrics", ""]
    lines += [f"- {metric}: unavailable (not collected)"
              for metric in UNAVAILABLE_METRICS]
    lines += ["", "## Diagnostics", ""]
    lines += ([f"- {display(item)}" for item in summary["diagnostics"]]
              or ["No recorded diagnostics."])
    for name, phase in summary["phases"].items():
        lines += [f"- {name} request {row['request_index']}: {display(row['error'])}"
                  for row in phase["request_errors"]]
    lines += ["", "## Original artifacts", ""]
    lines += [f"- `{display(path)}`" for path in summary["artifacts"]]
    return "\n".join(lines) + "\n"


def render_csv(summary: dict) -> str:
    output = io.StringIO(newline="")
    fields = (
        "run_id", "commit", "experiment_status", "cleanup_status", "record_type",
        "name", "status", "started_at", "ended_at", "elapsed_seconds", "exit_code",
        "requested", "completed", "succeeded", "failed", "missing",
        "generated_token_total", "successful_generated_token_total",
        "tokens_per_second", "latency_mean_seconds", "latency_p50_seconds",
        "latency_p95_seconds", "reason", "log_path", "value",
    )
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    common = {key: summary[key] for key in fields[:4]}
    for category, collection in (("step", "steps"), ("phase", "phases")):
        for name, row in summary[collection].items():
            writer.writerow({**common, **row, "record_type": category, "name": name})
            if category == "phase":
                for key, value in row["config"].items():
                    writer.writerow({
                        **common, "record_type": "request_config",
                        "name": f"{name}.{key}", "value": value,
                    })
            else:
                for index, attempt in enumerate(row.get("attempts", [])):
                    if isinstance(attempt, dict):
                        writer.writerow({
                            **common, **attempt, "record_type": "step_attempt",
                            "name": f"{name}[{index}]",
                        })
    for name, value in summary["configuration"].items():
        writer.writerow({**common, "record_type": "configuration",
                         "name": name, "value": value})
    for name in UNAVAILABLE_METRICS:
        writer.writerow({**common, "record_type": "uncollected_metric",
                         "name": name, "value": "unavailable"})
    for name in ("summary_status", "timing_definition", "throughput_definition",
                 "latency_definition"):
        writer.writerow({**common, "record_type": "summary",
                         "name": name, "value": summary[name]})
    for diagnostic in summary["diagnostics"]:
        writer.writerow({**common, "record_type": "diagnostic", "reason": diagnostic})
    for path in summary["artifacts"]:
        writer.writerow({**common, "record_type": "artifact", "log_path": path})
    return output.getvalue()


def generate(result_dir: Path | str, runtime: dict, stages: dict,
             diagnostics: list | None = None) -> dict:
    """Atomically write each summary, even for a failed or incomplete experiment.

    The caller holds the run lock. Exceptions writing the summaries must propagate
    so that the shell step can distinguish generation failure from experiment
    failure. Runtime/stage loading diagnostics can be supplied by that caller.
    """
    result_dir = Path(result_dir)
    diagnostics = list(diagnostics or [])
    run_id = runtime.get("RUN_ID") or stages.get("run_id")
    if not runtime.get("RUN_ID"):
        diagnostics.append("Prepared run ID is unavailable")
    runtime = {**runtime, "RUN_ID": run_id}
    source_steps = stages.get("steps", {})
    if stages.get("run_id") != run_id or not isinstance(source_steps, dict):
        diagnostics.append("Step state is missing, invalid, or belongs to another run")
        source_steps = {}
    steps = {}
    history_diagnostics = []
    for name in STEPS:
        step = source_steps.get(name, {})
        if not isinstance(step, dict):
            diagnostics.append(f"{name}: invalid step record")
            step = {}
        steps[name] = {
            "run_id": run_id, "status": "NOT_RUN", "started_at": None,
            "ended_at": None,
            "elapsed_seconds": None, "exit_code": None, "reason": None,
            "log_path": None, **step,
        }
        if not isinstance(steps[name].get("attempts", []), list):
            diagnostics.append(f"{name}: invalid attempt history")
            steps[name]["attempts"] = []
        if not number(steps[name]["elapsed_seconds"]):
            if steps[name]["elapsed_seconds"] is not None:
                diagnostics.append(f"{name}: invalid script elapsed_seconds")
            steps[name]["elapsed_seconds"] = None
        for attempt in steps[name].get("attempts", []):
            if isinstance(attempt, dict) and attempt.get("status") == "FAILED":
                history_diagnostics.append(
                    f"{name}: a recorded attempt failed: "
                    f"{attempt.get('reason') or 'no reason recorded'}; "
                    f"log: {attempt.get('log_path') or 'unavailable'}"
                )
        if step.get("status") == "RUNNING" and not step.get("reason"):
            steps[name]["reason"] = (
                "No completion record; this step is still running or was interrupted"
            )
    phase_records = {}
    phase_path = result_dir / "workload_phases.json"
    if phase_path.exists():
        try:
            phase_state = json.loads(phase_path.read_text(encoding="utf-8"))
            if not isinstance(phase_state, dict) or phase_state.get("run_id") != run_id:
                raise ValueError("phase state run ID mismatch")
            phase_records = phase_state["phases"]
            if not isinstance(phase_records, dict):
                raise ValueError("phase state is not an object")
        except (OSError, ValueError, KeyError, UnicodeError) as exc:
            diagnostics.append(f"Cannot read {phase_path}: {exc}")
            phase_records = {}
    phases = {}
    for name in ("warmup", "measure"):
        record = phase_records.get(name, {})
        if not isinstance(record, dict):
            diagnostics.append(f"{name}: invalid phase record")
            record = {}
        phases[name] = phase_summary(result_dir, name, runtime, record, diagnostics)
    experiment_steps = [steps[name] for name in STEPS if name not in (
        "stop_pdd", "summarize_results",
    )]
    if any(
        step["status"] == "FAILED" or any(
            isinstance(attempt, dict) and attempt.get("status") == "FAILED"
            for attempt in step.get("attempts", [])
        ) for step in experiment_steps
    ) or any(
        phase["status"] == "FAILED" for phase in phases.values()
    ):
        status = "FAILED"
    elif (
        all(steps[name]["status"] == "SUCCESS" for name in REQUIRED_STEPS)
        and steps["run_workload"]["status"] == "SUCCESS"
        and all(phase["status"] == "SUCCESS" for phase in phases.values())
        and not diagnostics
    ):
        status = "SUCCESS"
    else:
        status = "INCOMPLETE"
    diagnostics.extend(history_diagnostics)
    artifact_names = (
        "*.log", "*.env", "*.jsonl", "workload_phases.json", "stages.json",
        "workload_summary.txt", "correctness*.json", "correctness*.txt",
    )
    artifacts = sorted({str(path) for pattern in artifact_names
                        for path in result_dir.glob(pattern) if path.is_file()})
    summary = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary_status": "SUCCESS",
        "experiment_status": status,
        "cleanup_status": steps["stop_pdd"]["status"],
        "run_id": run_id,
        "commit": runtime.get("COMMIT"),
        "result_dir": str(result_dir),
        "configuration": {key: runtime[key] for key in CONFIG_KEYS if key in runtime},
        "steps": steps,
        "phases": phases,
        "timing_definition": TIMING_DEFINITION,
        "throughput_definition": THROUGHPUT_DEFINITION,
        "latency_definition": LATENCY_DEFINITION,
        "unavailable_metrics": {name: None for name in UNAVAILABLE_METRICS},
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }
    atomic_text(result_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_text(result_dir / "summary.csv", render_csv(summary))
    atomic_text(result_dir / "summary.md", render_markdown(summary))
    return summary
