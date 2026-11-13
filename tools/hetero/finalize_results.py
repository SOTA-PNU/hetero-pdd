# SPDX-License-Identifier: Apache-2.0
"""Export profiler summaries and raw observations from a sweep."""

import argparse
import csv
import sys
from pathlib import Path

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from tools.hetero.profiler.analysis import write_csv
from tools.hetero.profiler.common import live_rows, metric
from tools.hetero.profiler.expected import request_succeeded
from tools.hetero.profiler.log import log_line
from tools.hetero.profiler.resources import COLUMN_DOCS as PHASE_DOCS
from tools.hetero.profiler.resources import read_samples
from tools.hetero.profiler.sweep import outside_repo, read_json, validate_pairs

CLIENT = [
    "request_count",
    "success_count",
    "failure_count",
    "timeout_count",
    "retry_count",
    "incomplete_count",
    "completed_requests_per_second",
    "output_tokens_per_second",
    "makespan_seconds",
    "target_request_counts",
]
CLIENT += [
    f"{key}_{stat}_seconds"
    for key in (
        "ttft",
        "service_ttft",
        "tpot",
        "token_itl",
        "second_token_delay",
        "e2e",
        "queue",
    )
    for stat in ("mean", "p50", "p90", "p95", "p99")
]
TABLES = [
    "runs",
    "requests",
    "phase_resources",
    "resource_timeseries",
    "transfer",
    "memory_bw",
    "energy",
]
COLUMN_DOCS = {
    **PHASE_DOCS,
    "identity": "Scenario labels; run_id is the attempt path; workload is its name.",
    "client": "client.summary values; client_metrics retains accuracy metadata.",
    "server": "server.<role>.summary: overlapping queue/prefill/decode seconds.",
    "device_phase_active_ratio": "Union/duration per device; layout=H2D+reorder.",
    "kv_logical_mib_per_request": "L*2*Hkv*Dhead*dtype_bytes*input_tokens/2**20 (MiB).",
    "comparison": "Same validated pair/workload; latest successful attempts only.",
    "throughput_improvement_percent_vs_b0": "(PDD/B0-1)*100 (%).",
    "makespan_reduction_percent_vs_b0": "(1-PDD/B0)*100 (%).",
    "outputs_identical": "Complete request_index/token-ID maps; else not_evaluated.",
    "correctness": "workload/profile correctness.json; absent: not_evaluated.",
    "memory_types": "PLAN hardware: GPU GDDR7; NPU0/NPU1 GDDR6.",
    "runs": "All attempts/workloads, including failures. Relative source_summary.",
    "requests": "Raw requests.jsonl joined to client.requests by request_id.",
    "phase_resources": "summary.phase_resources.samples; time-weighted estimates.",
    "resource_timeseries": "MONOTONIC seconds; all-smi values, UUID/PCI mapping.",
    "transfer": "summary.transfer per server; request/aggregate metrics preserved.",
    "memory_bw": "summary.memory_bandwidth per device, including context_source.",
    "energy": "summary.energy per device/total: J, J/token, tokens/s/W, coverage.",
    "operators": "summary.operators, exported only when present.",
    "measurement_window": "First scheduled_arrival to last completed; no warm-up.",
    "accuracy": "Metric objects retain value/status/reason/source/unit/coverage.",
    "service": "dtype/KV dtype/context/batch: service/config.json or service_args.",
    "device_request_counts": "summary.per_device.<role>.summary.request_count.",
}


def kv_size(config, work):
    path = Path(config.get("model_path", "")) / "config.json"
    model = read_json(path) if config.get("model_path") else {}
    size = {"float16": 2, "bfloat16": 2, "float32": 4, "fp8": 1}.get(
        config.get("kv_cache_dtype")
    )
    value = None
    if model and size:
        head = (
            model.get("head_dim")
            or model["hidden_size"] // model["num_attention_heads"]
        )
        value = (
            model["num_hidden_layers"] * 2 * model["num_key_value_heads"] * head * size
        )
        value *= work["input_tokens"] / 2**20
    reason = "model_config_or_kv_dtype_missing"
    return metric(value, "MiB/request", "model/config.json", reason, "estimated")


def result_row(root, scenario, attempt, work):
    folder = root / attempt["path"] / work["name"]
    source = folder / "profile/summary.json"
    summary = read_json(source)
    config = read_json(root / attempt["path"] / "service/config.json")
    config = {**scenario["service_args"], **config}
    row = {
        k: scenario[k]
        for k in ("pair_id", "system", "comparison", "configuration", "mode", "model")
    }
    row.update(scenario=scenario["name"], run_id=attempt["path"], workload=work["name"])
    row.update(
        returncode=attempt.get("returncode", 0 if summary else None),
        error=attempt.get("error"),
    )
    state = "INCOMPLETE" if "returncode" in attempt else "NOT_RUN"
    state = "FAILED" if attempt.get("returncode") else state
    row["run_status"] = summary.get("run_status", state)
    row["analysis_status"] = summary.get("analysis_status", "INCOMPLETE")
    row.update(
        source_summary=str(source.relative_to(root)),
        failures=summary.get("failures", []),
    )
    row.update(
        input_tokens=work["input_tokens"],
        requested_output_tokens=work["output_tokens"],
        planned_requests=work["requests"],
        seed=work["seed"],
    )
    row.update(
        {
            k: config.get(k)
            for k in ("dtype", "kv_cache_dtype", "max_model_len", "max_num_seqs")
        }
    )
    row.update(
        npu_decode_bucket=config.get("max_num_seqs"),
        layout_reorder=config.get("layout_reorder", 0),
    )
    client = summary.get("client", {}).get("summary", {})
    for key in CLIENT:
        unit = "s" if key.endswith("_seconds") else "requests"
        if key.endswith("_per_second"):
            unit = "tokens/s" if key.startswith("output") else "requests/s"
        client.setdefault(
            key, metric(None, unit, "profile/summary.json", "missing_summary")
        )
    row.update({k: client.get(k, {}).get("value") for k in CLIENT})
    row["device_request_counts"] = {
        role: data["summary"]["request_count"]["value"]
        for role, data in summary.get("per_device", {}).items()
    }
    row.update(
        client_metrics=client,
        memory_types=dict(gpu="GDDR7", npu0="GDDR6", npu1="GDDR6"),
    )
    row["server"] = {r: d["summary"] for r, d in summary.get("server", {}).items()}
    row["request_phase_share"] = summary.get("request_phase_share", {}).get(
        "summary", {}
    )
    row["device_phase_active_ratio"] = summary.get("phase_resources", {}).get(
        "device_phase_active_ratio", {}
    )
    for key in ("transfer", "energy", "memory_bandwidth", "phase_resources"):
        row[key] = summary.get(key, {})
    logical = kv_size(config, work)
    row.update(kv_logical_mib_per_request=logical["value"], kv_logical_metadata=logical)
    # TODO(verify: IT-01): confirm the producer's optional correctness file schema.
    correctness = folder / "correctness.json"
    if not correctness.exists():
        correctness = folder / "profile/correctness.json"
    checked = read_json(correctness)
    row["correctness"] = checked if correctness.exists() else "not_evaluated"
    for key in ("correctness_for_performance", "strict_cross_device_correctness"):
        row[key] = checked.get(key, "not_evaluated")
    row.update(
        outputs_identical="not_evaluated", analysis_only=scenario["mode"] == "deep"
    )
    return row, summary, folder


def output_tokens(folder, work):
    rows = list(live_rows(folder / "requests.jsonl"))
    tokens = {r["request_index"]: r.get("streamed_token_ids") for r in rows}
    complete = len(rows) == len(tokens) == work["requests"] and all(
        request_succeeded(r)
        and r.get("streamed_token_ids") is not None
        and len(r["streamed_token_ids"]) == work["output_tokens"]
        for r in rows
    )
    return tokens if complete else None


def comparisons(records):
    latest = {}
    for row, summary, folder, scenario, work in records:
        if (
            row["run_status"] == row["analysis_status"] == "PASS"
            and row["returncode"] == 0
        ):
            latest[scenario["name"], work["name"]] = (row, folder, scenario, work)
    result = []
    for row, folder, scenario, work in latest.values():
        peers = [
            item
            for item in latest.values()
            if all(item[0][k] == row[k] for k in ("pair_id", "comparison", "workload"))
        ]
        reorder = [
            p
            for p in peers
            if p[0]["configuration"] == "pdd"
            and p[0]["layout_reorder"] != row["layout_reorder"]
        ]
        if row["configuration"] == "pdd" and reorder:
            a, b = output_tokens(folder, work), output_tokens(reorder[0][1], work)
            if a is not None and b is not None:
                row["outputs_identical"] = a == b
        baselines = [p[0] for p in peers if p[0]["configuration"] == "baseline"]
        if row["configuration"] != "pdd":
            continue
        for baseline in baselines:
            tp, span = (
                baseline["output_tokens_per_second"],
                baseline["makespan_seconds"],
            )
            rate, duration = row["output_tokens_per_second"], row["makespan_seconds"]
            if not tp or not span or None in (rate, duration):
                continue
            item = {
                k: row[k]
                for k in ("pair_id", "comparison", "workload", "analysis_only")
            }
            item.update(baseline=baseline["run_id"], candidate=row["run_id"])
            item.update(
                status="measured", reason=None, source="paired summaries", unit="%"
            )
            item["throughput_improvement_percent_vs_b0"] = (rate / tp - 1) * 100
            item["makespan_reduction_percent_vs_b0"] = (1 - duration / span) * 100
            result.append(item)
    return result


def datasets(records):
    tables = {name: [] for name in TABLES}
    for row, summary, folder, scenario, work in records:
        tag = {k: row[k] for k in ("run_id", "scenario", "pair_id", "workload")}
        tables["runs"].append(row)
        requests = list(live_rows(folder / "requests.jsonl"))
        metrics = {
            r["request_id"]: r["metrics"]
            for r in summary.get("client", {}).get("requests", [])
        }
        tables["requests"] += [
            dict(r, **tag, metrics=metrics.get(r["request_id"], {})) for r in requests
        ]
        tables["phase_resources"] += [
            dict(r, **tag)
            for r in summary.get("phase_resources", {}).get("samples", [])
        ]
        for table, key in (
            ("transfer", "transfer"),
            ("memory_bw", "memory_bandwidth"),
            ("energy", "energy"),
        ):
            tables[table] += [
                dict(tag, device=device, **values)
                for device, values in summary.get(key, {}).items()
            ]
        if "operators" in summary:
            tables.setdefault("operators", []).append(
                dict(tag, operators=summary["operators"])
            )
        links = read_json(folder.parent / "links.json")
        if requests and links:
            window = (
                min(r["scheduled_arrival_monotonic"] for r in requests),
                max(r["completed_monotonic"] for r in requests),
            )
            series = read_samples(
                live_rows(folder / "monitor/metrics.jsonl"), links, window
            )
            for (device, name), points in series.items():
                unit = "W" if name == "power" else "bytes" if name == "memory" else "%"
                tagged = dict(tag, device=device, metric=name)
                tables["resource_timeseries"] += [
                    dict(
                        tagged,
                        mono_seconds=t,
                        **metric(v, unit, "monitor/metrics.jsonl"),
                    )
                    for t, v in points
                ]
    return tables


def finalize(root, opprofile_dir=None):
    root = outside_repo(root)
    scenarios = read_json(root / "scenarios.json")["scenarios"]
    records = []
    for scenario in scenarios:
        attempts = scenario.get("attempts") or [dict(path="runs/" + scenario["name"])]
        for attempt in attempts:
            for work in scenario["workloads"]:
                records.append(
                    (*result_row(root, scenario, attempt, work), scenario, work)
                )
    valid, errors = [], []
    for pair in dict.fromkeys(s["pair_id"] for s in scenarios):
        members = [s for s in scenarios if s["pair_id"] == pair]
        try:
            validate_pairs(members)
            valid.extend(r for r in records if r[3]["pair_id"] == pair)
        except ValueError as exc:
            errors.append(str(exc))
            log_line("finalize", str(exc))
    compared = comparisons(valid)
    tables = datasets(records)
    if opprofile_dir is not None:
        for path in sorted(Path(opprofile_dir).rglob("operators*.csv")):
            if path.resolve() == (root / "dataset" / path.name).resolve():
                continue
            with path.open() as stream:
                rows = [
                    dict(r, operator_source=str(path.relative_to(opprofile_dir)))
                    for r in csv.DictReader(stream)
                ]
            tables.setdefault(path.stem, []).extend(rows)
    out = root / "dataset"
    out.mkdir(exist_ok=True)
    write_csv(root / "results.csv", tables["runs"])
    write_csv(root / "comparison.csv", compared)
    if not compared:
        columns = ["pair_id", "comparison", "workload", "baseline", "candidate"]
        columns += [k for k in COLUMN_DOCS if k.endswith("vs_b0")]
        (root / "comparison.csv").write_text(",".join(columns) + "\n")
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
        if not rows:
            (out / f"{name}.csv").write_text("run_id,workload\n")
    if "operators" not in tables:
        (out / "operators.csv").unlink(missing_ok=True)
    dictionary = [
        "# 데이터 사전",
        "",
        "중첩 셀은 JSON이며 지표의 정확도 표기를 보존한다.",
        "",
    ]
    dictionary += [f"- **{key}**: {value}" for key, value in COLUMN_DOCS.items()]
    for name, rows in tables.items():
        columns = sorted({k for row in rows for k in row})
        dictionary += ["", f"## {name}.csv", "", ", ".join(f"`{k}`" for k in columns)]
    (out / "DATA_DICTIONARY.md").write_text("\n".join(dictionary) + "\n")
    return int(bool(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--opprofile-dir", type=Path)
    args = parser.parse_args(argv)
    return finalize(args.sweep_dir, args.opprofile_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log_line("finalize", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
