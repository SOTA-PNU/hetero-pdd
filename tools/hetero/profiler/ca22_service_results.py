"""Audit service-suite bursts, physical block maps and full GPU output comparisons."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from tools.hetero.profiler.analysis import client_metrics
from tools.hetero.profiler.common import live_rows

# These service times start when the client sends its request. Makespan spans
# the whole burst from the earliest scheduled arrival to the final completion.
CLIENT = {
    "ttft_mean_ms": ("service_ttft_mean_seconds", 1000),
    "ttft_p95_ms": ("service_ttft_p95_seconds", 1000),
    "tpot_mean_ms": ("tpot_mean_seconds", 1000),
    "e2e_mean_ms": ("service_mean_seconds", 1000),
    "e2e_p95_ms": ("service_p95_seconds", 1000),
    "makespan_s": ("makespan_seconds", 1),
    "output_tokens_per_second": ("output_tokens_per_second", 1),
}
WORKER = ("receive_ms", "import_ms", "host_copy_calls_ms", "reorder_ms")


def read(path):
    return json.loads(path.read_text())


def unique(rows, key):
    indexed = {}
    for row in rows:
        if row[key] in indexed:
            raise ValueError(f"duplicate {key}: {row[key]}")
        indexed[row[key]] = row
    return indexed


def physical_runs(groups):
    runs = 0
    for group in groups:
        blocks = sorted(group)
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"duplicate physical block in worker map: {group}")
        runs += bool(blocks) + sum(b != a + 1 for a, b in zip(blocks, blocks[1:]))
    return runs


def worker_evidence(work, requests):
    # A native ID is assigned by vLLM at the destination. Match that ID and
    # endpoint, never a substring, row number or request arrival order.
    transfers = {
        (r["target"], r["transfer"]["native_id"]): r["transfer"]
        for r in requests
        if r.get("transfer")
    }
    events = defaultdict(dict)
    for ep in {ep for ep, _ in transfers}:
        for path in sorted((work / "events" / ep).glob("events_*.jsonl")):
            for ev in live_rows(path):
                key = ep, ev.get("req_id")
                kind = ev["kind"]
                if (
                    key not in transfers
                    or str(ev["rank"]) != "0"
                    or kind not in {"kv_xfer_start", "kv_h2d", "kv_xfer_done"}
                ):
                    continue
                if kind in events[key]:
                    raise ValueError(f"{key}: duplicate worker {kind}")
                events[key][kind] = ev
        for ev in live_rows(work / f"{ep}-admission.jsonl"):
            key = ep, ev.get("request_id")
            if key in transfers and ev["event"] == "PREPARE_ALLOCATED_RECV_QUEUED":
                if "allocation" in events[key]:
                    raise ValueError(f"{key}: ambiguous prepare allocation")
                events[key]["allocation"] = ev
    result = {}
    for key, t in transfers.items():
        raw = events[key]
        begin, ready, h2d = (
            raw[k] for k in ("kv_xfer_start", "kv_xfer_done", "kv_h2d")
        )
        allocation = raw.get("allocation")
        if not (
            t["local_block_ids"] == begin["local_block_ids"]
            and t["remote_block_ids"] == begin["remote_block_ids"]
            and t["remote_id"] == begin["remote_request_id"]
            and t["receive_start_ns"] == begin["monotonic_ns"]
            and t["host_ready_ns"] == ready["monotonic_ns"]
            and t["import_start_ns"] == h2d["start_ns"]
            and t["import_end_ns"] == h2d["end_ns"]
            and t["host_copy_calls_ms"] == h2d["h2d_ns"] / 1e6
            and t["receive_ms"] == (ready["monotonic_ns"] - begin["monotonic_ns"]) / 1e6
            and t["import_ms"] == (h2d["end_ns"] - h2d["start_ns"]) / 1e6
            and t["copy_calls"] == h2d["copy_calls"]
            and t["bytes"] == h2d["bytes_est"]
            and t["blocks"] == h2d["num_blocks"] == sum(map(len, t["local_block_ids"]))
            and (
                not allocation or allocation["local_block_ids"] == t["local_block_ids"]
            )
        ):
            raise ValueError(f"{key}: route evidence disagrees with worker/admission")
        result[key] = {
            **t,
            "physical_runs": physical_runs(begin["local_block_ids"]),
            "reorder_ms": h2d["reorder_ns"] / 1e6,
            "admission_allocation_ns": allocation["monotonic_ns"]
            if allocation
            else None,
            **{
                k: h2d.get(k)
                for k in (
                    "import_threads_requested",
                    "import_threads_previous",
                    "import_threads_applied",
                )
            },
        }
    return result


def distribution(rows):
    groups = defaultdict(list)
    for r in rows:
        if x := r["transfer"]:
            groups[(x["blocks"], x["physical_runs"], x["copy_calls"])].append(r)
    return [
        dict(
            blocks=k[0],
            contiguous_runs=k[1],
            copy_calls=k[2],
            requests=len(items),
            **{
                "median_" + f: statistics.median(r["transfer"][f] for r in items)
                for f in WORKER
            },
            example=dict(
                client_id=items[0]["request_id"],
                native_id=items[0]["transfer"]["native_id"],
                local_block_ids=items[0]["transfer"]["local_block_ids"],
                work=items[0]["work"],
            ),
        )
        for k, items in sorted(groups.items())
    ]


def counts(rows):
    checked = [r["gpu_reference"] for r in rows if r["gpu_reference"]]
    return dict(
        requests=len(rows),
        executed_paths=dict(Counter(r["executed_path"] for r in rows)),
        preload_selected=sum(r["preload_selected"] for r in rows),
        preload_approved=sum(r["preload_approved"] for r in rows),
        actual_overlap=sum(bool(r["overlap"]) for r in rows),
        applied_import_threads=dict(
            Counter(
                str(t.get("import_threads_applied"))
                if t.get("import_threads_applied") is not None
                else "unrecorded"
                for r in rows
                if (t := r["transfer"])
            )
        ),
        gpu_checked_requests=len(checked),
        gpu_exact_requests=sum(r["matching"] for r in checked),
        gpu_reference_tokens=sum(r["reference_output_length"] for r in checked),
        gpu_tokens_in_exact_requests=sum(
            r["reference_output_length"] for r in checked if r["matching"]
        ),
    )


def audit_run(cell_dir, run):
    work = cell_dir / f"repeat-{run['repeat']:02d}" / run["arm"]
    folder = work / "profile"
    routes = unique(read(folder / "route_evidence.json")["requests"], "request_id")
    transfers = worker_evidence(work, routes.values())
    gpu_file = folder / "gpu_correctness.json"
    gpu = unique(read(gpu_file)["requests"], "request_id") if gpu_file.exists() else {}
    measured = list(live_rows(folder / "requests.jsonl"))
    clients = client_metrics(measured)
    client_times = unique(clients["requests"], "request_id")
    if routes.keys() != client_times.keys() or (
        gpu_file.exists() and gpu.keys() != routes.keys()
    ):
        raise ValueError(f"{work}: client, route or GPU request identities disagree")
    rows = []
    for req in measured:
        rid = req["request_id"]
        r = routes[rid]
        timing = client_times[rid]["metrics"]
        rows.append(
            dict(
                cell=cell_dir.name,
                repeat=run["repeat"],
                arm=run["arm"],
                work=str(work),
                request_index=req["request_index"],
                request_id=rid,
                target=r["target"],
                executed_path=r["executed_path"],
                preload_selected=r["preload_selected"],
                preload_approved=r["preload_approved"],
                overlap=r.get("overlap"),
                transfer=transfers[(r["target"], r["transfer"]["native_id"])]
                if r.get("transfer")
                else None,
                gpu_reference=gpu.get(rid),
                input_tokens=req["actual_input_tokens"],
                output_tokens=req["actual_output_tokens"],
                **{
                    name: timing[field]["value"] * 1000
                    if timing[field]["value"] is not None
                    else None
                    for name, field in (
                        ("ttft_ms", "service_ttft_seconds"),
                        ("tpot_ms", "tpot_seconds"),
                        ("e2e_ms", "service_seconds"),
                    )
                },
            )
        )
    result = clients["summary"]
    metrics = {}
    for name, (field, scale) in CLIENT.items():
        value = result.get(field, {}).get("value")
        metrics[name] = None if value is None else value * scale
    for field in WORKER:
        values = [t[field] for t in transfers.values() if t[field] is not None]
        metrics["median_" + field] = statistics.median(values) if values else None
    return dict(
        arm=run["arm"],
        repeat=run["repeat"],
        cell=cell_dir.name,
        timing_valid=run["timing_valid"],
        comparison_valid=run["comparison_valid"],
        metrics=metrics,
        counts=counts(rows),
        allocations=distribution(rows),
    ), rows


def audit(cell_dir):
    source = read(cell_dir / "benchmark-summary.json")
    all_rows, runs = [], []
    for run in source["runs"]:
        item, rows = audit_run(cell_dir, run)
        all_rows.extend(rows)
        runs.append(item)
    arms = {}
    for arm in sorted({r["arm"] for r in runs}):
        valid = [r for r in runs if r["arm"] == arm and r["comparison_valid"]]
        all_arm = [r for r in all_rows if r["arm"] == arm]
        metrics = {}
        for k in {k for r in valid for k in r["metrics"]}:
            values = [r["metrics"][k] for r in valid if r["metrics"][k] is not None]
            if values:
                metrics[k] = dict(
                    n=len(values),
                    median=statistics.median(values),
                    min=min(values),
                    max=max(values),
                )
        arms[arm] = dict(
            valid_repeats=[r["repeat"] for r in valid],
            metrics=metrics,
            counts_all_measured=counts(all_arm),
            allocations_all_measured=distribution(all_arm),
        )
    return dict(errors=source["errors"], arms=arms, runs=runs), all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="a completed suite or cell")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.source.glob("*/benchmark-summary.json"))
    if (args.source / "benchmark-summary.json").exists():
        sources = [args.source / "benchmark-summary.json"]
    if not sources:
        parser.error("no completed benchmark-summary.json in input")
    results, rows = {}, []
    for path in sources:
        result, requests = audit(path.parent)
        results[path.parent.name] = result
        rows.extend(requests)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "requests.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
