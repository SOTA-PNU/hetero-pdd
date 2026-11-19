"""Join measured gateway requests with EPP choices and completed worker steps."""

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from statistics import mean

from tools.hetero.loadgen import request_succeeded
from tools.hetero.profiler.common import live_rows, union
from tools.hetero.profiler.expected import engine_request_base, server_request_id


def _one(records):
    return records[0] if len(records) == 1 else {}


def _log(path, selector, kinds):
    # EPP and Envoy interleave text diagnostics with JSON structured events.
    by_request = defaultdict(list)
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue  # writer can be between writes when auditing a live warmup
                if not line.startswith("{") or not any(k in line for k in kinds):
                    continue
                event = json.loads(line)
                if event.get(selector) in kinds:
                    by_request[event["request_id"]].append(event)
    return by_request


def _worker(work, endpoint):
    events = defaultdict(lambda: defaultdict(list))
    native_ids = defaultdict(set)
    steps = []
    for path in sorted((work / "events" / endpoint).glob("events_*.jsonl")):
        # Use calls and sampler steps from the same rank and worker stream.
        calls = []
        for event in live_rows(path):
            if str(event.get("rank")) != "0":
                continue
            kind = event["kind"]
            if kind == "model_call":
                calls.append(event)
                for native in event["request_ids"]:
                    events[native][event["phase"]].append(event)
                    native_ids[engine_request_base(native)].add(native)
            elif kind == "step":
                enclosed = [
                    c
                    for c in calls
                    if c["phase"] == "decode"
                    and event["span_start_ns"]
                    <= c["span_start_ns"]
                    <= c["span_end_ns"]
                    <= event["span_end_ns"]
                ]
                if (
                    event.get("phase") == "decode"
                    and event.get("sampled")
                    and len(enclosed) == 1
                ):
                    steps.append(
                        (
                            event["span_start_ns"],
                            event["span_end_ns"],
                            enclosed[0]["request_ids"],
                        )
                    )
                calls.clear()
            elif (native := event.get("req_id")) is not None:
                events[native][kind].append(event)
                native_ids[engine_request_base(native)].add(native)
    for event in live_rows(work / f"{endpoint}-admission.jsonl"):
        if native := event.get("request_id"):
            events[native][event["event"]].append(event)
            native_ids[engine_request_base(native)].add(native)
    steps.sort()
    return events, native_ids, steps, [step[0] for step in steps]


def _transfer(row, ep, producer, events, native):
    start = _one(events["kv_xfer_start"])
    done = _one(events["kv_xfer_done"])
    copied = _one(events["kv_h2d"])
    if not all((start, done, copied, producer)) or not events["decode"]:
        return None
    first_decode = min(c["span_start_ns"] for c in events["decode"])
    expected = math.ceil(row["actual_input_tokens"] / ep["engine"]["block_size"])
    headers = row["response_headers"]
    remote = start.get("remote_request_id")
    mapping = (
        start.get("engine_id") == ep["engine_id"]
        and start.get("remote_engine_id") == producer["engine_id"]
        and engine_request_base(remote) == server_request_id(row["request_id"])
        and remote == headers.get("x-kv-remote-request-id", remote)
        and remote != native
        and all(
            sum(map(len, start[k])) == expected
            for k in ("remote_block_ids", "local_block_ids")
        )
        and copied["num_blocks"] == expected
        and copied["bytes_est"] > 0
        and not events["prefill"]
    )
    begin, end = start["monotonic_ns"], done["monotonic_ns"]
    if not (
        mapping
        and begin <= end <= copied["start_ns"] < copied["end_ns"] <= first_decode
    ):
        return None
    return {
        "native_id": native,
        "remote_id": remote,
        "blocks": expected,
        "bytes": copied["bytes_est"],
        "local_block_ids": start["local_block_ids"],
        "remote_block_ids": start["remote_block_ids"],
        "receive_start_ns": begin,
        "host_ready_ns": end,
        "import_start_ns": copied["start_ns"],
        "import_end_ns": copied["end_ns"],
        "first_decode_ns": first_decode,
        "receive_ms": (end - begin) / 1e6,
        "import_ms": (copied["end_ns"] - copied["start_ns"]) / 1e6,
    }


def _overlap(transfer, worker, native, active_limit):
    events, _, steps, starts = worker
    own = events[native]
    alloc = _one(own["PREPARE_ALLOCATED_RECV_QUEUED"])
    grant = _one(own["IMPORT_GRANTED"])
    ready = _one(own["kv_host_ready"])
    if not all((alloc, grant, ready, transfer)):
        return None
    begin, end = transfer["receive_start_ns"], transfer["host_ready_ns"]
    if not (
        alloc["active_count"] > 0
        and not alloc["token_compute_scheduled"]
        and alloc["allocated_bytes"] == transfer["bytes"]
        and alloc["local_block_ids"] == transfer["local_block_ids"]
        and ready["monotonic_ns"] == end
        and alloc["monotonic_ns"] <= begin
        and grant["monotonic_ns"] <= transfer["import_start_ns"]
    ):
        return None
    # A receive envelope can span CPU gaps. Count only sample-complete decode
    # steps containing another request's real model call. Require an independent
    # NIXL progress/completion observation within one of those steps.
    intervals, witnesses, other_requests = [], [], set()
    i = max(0, bisect_left(starts, begin) - 1)
    while i < len(steps) and steps[i][0] < end:
        lo, hi, ids = steps[i]
        i += 1
        if hi <= begin or native in ids or not ids:
            continue
        # With one active credit the preceding request must terminate before
        # this import can be granted.
        if active_limit == 1 and not all(
            any(
                hi <= t["monotonic_ns"] <= grant["monotonic_ns"]
                for t in events[r]["REQUEST_TERMINAL"]
            )
            for r in ids
        ):
            continue
        intervals.append((max(lo, begin), min(hi, end)))
        other_requests.update(ids)
        if any(lo <= t <= hi for t in (end, ready.get("last_proc_ns") or -1)):
            witnesses.append({"step_start_ns": lo, "step_end_ns": hi, "requests": ids})
    if not witnesses:
        return None
    return {
        "previous_requests": sorted(other_requests),
        "receive_during_decode_ms": sum(b - a for a, b in union(intervals)) / 1e6,
        "witness": witnesses[0],
        "import_grant_ns": grant["monotonic_ns"],
    }


def analyze_routes(work: Path) -> dict:
    """Write per-request evidence and return counts and measured prediction errors."""
    endpoints = {
        e["id"]: e for e in json.loads((work / "run.json").read_text())["endpoints"]
    }
    epp = _log(work / "epp.log", "body", {"hetero_route", "hetero_route_prediction"})
    gateway = _log(work / "gateway.log", "event", {"gateway_access"})
    workers = {
        e: _worker(work, e) for e, meta in endpoints.items() if meta["kind"] == "npu"
    }
    decisions, actual = Counter(), Counter()
    errors = defaultdict(lambda: defaultdict(list))
    requests = []
    warmed = set()
    measured_paths = set()
    profile = work / "profile"
    measured_sample = next(live_rows(profile / "requests.jsonl"), {})
    shape = (
        measured_sample.get("requested_input_tokens"),
        measured_sample.get("requested_output_tokens"),
    )
    verification = work / "verification.json"
    if verification.exists():
        verify = json.loads(verification.read_text())
        if verify.get("smoke") == "passed":
            for req in verify["requests"]:
                if (len(req["prompt_ids"]), len(req["token_ids"])) != shape:
                    continue
                target = endpoints[req["target"]]
                is_pdd = req["path"] == "pdd"
                if is_pdd and not req.get("worker_import"):
                    continue
                worker = workers.get(req["target"])
                native_ids = (
                    worker[1].get(server_request_id(req["request_id"]), ())
                    if worker
                    else ()
                )
                staged = any(
                    worker[0][r]["PREPARE_ALLOCATED_RECV_QUEUED"] for r in native_ids
                )
                path = "pdd" if is_pdd else f"{target['kind']}_full"
                warmed.add((path, req["target"], staged))
    files = ((False, "warmup_requests.jsonl"), (True, "requests.jsonl"))
    records = chain.from_iterable(
        ((measured, row) for row in live_rows(work / "profile" / filename))
        for measured, filename in files
    )
    for measured_row, row in records:
        if (
            not measured_row
            and (row.get("requested_input_tokens"), row.get("requested_output_tokens"))
            != shape
        ):
            continue
        rid = row["request_id"]
        route = _one([r for r in epp[rid] if r["body"] == "hetero_route"])
        pred = _one(
            [
                r
                for r in epp[rid]
                if r["body"] == "hetero_route_prediction"
                and r["epp_request_id"] == route.get("epp_request_id")
            ]
        ).get("estimate", {})
        gw = _one(gateway[rid])
        choice = route.get("path")
        target = endpoints.get(route.get("target"), {})
        kind = target.get("kind")
        label = f"{kind}_full" if choice == "full" and kind else choice or "unknown"
        worker = workers.get(target.get("id"))
        native = None
        events = defaultdict(list)
        if worker:
            ids = worker[1].get(server_request_id(rid), set())
            if len(ids) == 1:
                native = next(iter(ids))
                events = worker[0][native]
        selected = pred.get("Preload") is True
        approved = bool(_one(events["PREPARE_ALLOCATED_RECV_QUEUED"]))
        transfer = None
        overlap = None
        dest_port = target.get(
            "sidecar_port" if "decode" in target.get("roles", []) else "port"
        )
        headers = row["response_headers"]
        matches = (
            target
            and choice in {"full", "pdd"}
            and request_succeeded(row)
            and gw.get("http_status") == 200
            and gw.get("path") == choice == headers.get("x-hetero-path", choice)
            and gw.get("target")
            == target["id"]
            == headers.get("x-hetero-target", target["id"])
            and gw.get("prefill") == (route.get("prefill") or None)
            and (gw.get("destination") or "").rsplit(":", 1)[-1] == str(dest_port)
        )
        path = "unverified"
        if matches and choice == "full" and not events["kv_xfer_start"]:
            path = label
        elif matches and native and choice == "pdd":
            transfer = _transfer(
                row, target, endpoints.get(route["prefill"]), events, native
            )
            if transfer:
                path = "pdd"
                active = (target["engine"].get("lookahead") or {}).get(
                    "active_decode_limit"
                )
                if selected and approved:
                    overlap = _overlap(transfer, worker, native, active)
        if path != "unverified":
            (measured_paths if measured_row else warmed).add(
                (path, target["id"], approved)
            )
        if not measured_row:
            continue
        decisions[label] += 1
        actual[path] += 1
        measured = {}
        if request_succeeded(row) and row.get("first_output_monotonic") is not None:
            measured["TTFTMs"] = 1000 * (
                row["first_output_monotonic"] - row["actual_send_monotonic"]
            )
            measured["FinishMs"] = 1000 * (
                row["completed_monotonic"] - row["actual_send_monotonic"]
            )
        if transfer:
            measured.update(
                TransferMs=transfer["receive_ms"], ImportMs=transfer["import_ms"]
            )
        for key, value in measured.items():
            if key in pred and path != "unverified":
                errors[path][key].append((pred[key], value))
        requests.append(
            dict(
                request_id=rid,
                requested_path=label,
                executed_path=path,
                target=target.get("id"),
                gateway_destination=gw.get("destination"),
                estimate=pred or None,
                observed_ms=measured,
                preload_selected=selected,
                preload_approved=approved,
                transfer=transfer,
                overlap=overlap,
            )
        )
    final_state = {}
    for endpoint in workers:
        state = None
        for event in live_rows(work / f"{endpoint}-admission.jsonl"):
            if event["event"] == "STATE":
                state = event
        if state is not None:
            final_state[endpoint] = {
                k: state[k]
                for k in (
                    "total_credits",
                    "pending",
                    "pending_bytes",
                    "quarantined_count",
                    "virtual_growth_reserved_blocks",
                )
            }
    summary = {
        "last_admission_state": final_state,
        "warmed_paths": sorted(warmed),
        "measured_unwarmed_paths": sorted(measured_paths - warmed),
        "requested_paths": dict(decisions),
        "executed_paths": dict(actual),
        "preload_selected": sum(r["preload_selected"] for r in requests),
        "preload_approved": sum(r["preload_approved"] for r in requests),
        "actual_overlap": sum(r["overlap"] is not None for r in requests),
        "cost_comparison_ms": {
            path: {
                k: {
                    "samples": len(pairs),
                    "predicted_mean": mean(a for a, b in pairs),
                    "observed_mean": mean(b for a, b in pairs),
                    "mean_absolute_error": mean(abs(b - a) for a, b in pairs),
                }
                for k, pairs in costs.items()
            }
            for path, costs in errors.items()
        },
    }
    report = dict(summary=summary, requests=requests)
    (work / "profile" / "route_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return summary
