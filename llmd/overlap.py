"""Verify one two-request host-staging sequence through the managed gateway."""

import argparse
import hashlib
import http.client
import json
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from urllib.parse import urlsplit

from tools.hetero.loadgen import make_prompt

from .runtime import confirm_listeners, fetch, save
from .verify import PromptTokenizer


def _rows(paths):
    rows = []
    for path in paths:
        if path.exists():
            # An open worker stream may be partway through the last record.
            lines = path.read_text().splitlines(keepends=True)
            rows.extend(json.loads(line) for line in lines if line.endswith("\n"))
    return rows


def _proof(work, decoder, first, following):
    trace = Path(decoder["events_dir"])
    rows = _rows(sorted(trace.glob("events_*.jsonl")))

    def find_one(kind, rid):
        found = [r for r in rows if r.get("kind") == kind and r.get("req_id") == rid]
        if len(found) != 1:
            raise ValueError(f"expected one {kind} event for {rid}, found {len(found)}")
        return found[0]

    def decoded(rid):
        return [
            r
            for r in rows
            if r.get("kind") == "model_call"
            and r.get("phase") == "decode"
            and rid in r["request_ids"]
        ]

    def internal(client):
        prefix = client["completion_id"] + "-"
        matches = {
            r["req_id"]
            for r in rows
            if r.get("kind") == "kv_xfer_start" and r["req_id"].startswith(prefix)
        }
        if len(matches) != 1:
            raise ValueError(
                f"expected one matching native read for {prefix}, found {matches}"
            )
        return matches.pop()

    a, b = internal(first), internal(following)
    a_steps, b_steps = decoded(a), decoded(b)
    if any(
        r.get("kind") == "model_call"
        and r.get("phase") == "prefill"
        and b in r["request_ids"]
        for r in rows
    ):
        raise ValueError("preloaded request executed local decoder prefill")
    estimates = [
        r
        for r in _rows([work / "epp.log"])
        if r.get("body") == "hetero_route_prediction"
        and r.get("request_id") == following["client_request_id"]
    ]
    if len(estimates) != 1 or estimates[0]["estimate"].get("Preload") is not True:
        raise ValueError("missing native EPP authorization for preloaded request")
    receive = find_one("kv_xfer_start", b)
    ready = find_one("kv_host_ready", b)
    copied = find_one("kv_h2d", b)
    if not a_steps or not b_steps:
        raise ValueError("both requests need real decoder forward-call envelopes")
    start, end = receive["monotonic_ns"], ready["monotonic_ns"]
    intersections = [
        min(end, step["span_end_ns"]) - max(start, step["span_start_ns"])
        for step in a_steps
    ]
    overlap = max(intersections)
    # RBLN's compiled forward returns a device tensor asynchronously. The
    # existing step event extends through sampling/synchronizing its result.
    # Attribute each relevant real step to its enclosed request-scoped call.
    proc = ready.get("last_proc_ns")
    joined = []
    for event in rows:
        if (
            event.get("kind") != "step"
            or event.get("phase") != "decode"
            or not event.get("sampled")
        ):
            continue
        lo, hi = event["span_start_ns"], event["span_end_ns"]
        if max(lo, start) >= min(hi, end):
            continue
        calls = [
            c for c in a_steps if lo <= c["span_start_ns"] <= c["span_end_ns"] <= hi
        ]
        if len(calls) == 1 and calls[0]["request_ids"] == [a]:
            joined.append((event, calls[0]))
    witness = next(
        (
            (step, call)
            for step, call in joined
            if step["span_start_ns"] <= end <= step["span_end_ns"]
            or (proc and step["span_start_ns"] <= proc <= step["span_end_ns"])
        ),
        None,
    )
    scheduler = _rows([work / f"{decoder['id']}-admission.jsonl"])
    alloc = [
        r
        for r in scheduler
        if r["event"] == "PREPARE_ALLOCATED_RECV_QUEUED" and r["request_id"] == b
    ]
    grant = [
        r for r in scheduler if r["event"] == "IMPORT_GRANTED" and r["request_id"] == b
    ]
    terminal = [
        r
        for r in scheduler
        if r["event"] == "REQUEST_TERMINAL" and r["request_id"] == a
    ]
    if len(alloc) != 1 or len(grant) != 1 or len(terminal) != 1:
        raise ValueError(
            "missing reservation, import permission, or previous decode completion"
        )
    if alloc[0]["active_count"] <= 0 or alloc[0]["token_compute_scheduled"]:
        raise ValueError(
            "next receive did not reserve a separate slot while decode ran"
        )
    if (
        (receive["remote_request_id"] != following["remote_request_id"])
        or receive["remote_engine_id"] != following["producer_engine"]
        or receive["engine_id"] != decoder["engine_id"]
        or receive["local_block_ids"] != alloc[0]["local_block_ids"]
        or not any(receive["remote_block_ids"])
        or alloc[0]["prompt_token_count"] != following["input_tokens"]
        or copied["num_blocks"] != alloc[0]["allocated_blocks"]
        or copied["bytes_est"] != alloc[0]["allocated_bytes"]
    ):
        raise ValueError(
            "remote lease, token range, block mapping or imported bytes differ"
        )
    if not (
        a_steps[0]["span_start_ns"]
        < start
        < end
        <= terminal[0]["monotonic_ns"]
        <= grant[0]["monotonic_ns"]
        <= copied["start_ns"]
        < copied["end_ns"]
        <= b_steps[0]["span_start_ns"]
    ):
        raise ValueError(
            "previous decode, host-ready, import and next decode are out of order"
        )
    if not witness:
        raise ValueError(
            "no independent NIXL progress/completion inside an identified decode step"
        )
    step, call = witness
    return dict(
        native_epp_estimate=estimates[0]["estimate"],
        first_decoder_request=a,
        next_decoder_request=b,
        transfer=receive,
        host_ready=ready,
        allocation=alloc[0],
        previous_decode_terminal=terminal[0],
        import_grant=grant[0],
        import_done=copied,
        first_next_decode=b_steps[0],
        max_model_call_receive_overlap_ns=overlap,
        decode_step=step,
        model_call_in_step=call,
        in_progress_inside_step=bool(
            proc and step["span_start_ns"] <= proc <= step["span_end_ns"]
        ),
        done_observed_inside_step=step["span_start_ns"] <= end <= step["span_end_ns"],
        step_receive_overlap_ns=min(end, step["span_end_ns"])
        - max(start, step["span_start_ns"]),
        host_dwell_ns=copied["start_ns"] - end,
        timing="host-monotonic model and sampler step, independent NIXL polling",
    )


def _handoff_timing(proof):
    # Compare the same local decoder boundary even if anchor generation time
    # drifts between repetitions. The host import remains on the critical path.
    terminal = proof["previous_decode_terminal"]["monotonic_ns"]
    begin = proof["transfer"]["monotonic_ns"]
    ready = proof["host_ready"]["monotonic_ns"]
    imported = proof["import_done"]
    decode = proof["first_next_decode"]["span_start_ns"]
    return dict(
        previous_terminal_to_next_decode_ms=(decode - terminal) / 1e6,
        next_receive_ms=(ready - begin) / 1e6,
        next_import_ms=(imported["end_ns"] - imported["start_ns"]) / 1e6,
    )


def _cancel_held_request(endpoint, scheduler_path, body):
    """Close a real gateway connection after observing its engine reservation."""
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
    rid = str(uuid.uuid4())

    def records():
        return [
            row
            for row in _rows([scheduler_path])
            if (row.get("request_id") or "").startswith(f"cmpl-{rid}-")
        ]

    try:
        connection.request(
            "POST",
            url.path,
            json.dumps(body),
            {"Content-Type": "application/json", "X-Request-Id": rid},
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(r["event"] == "PREPARE_ALLOCATED_RECV_QUEUED" for r in records()):
                break
            time.sleep(0.01)
        else:
            raise ValueError("client cancellation fixture never acquired host blocks")
        connection.sock.shutdown(socket.SHUT_RDWR)
    finally:
        connection.close()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        rows = records()
        aborted = any(
            r["event"] == "REQUEST_TERMINAL" and r["status"] == "FINISHED_ABORTED"
            for r in rows
        )
        released = any(
            r["event"] == "BLOCKS_RELEASED" and r["pending_credit_released"]
            for r in rows
        )
        if (
            aborted
            and released
            and not any(r["event"] == "IMPORT_GRANTED" for r in rows)
        ):
            return dict(client_request_id=rid, records=rows)
        time.sleep(0.05)
    raise ValueError(
        f"did not observe canceled reservation and safe block release: {records()}"
    )


def run(work, cancel=False, ordinary=False, arrival_offset_s=2.0):
    if not 0 < arrival_offset_s < 180:
        raise ValueError("arrival offset must be between zero and the request timeout")
    state = json.loads((work / "run.json").read_text())
    if state["status"] != "ready":
        raise ValueError("ready TikTok run required")
    confirm_listeners(state)
    endpoints = {ep["id"]: ep for ep in state["endpoints"]}
    decoders = [
        ep
        for ep in endpoints.values()
        if ep.get("events_dir") and ep["engine"].get("lookahead")
    ]
    if len(decoders) != 1:
        raise ValueError(
            "two-request verification requires one traced lookahead decoder"
        )
    decoder = decoders[0]
    tokenizer = PromptTokenizer(state["model"]["tokenizer_path"])
    result = {
        "run_id": state["run_id"],
        "endpoint": state["endpoint"],
        "status": "running",
    }
    path = work / (
        "ordinary-verification.json" if ordinary else "overlap-verification.json"
    )
    if cancel:
        path = work / "cancel-verification.json"
    anchor_sent = Event()
    anchor_start = []

    def payload(n, output, index):
        return dict(
            model=state["model"]["name"],
            prompt=make_prompt(tokenizer, n, index, 17),
            max_tokens=output,
            temperature=0,
            seed=0,
            ignore_eos=True,
            return_token_ids=True,
            stream=False,
        )

    def completion(n, output, index, anchor=False):
        rid = str(uuid.uuid4())
        body = payload(n, output, index)
        sent = time.monotonic()
        if anchor:
            anchor_start.append(sent)
            anchor_sent.set()
        data, headers = fetch(
            state["endpoint"],
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Request-Id": rid},
            timeout=180,
        )
        completed = time.monotonic()
        reply = json.loads(data)
        if (
            reply["choices"][0].get("finish_reason") != "length"
            or len(reply["choices"][0]["token_ids"]) != output
        ):
            raise ValueError("incomplete completion")
        if (
            headers.get("x-hetero-path") != "pdd"
            or headers.get("x-hetero-target") != decoder["id"]
        ):
            raise ValueError(
                f"native profile chose a different route for {rid}: {dict(headers)}"
            )
        return dict(
            client_request_id=rid,
            sent_monotonic=sent,
            completed_monotonic=completed,
            payload=body,
            completion_id=reply["id"],
            input_tokens=n,
            output_tokens=output,
            token_ids=reply["choices"][0]["token_ids"],
            producer=headers["x-hetero-prefill"],
            producer_engine=endpoints[headers["x-hetero-prefill"]]["engine_id"],
            remote_request_id=headers.get("x-kv-remote-request-id"),
        )

    save(path, result)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(completion, 128, 896, 0, True)
            if not anchor_sent.wait(20):
                raise RuntimeError("anchor did not start")
            # Both arms release the same prompt pair at the same client offset.
            # Sampling the decoder or scraping EPP in this loop perturbs arrival.
            time.sleep(max(0.0, anchor_start[0] + arrival_offset_s - time.monotonic()))
            if first.done():
                first.result()
                raise RuntimeError("anchor completed before the second arrival")
            if cancel:
                result["cancellation"] = _cancel_held_request(
                    state["endpoint"],
                    work / f"{decoder['id']}-admission.jsonl",
                    payload(64, 32, 1),
                )
                result["first"] = first.result()
                # The same producer/decoder must complete another genuine PD request.
                result["recovery"] = completion(128, 32, 2)
                result["status"] = "passed"
                return
            following = completion(64, 32, 1)
            result.update(first=first.result(), following=following)
        a, b = result["first"], result["following"]
        result["pair_timing"] = dict(
            requested_arrival_offset_ms=1000 * arrival_offset_s,
            actual_arrival_offset_ms=1000 * (b["sent_monotonic"] - a["sent_monotonic"]),
            anchor_service_ms=1000 * (a["completed_monotonic"] - a["sent_monotonic"]),
            following_service_ms=1000
            * (b["completed_monotonic"] - b["sent_monotonic"]),
            makespan_ms=1000
            * (
                max(a["completed_monotonic"], b["completed_monotonic"])
                - a["sent_monotonic"]
            ),
        )
        if ordinary:
            # Import metrics may be buffered until the worker's next event.
            # A post-pair unmeasured request also exercises slot reuse.
            result["recovery"] = completion(128, 32, 2)
            events = _rows(sorted(Path(decoder["events_dir"]).glob("events_*.jsonl")))
            sched = _rows([work / f"{decoder['id']}-admission.jsonl"])
            native = []
            for req in (a, b):
                ids = [
                    e["req_id"]
                    for e in events
                    if e["kind"] == "kv_xfer_start"
                    and e["remote_request_id"] == req["remote_request_id"]
                ]
                if len(ids) != 1 or any(
                    e["event"] == "PREPARE_ALLOCATED_RECV_QUEUED"
                    and e["request_id"] in ids
                    for e in sched
                ):
                    raise ValueError(
                        "ordinary control lacks a unique non-preloaded KV read"
                    )
                if not any(
                    e["kind"] == "kv_h2d" and e["req_id"] == ids[0] for e in events
                ):
                    raise ValueError("ordinary control lacks a completed KV import")
                native.append(ids[0])
            prior, following = native
            proof = {
                "previous_decode_terminal": next(
                    r
                    for r in sched
                    if r["event"] == "REQUEST_TERMINAL" and r["request_id"] == prior
                ),
                "first_next_decode": next(
                    r
                    for r in events
                    if r["kind"] == "model_call"
                    and r["phase"] == "decode"
                    and following in r["request_ids"]
                ),
            }
            for field, kind in (
                ("transfer", "kv_xfer_start"),
                ("host_ready", "kv_xfer_done"),
                ("import_done", "kv_h2d"),
            ):
                proof[field] = next(
                    r for r in events if r["kind"] == kind and r["req_id"] == following
                )
            if not (
                proof["previous_decode_terminal"]["monotonic_ns"]
                <= proof["transfer"]["monotonic_ns"]
                <= proof["host_ready"]["monotonic_ns"]
                <= proof["import_done"]["start_ns"]
                <= proof["import_done"]["end_ns"]
                <= proof["first_next_decode"]["span_start_ns"]
            ):
                raise ValueError("ordinary control received/imported KV out of order")
            result["proof"] = proof
            result["pair_timing"].update(_handoff_timing(proof))
            result["status"] = "passed"
            return
        # Also verify identical inputs on the ordinary PD route after both
        # credits return and the native data source samples the idle decoder.
        time.sleep(1.2)
        control = completion(64, 32, 1)
        result["ordinary_pd_reference"] = control
        if following["token_ids"] != control["token_ids"]:
            raise ValueError("preloaded output differs from ordinary PD token IDs")
        records = _rows([work / f"{decoder['id']}-admission.jsonl"])
        if any(
            r["event"] == "PREPARE_ALLOCATED_RECV_QUEUED"
            and (r["request_id"] or "").startswith(control["completion_id"] + "-")
            for r in records
        ):
            raise ValueError("reference request also used preloading")
        error = None
        for _ in range(15):
            try:
                result["proof"] = _proof(work, decoder, result["first"], following)
                break
            except ValueError as exc:
                error = exc
                time.sleep(0.2)
        else:
            raise error
        result["pair_timing"].update(_handoff_timing(result["proof"]))
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        save(path, result)


def compare_gpu(work, gpu_work):
    staged = json.loads((work / "overlap-verification.json").read_text())
    original = json.loads((work / "run.json").read_text())
    gpu = json.loads((gpu_work / "run.json").read_text())
    identity = (
        "name",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "tokenizer_sha256",
        "config_sha256",
    )
    if staged["status"] != "passed" or staged["run_id"] != original["run_id"]:
        raise ValueError("a successful matched overlap proof is required")
    if (
        gpu["status"] != "ready"
        or gpu["run_id"] == original["run_id"]
        or any(e["kind"] != "gpu" or e["roles"] != ["full"] for e in gpu["endpoints"])
        or any(gpu["model"][key] != original["model"][key] for key in identity)
    ):
        raise ValueError(
            "a separate ready GPU baseline of identical model/tokenizer is required"
        )
    for key in ("dtype", "kv_dtype", "max_model_len", "block_size"):
        if {e["engine"][key] for e in gpu["endpoints"]} != {
            e["engine"][key] for e in original["endpoints"]
        }:
            raise ValueError(f"different comparison condition: {key}")
    # The launcher fingerprints mutable exports; pinned Hub snapshots carry
    # their revision. Record the underlying shard bytes for this comparison.
    weights = {}
    for model in (original["model"], gpu["model"]):
        root = Path(model["path"]).resolve()
        if root in weights:
            continue
        shards = sorted([*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")])
        if not shards:
            raise ValueError(f"missing weight shards in {root}")
        digest = hashlib.sha256()
        for shard in shards:
            with shard.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
        weights[root] = digest.hexdigest()
    if len(set(weights.values())) != 1:
        raise ValueError("independent reference uses different model weight bytes")
    confirm_listeners(gpu)
    report = dict(
        weights_sha256=next(iter(weights.values())),
        overlap_run=original["run_id"],
        reference_run=gpu["run_id"],
        reference_path=str(gpu_work),
        result_by_request={},
    )
    path = work / "overlap-gpu-reference.json"
    for key in ("first", "following"):
        req = staged[key]
        rid = str(uuid.uuid4())
        raw, headers = fetch(
            gpu["endpoint"],
            data=json.dumps(req["payload"]).encode(),
            headers={"Content-Type": "application/json", "X-Request-Id": rid},
            timeout=180,
        )
        choice = json.loads(raw)["choices"][0]
        if (
            choice.get("finish_reason") != "length"
            or headers.get("x-hetero-path") != "full"
            or headers.get("x-request-id") != rid
            or len(choice["token_ids"]) != req["output_tokens"]
        ):
            raise ValueError("invalid independent GPU reference completion")
        ids = choice["token_ids"]
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(ids, req["token_ids"])) if x != y), None
        )
        report["result_by_request"][key] = dict(
            client_request_id=rid,
            target=headers["x-hetero-target"],
            token_ids=ids,
            exact=first_diff is None,
            first_different_token_index=first_diff,
        )
    report["correctness"] = (
        "passed"
        if all(r["exact"] for r in report["result_by_request"].values())
        else "failed"
    )
    save(path, report)
    if report["correctness"] != "passed":
        raise ValueError(f"GPU token mismatch, see {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cancel", action="store_true")
    mode.add_argument(
        "--ordinary",
        action="store_true",
        help="measure the matched preload-off PDD control",
    )
    parser.add_argument("--arrival-offset-s", type=float, default=2.0)
    mode.add_argument("--gpu-reference-run", type=Path)
    args = parser.parse_args()
    if args.gpu_reference_run:
        compare_gpu(args.work_dir.resolve(), args.gpu_reference_run.resolve())
    else:
        run(
            args.work_dir.resolve(),
            cancel=args.cancel,
            ordinary=args.ordinary,
            arrival_offset_s=args.arrival_offset_s,
        )
