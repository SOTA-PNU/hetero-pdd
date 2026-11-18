"""Send deterministic, model-tokenized completions through the common ingress."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.hetero.loadgen import make_prompt

from .engines import MODEL_IDENTITY
from .runtime import confirm_listeners, fetch, save


class PromptTokenizer:
    def __init__(self, snapshot):
        from tokenizers import Tokenizer

        path = Path(snapshot) / "tokenizer.json"
        if path.is_file():
            self.tokenizer = Tokenizer.from_file(str(path))
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                snapshot, local_files_only=True
            )

    def encode(self, text, *, add_special_tokens):
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return getattr(tokens, "ids", tokens)


def transferred(endpoint):
    raw, _ = fetch(endpoint["metrics"])
    # Prometheus exposes NIXL transfer telemetry as a histogram.
    totals = {}
    for key in ("sum", "count"):
        name = b"^vllm:nixl_bytes_transferred_" + key.encode()
        rows = re.findall(name + rb"(?:\{[^\n]*?\})?\s+([^\s]+)", raw, re.MULTILINE)
        values = [float(x) for x in rows]
        if not values or not all(math.isfinite(x) and x >= 0 for x in values):
            raise RuntimeError(f"missing NIXL {key} telemetry: {endpoint['metrics']}")
        totals[key] = sum(values)
    return totals


def reference_output(reference, model, run_id, prompts, conditions):
    if reference is None:
        return None
    source = json.loads(reference.read_text())
    if (
        source.get("schema_version") != 2
        or source.get("status") not in {"smoke_passed", "verified"}
        or source.get("correctness") not in {"not_evaluated", "passed"}
        or source.get("smoke") != "passed"
        or not source.get("gpu_only")
        or source.get("run_id") == run_id
        or source.get("conditions") != conditions
        or not source.get("requests")
        or any(source["model"].get(key) != model.get(key) for key in MODEL_IDENTITY)
        or any(r.get("path") != "full" or "error" in r for r in source["requests"])
    ):
        raise ValueError(
            "reference model, tokenizer, GPU full route or generation conditions differ"
        )
    if [r["prompt_ids"] for r in source["requests"]] != prompts:
        raise ValueError("reference prompts or request count differ")
    return source["requests"]


def worker_import(endpoint, row):
    # Collect evidence emitted by this consumer's connector, correlated with the
    # actual upstream vLLM producer handle and the request seen by the gateway.
    rows = []
    for path in Path(endpoint["events_dir"]).glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # a concurrent writer may have an incomplete final line
            if row["request_id"] in ev.get("req_id", ""):
                rows.append(ev)
    start = next(
        (
            v
            for v in rows
            if v["kind"] == "kv_xfer_start"
            and v.get("remote_engine_id") == row["remote_engine_id"]
            and v.get("remote_request_id") == row["remote_request_id"]
            and v.get("engine_id") == endpoint["engine_id"]
        ),
        None,
    )
    imported = next(
        (v for v in rows if v["kind"] == "kv_h2d" and v.get("num_blocks", 0) > 0), None
    )
    if (
        not start
        or not imported
        or start["req_id"] != imported["req_id"]
        or start["monotonic_ns"] > imported["monotonic_ns"]
    ):
        return None
    coverage = math.ceil(len(row["prompt_ids"]) / endpoint["engine"]["block_size"])
    if (
        row["blocks"] != coverage
        or imported["num_blocks"] != coverage
        or imported.get("bytes_est", 0) <= 0
        or any(
            sum(map(len, start[k])) != coverage
            for k in ("remote_block_ids", "local_block_ids")
        )
    ):
        return None
    return dict(
        consumer_request=start["req_id"],
        remote_blocks=start["remote_block_ids"],
        local_blocks=start["local_block_ids"],
        imported_blocks=imported["num_blocks"],
        bytes_est=imported["bytes_est"],
        h2d_ns=imported["h2d_ns"],
    )


def verify(work, state, reference, *, collect=None):
    model = state["model"]
    cfg = state["experiment"]["verify"]
    eps = {e["id"]: e for e in state["endpoints"]}
    conditions = dict(
        temperature=0,
        seed=0,
        max_tokens=cfg["output_tokens"],
        ignore_eos=True,
        generation_config="vllm",
        stream=False,
        dtypes=sorted({e["engine"]["dtype"] for e in eps.values()}),
        kv_dtypes=sorted({e["engine"]["kv_dtype"] for e in eps.values()}),
    )
    report = dict(
        schema_version=2,
        status="running",
        smoke="pending",
        transfer="not_evaluated",
        correctness="not_evaluated",
        run_id=state["run_id"],
        profile=state["experiment"]["profile"],
        model=model,
        conditions=conditions,
        gpu_only=all(e["kind"] == "gpu" for e in eps.values()),
        requests=[],
        reference=str(reference) if reference else None,
    )
    path = work / "verification.json"
    save(path, report)
    try:
        if state["status"] != "ready":
            raise ValueError(f"verification requires ready, run is {state['status']}")
        if collect is None:
            confirm_listeners(state)
        tokenizer = PromptTokenizer(model["tokenizer_path"])
        prompts = [
            make_prompt(tokenizer, cfg["input_tokens"], i, 17)
            for i in range(cfg["requests"])
        ]
        ref = reference_output(reference, model, state["run_id"], prompts, conditions)
        decoders = {k: e for k, e in eps.items() if "decode" in e["roles"]}
        before = {k: transferred(e) for k, e in decoders.items()}

        def request(i):
            rid = str(uuid.uuid4())
            row = dict(request_id=rid, prompt_ids=prompts[i])
            try:
                body = {
                    k: conditions[k]
                    for k in (
                        "temperature",
                        "seed",
                        "max_tokens",
                        "ignore_eos",
                        "stream",
                    )
                }
                body.update(
                    model=model["name"], prompt=prompts[i], return_token_ids=True
                )
                raw, h = fetch(
                    state["endpoint"],
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Request-Id": rid,
                        "X-Pdd-Request-Index": str(i),
                    },
                    timeout=180,
                )
                row.update(
                    {
                        k: h.get("x-hetero-" + k)
                        for k in (
                            "policy",
                            "reason",
                            "target",
                            "prefill",
                            "preload",
                            "path",
                        )
                    }
                )
                if h.get("x-request-id") != rid or row["target"] not in eps:
                    raise ValueError(
                        "missing request identity or selected EPP target/policy"
                    )
                ep = eps[row["target"]]
                if row["path"] == "pdd":
                    row.update(
                        remote_request_id=h.get("x-kv-remote-request-id", ""),
                        remote_engine_id=h.get("x-kv-remote-engine-id"),
                        blocks=int(h.get("x-kv-remote-blocks", 0)),
                    )
                    if (
                        row["prefill"] not in ep.get("peers", [])
                        or row["blocks"] <= 0
                        or row["remote_engine_id"] != eps[row["prefill"]]["engine_id"]
                        or not re.fullmatch(
                            r"cmpl-" + re.escape(rid) + r"-0(?:-[0-9a-f]{8})?",
                            row["remote_request_id"],
                        )
                    ):
                        raise ValueError(
                            "KV identity differs from selected pair or request"
                        )
                elif (
                    row["path"] != "full" or row["prefill"] or "full" not in ep["roles"]
                ):
                    raise ValueError("invalid full route")
                data = json.loads(raw)
                choice = data["choices"][0]
                ids = row["token_ids"] = choice["token_ids"]
                if (
                    choice.get("finish_reason") != "length"
                    or not isinstance(ids, list)
                    or len(ids) != cfg["output_tokens"]
                    or any(type(t) is not int or t < 0 for t in ids)
                    or data["usage"]["prompt_tokens"] != len(prompts[i])
                ):
                    raise ValueError(
                        "unexpected generation length, token IDs or finish reason"
                    )
            except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
                OSError,
                RuntimeError,
            ) as exc:
                row["error"] = str(exc)
            return row

        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            report["requests"] = list(pool.map(request, range(len(prompts))))
        if collect is not None:
            collect()
        # The EPP decision and the Envoy response must agree for each request.
        routes = {}
        predictions = {}
        with (work / "epp.log").open() as log:
            for line in log:
                if '"hetero_route_prediction"' in line:
                    pred = json.loads(line)
                    predictions[(pred["request_id"], pred["epp_request_id"])] = pred[
                        "estimate"
                    ]
                elif '"hetero_route"' in line:
                    event = json.loads(line)
                    routes.setdefault(event.get("request_id"), []).append(event)
        for row in report["requests"]:
            if row.get("error"):
                continue
            matches = [
                e
                for e in routes.get(row["request_id"], [])
                if e.get("target") == row["target"]
                and e.get("path") == row["path"]
                and e.get("policy") == row.get("policy")
                and row.get("reason")
                and (e.get("prefill") or None) == (row["prefill"] or None)
            ]
            if len(matches) != 1:
                row["error"] = "no unique native EPP decision matching gateway response"
                continue
            event = matches[0]
            row.update(
                policy=event["policy"],
                estimate=predictions.get((row["request_id"], event["epp_request_id"])),
                epp_request_id=event["epp_request_id"],
            )
        if errors := [r for r in report["requests"] if "error" in r]:
            raise RuntimeError(f"{len(errors)} request(s) failed: {errors[0]['error']}")
        observed = {r["path"] for r in report["requests"]}
        report["observed_paths"] = sorted(observed)
        if cfg.get("paths") and observed != set(cfg["paths"]):
            raise RuntimeError(
                f"expected {cfg['paths']} route(s), observed {sorted(observed)}"
            )
        report["smoke"] = "passed"
        pdd = [r for r in report["requests"] if r["path"] == "pdd"]
        if pdd:
            report["transfer"] = "failed"
            selected = Counter(r["target"] for r in pdd)
            deadline = time.monotonic() + 35
            while True:
                after = {k: transferred(eps[k]) for k in selected}
                delta = {k: after[k]["sum"] - before[k]["sum"] for k in selected}
                counts = {k: after[k]["count"] - before[k]["count"] for k in selected}
                needs = [r for r in pdd if eps[r["target"]].get("events_dir")]
                for row in needs:
                    row["worker_import"] = worker_import(eps[row["target"]], row)
                if all(
                    delta[k] > 0 and counts[k] >= selected[k] for k in selected
                ) and all(r["worker_import"] for r in needs):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"incomplete NIXL or worker import: {delta}, {counts}"
                    )
                time.sleep(0.5)
            report.update(
                transfer="passed" if len(needs) == len(pdd) else "aggregate_observed",
                nixl_bytes_delta_by_decoder=delta,
                nixl_transfer_count_delta_by_decoder=counts,
            )
        else:
            report["transfer"] = "not_applicable"
        if ref:
            mismatches = [
                i
                for i, r in enumerate(report["requests"])
                if r["token_ids"] != ref[i]["token_ids"]
            ]
            if mismatches:
                report.update(correctness="failed", mismatched_requests=mismatches)
                raise RuntimeError(
                    f"token IDs differ from reference for request(s) {mismatches}"
                )
            report["correctness"] = "passed"
        report["status"] = "verified" if ref else "smoke_passed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        if report["smoke"] == "pending":
            report["smoke"] = "failed"
        raise
    finally:
        save(path, report)
