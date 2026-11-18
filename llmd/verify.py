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

from .runtime import confirm_listeners, fetch, save


class PromptTokenizer:
    def __init__(self, snapshot):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))

    def encode(self, text, *, add_special_tokens):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


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


def reference_output(reference, model, run_id, prompts, output_tokens):
    if reference is None:
        return None
    source = json.loads(reference.read_text())
    same_model = all(source["model"][key] == model[key] for key in ("name", "revision"))
    if (
        not same_model
        or source["policy"] != "baseline"
        or source["status"] != "passed"
        or not source.get("gpu_only")
        or source["run_id"] == run_id
    ):
        raise ValueError(
            "reference must be a passed GPU baseline of the same model and revision"
        )
    if [row["prompt_ids"] for row in source["requests"]] != prompts:
        raise ValueError("reference input tokens / number of requests differ")
    if source["output_tokens"] != output_tokens:
        raise ValueError("reference output length differs")
    return source["requests"]


def verify(work, state, reference):
    model = state.get("model")
    config = state["experiment"]["verify"]
    policy = state["experiment"]["policy"]
    eps = {e["id"]: e for e in state["endpoints"]}
    prefix = "verify-" + uuid.uuid4().hex
    report = {
        "status": "running",
        "run_id": state["run_id"],
        "request_prefix": prefix,
        "policy": policy,
        "model": model,
        "gpu_only": all(e["kind"] == "gpu" for e in eps.values()),
        "output_tokens": config["output_tokens"],
        "requests": [],
        "reference": str(reference) if reference else None,
    }
    path = work / "verification.json"
    save(path, report)
    try:
        if state["status"] != "ready":
            raise ValueError(f"run is {state['status']}, verify requires ready")
        confirm_listeners(state)
        tokenizer = PromptTokenizer(model["path"])
        prompts = [
            make_prompt(tokenizer, config["input_tokens"], i, 17)
            for i in range(config["requests"])
        ]
        source = reference_output(
            reference, model, state["run_id"], prompts, config["output_tokens"]
        )
        decoders = {k: e for k, e in eps.items() if e["role"] == "decode"}
        before = {k: transferred(e) for k, e in decoders.items()}

        def request(i):
            rid = f"{prefix}-{i}"
            payload = {
                "model": model["name"],
                "prompt": prompts[i],
                "temperature": 0,
                "seed": 0,
                "max_tokens": config["output_tokens"],
                "ignore_eos": True,
                "return_token_ids": True,
                "stream": False,
            }
            row = {"request_id": rid, "prompt_ids": prompts[i]}
            try:
                raw, headers = fetch(
                    state["endpoint"],
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "X-Request-Id": rid},
                    timeout=180,
                )
                data = json.loads(raw)
                target = headers.get("X-EPP-Target")
                prefill = headers.get("X-EPP-Prefill")
                row.update(
                    target=target,
                    prefill=prefill,
                    remote_request_id=headers.get("X-KV-Remote-Request-Id"),
                )
                if headers.get("X-Request-Id") != rid or target not in eps:
                    raise ValueError("missing or invalid request ID or EPP target")
                if policy == "fixed-pdd":
                    if (
                        prefill not in eps
                        or eps[prefill]["role"] != "prefill"
                        or target not in decoders
                        or not row["remote_request_id"]
                    ):
                        raise ValueError(
                            "missing selected GPU prefill / NPU decode / KV request ID"
                        )
                elif prefill or eps[target]["role"] != "full":
                    raise ValueError("expected one whole-request engine")
                if data["choices"][0].get("finish_reason") != "length":
                    raise ValueError("engine returned an incomplete generation")
                ids = data["choices"][0]["token_ids"]
                row["token_ids"] = ids
                if (
                    not isinstance(ids, list)
                    or any(type(token) is not int or token < 0 for token in ids)
                    or len(ids) != config["output_tokens"]
                    or data["usage"]["prompt_tokens"] != len(prompts[i])
                ):
                    raise ValueError(f"unexpected input/output token lengths for {rid}")
                if source and ids != source[i]["token_ids"]:
                    raise ValueError("generated token IDs differ from the GPU baseline")
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

        with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            report["requests"] = list(pool.map(request, range(len(prompts))))
        if errors := [r for r in report["requests"] if "error" in r]:
            raise RuntimeError(
                f"{len(errors)} request(s) failed; first: {errors[0]['error']}"
            )
        if decoders:
            selected = Counter(r["target"] for r in report["requests"])
            deadline = time.monotonic() + 30
            while True:
                after = {k: transferred(decoders[k]) for k in selected}
                delta = {k: after[k]["sum"] - before[k]["sum"] for k in selected}
                counts = {k: after[k]["count"] - before[k]["count"] for k in selected}
                if all(delta[k] > 0 and counts[k] >= selected[k] for k in selected):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"insufficient decoder NIXL bytes/count: {delta}, {counts}"
                    )
                time.sleep(1)
            report["nixl_bytes_delta_by_decoder"] = delta
            report["nixl_transfer_count_delta_by_decoder"] = counts
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        save(path, report)
