"""Run the native policy profiles through the actual Envoy, EPP and PD sidecar binaries.

The instrumented backends below serve synthetic OpenAI, Prometheus and KV fixtures.
Run from the repository root with python -m llmd.epp.tests.policy_process.
"""

import argparse
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from pathlib import Path
from urllib.request import urlopen

import yaml

from llmd.serving import render
from llmd.serving.protocol_test import fixture, get_port, request

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


class Traffic(list):
    def __init__(self):
        super().__init__()
        self.holds = {}

    def append(self, row):
        super().append(row)
        if hold := self.holds.get(row["request_id"]):
            hold.wait(15)


def serve_fixture(ep, traffic, state, failures):
    server = fixture(ep, traffic, {})
    handler = server.RequestHandlerClass
    upstream_reply = handler.reply

    def get(self):
        if self.path != "/metrics":
            return upstream_reply(self, b"{}")
        gauges = state[ep["id"]]
        wire = "".join(
            f"# TYPE {key} gauge\n{key} {val}\n" for key, val in gauges.items()
        )
        upstream_reply(self, wire.encode(), mime="text/plain")

    def reply(self, data, code=200, mime="application/json"):
        rid = self.headers.get("x-request-id")
        if rid in failures:
            return upstream_reply(self, b'{"error":"injected engine error"}', 500)
        if self.command == "POST":
            result = json.loads(data)
            if kv := result.get("kv_transfer_params"):
                size = ep["engine"]["block_size"]
                tokens = len(result["choices"][0]["prompt_token_ids"])
                kv["remote_num_tokens"] = tokens
                kv["remote_block_ids"] = [list(range((tokens + size - 1) // size))]
                result["usage"]["prompt_tokens"] = tokens
                data = json.dumps(result).encode()
        with suppress(BrokenPipeError, ConnectionResetError):
            upstream_reply(self, data, code, mime)

    handler.do_GET = get
    handler.reply = reply
    return server


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def wait_for(condition, message, timeout=12):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def exercise(policy, work, bin_dir, *, flip=False):
    work.mkdir(parents=True)
    available = set()
    while len(available) < 10:
        available.add(get_port())
    p = iter(available)
    ports = {
        k: next(p) for k in ("gateway", "epp", "epp_health", "epp_metrics", "admin")
    }
    endpoints = []
    state = {}
    traffic, failures, results = Traffic(), set(), []
    for order, (name, role, backend) in enumerate(
        [("gpu", "prefill", "cuda"), ("npu", "decode", "rbln")]
    ):
        labels = {
            "id": name,
            "backend": backend,
            "model": "demo",
            "kv-group": "demo",
            "slots": "1",
            "order": str(order),
            "full": "true",
            "producer" if role == "prefill" else "consumer": "true",
        }
        if role == "decode":
            labels["peer-gpu"] = "true"
        endpoints.append(
            dict(
                id=name,
                host="127.0.0.1",
                port=next(p),
                kv_port=next(p),
                sidecar_port=next(p) if role == "decode" else None,
                roles=[role, "full"],
                engine_id="fixture-" + name,
                tp=1,
                engine={"block_size": 2, "num_blocks": 256},
                labels={"pdd.sota.dev/" + k: v for k, v in labels.items()},
            )
        )
        state[name] = {
            "vllm:num_requests_waiting": 0,
            "vllm:num_requests_running": 0,
            "vllm:kv_cache_usage_perc": 0.1,
        }
    if policy == "least-queue":
        state["gpu"]["vllm:num_requests_waiting"] = 4
    if policy == "tiktok":
        # A cold multiprocess vLLM worker may publish only the scheduler's
        # actual RBLN observations until its first vLLM request completes.
        state["npu"] = {"hetero_pdd_waiting_requests": 0, "hetero_pdd_kv_usage": 0.1}
        state["npu"].update(
            {
                "hetero_pdd_running_requests": 1,
                "hetero_pdd_prepare_slots_free": 1,
                "hetero_pdd_prepare_bytes_free": 134217728,
            }
        )
    native = yaml.safe_load((PROFILES / f"{policy}.yaml").read_text())
    if policy == "trace-allocation":
        replay = next(
            x["parameters"]
            for x in native["plugins"]
            if x["type"] == "trace-allocation-handler"
        )
        replay["allocation"] = {"0": "gpu", "1": "npu", "2": "npu"}
        replay["pddTargets"] = ["npu"]
    if flip:
        for item in native["schedulingProfiles"][0]["plugins"]:
            if item["pluginRef"] == "prefer-rbln":
                item["weight"] = 4
    specs = render(
        work, native=native, endpoints=endpoints, ports=ports, bin_dir=bin_dir
    )
    url = f"http://127.0.0.1:{ports['gateway']}/v1/completions"

    def call(tokens=3, output=32, rid=None, expect=200, headers=None):
        rid = rid or str(uuid.uuid4())
        code, head, raw = request(
            url,
            {"model": "fixture", "prompt": [1] * tokens, "max_tokens": output},
            rid,
            headers,
        )
        if code != expect:
            raise AssertionError(
                f"{policy} {rid}: expected {expect}, HTTP {code}: {raw[:350]}"
            )
        row = {
            "request_id": rid,
            "status": code,
            "input_tokens": tokens,
            **{
                k: head.get("x-hetero-" + k)
                for k in ("path", "prefill", "target", "policy", "preload", "reason")
            },
            "stages": [r for r in traffic if r["request_id"] == rid],
        }
        results.append(row)
        if expect == 200:
            assert row["policy"] == policy and row["reason"], row
        return row

    def wait_healthy():
        try:
            with urlopen(
                f"http://127.0.0.1:{ports['admin']}/clusters", timeout=0.5
            ) as f:
                return (
                    f"ext_proc::127.0.0.1:{ports['epp']}::health_flags::healthy".encode()
                    in f.read()
                )
        except OSError:
            return False

    with ExitStack() as cleanup:
        for ep in endpoints:
            server = serve_fixture(ep, traffic, state, failures)
            cleanup.callback(server.server_close)
            cleanup.callback(server.shutdown)
        for spec in specs:
            log = cleanup.enter_context((work / (spec["name"] + ".log")).open("w"))
            proc = subprocess.Popen(
                spec["argv"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OTEL_TRACES_SAMPLER": "always_off", **spec["env"]},
            )
            cleanup.callback(stop_process, proc)
        try:
            wait_for(wait_healthy, "Envoy/EPP readiness", 55)
            time.sleep(1.5)
            if policy == "baseline":
                rows = [call() for _ in range(5)]
                assert all(r["path"] == "full" and len(r["stages"]) == 1 for r in rows)
            elif policy == "round-robin":
                assert [call()["target"] for _ in range(4)] == [
                    "gpu",
                    "npu",
                    "gpu",
                    "npu",
                ]
            elif policy == "example-weighted":
                assert call()["target"] == ("npu" if flip else "gpu")
            elif policy == "least-queue":
                assert call()["target"] == "npu"
                state["npu"]["vllm:num_requests_waiting"] = 6
                time.sleep(2)
                assert call()["target"] == "gpu"
                for signals in state.values():
                    del signals["vllm:num_requests_waiting"]
                time.sleep(2)
                assert not call(expect=503)["stages"]
            elif policy == "selective":
                assert call(3)["path"] == "full"
                split = call(128)
                assert split["path"] == "pdd" and [
                    r["stage"] for r in split["stages"]
                ] == ["produce", "consume"], split
                assert not split["stages"][-1]["transfer"].get("hetero_preload")
            elif policy == "tiktok":
                split = call(40)
                assert (
                    split["path"] == "pdd"
                    and split["stages"][-1]["transfer"]["hetero_preload"]
                ), split
                state["npu"]["hetero_pdd_prepare_slots_free"] = 0
                time.sleep(2)
                split = call(40)
                assert not split["stages"][-1]["transfer"].get("hetero_preload"), split
            elif policy == "trace-allocation":
                rows = [call(headers={"x-pdd-request-index": str(i)}) for i in range(3)]
                assert [(r["target"], r["path"]) for r in rows] == [
                    ("gpu", "full"),
                    ("npu", "pdd"),
                    ("npu", "full"),
                ]
                assert not call(headers={"x-pdd-request-index": "9"}, expect=429)[
                    "stages"
                ]
            elif policy == "fixed-pdd":
                assert [r["stage"] for r in call()["stages"]] == ["produce", "consume"]
            elif policy == "greedy":
                greedy_burst(call, traffic, failures, ports["gateway"])
        finally:
            (work / "requests.json").write_text(json.dumps(results, indent=2))
    return {"policy": policy, "requests": len(results), "status": "passed"}


def greedy_burst(call, traffic, failures, port):
    ids = [str(uuid.uuid4()) for _ in range(5)]
    traffic.holds.update({rid: threading.Event() for rid in ids[:3]})

    def seen(i):
        return [row for row in traffic if row["request_id"] == ids[i]]

    with ThreadPoolExecutor(4) as pool:
        try:
            a = pool.submit(call, rid=ids[0])
            wait_for(lambda: seen(0), "first slot")
            b = pool.submit(call, rid=ids[1], expect=500)
            wait_for(lambda: seen(1), "second slot")
            # Close an HTTP client while both slots are leased. A disconnected
            # waiter must not consume the next wakeup.
            cancelled = str(uuid.uuid4())
            payload = b'{"model":"fixture","prompt":[1],"max_tokens":1}'
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.sendall(
                    (
                        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                        f"x-request-id: {cancelled}\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(payload)}\r\n\r\n"
                    ).encode()
                    + payload
                )
                time.sleep(0.15)
            c = pool.submit(call, rid=ids[2])
            time.sleep(0.15)
            d = pool.submit(call, rid=ids[3])
            time.sleep(0.15)
            assert not seen(2) and not seen(3)
            failures.add(ids[1])
            traffic.holds[ids[1]].set()
            b.result(8)
            wait_for(lambda: seen(2), "error completion did not free second slot")
            assert seen(2)[0]["endpoint"] == seen(1)[0]["endpoint"] and not seen(3)
            traffic.holds[ids[0]].set()
            a.result(8)
            assert d.result(8)["target"] == seen(0)[0]["endpoint"]
            traffic.holds[ids[2]].set()
            c.result(8)
            assert not any(r["request_id"] == cancelled for r in traffic)
            call(rid=ids[4])
        finally:
            for event in traffic.holds.values():
                event.set()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--bin-dir")
    parser.add_argument("policies", nargs="*")
    opts = parser.parse_args()
    selected = opts.policies or [
        "baseline",
        "greedy",
        "round-robin",
        "least-queue",
        "fixed-pdd",
        "selective",
        "tiktok",
        "example-weighted",
        "trace-allocation",
    ]
    opts.work.mkdir(parents=True, exist_ok=False)
    summary = []
    for policy in selected:
        for flip in (False, True) if policy == "example-weighted" else (False,):
            sub = opts.work / (policy + ("-reweighted" if flip else ""))
            try:
                result = exercise(policy, sub, opts.bin_dir, flip=flip)
            except Exception as err:
                result = {"policy": policy, "status": "failed", "error": repr(err)}
            summary.append({**result, "directory": str(sub)})
            print(json.dumps(summary[-1]), flush=True)
            (opts.work / "policy-results.json").write_text(
                json.dumps(summary, indent=2)
            )
    return int(any(r["status"] != "passed" for r in summary))


if __name__ == "__main__":
    raise SystemExit(main())
