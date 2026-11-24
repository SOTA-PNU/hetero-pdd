"""Exercise real Envoy, EPP and PD sidecars against HTTP engine protocol fixtures.

Run after build.sh. A fixture returns vLLM-shaped leases and deterministic IDs;
this does not exercise NIXL DMA, engine scheduling or model execution.
"""

import argparse
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from llmd.serving import HERE, render


def get_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def fixture(ep, records, faults):
    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, data, code=200, mime="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                (
                    b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                    b"vllm:kv_cache_usage_perc 0\n"
                )
                if self.path == "/metrics"
                else b"{}",
                mime="text/plain",
            )

        def do_POST(self):
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                chunks = []
                while size := int(self.rfile.readline().split(b";", 1)[0], 16):
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()
                self.rfile.readline()
                raw = b"".join(chunks)
            else:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            rid = self.headers["x-request-id"]
            kv = body.get("kv_transfer_params", {})
            result = {
                "id": "cmpl-" + rid,
                "object": "text_completion",
                "model": body["model"],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
                "choices": [
                    {
                        "text": "fixture",
                        "index": 0,
                        "finish_reason": "length",
                        "token_ids": [42],
                    }
                ],
            }
            if kv.get("do_remote_decode"):
                stage = "produce"
                result["choices"][0]["prompt_token_ids"] = faults.get(
                    "prompt_ids", body["prompt"]
                )
                handle = dict(
                    do_remote_prefill=True,
                    do_remote_decode=False,
                    remote_block_ids=[[4, 7]],
                    remote_engine_id=ep["engine_id"],
                    remote_request_id=result["id"] + "-0-12345678",
                    remote_host=ep["host"],
                    remote_port=ep["kv_port"],
                    tp_size=1,
                    remote_num_tokens=3,
                )
                handle.update(
                    {
                        k: v
                        for k, v in faults.items()
                        if k not in {"phase", "prompt_ids", "no_handle"}
                    }
                )
                if not faults.get("no_handle"):
                    result["kv_transfer_params"] = handle
            elif kv.get("do_remote_prefill"):
                stage = "consume"
            else:
                stage = "full"
            records.append(
                dict(endpoint=ep["id"], stage=stage, request_id=rid, transfer=kv)
            )
            if faults.get("phase") == stage:
                self.reply(b'{"error":"fixture-engine-unavailable"}', 503)
            else:
                self.reply(json.dumps(result).encode())

    server = ThreadingHTTPServer(("0.0.0.0", ep["port"]), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def request(url, body, rid, headers=None):
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-request-id": rid,
            **(headers or {}),
        },
    )
    try:
        response = urlopen(req, timeout=12)
    except HTTPError as err:
        response = err
    with response:
        return (
            response.status,
            dict((k.lower(), v) for k, v in response.headers.items()),
            response.read().decode(),
        )


def protocol(work, bin_dir=None):
    work.mkdir(parents=True, exist_ok=False)
    free = set()
    while len(free) < 17:
        free.add(get_port())
    p = iter(free)
    ports = {
        key: next(p) for key in ("gateway", "epp", "epp_health", "epp_metrics", "admin")
    }
    eps, traffic, faults, results = [], [], {}, []
    for idx, name in enumerate(("p0", "p1", "d0", "d1")):
        is_p = name.startswith("p")
        labels = {
            "id": name,
            "order": str(idx),
            "model": "fixture",
            "kv-group": "fixture",
            "slots": "1",
            "producer" if is_p else "consumer": "true",
        }
        if not is_p:
            labels.update({"peer-p0": "true", "peer-p1": "true"})
        eps.append(
            {
                "id": name,
                "host": "127.0.0.1",
                "roles": ["prefill" if is_p else "decode"],
                "port": next(p),
                "sidecar_port": None if is_p else next(p),
                "kv_port": next(p),
                "sidecar_metrics_port": 0 if is_p else next(p),
                "engine_id": "fixture-engine-" + name,
                "tp": 1,
                "engine": {"block_size": 2, "num_blocks": 256},
                "labels": {"pdd.sota.dev/" + k: v for k, v in labels.items()},
            }
        )
    native = yaml.safe_load((HERE.parent / "epp/profiles/fixed-pdd.yaml").read_text())
    contract = next(pl for pl in native["plugins"] if pl["type"] == "route-contract")
    contract.setdefault("parameters", {})["servedModel"] = "fixture"
    specs = render(work, native=native, endpoints=eps, ports=ports, bin_dir=bin_dir)
    with ExitStack() as cleanup:
        for ep in eps:
            engine = fixture(ep, traffic, faults)
            cleanup.callback(engine.server_close)
            cleanup.callback(engine.shutdown)
        processes = []
        for spec in specs:
            log = cleanup.enter_context((work / (spec["name"] + ".log")).open("w"))
            proc = subprocess.Popen(
                spec["argv"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **spec["env"]},
            )
            cleanup.callback(proc.wait, 12)
            cleanup.callback(proc.terminate)
            processes.append(proc)
        dst = f"http://127.0.0.1:{ports['gateway']}/v1/completions"
        body = {
            "model": "fixture",
            "prompt": [1, 2, 3],
            "max_tokens": 1,
            "stream": False,
        }
        deadline = time.monotonic() + 35
        while True:
            if any(proc.poll() is not None for proc in processes):
                raise RuntimeError(f"a serving process exited, see {work}")
            try:
                # Wait for Envoy to finish its initial gRPC health check on EPP.
                with urlopen(
                    f"http://127.0.0.1:{ports['admin']}/clusters", timeout=1
                ) as r:
                    if (
                        b"ext_proc::127.0.0.1:"
                        + str(ports["epp"]).encode()
                        + b"::health_flags::healthy"
                        in r.read()
                    ):
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"gateway/ext_proc readiness timeout, see {work}")
            time.sleep(0.1)

        def exercise(label, payload, status, changed=None):
            rid = str(uuid.uuid4())
            code, headers, raw = request(dst, payload, rid, changed)
            seen = [row for row in traffic if row["request_id"] == rid]
            results.append(
                dict(
                    case=label,
                    status=code,
                    request_id=rid,
                    headers=headers,
                    stages=seen,
                    response=raw,
                )
            )
            if code != status:
                raise AssertionError(
                    f"{label} expected {status}, got {code}: {raw[:450]}"
                )
            return seen, headers

        try:
            seen, headers = exercise(
                "pair",
                body,
                200,
                {"x-hetero-preload": "1", "x-prefiller-host-port": "127.0.0.1:1"},
            )
            assert [r["stage"] for r in seen] == ["produce", "consume"], seen
            assert "hetero_preload" not in seen[1]["transfer"]
            assert (
                headers["x-hetero-path"] == "pdd"
                and headers["x-hetero-target"] == seen[1]["endpoint"]
            ), headers
            assert (
                headers["x-hetero-prefill"] == seen[0]["endpoint"]
                and headers["x-kv-remote-blocks"] == "2"
            ), headers
            inventory_path = work / "endpoints.yaml"
            endpoints = yaml.safe_load(inventory_path.read_text())
            removed = {row["endpoint"] for row in seen}
            endpoints["endpoints"] = [
                ep for ep in endpoints["endpoints"] if ep["name"] not in removed
            ]
            temp = inventory_path.with_suffix(".tmp")
            temp.write_text(yaml.safe_dump(endpoints))
            temp.replace(inventory_path)
            for _ in range(50):
                seen, _ = exercise("reloaded-pair", body, 200)
                if all(row["endpoint"] not in removed for row in seen):
                    break
                time.sleep(0.1)
            else:
                raise AssertionError(
                    "file-discovery did not move traffic to the second P/D pair"
                )
            for field, invalid in (
                ("prompt_ids", [1, 9, 3]),
                ("no_handle", True),
                ("remote_engine_id", "other"),
                ("remote_host", "192.0.2.1"),
                ("remote_port", 1),
                ("remote_request_id", "stale"),
                ("remote_block_ids", [[7]]),
                ("remote_block_ids", [[5, 257]]),
                ("remote_num_tokens", 1),
                ("do_remote_prefill", False),
            ):
                faults[field] = invalid
                case = "reject-" + field
                if field == "remote_block_ids" and len(invalid[0]) == 2:
                    case += "-capacity"
                seen, _ = exercise(case, body, 502)
                assert [r["stage"] for r in seen] == ["produce"], seen
                faults.clear()
            for stage in ("produce", "consume"):
                faults["phase"] = stage
                seen, _ = exercise(stage + "-unavailable", body, 503)
                assert [r["stage"] for r in seen] == (
                    ["produce"] if stage == "produce" else ["produce", "consume"]
                )
                faults.clear()
            rid = str(uuid.uuid4())
            consumer = next(ep for ep in eps if "decode" in ep["roles"])
            sidecar = f"http://127.0.0.1:{consumer['sidecar_port']}/v1/completions"
            code, headers, raw = request(
                sidecar,
                body,
                rid,
                {"x-hetero-path": "pdd", "x-hetero-target": consumer["id"]},
            )
            seen = [row for row in traffic if row["request_id"] == rid]
            results.append(
                dict(
                    case="missing-producer-route",
                    request_id=rid,
                    status=code,
                    response=raw,
                    stages=seen,
                )
            )
            assert code == 400 and not seen
            for ep in eps:
                if ep["sidecar_metrics_port"]:
                    url = f"http://127.0.0.1:{ep['sidecar_metrics_port']}/metrics"
                    with urlopen(url, timeout=4) as response:
                        metrics = response.read()
                    assert b"llm_d" in metrics
                    (work / (ep["id"] + "-sidecar.prom")).write_bytes(metrics)
            seen, _ = exercise(
                "reject-reserved-input", {**body, "kv_transfer_params": {}}, 400
            )
            assert not seen, seen
            seen, _ = exercise("reject-model", {**body, "model": "other"}, 400)
            assert not seen, seen
        finally:
            (work / "protocol-results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--bin-dir")
    opts = parser.parse_args()
    rows = protocol(opts.work, opts.bin_dir)
    print(json.dumps({"checks": len(rows), "passed": True, "path": str(opts.work)}))
