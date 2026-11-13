# SPDX-License-Identifier: Apache-2.0
"""Loopback HTTP lifecycle and vLLM/all-smi doubles shared by WP13 fixtures."""

from __future__ import annotations

import json
import signal
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from tests.hetero.mock_protocol import DECODE_KV, completion_payload, metrics_text, sse


def post_completion(url, body, request_id):
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    return urlopen(
        Request(url + "/v1/completions", json.dumps(body).encode(), headers), timeout=5
    )


class MockHTTPServer(ThreadingHTTPServer):
    def __init__(self, handler, port=0):
        super().__init__(("127.0.0.1", port), handler)
        self.url = f"http://127.0.0.1:{self.server_port}"
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.shutdown()
        self.thread.join(timeout=5)
        self.server_close()


class Handler(BaseHTTPRequestHandler):
    server: MockHTTPServer

    def log_message(self, *_args):
        pass

    def read_request(self):
        received = time.monotonic()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        record = dict(
            request_id=self.headers.get("X-Request-Id"),
            body=body,
            headers={k.lower(): v for k, v in self.headers.items()},
            received_monotonic=received,
        )
        with self.server.lock:
            self.server.requests.append(record)
        return body

    def respond(self, data=b"{}", status=200, **headers):
        self.send_response(status)
        for key, value in {"Content-Type": "application/json", **headers}.items():
            self.send_header(key, value)
        self.end_headers()
        if data is not None:
            self.wfile.write(data)


def serve_until_signal(server):
    """Install handlers before announcing readiness; shutdown runs off serve_forever."""
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with server:
        print(server.url, flush=True)
        stopped.wait()


@dataclass(kw_only=True)
class MockServer(MockHTTPServer):
    port: int = 0
    role: str = "vllm"
    metrics: bool = True
    first_token_delay: float = 0
    token_delay: float = 0
    fail_ids: tuple[str, ...] = ()
    durations: list[float] = field(default_factory=list, init=False)
    running: int = field(default=0, init=False)

    def __post_init__(self):
        super().__init__(CompletionHandler, self.port)


class CompletionHandler(Handler):
    server: MockServer

    def do_GET(self):
        if self.path == "/health" or (
            self.path == "/-/ready" and self.server.role == "all-smi"
        ):
            self.respond()
        elif self.path == "/metrics" and self.server.metrics:
            with self.server.lock:
                data = metrics_text(
                    self.server.role, self.server.durations, self.server.running
                )
            self.respond(data, **{"Content-Type": "text/plain; version=0.0.4"})
        else:
            self.respond(status=404)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.respond(status=404)
            return
        body = self.read_request()
        rid = self.headers.get("X-Request-Id", "mock-request")
        server = self.server
        if rid in server.fail_ids:
            self.respond(b'{"error":"injected HTTP 500"}', status=500)
            return
        started = time.monotonic()
        with server.lock:
            server.running += 1
        try:
            count = body["max_tokens"]
            tokens = list(range(100, 100 + count))
            if body.get("stream"):
                self.respond(
                    None, **{"Content-Type": "text/event-stream", "X-Request-Id": rid}
                )
                for i, token in enumerate(tokens):
                    time.sleep(
                        server.first_token_delay if i == 0 else server.token_delay
                    )
                    reason = "length" if i == count - 1 else None
                    self.wfile.write(sse(completion_payload(body, [token], reason)))
                    self.wfile.flush()
                if body.get("stream_options", {}).get("include_usage"):
                    payload = completion_payload(body, [], None, usage=True)
                    payload["choices"] = []
                    self.wfile.write(sse(payload))
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                time.sleep(
                    server.first_token_delay + server.token_delay * max(0, count - 1)
                )
                payload = completion_payload(body, tokens, "length", usage=True)
                if server.role == "prefill":
                    payload["kv_transfer_params"] = DECODE_KV
                self.respond(json.dumps(payload).encode())
            with server.lock:
                server.durations.append(time.monotonic() - started)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A test client may deliberately disconnect.
        finally:
            with server.lock:
                server.running -= 1
