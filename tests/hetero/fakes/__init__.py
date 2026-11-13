# SPDX-License-Identifier: Apache-2.0
"""Temporary PATH executables and a Popen double; no real tool installation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import Mock

from tests.hetero.mock_server import (
    Handler,
    MockHTTPServer,
    MockServer,
    serve_until_signal,
)


def install_fake_tools(directory, versions=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    entries = {"all-smi": "fake_allsmi", "otelcol": "fake_otelcol"}
    entries.update({name: None for name in (versions or {}) if name not in entries})
    for name, entrypoint in entries.items():
        path = directory / name
        path.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "if sys.argv[1:] == ['--version']:\n"
            f"    print({(versions or {}).get(name, 'unversioned')!r})\n"
            "    raise SystemExit(0)\n"
            f"sys.path.insert(0, {str(repo)!r})\n"
            + (
                f"from tests.hetero.fakes import {entrypoint}\n{entrypoint}()\n"
                if entrypoint
                else "raise SystemExit('unsupported fake command')\n"
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)


def fake_allsmi():
    parser = argparse.ArgumentParser(description="Temporary all-smi test executable")
    parser.add_argument("command", choices=("api",))
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--interval", required=True, type=float)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    serve_until_signal(MockServer(role="all-smi", port=args.port))


class TraceHandler(Handler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.respond(status=200 if self.path == "/v1/traces" else 404)


def fake_otelcol():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error("--config must name an existing file")
    spans = Path(os.environ["FAKE_OTEL_SPANS"]).read_bytes()
    outdir = Path(os.environ["OTEL_OUT"])
    serve_until_signal(MockHTTPServer(TraceHandler, int(os.environ["OTLP_PORT"])))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "traces.jsonl").write_bytes(spans)


class FakePopen:
    def __init__(self):
        self.calls: list[dict] = []
        self.processes: list[Mock] = []

    def __call__(self, argv, *, env=None, **kwargs):
        if isinstance(argv, (str, bytes)) or kwargs.get("shell"):
            raise ValueError("fake Popen requires an argv sequence without a shell")
        environment = dict(os.environ if env is None else env)
        self.calls.append(dict(argv=list(argv), env=environment, kwargs=kwargs))
        process = Mock(
            spec=["args", "pid", "returncode", "poll"],
            args=list(argv),
            pid=10000 + len(self.processes),
            returncode=None,
        )
        process.poll.side_effect = lambda: process.returncode
        self.processes.append(process)
        return process
