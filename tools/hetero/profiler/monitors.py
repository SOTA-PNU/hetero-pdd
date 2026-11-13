# SPDX-License-Identifier: Apache-2.0
import errno
import os
import socket
import subprocess
import sys
from pathlib import Path

from tools.hetero.profiler.expected import live_rows
from tools.hetero.profiler.log import log_line
from tools.hetero.profiler.scrape import fetch, wait_until

REPO = Path(__file__).resolve().parents[3]


def free_port(port=0, timeout=0):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        def available():
            try:
                listener.bind(("127.0.0.1", port))
                return True
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                return False

        wait_until(available, timeout, f"port {port}")
        return listener.getsockname()[1]


def port_ready(port):
    with socket.socket() as client:
        client.settimeout(0.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


def wait_for_layout(root, timeout=30):
    def ready():
        found = set()
        for role in ("decode0", "decode1"):
            for path in (root / "rbln_metrics" / role).glob("events_*.jsonl"):
                for row in live_rows(path):
                    if row.get("kind") == "kv_layout":
                        if row.get("use_host_buffer") is not True:
                            raise RuntimeError(f"{role}: host buffer is not enabled")
                        found.add(role)
        return len(found) == 2

    wait_until(ready, timeout, "kv_layout")


class Monitors:
    def __init__(self, root):
        self.root, self.processes = Path(root), {}

    def start(self, name, argv, env=None, probe=None):
        directory = self.root / "monitor"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{name}.log").open("ab") as output:
            io = dict(stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            process = subprocess.Popen(argv, env=env, cwd=REPO, stdout=output, **io)
        self.processes[name] = process
        log_line(name, f"start: {argv}")

        def ready():
            if process.poll() is not None:
                raise RuntimeError(f"{name}: exited with {process.returncode}")
            return probe()

        if probe:
            wait_until(ready, 60, name)

    def collector(self):
        port, out = free_port(), self.root / "otel"
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OTLP_PORT=str(port), OTEL_OUT=str(out))
        binary = os.environ.get("OTELCOL_BIN", "otelcol-contrib")
        argv = [binary, "--config", str(Path(__file__).with_name("otelcol.yaml"))]
        self.start("otelcol", argv, env, lambda: port_ready(port))
        return port

    def allsmi(self):
        port = free_port(int(os.environ.get("PDD_ALLSMI_PORT", "19090")), 30)
        url = f"http://127.0.0.1:{port}"
        binary = os.environ.get("ALLSMI_BIN", "all-smi")
        argv = [binary, "api", "--port", str(port), "--interval", "1"]
        ready = lambda: fetch(("ready", url + "/-/ready"))["status"] == 200
        self.start("all-smi", argv, probe=ready)
        return url + "/metrics"

    def scrape(self, endpoints, out):
        argv = [sys.executable, "-m", "tools.hetero.profiler.scrape", "--out", str(out)]
        for name, url in endpoints.items():
            argv += ["--endpoint", f"{name}={url}"]
        self.start("scrape", argv)

    def stop(self, name):
        if (process := self.processes.get(name)) is not None:
            if process.poll() is None:
                process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log_line(name, "stop timeout: SIGKILL")
                process.kill()
                code = process.wait(timeout=5)
            del self.processes[name]
            log_line(name, f"exit: {code}")
            return code
