"""Linux process ownership and readiness for one persistent experiment."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def run_lock(work, *, create=False):
    if create:
        work.mkdir(parents=True, exist_ok=True)
    with (work / ".llmd.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another lifecycle command is active in {work}"
            ) from exc
        yield


@contextlib.contextmanager
def claim_devices(endpoints):
    # Close the race between two local launches before the driver reports a
    # context. The kernel drops these transient Linux socket names on exit.
    with contextlib.ExitStack() as stack:
        for ep in endpoints:
            device = f"{ep['kind']}{ep['device']}"
            sock = stack.enter_context(socket.socket(socket.AF_UNIX))
            try:
                sock.bind(f"\0hetero-llmd-{os.getuid()}-{device}")
            except OSError as exc:
                raise RuntimeError(f"another local launch is using {device}") from exc
        yield


def check_ports(ports):
    for port in ports:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"port {port} unavailable: {exc}") from exc


def spawn(name, argv, env, work, state):
    with (work / f"{name}.log").open("xb") as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env={**env, "HETERO_LLMD_RUN_ID": state["run_id"]},
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    state["processes"][name] = {"pid": proc.pid, "log": str(log.name)}
    return proc


def fetch(url, *, data=None, headers=None, timeout=5):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{url} returned HTTP {exc.code}: {exc.read(1024).decode(errors='replace')}"
        ) from exc


def wait_ready(checks, processes, timeout):
    deadline = time.monotonic() + timeout
    pending = dict(checks)
    while pending:
        for name, proc in processes.items():
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{name} exited with code {proc.returncode}; see {name}.log"
                )
        for name, url in list(pending.items()):
            try:
                fetch(url, timeout=2)
            except (OSError, RuntimeError):
                continue
            del pending[name]
        if pending:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"readiness timeout: {pending}")
            time.sleep(0.5)


def owned_processes(run_id, sig=None, *, only=None):
    # vLLM workers may create separate sessions. Open pidfds before checking the
    # inherited run marker so PID reuse cannot redirect a signal to another job.
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or not all(c in "0123456789abcdef" for c in run_id)
    ):
        raise ValueError("invalid ownership token in run.json")
    marker = f"HETERO_LLMD_RUN_ID={run_id}".encode()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if only is not None and int(entry.name) not in only:
                continue
            if entry.stat().st_uid != os.getuid():
                continue
            pid = int(entry.name)
            fd = os.pidfd_open(pid)
            try:
                if marker in (entry / "environ").read_bytes().split(b"\0"):
                    found.append(pid)
                    if sig is not None:
                        signal.pidfd_send_signal(fd, sig)
            finally:
                os.close(fd)
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    return found


def confirm_listeners(state):
    """Confirm that the processes for this run own all configured HTTP ports."""
    expected = {state["experiment"]["port"], *(ep["port"] for ep in state["endpoints"])}
    sockets = {}
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        columns = line.split()
        if columns[3] == "0A":
            port = int(columns[1].split(":")[1], 16)
            if port in expected:
                sockets[f"socket:[{columns[9]}]"] = port
    found = set()
    for pid in owned_processes(state["run_id"]):
        try:
            descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
            except (FileNotFoundError, PermissionError):
                continue
            if target in sockets:
                found.add(sockets[target])
    if missing := expected - found:
        raise RuntimeError(f"run no longer owns listeners on ports {sorted(missing)}")


def stop_processes(run_id, processes=None):
    # Let API servers drain and reap their engine workers/IPC trackers first.
    # The marker scan also covers workers detached from a dead or missing leader.
    leaders = {p["pid"] for p in (processes or {}).values()}
    stages = [(None, signal.SIGTERM, 6), (None, signal.SIGKILL, 5)]
    if leaders:
        stages.insert(0, (leaders, signal.SIGTERM, 8))
    for only, sig, grace in stages:
        if not owned_processes(run_id, sig, only=only):
            if only is None:
                return
            continue
        until = time.monotonic() + grace
        while time.monotonic() < until:
            if not owned_processes(run_id):
                return
            time.sleep(0.2)
    if remaining := owned_processes(run_id):
        raise RuntimeError(f"owned processes still active: {remaining}")
