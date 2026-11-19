"""Run an isolated native picker and gateway on a managed resident engine pair."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

from llmd import runtime, serving


def _worker_files(resident):
    return [
        *resident.glob("events/*/events_*.jsonl"),
        *resident.glob("*-admission.jsonl"),
    ]


def isolated_ports(cfg):
    # Select distinct free ports together; confirm newly spawned process ownership
    # before sending requests even if another process races to bind one.
    result = {}
    with ExitStack() as stack:
        for key in cfg["ports"]:
            host = cfg["host"] if key == "gateway" else "127.0.0.1"
            sock = stack.enter_context(socket.socket())
            sock.bind((host, 0))
            result[key] = sock.getsockname()[1]
    return result


@contextmanager
def gateway(resident: Path, cfg: dict, work: Path):
    """Yield (run state, worker-log snapshot callback). Leave resident engines live."""
    resident, work = Path(resident).resolve(), Path(work).resolve()
    source = json.loads((resident / "run.json").read_text())
    if source["status"] != "ready":
        raise ValueError(f"resident service is {source['status']}")
    for key in ("model", "endpoints", "instrumentation"):
        if cfg[key] != source["experiment"][key]:
            raise ValueError(f"resident {key} differs from the requested experiment")
    runtime.confirm_listeners(source)
    cfg = deepcopy(cfg)
    ports = cfg["ports"] = isolated_ports(cfg)
    if work.exists() and any(work.iterdir()):
        raise ValueError(f"isolated gateway needs an empty directory: {work}")
    for plugin in cfg["native"]["plugins"]:
        if plugin["type"] == "route-contract":
            plugin.setdefault("parameters", {})["servedModel"] = source["model"]["name"]
    specs = serving.render(
        work,
        native=cfg["native"],
        endpoints=source["endpoints"],
        ports=ports,
        host=cfg["host"],
    )
    control = [s for s in specs if s["name"] in ("epp", "gateway")]
    state = dict(
        source,
        run_id=uuid.uuid4().hex,
        work_dir=str(work),
        experiment=cfg,
        resident_run=source["run_id"],
        processes={},
        control_plane=control,
        health=control[-1]["health"],
        status="starting",
        metrics=f"http://127.0.0.1:{ports['epp_metrics']}/metrics",
        endpoint=f"http://{cfg['host']}:{ports['gateway']}/v1/completions",
        owned_listeners=[
            (cfg["host"] if k == "gateway" else "127.0.0.1", port)
            for k, port in ports.items()
        ],
        resolved_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in work.glob("*.yaml")
        },
        binary_sha256=dict(source["binary_sha256"]),
    )
    state.pop("image_archive_digests", None)
    offsets = {p: p.stat().st_size for p in _worker_files(resident)}

    def snapshot():
        for src in _worker_files(resident):
            dst = work / src.relative_to(resident)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as reader, dst.open("wb") as writer:
                reader.seek(offsets.get(src, 0))
                shutil.copyfileobj(reader, writer)

    live = {}
    runtime.save(work / "run.json", state)
    try:
        for spec in control:
            name = spec["name"]
            live[name] = runtime.spawn(
                name, spec["argv"], os.environ | spec["env"], work, state
            )
            binary = Path(spec["argv"][0])
            state["binary_sha256"][binary.name] = hashlib.sha256(
                binary.read_bytes()
            ).hexdigest()
            runtime.save(work / "run.json", state)
            runtime.wait_ready({name: spec["health"]}, live, 30)
        runtime.gateway_ready(ports, 30)
        runtime.confirm_listeners(state)
        state["status"] = "ready"
        runtime.save(work / "run.json", state)
        # Allow the standard native one-second metrics poll to populate the picker.
        time.sleep(1.1)
        yield state, snapshot
    finally:
        runtime.stop_processes(state["run_id"], state["processes"])
        for proc in live.values():
            proc.wait(timeout=5)
        snapshot()
        state["status"] = "stopped"
        runtime.save(work / "run.json", state)
