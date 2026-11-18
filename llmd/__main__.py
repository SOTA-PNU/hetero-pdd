"""CLI for an isolated local llm-d service. Run from the repository root."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

from . import config, engines, runtime
from .verify import verify


def start(cfg, work, timeout):
    if any(p.name != ".llmd.lock" for p in work.iterdir()):
        raise ValueError(f"start needs a new or empty work directory: {work}")
    engine, model = cfg["engine"], cfg["model"]
    base = f"http://127.0.0.1:{cfg['port']}"
    state = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "status": "starting",
        "work_dir": str(work),
        "endpoint": base + "/v1/completions",
        "health": base + "/health",
        "experiment": cfg,
        "endpoints": [],
        "processes": {},
        "versions": {},
    }
    state_file = work / "run.json"
    runtime.save(state_file, state)
    processes = {}
    claims = contextlib.ExitStack()
    try:
        binary = shutil.which(os.getenv("LLMD_ROUTER_BIN", "hetero-epp"))
        if not binary:
            raise ValueError(
                "build llmd/epp/cmd/serve and set LLMD_ROUTER_BIN to its path"
            )
        ports = [cfg["port"]] + [
            ep[k] for ep in cfg["endpoints"] for k in ("port", "kv_port") if k in ep
        ]
        runtime.check_ports(ports)
        claims.enter_context(runtime.claim_devices(cfg["endpoints"]))
        state["model"] = engines.model_info(model, engine)
        build = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, timeout=10, check=True
        )
        state["versions"] = json.loads(build.stdout)
        expected = {
            "epp": "v0.5.1",
            "gateway_inference_extension": "v1.3.1",
            "sidecar_protocol": "NIXLv2-v0.5.1",
        }
        if any(state["versions"].get(k) != v for k, v in expected.items()):
            raise ValueError(f"incompatible router binary: {state['versions']}")

        versions = engines.preflight(model, engine, cfg["endpoints"])
        devices = versions["endpoints"]
        producer_device = next(
            (ep["device"] for ep in cfg["endpoints"] if ep["role"] == "prefill"),
            None,
        )
        specs = {}
        for ep in cfg["endpoints"]:
            name = ep["id"]
            url = f"http://127.0.0.1:{ep['port']}"
            spec = engines.engine_spec(
                model,
                engine,
                ep,
                run_dir=work,
                run_id=state["run_id"],
                cuda_device=ep["device"] if ep["kind"] == "gpu" else producer_device,
                instrumentation=cfg["instrumentation"],
            )
            specs[name] = spec
            info = {
                **ep,
                "slots": ep.get("slots", 1),
                "url": url,
                "metrics": url + "/metrics",
                "log": str(work / f"{name}.log"),
                "engine_id": spec["engine_id"],
                "runtime": devices[name],
                "engine_env": {
                    k: v
                    for k, v in spec["env"].items()
                    if k
                    in (
                        "CUDA_VISIBLE_DEVICES",
                        "RBLN_DEVICES",
                        "VLLM_KV_CACHE_LAYOUT",
                        "VLLM_RBLN_PDD_LAYOUT_REORDER",
                        "UCX_NET_DEVICES",
                        "VLLM_CACHE_ROOT",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                        "RBLN_OFFLOAD_DIR",
                    )
                },
                "argv": spec["argv"],
                "trace_endpoint": cfg["instrumentation"].get("otlp_endpoint"),
                "events_dir": spec["env"].get("VLLM_RBLN_METRICS_DIR"),
            }
            state["endpoints"].append(info)
        router_config = {
            "listen": f"127.0.0.1:{cfg['port']}",
            "model": state["model"]["name"],
            "policy": cfg["policy"],
            "endpoints": [
                {k: ep[k] for k in ("id", "url", "role", "slots")}
                for ep in state["endpoints"]
            ],
        }
        if cfg["policy"] == "fixed-pdd":
            router_config["producers"] = {
                ep["id"]: {
                    "engine_id": ep["engine_id"],
                    "host": "127.0.0.1",
                    "port": ep["kv_port"],
                }
                for ep in state["endpoints"]
                if ep["role"] == "prefill"
            }
        runtime.save(work / "router.json", router_config)
        runtime.save(state_file, state)
        for name, spec in specs.items():
            processes[name] = runtime.spawn(
                name, spec["argv"], spec["env"], work, state
            )
            runtime.save(state_file, state)
        checks = {ep["id"]: ep["url"] + "/health" for ep in state["endpoints"]}
        runtime.wait_ready(checks, processes, timeout)
        for ep in state["endpoints"]:
            raw, _ = runtime.fetch(ep["url"] + "/v1/models")
            if state["model"]["name"] not in {m["id"] for m in json.loads(raw)["data"]}:
                raise RuntimeError(
                    f"{ep['id']}: served model differs from the experiment"
                )
        processes["router"] = runtime.spawn(
            "router",
            [binary, "-config", str(work / "router.json")],
            os.environ,
            work,
            state,
        )
        runtime.save(state_file, state)
        runtime.wait_ready({"router": state["health"]}, processes, 30)
        runtime.confirm_listeners(state)
        state["status"] = "ready"
        runtime.save(state_file, state)
    except BaseException as exc:
        state.update(status="failed", error=str(exc) or type(exc).__name__)
        try:
            runtime.stop_processes(state["run_id"], state["processes"])
            for proc in processes.values():
                proc.wait(timeout=5)
        finally:
            try:
                runtime.save(state_file, state)
            except OSError as write_error:
                print(f"failed to persist {state_file}: {write_error}", file=sys.stderr)
        raise
    finally:
        claims.close()
    print(
        json.dumps(
            {
                "status": state["status"],
                "endpoint": state["endpoint"],
                "run": str(state_file),
            }
        )
    )


def cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("validate", "start", "verify", "stop"):
        p = sub.add_parser(command)
        if command in ("validate", "start"):
            p.add_argument("experiment", type=Path)
        if command != "validate":
            p.add_argument("--work-dir", required=True, type=Path)
        if command == "start":
            p.add_argument(
                "--timeout",
                type=int,
                default=2400,
                help="engine startup timeout in seconds",
            )
        if command == "verify":
            p.add_argument(
                "--reference",
                type=Path,
                help="verification.json from a matching GPU-only baseline",
            )
    args = parser.parse_args()
    if args.action in ("validate", "start"):
        cfg = config.load(args.experiment)
    if args.action == "validate":
        print(json.dumps(cfg))
        return
    work = args.work_dir.resolve()
    with runtime.run_lock(work, create=args.action == "start"):
        if args.action == "start":
            if args.timeout <= 0:
                raise ValueError("--timeout must be positive")
            start(cfg, work, args.timeout)
            return
        path = work / "run.json"
        state = json.loads(path.read_text())
        if args.action == "stop":
            runtime.stop_processes(state["run_id"], state["processes"])
            state["status"] = "stopped"
            runtime.save(path, state)
        else:
            verify(work, state, args.reference)
        print(
            str(work / ("verification.json" if args.action == "verify" else "run.json"))
        )


def interrupted(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        cli()
    except KeyboardInterrupt:
        sys.exit(130)
    except (
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ) as exc:
        sys.exit(f"llmd: {exc}")
