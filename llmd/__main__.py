"""Manage one model served through the official llm-d v1 deployment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

from . import config, engines, runtime, serving
from .verify import verify


def resolved(cfg, run_id):
    eps = [dict(ep, engine_id=run_id + "-" + ep["id"]) for ep in cfg["endpoints"]]
    for plugin in cfg["native"].get("plugins", []):
        if plugin["type"] == "route-contract":
            plugin.setdefault("parameters", {})["servedModel"] = cfg["model"]["name"]
    return eps


def local_render(cfg, work, eps):
    return serving.render(
        work, native=cfg["native"], endpoints=eps, ports=cfg["ports"], host=cfg["host"]
    )


def launch_dirs(env):
    for key in (
        "RBLN_OFFLOAD_DIR",
        "VLLM_RBLN_METRICS_DIR",
        "PROMETHEUS_MULTIPROC_DIR",
    ):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)


def start(cfg, work, timeout):
    if any(p.name != ".llmd.lock" for p in work.iterdir()):
        raise ValueError(f"start requires a new or empty work directory: {work}")
    run_id = uuid.uuid4().hex
    eps = resolved(cfg, run_id)
    base = f"http://{cfg['host']}:{cfg['ports']['gateway']}"
    path = work / "run.json"
    state = dict(
        schema_version=2,
        run_id=run_id,
        status="starting",
        work_dir=str(work),
        endpoint=base + "/v1/completions",
        experiment=cfg,
        model=engines.model_info(cfg["model"], cfg),
        endpoints=eps,
        processes={},
        versions={
            "llm_d": "v1.0.0",
            "router": "v1.0.0",
            "llm_d_commit": serving.LLMD_COMMIT,
            "router_commit": serving.ROUTER_COMMIT,
            "gie": "v1.6.2",
        },
        metrics=f"http://127.0.0.1:{cfg['ports']['epp_metrics']}/metrics",
    )
    runtime.save(path, state)
    processes = {}

    def spawn(name, argv, env):
        processes[name] = runtime.spawn(name, argv, env, work, state)
        runtime.save(path, state)

    try:
        if len({e["node"] for e in eps}) > 1:
            raise ValueError(
                "use Kubernetes to launch engines on multiple physical nodes"
            )
        control = local_render(cfg, work, eps)
        state["control_plane"] = control
        state["resolved_files"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in work.glob("*.yaml")
        }
        for spec in control:
            binary = Path(spec["argv"][0])
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise ValueError(f"missing {binary}; run llmd/serving/build.sh")
        state["binary_sha256"] = {}
        state["image_archive_digests"] = {}
        state["health"] = control[-1]["health"]
        state["owned_listeners"] = [
            (cfg["host"], cfg["ports"]["gateway"]),
            ("127.0.0.1", cfg["ports"]["admin"]),
            *(
                ("127.0.0.1", cfg["ports"][key])
                for key in ("epp", "epp_health", "epp_metrics")
            ),
        ]
        state["owned_listeners"].extend((ep["host"], ep["port"]) for ep in eps)
        state["owned_listeners"].extend(
            ("127.0.0.1", ep["sidecar_port"]) for ep in eps if "decode" in ep["roles"]
        )
        for ep in eps:
            if ep.get("sidecar_metrics_port"):
                state["owned_listeners"].append(
                    ("127.0.0.1", ep["sidecar_metrics_port"])
                )
                ep["sidecar_metrics"] = (
                    f"http://{ep['host']}:{ep['sidecar_metrics_port']}/metrics"
                )
        kv_ports = [
            (ep["host"], ep["kv_port"] + i)
            for ep in eps
            if "kv_port" in ep
            for i in range(ep["tp"])
        ]
        runtime.check_ports([*state["owned_listeners"], *kv_ports])
        with runtime.claim_devices(eps), contextlib.ExitStack() as caches:
            hw = engines.preflight(state["model"], eps)
            cache_root = cfg.get("cache_root") or os.environ.get(
                "HETERO_COMPILE_CACHE", "~/.cache/hetero-pdd/compiled"
            )
            publishers = {}
            end = time.monotonic() + timeout
            for ep in eps:
                name = ep["id"]
                cache, ep["compile_cache"], publishers[name] = caches.enter_context(
                    engines.compilation_cache(cache_root, state["model"], ep, hw[name])
                )
                peers = [
                    device
                    for p in eps
                    if p["id"] in ep.get("peers", [])
                    for device in p["devices"]
                ]
                spec = engines.engine_spec(
                    state["model"],
                    ep,
                    work=work,
                    engine_id=ep["engine_id"],
                    cache=cache,
                    peer_devices=peers,
                    instrumentation=cfg["instrumentation"],
                )
                if "decode" in ep["roles"] and ep["host"] != "127.0.0.1":
                    # The official sidecar connects to localhost. Keep the real
                    # interface IP in discovery and the NIXL handshake.
                    spec["argv"][spec["argv"].index("--host") + 1] = "0.0.0.0"
                env = spec["env"]
                launch_dirs(env)
                url = f"http://{ep['host']}:{ep['port']}"
                ep.update(
                    url=url,
                    metrics=url + "/metrics",
                    log=str(work / (name + ".log")),
                    runtime=hw[name],
                    argv=spec["argv"],
                    engine_env={
                        k: v
                        for k, v in env.items()
                        if k.startswith(
                            (
                                "VLLM_",
                                "RBLN_",
                                "CUDA_VISIBLE",
                                "UCX_",
                                "PROMETHEUS_",
                                "TORCHINDUCTOR_",
                                "TRITON_",
                                "OTEL_",
                            )
                        )
                    },
                    events_dir=env.get("VLLM_RBLN_METRICS_DIR"),
                    trace_endpoint=cfg["instrumentation"].get("otlp_endpoint"),
                )
                spawn(name, spec["argv"], env)
            runtime.wait_ready(
                {e["id"]: e["url"] + "/health" for e in eps},
                processes,
                end - time.monotonic(),
            )
            for ep in eps:
                data, _ = runtime.fetch(ep["url"] + "/v1/models")
                if state["model"]["name"] not in {
                    m["id"] for m in json.loads(data)["data"]
                }:
                    raise RuntimeError(f"{ep['id']} serves a different model")
                publishers[ep["id"]]()
        for spec in control:
            spawn(spec["name"], spec["argv"], {**os.environ, **spec["env"]})
            binary = Path(spec["argv"][0])
            proc_info = state["processes"][spec["name"]]
            sha = hashlib.sha256(
                Path(f"/proc/{proc_info['pid']}/exe").read_bytes()
            ).hexdigest()
            proc_info["exe_sha256"] = sha
            state["binary_sha256"][binary.name] = sha
            checks = binary.parent / "binaries.sha256"
            digest = binary.with_suffix(".digest")
            if (
                checks.is_file()
                and digest.is_file()
                and f"{sha}  {binary.name}" in checks.read_text().splitlines()
            ):
                state["image_archive_digests"][binary.name] = digest.read_text().strip()
            runtime.wait_ready(
                {spec["name"]: spec["health"]}, processes, end - time.monotonic()
            )
        runtime.gateway_ready(cfg["ports"], end - time.monotonic())
        runtime.confirm_listeners(state)
        state["status"] = "ready"
        runtime.save(path, state)
        return {"status": "ready", "endpoint": state["endpoint"], "run": str(path)}
    except BaseException as exc:
        state.update(status="failed", error=str(exc) or type(exc).__name__)
        runtime.save(path, state)
        runtime.stop_processes(run_id, state["processes"])
        for proc in processes.values():
            proc.wait(timeout=5)
        raise


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "render", "start", "ready", "verify", "stop"):
        p = sub.add_parser(name)
        if name in ("validate", "render", "start"):
            p.add_argument("experiment", type=Path)
        if name != "validate":
            p.add_argument("--work-dir", required=True, type=Path)
        if name == "render":
            p.add_argument("--target", choices=["local", "kubernetes"], default="local")
            p.add_argument("--namespace", default="hetero-pdd")
            p.add_argument(
                "--site",
                type=Path,
                help="Kubernetes images and placement/storage settings YAML",
            )
        if name == "start":
            p.add_argument("--timeout", type=int, default=2400)
        if name == "verify":
            p.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.action in ("validate", "render", "start"):
        cfg = config.load(args.experiment)
        if args.action == "validate":
            return cfg
    work = args.work_dir.resolve()
    with runtime.run_lock(work, create=args.action in ("render", "start")):
        if args.action == "render":
            if any((work / f).exists() for f in ("run.json", "kube-run.json")):
                raise ValueError("render needs a directory separate from a managed run")
            eps = resolved(cfg, uuid.uuid4().hex)
            if args.target == "local":
                return local_render(cfg, work, eps)
            if args.site:
                site = config.read_yaml(args.site)
                config.fields(site, {"cluster", "images"}, set(), "site")
                cfg.update(site)
            from .cluster import render

            return render(cfg, eps, work, args.namespace)
        if args.action == "start":
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            return start(cfg, work, args.timeout)
        path = work / "run.json"
        state = json.loads(path.read_text())
        if args.action == "stop":
            runtime.stop_processes(state["run_id"], state["processes"])
            state["status"] = "stopped"
            runtime.save(path, state)
        elif args.action == "ready":
            if state["status"] != "ready":
                raise RuntimeError(f"run is {state['status']}")
            runtime.confirm_listeners(state)
            runtime.fetch(state["health"])
            runtime.gateway_ready(state["experiment"]["ports"], 5)
            for e in state["endpoints"]:
                runtime.fetch(e["url"] + "/health")
        else:
            verify(work, state, args.reference)
            return {"result": str(work / "verification.json")}
        return {"status": state["status"], "run": str(path)}


def interrupted(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print(json.dumps(cli(), ensure_ascii=False))
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
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), "type": type(exc).__name__}
            ),
            file=sys.stderr,
        )
        sys.exit(1)
