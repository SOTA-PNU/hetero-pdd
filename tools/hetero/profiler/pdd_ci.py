# SPDX-License-Identifier: Apache-2.0
"""Attach profiling to persistent tools/pdd servers; never launch a server."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[3])

from tools.hetero.loadgen import make_tokenizer
from tools.hetero.profiler import analysis, expected, run
from tools.hetero.profiler.common import live_rows
from tools.hetero.profiler.monitors import Monitors, wait_for_layout
from tools.hetero.profiler.scrape import boundary_snapshot
from tools.hetero.profiler.tools import capture_links, check_tools
from tools.pdd.runner import atomic_json, execution_env, proc_identity, read_env


def service(runtime):
    endpoints = {
        r: f"http://127.0.0.1:{runtime[k + '_PORT']}"
        for r, k in (("prefill", "PREFILL"), ("decode0", "DECODE"), ("proxy", "PROXY"))
    }
    devices = dict(prefill="gpu", decode0="npu" + runtime["RBLN_DEVICE"])
    return dict(
        configuration="pdd",
        endpoints=endpoints,
        metrics_endpoints={
            r: u + "/metrics" for r, u in endpoints.items() if r != "proxy"
        },
        model_name=runtime["MODEL_NAME"],
        model_path=runtime["MODEL_PATH"],
        dtype=runtime["MODEL_DTYPE"],
        kv_cache_dtype=runtime["KV_CACHE_DTYPE"],
        max_model_len=1024,
        max_num_seqs=1,
        block_size=128,
        num_blocks=32,
        role_devices=devices,
        active_devices=list(devices.values()),
        cuda_device=runtime["CUDA_DEVICE"],
        rbln_device=runtime["RBLN_DEVICE"],
        kv_buffer_device=dict(prefill="cuda", decode0="cpu"),
        layout_reorder=int(runtime.get("VLLM_RBLN_PDD_LAYOUT_REORDER", "0")),
    )


class SavedProcess:
    """Signal only the recorded isolated session, never a recycled PID."""

    def __init__(self, record):
        self.pid, self.identity = record["pid"], record["identity"]

    def poll(self):
        now = proc_identity(self.pid)
        if now and any(
            now[k] != self.identity[k] for k in ("PGID", "SID", "STARTTIME")
        ):
            raise RuntimeError(f"profiler PID {self.pid}: identity changed")
        for path in (Path(os.sep) / "proc").iterdir():
            member = proc_identity(path.name) if path.name.isdigit() else None
            if (
                member
                and member["SID"] == self.identity["SID"]
                and member["state"] not in ("Z", "X")
            ):
                return None
        return 0

    def send(self, sig):
        if self.poll() is None:
            os.killpg(self.pid, sig)

    def terminate(self):
        self.send(signal.SIGTERM)

    def kill(self):
        self.send(signal.SIGKILL)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.05)
        return 0


def monitors(root):
    path = root / "processes.json"
    records = json.loads(path.read_text()) if path.exists() else {}

    def saved(name, process):
        identity = proc_identity(process.pid)
        if not identity or identity["SID"] != str(process.pid):
            raise RuntimeError(f"{name}: cannot identify profiler session")
        records[name] = dict(pid=process.pid, identity=identity)
        atomic_json(path, records)

    manager = Monitors(root, on_start=saved)
    manager.processes.update({k: SavedProcess(v) for k, v in records.items()})
    return manager


def stop(manager, names):
    errors = []
    for name in names:
        try:
            manager.stop(name)
            path = manager.root / "processes.json"
            records = json.loads(path.read_text()) if path.exists() else {}
            records.pop(name, None)
            atomic_json(path, records)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))


def check_import_sources(paths, checkout):
    for path in paths.values():
        if not Path(path).resolve().is_relative_to(Path(checkout).resolve()):
            raise RuntimeError(f"vllm_rbln이 체크아웃이 아닌 {path}에서 import됨")


def verify_imports(root, runtime):
    script = (
        "import json, vllm_rbln, vllm_rbln.v1.worker.metrics_v2 as metrics; "
        "print(json.dumps({'vllm_rbln': vllm_rbln.__file__, "
        "'metrics_v2': metrics.__file__}))"
    )
    output = subprocess.check_output(
        [runtime["RBLN_PY"], "-c", script],
        env=execution_env(runtime),
        cwd=runtime["REPO"],
        text=True,
    )
    paths = json.loads(output.splitlines()[-1])
    atomic_json(root / "import_sources.json", paths)
    check_import_sources(paths, runtime["REPO"])


def prepare(root, runtime, manager):
    mode = os.environ.get("PDD_PROFILE_MODE", "trace")
    if mode not in ("none", "monitor", "trace"):
        raise ValueError(f"unsupported PDD_PROFILE_MODE: {mode}")
    atomic_json(root / "run.json", dict(mode=mode, commit=runtime["COMMIT"]))
    verify_imports(root, runtime)
    check_tools(mode)
    port = manager.collector() if mode == "trace" else None
    env, args = run.role_options("pdd", mode, root, port)
    options = {
        role: dict(env=env.get(role, {}), args=args.get(role, []))
        for role in ("prefill", "decode0")
    }
    atomic_json(root / "server_options.json", options)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pdd"))
    from workload import Workload

    if not Path(runtime["MODEL_PATH"]).is_dir():
        from huggingface_hub import snapshot_download

        runtime["MODEL_PATH"] = snapshot_download(
            runtime["MODEL_PATH"], local_files_only=True
        )
    prompt = json.loads(Workload(runtime).body_bytes)["prompt"]
    length = (
        int(os.environ["INPUT_TOKENS"])
        if os.environ.get("INPUT_TOKENS")
        else len(make_tokenizer(runtime["MODEL_PATH"]).encode(prompt))
    )
    work = dict(
        name="measure",
        input_tokens=length,
        output_tokens=int(runtime["MAX_TOKENS"]),
        requests=int(runtime["MEASURE_REQUESTS"]),
        concurrency=int(runtime["CONCURRENCY"]),
        warmup=int(runtime["WARMUP_REQUESTS"]),
        seed=int(runtime["SEED"]),
    )
    run.validate_workloads([work], "pdd")
    if length + work["output_tokens"] > int(runtime["MAX_MODEL_LEN"]):
        raise ValueError("INPUT_TOKENS + MAX_TOKENS exceeds MAX_MODEL_LEN")
    atomic_json(root / "workload.json", work)
    atomic_json(
        root / "input_length.json",
        dict(
            input_tokens=length,
            source="INPUT_TOKENS"
            if os.environ.get("INPUT_TOKENS")
            else "legacy prompt tokenizer.encode",
            prompt_mode=runtime["PROMPT_MODE"],
            model=runtime["MODEL_PATH"],
        ),
    )


def collect(root, runtime, manager, mode):
    runtime["MODEL_PATH"] = json.loads((root / "input_length.json").read_text())[
        "model"
    ]
    config = service(runtime)
    atomic_json(root / "service.json", config)
    if mode == "trace":
        wait_for_layout(root, roles=("decode0",))
    if mode != "none":
        capture_links(runtime["CUDA_DEVICE"], root / "links.json")
        endpoints = dict(config["metrics_endpoints"], **{"all-smi": manager.allsmi()})
        manager.scrape(endpoints, root / "measure/monitor/metrics.jsonl")


def workload(root, phase, mode):
    work = json.loads((root / "workload.json").read_text())
    config = expected.service_config(root)
    argv = run.loadgen_args(work, config)
    for part in ("warmup", "measure") if phase == "all" else (phase,):
        if part == "warmup":
            subprocess.run(
                argv
                + ["--warmup-only"]
                + run.flags(
                    warmup_per_endpoint=work["warmup"], out_dir=root / "warmup"
                ),
                check=True,
            )
            (root / "warmup.done").touch()
        else:
            if not (root / "warmup.done").exists():
                raise ValueError("measure requires successfully completed warmup")
            if phase == "all":
                time.sleep(5)
            boundary = root / "measure/monitor/boundaries.jsonl"
            try:
                if mode != "none":
                    boundary_snapshot(config["metrics_endpoints"], boundary, "before")
                subprocess.run(
                    argv + run.flags(warmup_per_endpoint=0, out_dir=root / "measure"),
                    check=True,
                )
            finally:
                if mode != "none":
                    boundary_snapshot(config["metrics_endpoints"], boundary, "after")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=(
            "prepare",
            "collect",
            "workload",
            "before_stop",
            "after_stop",
            "analyze",
        ),
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("warmup", "measure", "all"), default="all")
    args = parser.parse_args()
    root = args.result_dir / "profile"
    root.mkdir(exist_ok=True)
    manager = monitors(root)
    runtime = read_env(args.result_dir / "runtime.env")
    mode = (
        json.loads((root / "run.json").read_text())["mode"]
        if (root / "run.json").exists()
        else "trace"
    )
    try:
        if args.step == "prepare":
            prepare(root, runtime, manager)
        elif args.step == "collect":
            collect(root, runtime, manager, mode)
        elif args.step == "workload":
            workload(root, args.phase, mode)
        elif args.step == "before_stop":
            try:
                if mode == "trace" and (root / "service.json").exists():
                    rows = list(live_rows(root / "measure/requests.jsonl"))
                    ids = expected.expected_ids(rows, expected.service_config(root))
                    if not expected.wait_for_spans(root, ids)["complete"]:
                        raise RuntimeError(
                            "profile/trace_wait.json: missing expected spans"
                        )
            finally:
                stop(manager, ("scrape", "all-smi"))
        elif args.step == "after_stop":
            stop(manager, ("scrape", "all-smi", "otelcol"))
        else:
            return analysis.analyze(root, "measure")
    except BaseException:
        if args.step == "prepare":
            stop(manager, tuple(manager.processes))
        elif args.step == "collect":
            stop(manager, ("scrape", "all-smi"))
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"pdd profiler: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
