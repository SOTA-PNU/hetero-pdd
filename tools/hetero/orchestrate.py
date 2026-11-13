# SPDX-License-Identifier: Apache-2.0
"""Start/stop the existing vLLM and vLLM-RBLN servers for this experiment.

This is deliberately only a process launcher. It does not alter inference,
scheduling, compilation, or KV-transfer code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = {
    "baseline": {"gpu", "npu0", "npu1"},
    "pdd": {"prefill", "decode0", "decode1", "proxy"},
}
ALLOWED_ARGS = {"--otlp-traces-endpoint", "--profiler-config"}


def python_path(value, variable: str) -> Path:
    value = value or os.environ.get(variable)
    if not value:
        option = variable.lower().replace("_py", "-python")
        raise ValueError(f"Python path required: --{option} or {variable}")
    # Do not resolve symlinks: a venv Python may point to the system interpreter.
    return Path(value).expanduser().absolute()


def validate_role_options(args):
    for values in (args.role_env_json, args.role_args_json):
        if not isinstance(values, dict) or values.keys() - ROLES[args.configuration]:
            raise ValueError(
                "role options must use roles from the selected configuration"
            )
    for values in args.role_env_json.values():
        if not isinstance(values, dict):
            raise ValueError("role environment must be a JSON object")
        for name, value in values.items():
            if (
                not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name)
                or not isinstance(value, str)
                or any(c in value for c in "\n\r\0")
            ):
                raise ValueError(f"invalid role environment: {name}")
    for values in args.role_args_json.values():
        if not isinstance(values, list) or len(values) % 2:
            raise ValueError("role arguments must be flag/value pairs")
        for flag, value in zip(values[::2], values[1::2]):
            if (
                not isinstance(flag, str)
                or flag not in ALLOWED_ARGS
                or not isinstance(value, str)
                or value.startswith("--")
                or any(c in value for c in "\n\r\0")
            ):
                raise ValueError("invalid or disallowed role argument")


def apply_role_options(args, role, env, delta, argv):
    delta.update(args.role_env_json.get(role, {}))
    env.update(delta)
    argv.extend(args.role_args_json.get(role, []))


def run_checked(argv: list[str], *, text: bool = True) -> str:
    completed = subprocess.run(
        argv, check=True, capture_output=True, text=text, timeout=30
    )
    return completed.stdout.strip() if text else completed.stdout


def verify_environment(
    repo: Path, cuda_py: Path, rbln_py: Path, expected_compiler=None
) -> dict[str, str]:
    commit = run_checked(["git", "-C", str(repo), "rev-parse", "HEAD"])
    branch = run_checked(["git", "-C", str(repo), "branch", "--show-current"])
    dirty = run_checked(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"]
    )
    if dirty:
        raise RuntimeError(
            f"repository mismatch: commit={commit}, branch={branch}, "
            f"dirty={bool(dirty)}"
        )
    compiler = run_checked(
        [
            str(rbln_py),
            "-c",
            "import importlib.metadata as m; print(m.version('rebel-compiler'))",
        ]
    )
    cuda_vllm = run_checked(
        [str(cuda_py), "-c", "import importlib.metadata as m; print(m.version('vllm'))"]
    )
    rbln_vllm = run_checked(
        [str(rbln_py), "-c", "import importlib.metadata as m; print(m.version('vllm'))"]
    )
    if expected_compiler is not None and compiler != expected_compiler:
        raise RuntimeError(f"compiler mismatch: {compiler}")
    if cuda_vllm != "0.22.0" or rbln_vllm != "0.22.0+cpu":
        raise RuntimeError(f"vLLM mismatch: cuda={cuda_vllm}, rbln={rbln_vllm}")
    return {
        "repo": str(repo),
        "commit": commit,
        "branch": branch,
        "cuda_python": str(cuda_py),
        "rbln_python": str(rbln_py),
        "compiler": compiler,
        "cuda_vllm": cuda_vllm,
        "rbln_vllm": rbln_vllm,
    }


def verify_idle() -> dict[str, object]:
    gpu_apps = run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rbln = json.loads(run_checked(["rbln-smi", "--json"]))
    if gpu_apps:
        raise RuntimeError(f"GPU already has compute processes: {gpu_apps}")
    if rbln.get("contexts"):
        raise RuntimeError(f"NPU already has contexts: {rbln['contexts']}")
    return {"gpu_compute_apps": [], "rbln_before": rbln}


def reserve_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            sockets.append(sock)
        ports = [sock.getsockname()[1] for sock in sockets]
        if len(set(ports)) != count:
            raise RuntimeError("duplicate reserved port")
        return ports
    finally:
        for sock in sockets:
            sock.close()


def proc_starttime(pid: int) -> int:
    stat = (Path(os.sep) / "proc" / str(pid) / "stat").read_text(encoding="utf-8")
    return int(stat.rsplit(") ", 1)[1].split()[19])


def wait_http(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"process {process.pid} exited with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"readiness timeout: {url}")


def clean_base_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("RBLN_") or name.startswith("VLLM_RBLN_"):
            env.pop(name, None)
    for name in (
        "VLLM_PLUGINS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "VLLM_RBLN_ENFORCE_MODEL_FP32",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def common_server_args(args: argparse.Namespace, port: int) -> list[str]:
    return [
        str(args.model_path),
        "--served-model-name",
        args.model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        args.dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--tensor-parallel-size",
        "1",
        "--block-size",
        str(args.block_size),
        "--num-gpu-blocks-override",
        str(args.num_blocks),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
        "--generation-config",
        "vllm",
    ]


def write_commands(
    path: Path, commands: list[tuple[str, dict[str, str], list[str]]]
) -> None:
    """Preserve the original audit script; process execution always uses argv."""
    lines = ["#!/usr/bin/env bash", "set -Eeuo pipefail", ""]
    for role, env_delta, argv in commands:
        lines.append(f"# {role}")
        env_words = [
            f"{name}={shlex.quote(value)}" for name, value in sorted(env_delta.items())
        ]
        lines.append("env " + " ".join(env_words + [shlex.join(argv)]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def launch(
    run_dir: Path,
    role: str,
    argv: list[str],
    env: dict[str, str],
    env_delta: dict[str, str],
    health_url: str,
    timeout: float,
    records: list[dict[str, object]],
) -> None:
    log_path = run_dir / "logs" / f"{role}.log"
    log_handle = log_path.open("ab", buffering=0)
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        time.sleep(1)
        wait_http(health_url, process, timeout)
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    records.append(
        {
            "role": role,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid),
            "starttime": proc_starttime(process.pid),
            "health_url": health_url,
            "startup_seconds": time.monotonic() - started,
            "log": str(log_path),
            "argv": argv,
            "env_delta": env_delta,
        }
    )
    (run_dir / "processes.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stop_records(run_dir: Path) -> None:
    process_file = run_dir / "processes.json"
    if not process_file.exists():
        return
    records = json.loads(process_file.read_text(encoding="utf-8"))
    live: list[dict[str, object]] = []
    for record in reversed(records):
        pid = int(record["pid"])
        try:
            if proc_starttime(pid) != int(record["starttime"]):
                continue
            if os.getpgid(pid) != int(record["pgid"]) or int(record["pgid"]) != pid:
                continue
            os.killpg(pid, signal.SIGTERM)
            live.append(record)
        except (FileNotFoundError, ProcessLookupError):
            pass
    deadline = time.monotonic() + 20
    while live and time.monotonic() < deadline:
        live = [
            record
            for record in live
            if (Path(os.sep) / "proc" / str(record["pid"])).exists()
        ]
        if live:
            time.sleep(0.25)
    for record in live:
        pid = int(record["pid"])
        try:
            if proc_starttime(pid) == int(record["starttime"]):
                os.killpg(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
    after = json.loads(run_checked(["rbln-smi", "--json"]))
    (run_dir / "rbln_smi_after_stop.json").write_text(
        json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def start(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else run_dir / "cache"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"run directory is not empty: {run_dir}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cache_dir / "gpu").mkdir(parents=True, exist_ok=True)
    (cache_dir / "npu_shared").mkdir(parents=True, exist_ok=True)

    environment = verify_environment(
        REPO_ROOT, args.cuda_python, args.rbln_python, args.expected_compiler
    )
    idle = verify_idle()
    if not args.model_path.is_dir():
        raise RuntimeError(f"local model snapshot is missing: {args.model_path}")

    records: list[dict[str, object]] = []
    commands: list[tuple[str, dict[str, str], list[str]]] = []
    config: dict[str, object] = {
        "configuration": args.configuration,
        "environment": environment,
        "idle_check": idle,
        "model_path": str(args.model_path.resolve()),
        "model_name": args.model_name,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "prefix_caching": False,
        "chunked_prefill": True,
        "started_unix": time.time(),
        "kv_buffer_device": {},
    }

    try:
        if args.configuration == "baseline":
            gpu_port, npu0_port, npu1_port = reserve_ports(3)
            endpoints = {
                "gpu": f"http://127.0.0.1:{gpu_port}",
                "npu0": f"http://127.0.0.1:{npu0_port}",
                "npu1": f"http://127.0.0.1:{npu1_port}",
            }
            config["endpoints"] = endpoints
            specs = [
                ("gpu", args.cuda_python, gpu_port, "0", None),
                ("npu0", args.rbln_python, npu0_port, "", "0"),
                ("npu1", args.rbln_python, npu1_port, "", "1"),
            ]
            for role, python, port, cuda_devices, rbln_device in specs:
                env = clean_base_env()
                delta = {
                    "PYTHONPATH": str(REPO_ROOT),
                    "CUDA_VISIBLE_DEVICES": cuda_devices,
                    "VLLM_CACHE_ROOT": str(
                        cache_dir / ("gpu" if role == "gpu" else "npu_shared")
                    ),
                }
                if role == "gpu":
                    delta["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                if rbln_device is not None:
                    delta.update(
                        {
                            "RBLN_DEVICES": rbln_device,
                            "VLLM_RBLN_USE_VLLM_MODEL": "1",
                            "VLLM_RBLN_USE_DEVICE_TENSOR": "1",
                            "VLLM_RBLN_COMPILE_MODEL": "1",
                            "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
                            "VLLM_RBLN_SAMPLER": "0",
                            "RBLN_USE_CUSTOM_KERNEL": "0",
                            "RBLN_ROOT_IP": "127.0.0.1",
                            "RBLN_LOCAL_IP": "127.0.0.1",
                        }
                    )
                argv = [str(python), "-m", "vllm.entrypoints.cli.main", "serve"]
                argv += common_server_args(args, port)
                apply_role_options(args, role, env, delta, argv)
                commands.append((role, delta, argv))
                launch(
                    run_dir,
                    role,
                    argv,
                    env,
                    delta,
                    endpoints[role] + "/health",
                    2400,
                    records,
                )
        else:
            (
                prefill_port,
                decode0_port,
                decode1_port,
                proxy_port,
                prefill_side,
                decode0_side,
                decode1_side,
            ) = reserve_ports(7)
            run_token = f"{int(time.time())}-{os.getpid()}"
            producer = {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_producer",
                "kv_buffer_device": "cuda",
                "kv_load_failure_policy": "fail",
                "engine_id": f"cuda-prefill-{run_token}",
                "kv_connector_extra_config": {"kv_recompute_threshold": 0},
            }
            consumers = []
            for index in (0, 1):
                consumers.append(
                    {
                        "kv_connector": "RblnNixlConnector",
                        "kv_role": "kv_consumer",
                        "kv_buffer_device": "cpu",
                        "kv_load_failure_policy": "fail",
                        "engine_id": f"rbln-decode{index}-{run_token}",
                        "kv_connector_extra_config": {"kv_recompute_threshold": 0},
                    }
                )
            endpoints = {
                "prefill": f"http://127.0.0.1:{prefill_port}",
                "decode0": f"http://127.0.0.1:{decode0_port}",
                "decode1": f"http://127.0.0.1:{decode1_port}",
                "proxy": f"http://127.0.0.1:{proxy_port}",
            }
            config.update(
                {
                    "endpoints": endpoints,
                    "producer_config": producer,
                    "consumer_configs": consumers,
                    "kv_buffer_device": {
                        f"decode{i}": c["kv_buffer_device"]
                        for i, c in enumerate(consumers)
                    },
                    "transfer_path": "CUDA_VRAM_TO_UCX_HOST_DRAM_TO_RBLN",
                    "direct_rdma": False,
                    "proxy_policy": "round_robin_decoders",
                }
            )
            pdd_specs = [
                (
                    "prefill",
                    args.cuda_python,
                    prefill_port,
                    prefill_side,
                    "0",
                    None,
                    producer,
                ),
                (
                    "decode0",
                    args.rbln_python,
                    decode0_port,
                    decode0_side,
                    "0",
                    "0",
                    consumers[0],
                ),
                (
                    "decode1",
                    args.rbln_python,
                    decode1_port,
                    decode1_side,
                    "0",
                    "1",
                    consumers[1],
                ),
            ]
            for (
                role,
                python,
                port,
                side_port,
                cuda_devices,
                rbln_device,
                kv_config,
            ) in pdd_specs:
                env = clean_base_env()
                delta = {
                    "PYTHONPATH": str(REPO_ROOT),
                    "CUDA_VISIBLE_DEVICES": cuda_devices,
                    "VLLM_CACHE_ROOT": str(
                        cache_dir / ("gpu" if role == "prefill" else "npu_shared")
                    ),
                    "VLLM_KV_CACHE_LAYOUT": "HND",
                    "UCX_NET_DEVICES": "all",
                    "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_port),
                }
                if role == "prefill":
                    delta["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                if rbln_device is not None:
                    delta.update(
                        {
                            "RBLN_DEVICES": rbln_device,
                            "VLLM_RBLN_USE_VLLM_MODEL": "1",
                            "VLLM_RBLN_USE_DEVICE_TENSOR": "1",
                            "VLLM_RBLN_COMPILE_MODEL": "1",
                            "VLLM_RBLN_COMPILE_STRICT_MODE": "1",
                            "VLLM_RBLN_SAMPLER": "0",
                            "VLLM_RBLN_PDD_LAYOUT_REORDER": str(args.layout_reorder),
                            "RBLN_USE_CUSTOM_KERNEL": "0",
                            "RBLN_ROOT_IP": "127.0.0.1",
                            "RBLN_LOCAL_IP": "127.0.0.1",
                        }
                    )
                argv = [str(python), "-m", "vllm.entrypoints.cli.main", "serve"]
                argv += common_server_args(args, port)
                argv += [
                    "--kv-transfer-config",
                    json.dumps(kv_config, separators=(",", ":")),
                ]
                apply_role_options(args, role, env, delta, argv)
                commands.append((role, delta, argv))
                launch(
                    run_dir,
                    role,
                    argv,
                    env,
                    delta,
                    endpoints[role] + "/health",
                    2400,
                    records,
                )

            proxy_env = clean_base_env()
            proxy_delta = {
                "PYTHONPATH": str(REPO_ROOT),
                "CUDA_VISIBLE_DEVICES": "0",
            }
            proxy_script = (
                REPO_ROOT
                / "tests/torch_compile/e2e/v1/kv_connector"
                / "nixl_integration/toy_proxy_server.py"
            )
            proxy_argv = [
                str(args.cuda_python),
                str(proxy_script),
                "--host",
                "127.0.0.1",
                "--port",
                str(proxy_port),
                "--prefiller-hosts",
                "127.0.0.1",
                "--prefiller-ports",
                str(prefill_port),
                "--decoder-hosts",
                "127.0.0.1",
                "127.0.0.1",
                "--decoder-ports",
                str(decode0_port),
                str(decode1_port),
            ]
            apply_role_options(args, "proxy", proxy_env, proxy_delta, proxy_argv)
            commands.append(("proxy", proxy_delta, proxy_argv))
            launch(
                run_dir,
                "proxy",
                proxy_argv,
                proxy_env,
                proxy_delta,
                endpoints["proxy"] + "/healthcheck",
                60,
                records,
            )

        config["metrics_endpoints"] = {
            role: url + "/metrics" for role, url in endpoints.items() if role != "proxy"
        }
        config["decoder_roles"] = {
            urlsplit(url).netloc: role
            for role, url in endpoints.items()
            if role.startswith("decode")
        }
        config["processes"] = records
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_commands(run_dir / "commands.sh", commands)
        print(json.dumps({"run_dir": str(run_dir), "endpoints": config["endpoints"]}))
    except Exception:
        stop_records(run_dir)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument(
        "--configuration", choices=("baseline", "pdd"), required=True
    )
    start_parser.add_argument("--run-dir", type=Path, required=True)
    start_parser.add_argument("--model-path", type=Path, required=True)
    start_parser.add_argument("--model-name", required=True)
    start_parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    start_parser.add_argument(
        "--kv-cache-dtype", choices=("float16", "bfloat16"), default="float16"
    )
    start_parser.add_argument("--max-model-len", type=int, default=1024)
    start_parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    start_parser.add_argument("--max-num-seqs", type=int, default=1)
    start_parser.add_argument("--block-size", type=int, default=128)
    start_parser.add_argument("--num-blocks", type=int, default=32)
    start_parser.add_argument("--cuda-python", type=Path)
    start_parser.add_argument("--rbln-python", type=Path)
    start_parser.add_argument("--expected-compiler")
    start_parser.add_argument("--layout-reorder", type=int, choices=(0, 1), default=0)
    start_parser.add_argument("--cache-dir", type=Path)
    start_parser.add_argument(
        "--role-env-json",
        type=json.loads,
        default={},
        help="JSON role -> environment object",
    )
    start_parser.add_argument(
        "--role-args-json",
        type=json.loads,
        default={},
        help="JSON role -> flag/value list",
    )
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "start":
        args.cuda_python = python_path(args.cuda_python, "CUDA_PY")
        args.rbln_python = python_path(args.rbln_python, "RBLN_PY")
        validate_role_options(args)
    return args


def main() -> None:
    args = parse_args()
    if args.action == "start":
        start(args)
    else:
        stop_records(args.run_dir.resolve())
        print(f"STOPPED {args.run_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
