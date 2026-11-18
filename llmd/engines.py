"""Native TP1 vLLM launch specifications for the local llm-d EPP."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from tools.hetero import orchestrate

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "qwen3-0.6b": {
        "name": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    "qwen3-1.7b": {
        "name": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    },
}
VERSIONS = {
    "gpu": {"vllm": "0.22.0", "torch": "2.11.0", "nixl": "1.4.1"},
    "npu": {
        "vllm": "0.22.0+cpu",
        "torch": "2.11.0+cpu",
        "nixl": "1.4.1",
        "optimum-rbln": "0.11.1",
        "torch-rbln": "0.3.0",
        "rebel-compiler": "0.11.1.post2.dev2+g2995098f.prod",
    },
}


def model_info(preset, engine):
    model = PRESETS[preset]
    default = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    cache = Path(
        engine.get("hub_cache") or os.getenv("HF_HUB_CACHE") or default
    ).expanduser()
    path = (
        cache.absolute()
        / ("models--" + model["name"].replace("/", "--"))
        / "snapshots"
        / model["revision"]
    )
    return {**model, "path": str(path)}


def validate_settings(preset, endpoints, max_model_len):
    if preset not in PRESETS:
        raise ValueError(f"unknown model preset: {preset}")
    if not 128 <= max_model_len <= 4096:
        raise ValueError("supported TP1 FP16 context length is 128..4096")
    for endpoint in endpoints:
        if endpoint["kind"] == "npu" and endpoint["role"] not in {"full", "decode"}:
            raise ValueError("RBLN supports full and decode roles")


def _python(engine, kind):
    return str(
        orchestrate.python_path(
            engine.get(kind + "_python"), "CUDA_PY" if kind == "gpu" else "RBLN_PY"
        )
    )


def _checked(argv, *, env=None):
    proc = subprocess.run(
        argv, env=env, cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    if proc.returncode:
        raise RuntimeError(
            f"{argv[0]} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def _snapshot(model):
    path = Path(model["path"])
    if not path.is_dir():
        raise ValueError(
            f"missing pinned snapshot {path}. In the configured hub_cache run "
            f"hf download {model['name']} --revision {model['revision']}"
        )
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get(
        "quantization_config"
    ):
        raise ValueError(f"expected an unquantized Qwen3ForCausalLM at {path}")
    files = ["tokenizer.json", "tokenizer_config.json"]
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            raise ValueError(f"empty weight index in {index}")
        files.extend(shards)
    else:
        files.append("model.safetensors")
    for filename in files:
        if not (path / filename).is_file():
            raise ValueError(f"missing pinned model file {path / filename}")


def _runtime_packages(python, kind, needs_transfer):
    probe = """import importlib.metadata as m, importlib.util, json, sys
platforms = m.entry_points(group="vllm.platform_plugins")
print(json.dumps({
    "packages": {p: m.version(p) for p in sys.argv[1:]},
    "platforms": {p.name: p.value for p in platforms},
    "nixl_rbln": importlib.util.find_spec("nixl_rbln") is not None,
}))
"""
    result = json.loads(_checked([python, "-c", probe, *VERSIONS[kind]]))
    if result["packages"] != VERSIONS[kind]:
        raise ValueError(
            f"{kind} package mismatch: expected {VERSIONS[kind]}, "
            f"got {result['packages']}"
        )
    if kind == "npu":
        if result["platforms"].get("rbln") != "vllm_rbln:register":
            raise ValueError(f"{python}: missing vllm_rbln platform entry point")
        if needs_transfer and result["nixl_rbln"]:
            raise ValueError(
                "nixl-rbln selects a different backend. The UCX preset needs "
                "an NPU venv without nixl-rbln."
            )
    return result["packages"]


def _cuda_uuid(python, ordinal):
    # vLLM 0.22 casts CUDA_VISIBLE_DEVICES entries to int. Verify the selected
    # ordinal in CUDA itself before trusting nvidia-smi's device assignment.
    probe = """from cuda.bindings import driver
from uuid import UUID
def checked(call):
    rc, *values = call
    if rc != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA device lookup returned {rc}")
    return values
checked(driver.cuInit(0))
device, = checked(driver.cuDeviceGet(0))
identity, = checked(driver.cuDeviceGetUuid(device))
print("GPU-" + str(UUID(bytes=bytes(identity.bytes))))
"""
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(ordinal), CUDA_DEVICE_ORDER="PCI_BUS_ID"
    )
    return _checked([python, "-c", probe], env=env)


def preflight(preset, engine, endpoints):
    model = model_info(preset, engine)
    validate_settings(preset, endpoints, engine.get("max_model_len", 1024))
    _snapshot(model)
    packages = {}
    details = {}
    plugin = None
    for endpoint in endpoints:
        kind = endpoint["kind"]
        index = endpoint["device"]
        python = _python(engine, kind)
        if kind not in packages:
            needs_transfer = any(ep["role"] == "decode" for ep in endpoints)
            packages[kind] = _runtime_packages(python, kind, needs_transfer)
        if kind == "gpu":
            raw = _checked(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    "--query-gpu=name,uuid,pci.bus_id",
                    "--format=csv,noheader",
                ]
            )
            name, uuid, bus = next(csv.reader([raw], skipinitialspace=True))
            visible = _cuda_uuid(python, index)
            if visible.lower() != uuid.lower():
                raise ValueError(
                    f"GPU {index}: CUDA resolves to {visible}, nvidia-smi to {uuid}"
                )
            busy = _checked(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ]
            )
            if busy:
                raise ValueError(f"GPU {index} already has a compute process ({busy})")
        else:
            if plugin is None:
                plugin = {
                    "source": str(ROOT / "vllm_rbln"),
                    "git_tree": _checked(["git", "rev-parse", "HEAD:vllm_rbln"]),
                    "modified": bool(
                        _checked(["git", "status", "--porcelain", "--", "vllm_rbln"])
                    ),
                }
            doc = json.loads(_checked(["rbln-smi", "--json"]))
            device = next((d for d in doc["devices"] if d["npu"] == index), {})
            if device.get("status") != "normal" or device.get("name") != "RBLN-CA22":
                raise ValueError(f"NPU {index}: expected a ready RBLN-CA22")
            if any(
                str(c.get("npu", c.get("device", ""))) in {str(index), f"rbln{index}"}
                for c in doc["contexts"]
            ):
                raise ValueError(f"NPU {index} already has a context")
            name, uuid, bus = device["name"], device["uuid"], device["pci"]["bus_id"]
        details[endpoint["id"]] = {
            "python": python,
            "packages": packages[kind],
            "name": name,
            "index": index,
            "uuid": uuid,
            "pci_bus": bus,
        }
        if kind == "npu":
            details[endpoint["id"]].update(
                rbln_plugin=plugin, kmd=doc["KMD_version"], firmware=device["fw_ver"]
            )
    return {"model_path": model["path"], "endpoints": details}


def engine_spec(
    preset, engine, endpoint, *, run_dir, run_id, cuda_device, instrumentation=None
):
    kind, role, name = endpoint["kind"], endpoint["role"], endpoint["id"]
    is_gpu = kind == "gpu"
    python = _python(engine, kind)
    model = model_info(preset, engine)
    run_dir = Path(run_dir).absolute()
    # Never open a compile artifact belonging to another launch/model/revision.
    cache = run_dir / "cache" / name
    context = engine.get("max_model_len", 1024)
    # vLLM reserves its null block; leave a full context's worth available.
    num_blocks = max(32, (context + 127) // 128 + 1)
    options = SimpleNamespace(
        model_path=model["path"],
        model_name=model["name"],
        dtype="float16",
        kv_cache_dtype="float16",
        max_model_len=context,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        block_size=128,
        num_blocks=num_blocks,
    )
    argv = [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        *orchestrate.common_server_args(options, endpoint["port"]),
        "--enable-request-id-headers",
    ]
    env = orchestrate.clean_base_env()
    # Clear inherited vLLM backend and cache settings.
    env = {k: v for k, v in env.items() if not k.startswith("VLLM_")}
    env.update(
        PYTHONPATH=str(ROOT),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(cuda_device) if is_gpu else "",
        VLLM_CACHE_ROOT=str(cache),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        PATH=f"{Path(python).parent}{os.pathsep}{env['PATH']}",
    )
    engine_id = f"{run_id}-{name}"
    if is_gpu:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    else:
        offload = cache / "offload"
        offload.mkdir(parents=True, exist_ok=True)
        env.update(
            RBLN_DEVICES=str(endpoint["device"]),
            RBLN_OFFLOAD_DIR=str(offload),
            VLLM_RBLN_USE_VLLM_MODEL="1",
            VLLM_RBLN_USE_DEVICE_TENSOR="1",
            VLLM_RBLN_COMPILE_MODEL="1",
            VLLM_RBLN_COMPILE_STRICT_MODE="1",
            VLLM_RBLN_SAMPLER="0",
            VLLM_RBLN_PDD_LAYOUT_REORDER="0",
            RBLN_USE_CUSTOM_KERNEL="0",
            RBLN_ROOT_IP="127.0.0.1",
            RBLN_LOCAL_IP="127.0.0.1",
        )
    if role in ("prefill", "decode"):
        connector = {
            "kv_connector": "NixlConnector" if is_gpu else "RblnNixlConnector",
            "kv_role": "kv_producer" if is_gpu else "kv_consumer",
            "kv_buffer_device": "cuda" if is_gpu else "cpu",
            "kv_load_failure_policy": "fail",
            "engine_id": engine_id,
            "kv_connector_extra_config": {"kv_recompute_threshold": 0},
        }
        argv += ["--kv-transfer-config", json.dumps(connector)]
        env.update(
            VLLM_KV_CACHE_LAYOUT="HND",
            UCX_NET_DEVICES="all",
            VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1",
            VLLM_NIXL_SIDE_CHANNEL_PORT=str(endpoint["kv_port"]),
        )
        if (
            not is_gpu
        ):  # UCX needs to see the source VRAM; the receiver stages in host DRAM.
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    instrumentation = instrumentation or {}
    if trace := instrumentation.get("otlp_endpoint"):
        argv += ["--otlp-traces-endpoint", trace]
        env.update(
            OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf", OTEL_SERVICE_NAME=name
        )
    if not is_gpu and instrumentation.get("rbln_events"):
        events = run_dir / "events" / name
        events.mkdir(parents=True, exist_ok=True)
        env.update(VLLM_RBLN_METRICS="1", VLLM_RBLN_METRICS_DIR=str(events))
    return {"argv": argv, "env": env, "engine_id": engine_id}
