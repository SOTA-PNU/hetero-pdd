# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned vLLM backend launch and model/compile artifact provenance."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import uuid as uuidlib
from pathlib import Path

from tools.hetero import orchestrate

from .config import fingerprint

ROOT = Path(__file__).resolve().parents[1]
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


MODEL_IDENTITY = (
    "name",
    "revision",
    "tokenizer",
    "tokenizer_revision",
    "tokenizer_sha256",
    "tokenizer_config_sha256",
    "config_sha256",
    "weights_sha256",
)


def model_info(model, cfg):
    default = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    cache = (
        Path(cfg.get("hub_cache") or os.getenv("HF_HUB_CACHE") or default)
        .expanduser()
        .absolute()
    )

    def snapshot(key, revision):
        return (
            cache
            / ("models--" + model[key].replace("/", "--"))
            / "snapshots"
            / model[revision]
        )

    path = Path(model.get("path") or snapshot("name", "revision"))
    tok = Path(
        model.get("tokenizer_path")
        or (
            path
            if model["name"] == model["tokenizer"]
            and model["revision"] == model["tokenizer_revision"]
            else snapshot("tokenizer", "tokenizer_revision")
        )
    )
    return {**model, "path": str(path), "tokenizer_path": str(tok)}


def _python(ep):
    return str(
        orchestrate.python_path(
            ep["engine"].get("python"), "CUDA_PY" if ep["kind"] == "gpu" else "RBLN_PY"
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
    if not config.get("architectures"):
        raise ValueError(f"{path}: missing model architecture")
    tokenizer = Path(model["tokenizer_path"])
    tokenizer_identity(tokenizer)
    files = []
    index = path / "model.safetensors.index.json"
    binary = path / "pytorch_model.bin"
    if not index.exists() and not (path / "model.safetensors").exists():
        index = path / "pytorch_model.bin.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            raise ValueError(f"empty weight index in {index}")
        files.extend(shards)
    else:
        files.append("pytorch_model.bin" if binary.is_file() else "model.safetensors")
    for filename in files:
        if not (path / filename).is_file():
            raise ValueError(f"missing pinned model file {path / filename}")


def tokenizer_identity(snapshot):
    root = Path(snapshot)
    if not (root / "tokenizer_config.json").is_file():
        raise ValueError(f"missing tokenizer configuration at {root}")
    fast = root / "tokenizer.json"
    if fast.is_file():
        return hashlib.sha256(fast.read_bytes()).hexdigest()
    models = sorted(
        {*root.glob("*.model"), *root.glob("vocab.*"), *root.glob("merges.txt")}
    )
    if not models:
        raise ValueError(f"missing tokenizer vocabulary at {root}")
    return fingerprint(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*models, root / "tokenizer_config.json"]
        }
    )


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


def source_identity():
    # Includes uncommitted engine/connector changes on development checkouts.
    h = hashlib.sha256()
    for file in sorted((ROOT / "vllm_rbln").rglob("*.py")):
        h.update(str(file.relative_to(ROOT)).encode() + b"\0" + file.read_bytes())
    return h.hexdigest()


def preflight(model, endpoints, *, cluster=False):
    _snapshot(model)
    model_config = json.loads((Path(model["path"]) / "config.json").read_text())
    model["architecture"] = model_config["architectures"]
    model["tokenizer_sha256"] = tokenizer_identity(model["tokenizer_path"])
    model["config_sha256"] = fingerprint(model_config)
    model["tokenizer_config_sha256"] = hashlib.sha256(
        (Path(model["tokenizer_path"]) / "tokenizer_config.json").read_bytes()
    ).hexdigest()
    if any("decode" in ep["roles"] and ep["backend"] == "rbln" for ep in endpoints):
        text = model_config.get("text_config", model_config)
        heads = text.get("num_attention_heads")
        kv_heads = text.get("num_key_value_heads", heads)
        layers = text.get("num_hidden_layers")
        if not all(type(v) is int and v > 0 for v in (heads, kv_heads, layers)):
            raise ValueError(
                "RBLN host import needs known attention heads, KV heads and layer count"
            )
        if (
            text.get("quantization_config")
            or text.get("use_sliding_window")
            or any(t != "full_attention" for t in text.get("layer_types", []))
            or text.get("is_encoder_decoder")
            or text.get("kv_lora_rank")
        ):
            raise ValueError(
                "RBLN host import requires dense unquantized decoder self-attention"
            )
        model["kv_geometry"] = {
            "kv_heads": kv_heads,
            "layers": layers,
            "head_dim": text.get("head_dim") or text["hidden_size"] // heads,
        }
    # Verify loaded weight bytes, including snapshots on separate cluster mounts.
    digest = hashlib.sha256()
    for segment in sorted(
        [
            *Path(model["path"]).glob("*.safetensors"),
            *Path(model["path"]).glob("pytorch_model*.bin"),
        ]
    ):
        with segment.open("rb") as stream:
            for data in iter(lambda: stream.read(8 << 20), b""):
                digest.update(data)
    model["weights_sha256"] = digest.hexdigest()
    details, packages = {}, {}
    for ep in endpoints:
        python, kind = _python(ep), ep["kind"]
        signature = (python, kind, "decode" in ep["roles"])
        if signature not in packages:
            packages[signature] = _runtime_packages(*signature)
        devices = []
        if kind == "npu":
            doc = json.loads(_checked(["rbln-smi", "--json"]))
        for index in ep["devices"]:
            if kind == "gpu":
                raw = _checked(
                    [
                        "nvidia-smi",
                        "-i",
                        str(index),
                        "--query-gpu=name,uuid,pci.bus_id,driver_version",
                        "--format=csv,noheader",
                    ]
                )
                name, uid, bus, version = next(csv.reader([raw], skipinitialspace=True))
                if _cuda_uuid(python, index).lower() != uid.lower():
                    raise ValueError(
                        f"CUDA ordinal {index} does not resolve to NVML UUID {uid}"
                    )
                if not cluster and _checked(
                    [
                        "nvidia-smi",
                        "-i",
                        str(index),
                        "--query-compute-apps=pid",
                        "--format=csv,noheader",
                    ]
                ):
                    raise ValueError(f"GPU {index} already has a compute process")
            else:
                d = next((d for d in doc["devices"] if d["npu"] == index), {})
                if d.get("status") != "normal":
                    raise ValueError(f"RBLN device {index} missing or unhealthy")
                if any(
                    str(c.get("npu", c.get("device", "")))
                    in {str(index), f"rbln{index}"}
                    for c in doc["contexts"]
                ):
                    raise ValueError(f"RBLN device {index} already has a context")
                name, uid, bus, version = (
                    d["name"],
                    d["uuid"],
                    d["pci"]["bus_id"],
                    [doc["KMD_version"], d["fw_ver"]],
                )
            devices.append(
                dict(index=index, name=name, uuid=uid, pci_bus=bus, driver=version)
            )
        details[ep["id"]] = dict(
            python=python,
            packages=packages[signature],
            devices=devices,
            plugin_sha256=source_identity() if kind == "npu" else None,
        )
    return details


@contextlib.contextmanager
def compilation_cache(root, model, ep, hardware):
    """Publish a compatible generation after model readiness."""
    contract = {
        "format": 1,
        "model": model,
        "engine": {k: v for k, v in ep["engine"].items() if k != "coalesce_kv_import"},
        "backend": ep["backend"],
        "node": ep["node"],
        "roles": ep["roles"],
        "runtime": hardware,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_HOME", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CC", "CXX")
        },
    }
    key = fingerprint(contract)
    base = Path(root).expanduser().resolve() / key
    base.mkdir(parents=True, exist_ok=True)
    with (base / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = base / "ready.json"
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved["contract"] != contract or not re_full_generation(
                saved["generation"]
            ):
                raise ValueError(f"invalid compile manifest at {metadata}")
            cache = base / saved["generation"]
            if not cache.is_dir():
                raise ValueError(f"missing compiled generation {cache}")
            reused = True
        else:
            cache = base / uuidlib.uuid4().hex
            cache.mkdir()
            reused = False

        def ready():
            if not reused:
                from .runtime import save

                save(metadata, {"contract": contract, "generation": cache.name})

        try:
            yield (
                cache,
                {"key": key, "reused": reused, "manifest": str(metadata)},
                ready,
            )
        finally:
            if not reused and not metadata.exists():
                shutil.rmtree(cache, ignore_errors=True)


def re_full_generation(name):
    return (
        isinstance(name, str)
        and len(name) == 32
        and all(c in "0123456789abcdef" for c in name)
    )


def engine_spec(model, ep, *, work, engine_id, cache, peer_devices, instrumentation):
    opts, kind = ep["engine"], ep["kind"]
    gpu = kind == "gpu"
    python = _python(ep)
    argv = [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model["path"],
        "--served-model-name",
        model["name"],
        "--tokenizer",
        model["tokenizer_path"],
        "--host",
        ep["host"],
        "--port",
        str(ep["port"]),
        "--dtype",
        opts["dtype"],
        "--kv-cache-dtype",
        opts["kv_dtype"],
        "--tensor-parallel-size",
        str(opts["tp"]),
        "--pipeline-parallel-size",
        str(opts["pp"]),
        "--data-parallel-size",
        str(opts["dp"]),
        "--max-model-len",
        str(opts["max_model_len"]),
        "--max-num-batched-tokens",
        str(opts["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(opts["max_num_seqs"]),
        "--block-size",
        str(opts["block_size"]),
        "--num-gpu-blocks-override",
        str(opts["num_blocks"]),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
        "--generation-config",
        "vllm",
        "--enable-request-id-headers",
    ]
    if opts.get("compile_options"):
        argv.extend(["--compilation-config", json.dumps(opts["compile_options"])])
    argv.extend(opts.get("extra_args", []))
    devices = ",".join(map(str, ep["devices"]))
    env = {
        k: v
        for k, v in orchestrate.clean_base_env().items()
        if not k.startswith("VLLM_")
    }
    env.update(
        PYTHONPATH=str(ROOT),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=devices if gpu else ",".join(map(str, peer_devices)),
        VLLM_CACHE_ROOT=str(cache),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        PATH=f"{Path(python).parent}{os.pathsep}{env['PATH']}",
        VLLM_KV_CACHE_LAYOUT=opts["layout"],
    )
    if gpu:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    else:
        offload = work / "offload" / ep["id"]
        env.update(
            RBLN_DEVICES=devices,
            RBLN_OFFLOAD_DIR=str(offload),
            VLLM_RBLN_USE_VLLM_MODEL="1",
            VLLM_RBLN_USE_DEVICE_TENSOR="1",
            VLLM_RBLN_COMPILE_MODEL="1",
            VLLM_RBLN_COMPILE_STRICT_MODE="1",
            VLLM_RBLN_SAMPLER="0",
            VLLM_RBLN_PDD_LAYOUT_REORDER="0",
            VLLM_RBLN_COALESCE_KV_IMPORT="1" if opts.get("coalesce_kv_import") else "0",
            RBLN_USE_CUSTOM_KERNEL="0",
            RBLN_ROOT_IP=ep["host"],
            RBLN_LOCAL_IP=ep["host"],
        )
    if "prefill" in ep["roles"] or "decode" in ep["roles"]:
        extra = {"kv_recompute_threshold": 0}
        if "lookahead" in opts:
            extra["rbln_lookahead"] = {
                **opts["lookahead"],
                "events_path": str(work / f"{ep['id']}-admission.jsonl"),
            }
        connector = {
            "kv_connector": "NixlConnector" if gpu else "RblnNixlConnector",
            "kv_role": "kv_producer" if "prefill" in ep["roles"] else "kv_consumer",
            "kv_buffer_device": "cuda" if gpu else "cpu",
            "kv_load_failure_policy": "fail",
            "engine_id": engine_id,
            "kv_connector_extra_config": extra,
        }
        argv.extend(["--kv-transfer-config", json.dumps(connector)])
        env.update(
            UCX_NET_DEVICES="all",
            VLLM_NIXL_SIDE_CHANNEL_HOST=ep["host"],
            VLLM_NIXL_SIDE_CHANNEL_PORT=str(ep["kv_port"]),
        )
    if trace := instrumentation.get("otlp_endpoint"):
        argv += ["--otlp-traces-endpoint", trace]
        env.update(
            OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf",
            OTEL_SERVICE_NAME=ep["id"],
        )
    if not gpu and (instrumentation.get("rbln_events") or "lookahead" in opts):
        env.update(
            VLLM_RBLN_METRICS="1", VLLM_RBLN_METRICS_DIR=str(work / "events" / ep["id"])
        )
    if "lookahead" in opts:
        env["PROMETHEUS_MULTIPROC_DIR"] = str(work / "prom" / ep["id"])
    return {"argv": argv, "env": env}
