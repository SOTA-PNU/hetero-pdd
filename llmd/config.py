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
"""Compile an experiment and backend capabilities into endpoint labels and KV pairs."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import yaml


class UniqueKeys(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for raw_key, raw_value in node.value:
        key = loader.construct_object(raw_key)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"expected a unique string key, found {key!r}")
        result[key] = loader.construct_object(raw_value, deep=True)
    return result


UniqueKeys.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def read_yaml(path):
    value = yaml.load(Path(path).read_text(), Loader=UniqueKeys)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return value


def fields(value, allowed, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if missing := set(required) - value.keys():
        raise ValueError(f"{name} missing {', '.join(sorted(missing))}")
    if extra := value.keys() - set(allowed):
        raise ValueError(f"{name} unknown keys: {', '.join(sorted(extra))}")


def integer(value, name, low=1, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"invalid {name}: {value!r}")
    return value


OVERLAP_PROBE_DEFAULTS = {
    "anchor_input_tokens": 128,
    "anchor_output_tokens": 896,
    "following_input_tokens": 64,
    "following_output_tokens": 32,
    "arrival_offset_ms": 2000,
}


def overlap_probe_settings(raw):
    fields(raw, OVERLAP_PROBE_DEFAULTS, set(), "verify.overlap_probe")
    probe = {**OVERLAP_PROBE_DEFAULTS, **raw}
    for key, value in probe.items():
        integer(
            value,
            f"verify.overlap_probe.{key}",
            high=179999 if key == "arrival_offset_ms" else None,
        )
    return probe


def check_probe_context(probe, capacity):
    for key in ("anchor", "following"):
        if probe[f"{key}_input_tokens"] + probe[f"{key}_output_tokens"] > capacity:
            raise ValueError(
                f"verify.overlap_probe.{key} exceeds engine context {capacity}"
            )


def fingerprint(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Backend adapter capabilities, not scheduler policies. TP/PP/DP are dimensions
# of one service instance. vLLM uses one accelerator per rank on this launcher.
BACKENDS = {
    "cuda": {
        "kind": "gpu",
        "roles": {"full", "prefill", "decode"},
        "resource": "nvidia.com/gpu",
    },
    "rbln": {
        "kind": "npu",
        "roles": {"full", "decode"},
        "resource": "rebellions.ai/npu",
    },
}
DEFAULT_ENGINE = {
    "dtype": "float16",
    "kv_dtype": "float16",
    "layout": "HND",
    "block_size": 128,
    "tp": 1,
    "pp": 1,
    "dp": 1,
    "max_model_len": 1024,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 128,
    "num_blocks": 32,
}
ENGINE_KEYS = set(DEFAULT_ENGINE) | {
    "python",
    "lookahead",
    "extra_args",
    "compile_options",
    "coalesce_kv_import",
}


def model_definition(model, directory):
    if isinstance(model, str):
        path = (directory / model).resolve()
        directory, model = path.parent, read_yaml(path)
    fields(
        model,
        {
            "name",
            "revision",
            "path",
            "tokenizer",
            "tokenizer_revision",
            "tokenizer_path",
        },
        {"name", "revision"},
        "model",
    )
    result = dict(model)
    result.setdefault("tokenizer", result["name"])
    if result["tokenizer"] == result["name"]:
        result.setdefault("tokenizer_revision", result["revision"])
    for key in ("revision", "tokenizer_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(result.get(key, ""))):
            raise ValueError(f"model.{key} requires an immutable 40-character revision")
    for key in ("name", "tokenizer"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"model.{key} must be nonempty")
    for key in ("path", "tokenizer_path"):
        if key in result:
            result[key] = str((directory / Path(result[key]).expanduser()).resolve())
    return result


def wire(model, ep):
    cfg = ep["engine"]
    return {
        "model": model["name"],
        "revision": model["revision"],
        "tokenizer": model["tokenizer"],
        "tokenizer_revision": model["tokenizer_revision"],
        "protocol": "nixlv2-0.22",
        **{
            k: cfg[k]
            for k in ("dtype", "kv_dtype", "layout", "block_size", "tp", "pp", "dp")
        },
    }


def pair_error(p, d, model):
    if wire(model, p) != wire(model, d):
        return "KV geometry, model or shard mismatch; redistribution is unsupported"
    cfg = d["engine"]
    if cfg["pp"] != 1 or cfg["dp"] != 1:
        return "NIXL lease mapping requires one pipeline and data-parallel rank"
    if d["backend"] == "rbln":
        if p["backend"] != "cuda" or cfg["tp"] != 1:
            return "RBLN host-staging import supports a CUDA producer and TP1 shards"
        if p["node"] != d["node"]:
            return "RBLN CUDA IPC requires a visible producer GPU on the same node"
        if (
            cfg["layout"] != "HND"
            or cfg["kv_dtype"] not in ("float16", "bfloat16")
            or cfg["dtype"] != cfg["kv_dtype"]
        ):
            return "RBLN host-staging import requires matching native FP16/BF16 HND"
    return None


def load(path):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    if cfg.get("apiVersion") != "hetero-pdd/v2":
        raise ValueError(
            "expected apiVersion: hetero-pdd/v2 with an official native profile"
        )
    fields(
        cfg,
        {
            "apiVersion",
            "model",
            "profile",
            "ports",
            "host",
            "engine",
            "endpoints",
            "verify",
            "benchmark",
            "instrumentation",
            "cache_root",
            "hub_cache",
            "images",
            "cluster",
        },
        {"apiVersion", "model", "profile", "endpoints", "verify"},
        "experiment",
    )
    model = cfg["model"] = model_definition(cfg["model"], path.parent)
    cfg["profile"] = str((path.parent / cfg["profile"]).resolve())
    native = read_yaml(cfg["profile"])
    cfg["native"] = native
    cfg.setdefault("host", "127.0.0.1")
    cfg["host"] = str(ipaddress.IPv4Address(cfg["host"]))
    defaults = {
        "gateway": 18000,
        "admin": 19000,
        "epp": 19001,
        "epp_health": 19002,
        "epp_metrics": 19003,
    }
    fields(cfg.get("ports", {}), defaults, set(), "ports")
    cfg["ports"] = defaults | cfg.get("ports", {})
    occupied = set()

    def port(host, val):
        key = (host, integer(val, "port", high=65535))
        if key in occupied:
            raise ValueError(f"duplicate listen address {key}")
        occupied.add(key)

    for val in cfg["ports"].values():
        port(cfg["host"], val)

    common = cfg.get("engine", {})
    fields(common, ENGINE_KEYS, set(), "engine")
    verify = cfg["verify"]
    keys = {"input_tokens", "output_tokens", "requests", "concurrency"}
    fields(verify, keys | {"paths", "overlap_probe"}, keys, "verify")
    if "overlap_probe" in verify:
        verify["overlap_probe"] = overlap_probe_settings(verify["overlap_probe"])
    for key in keys:
        integer(verify[key], f"verify.{key}")
    paths = verify.get("paths")
    if paths is not None and (
        not isinstance(paths, list) or not paths or not set(paths) <= {"pdd", "full"}
    ):
        raise ValueError("verify.paths requires full and/or pdd")
    if verify["concurrency"] > verify["requests"]:
        raise ValueError("verify.concurrency exceeds requests")
    if "benchmark" in cfg:
        bench = cfg["benchmark"]
        fields(
            bench,
            keys | {"seed", "warmup", "repeats", "comparisons", "ci"},
            keys,
            "benchmark",
        )
        for key in keys:
            integer(bench[key], f"benchmark.{key}")
        if bench["output_tokens"] < 2 or bench["concurrency"] > bench["requests"]:
            raise ValueError(
                "benchmark needs output_tokens >= 2 and requests >= concurrency"
            )
        for key, default, low in (
            ("seed", 20260924, 0),
            ("warmup", 2, 0),
            ("repeats", 1, 1),
        ):
            integer(bench.setdefault(key, default), f"benchmark.{key}", low=low)
        if bench["seed"] > 2**63 - 1:
            raise ValueError("benchmark.seed exceeds the engine's signed 64-bit range")
        if type(bench.setdefault("ci", True)) is not bool:
            raise ValueError("benchmark.ci must be a boolean")
        comparisons = bench.setdefault("comparisons", {})
        if not isinstance(comparisons, dict):
            raise ValueError(
                "benchmark.comparisons requires a label-to-profile mapping"
            )
        for label, filename in comparisons.items():
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", label) or label == "candidate":
                raise ValueError(
                    f"invalid or reserved benchmark.comparisons label: {label}"
                )
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{label}: expected a comparison profile filename")
            comparisons[label] = str((path.parent / filename).resolve())
    instrumentation = cfg.setdefault("instrumentation", {})
    fields(instrumentation, {"rbln_events", "otlp_endpoint"}, set(), "instrumentation")
    if type(instrumentation.get("rbln_events", False)) is not bool:
        raise ValueError("rbln_events must be boolean")
    endpoints = cfg["endpoints"]
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("nonempty endpoints list required")
    ids, devices = set(), set()
    for index, raw in enumerate(endpoints):
        ep = endpoints[index] = deepcopy(raw)
        required = {"id", "backend", "roles", "devices", "port"}
        fields(
            ep,
            required
            | {
                "node",
                "host",
                "kv_port",
                "sidecar_port",
                "slots",
                "labels",
                "engine",
                "resource",
                "nodeSelector",
                "sidecar_metrics_port",
            },
            required,
            "endpoint",
        )
        name = ep["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-z](?:[a-z0-9-]{0,38}[a-z0-9])?", name)
            or name in ids
        ):
            raise ValueError(f"invalid or duplicate endpoint ID {name}")
        ids.add(name)
        backend = BACKENDS.get(ep["backend"])
        if not backend:
            raise ValueError(f"{name}: missing backend adapter for {ep['backend']}")
        roles = ep["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or len(set(roles)) != len(roles)
            or not set(roles) <= backend["roles"]
        ):
            raise ValueError(
                f"{name}: {ep['backend']} supports {sorted(backend['roles'])}"
            )
        if {"prefill", "decode"} <= set(roles):
            raise ValueError(
                f"{name}: NIXL adapter needs separate producer and consumer instances"
            )
        ep.setdefault("node", "local")
        if (
            not isinstance(ep["node"], str)
            or len(ep["node"]) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", ep["node"])
        ):
            raise ValueError(
                f"{name}: node needs a Kubernetes-compatible node-group label"
            )
        if not isinstance(ep.get("nodeSelector", {}), dict) or not all(
            isinstance(v, str) for v in ep.get("nodeSelector", {}).values()
        ):
            raise ValueError(f"{name}: nodeSelector requires string labels")
        ep.setdefault("host", "127.0.0.1")
        ep["host"] = str(ipaddress.IPv4Address(ep["host"]))
        if ep["host"] == "0.0.0.0":
            raise ValueError(f"{name}: host requires an advertised interface address")
        ep["kind"] = backend["kind"]
        ep.setdefault("resource", backend["resource"])
        ep["slots"] = integer(ep.get("slots", 1), f"{name}.slots")
        personal = ep.get("engine", {})
        fields(personal, ENGINE_KEYS, set(), f"{name}.engine")
        params = ep["engine"] = DEFAULT_ENGINE | common | personal
        for k in (
            "tp",
            "pp",
            "dp",
            "max_model_len",
            "block_size",
            "num_blocks",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            integer(params[k], f"{name}.{k}")
        if ep["slots"] > params["max_num_seqs"]:
            raise ValueError(f"{name}: slots exceed max_num_seqs")
        if (params["num_blocks"] - 1) * params["block_size"] < params["max_model_len"]:
            raise ValueError(
                f"{name}: KV blocks must include a null block plus the maximum context"
            )
        if verify["input_tokens"] + verify["output_tokens"] > params["max_model_len"]:
            raise ValueError(f"{name}: verification exceeds engine context")
        if "overlap_probe" in verify:
            check_probe_context(verify["overlap_probe"], params["max_model_len"])
        if "benchmark" in cfg:
            bench = cfg["benchmark"]
            if bench["input_tokens"] + bench["output_tokens"] > params["max_model_len"]:
                raise ValueError(f"{name}: benchmark exceeds engine context")
        if params["layout"] not in ("HND", "NHD"):
            raise ValueError(f"{name}: vLLM cache layout must be HND or NHD")
        if ep["backend"] == "rbln" and (params["pp"] != 1 or params["dp"] != 1):
            raise ValueError(f"{name}: RBLN adapter supports TP, lacks PP/DP placement")
        ds = ep["devices"]
        if (
            not isinstance(ds, list)
            or len(ds) != params["tp"] * params["pp"] * params["dp"]
        ):
            raise ValueError(f"{name}: physical device count must match TP × PP × DP")
        for device in ds:
            key = (ep["node"], ep["backend"], integer(device, f"{name}.devices", low=0))
            if key in devices:
                raise ValueError(f"{name}: duplicate device reservation {key}")
            devices.add(key)
        port(ep["host"], ep["port"])
        if "prefill" in roles or "decode" in roles:
            if "kv_port" not in ep:
                raise ValueError(f"{name}: NIXL role requires kv_port")
            for rank in range(params["tp"]):
                port(ep["host"], ep["kv_port"] + rank)
        if "decode" in roles:
            if "sidecar_port" not in ep:
                raise ValueError(f"{name}: decode role requires sidecar_port")
            port(ep["host"], ep["sidecar_port"])
            if "sidecar_metrics_port" in ep:
                port(ep["host"], ep["sidecar_metrics_port"])
        extra = params.get("extra_args", [])
        if not isinstance(extra, list) or not all(
            isinstance(arg, str) for arg in extra
        ):
            raise ValueError(f"{name}: extra_args must be an argv list")
        # Preserve the identity and geometry advertised to discovery.
        owned = {
            "--model",
            "--host",
            "--port",
            "--served-model-name",
            "--revision",
            "--tokenizer",
            "--tokenizer-revision",
            "--dtype",
            "--kv-cache-dtype",
            "--block-size",
            "--num-gpu-blocks-override",
            "--kv-transfer-config",
            "--max-model-len",
            "--max-num-seqs",
            "--max-num-batched-tokens",
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "--data-parallel-size",
            "-tp",
            "-pp",
            "-dp",
        }
        owned |= {
            "--compilation-config",
            "--seed",
            "--generation-config",
            "--enable-request-id-headers",
            "--no-enable-request-id-headers",
            "--enable-prefix-caching",
            "--no-enable-prefix-caching",
            "--enable-chunked-prefill",
            "--no-enable-chunked-prefill",
        }
        if any(x.split("=")[0].replace("_", "-") in owned for x in extra):
            raise ValueError(f"{name}: extra_args overrides managed options")
        if "lookahead" in params and (ep["backend"] != "rbln" or "decode" not in roles):
            raise ValueError(f"{name}: async host receive requires an RBLN consumer")
        if "lookahead" in params:
            look = params["lookahead"]
            limits = {"active_decode_limit", "lookahead_depth", "prefetch_byte_budget"}
            fields(look, limits | {"receive_timeout_s"}, limits, "lookahead")
            integer(
                look["active_decode_limit"],
                "active_decode_limit",
                high=params["max_num_seqs"],
            )
            integer(look["lookahead_depth"], "lookahead_depth", low=0)
            integer(look["prefetch_byte_budget"], "prefetch_byte_budget")
            t = look.get("receive_timeout_s", 30)
            if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
                raise ValueError("receive_timeout_s must be finite and positive")
        if params.get("compile_options") is not None and not isinstance(
            params["compile_options"], dict
        ):
            raise ValueError("compile_options requires vLLM compilation-config object")
        if "coalesce_kv_import" in params and (
            type(params["coalesce_kv_import"]) is not bool
            or ep["backend"] != "rbln"
            or "decode" not in roles
        ):
            raise ValueError(
                "coalesce_kv_import requires a boolean on an RBLN consumer"
            )
        labels = ep.setdefault("labels", {})
        if not isinstance(labels, dict) or any(
            not isinstance(v, str) or k.startswith("pdd.sota.dev/")
            for k, v in labels.items()
        ):
            raise ValueError(
                "endpoint labels must be strings; pdd.sota.dev/ labels are generated"
            )
        labels.update(
            {
                "pdd.sota.dev/" + k: v
                for k, v in {
                    "id": name,
                    "model": fingerprint(model["name"])[:16],
                    "backend": ep["backend"],
                    "order": str(index),
                    "slots": str(ep["slots"]),
                    "kv-group": fingerprint(wire(model, ep))[:20],
                    **dict.fromkeys(
                        [
                            {"prefill": "producer", "decode": "consumer"}.get(r, r)
                            for r in roles
                        ],
                        "true",
                    ),
                }.items()
            }
        )
        if "lookahead" in params:
            look = params["lookahead"]
            labels["pdd.sota.dev/lookahead-depth"] = str(look["lookahead_depth"])
            labels["pdd.sota.dev/lookahead-bytes"] = str(look["prefetch_byte_budget"])
        ep["tp"] = params["tp"]
    producers = [e for e in endpoints if "prefill" in e["roles"]]
    consumers = [e for e in endpoints if "decode" in e["roles"]]
    rejected = {}
    for d in consumers:
        d["peers"] = []
        for p in producers:
            why = pair_error(p, d, model)
            if why:
                rejected[p["id"] + "->" + d["id"]] = why
                continue
            d["peers"].append(p["id"])
            d["labels"]["pdd.sota.dev/peer-" + p["id"]] = "true"
        if not d["peers"]:
            raise ValueError(
                f"{d['id']}: no compatible producer: {rejected or 'empty pool'}"
            )
    if producers and not consumers:
        raise ValueError("producer pool has no decode candidates")
    for filename in cfg.get("benchmark", {}).get("comparisons", {}).values():
        validate_profile(read_yaml(filename), endpoints)
    validate_profile(native, endpoints)
    cfg["rejected_pairs"] = rejected
    return cfg


def validate_profile(native, endpoints):
    if (
        native.get("apiVersion") != "llm-d.ai/v1"
        or native.get("kind") != "EndpointPickerConfig"
    ):
        raise ValueError("profile requires a native llm-d.ai/v1 EndpointPickerConfig")
    for prof in native.get("schedulingProfiles", []):
        refs = {p["pluginRef"] for p in prof.get("plugins", [])}
        for plugin in native.get("plugins", []):
            if (
                plugin.get("name", plugin["type"]) in refs
                and plugin["type"] == "label-selector-filter"
            ):
                match = plugin.get("parameters", {}).get("matchLabels", {})
                if match and not any(
                    all(e["labels"].get(k) == v for k, v in match.items())
                    for e in endpoints
                ):
                    raise ValueError(f"{prof['name']}: no candidates for {match}")
