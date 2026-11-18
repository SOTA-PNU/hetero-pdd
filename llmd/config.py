"""Experiment YAML. Unknown keys and ambiguous device assignments are errors."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

POLICIES = ("baseline", "greedy", "round-robin", "least-queue", "fixed-pdd")


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


def fields(value, allowed, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if missing := set(required) - value.keys():
        raise ValueError(f"{name} missing {', '.join(sorted(missing))}")
    if extra := value.keys() - set(allowed):
        raise ValueError(f"{name} unknown keys: {', '.join(sorted(extra))}")


def integer(value, name, low=1, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(
            f"{name} must be an integer >= {low}" + (f" and <= {high}" if high else "")
        )
    return value


def load(path):
    from . import engines

    config = yaml.load(Path(path).read_text(), Loader=UniqueKeys)
    required = {"model", "policy", "endpoints", "verify"}
    fields(
        config, required | {"port", "engine", "instrumentation"}, required, "experiment"
    )
    policy = config["policy"]
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; supported: {', '.join(POLICIES)}")
    config["port"] = integer(config.get("port", 18000), "port", high=65535)
    endpoints = config["endpoints"]
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("endpoints must be a nonempty list")
    ids, devices, ports, roles = set(), set(), {config["port"]}, set()
    for ep in endpoints:
        required = {"id", "kind", "role", "device", "port"}
        fields(ep, required | {"slots", "kv_port"}, required, "endpoint")
        name, kind, role = ep["id"], ep["kind"], ep["role"]
        if (
            not isinstance(name, str)
            or not re.fullmatch("[a-z][a-z0-9-]{0,47}", name)
            or name == "router"
            or name in ids
        ):
            raise ValueError(f"invalid, reserved or duplicate endpoint id {name!r}")
        ids.add(name)
        if kind not in ("gpu", "npu"):
            raise ValueError(f"{name}: kind must be gpu or npu")
        device = integer(ep["device"], f"{name}.device", low=0)
        if (kind, device) in devices:
            raise ValueError(f"duplicate physical device {kind} {device}")
        devices.add((kind, device))
        roles.add(role)
        if policy == "fixed-pdd":
            if (kind, role) not in (
                ("gpu", "prefill"),
                ("npu", "decode"),
            ) or "kv_port" not in ep:
                raise ValueError(
                    f"{name}: fixed-pdd requires GPU prefill / NPU decode with kv_port"
                )
        elif role != "full" or "kv_port" in ep:
            raise ValueError(f"{name}: {policy} needs full role without kv_port")
        if "slots" in ep and policy != "greedy":
            raise ValueError(f"{name}: slots applies only to greedy")
        if policy == "greedy":
            ep["slots"] = integer(ep.get("slots", 1), f"{name}.slots", high=65535)
        for key in ("port", "kv_port"):
            if key in ep:
                port = integer(ep[key], f"{name}.{key}", high=65535)
                if port in ports:
                    raise ValueError(f"duplicate port {port}")
                ports.add(port)
    if policy == "fixed-pdd" and roles != {"prefill", "decode"}:
        raise ValueError(
            "fixed-pdd requires at least one GPU prefiller and one NPU decoder"
        )
    v = config["verify"]
    keys = {"input_tokens", "output_tokens", "concurrency", "requests"}
    fields(v, keys, keys, "verify")
    for key in keys:
        integer(v[key], f"verify.{key}")
    if v["concurrency"] > v["requests"]:
        raise ValueError("verify.concurrency exceeds verify.requests")
    if policy == "fixed-pdd" and sum(ep["role"] == "prefill" for ep in endpoints) != 1:
        raise ValueError(
            "native host-staging deployment supports one GPU prefiller per experiment"
        )
    engine = config.setdefault("engine", {})
    fields(
        engine,
        {"gpu_python", "npu_python", "hub_cache", "max_model_len"},
        set(),
        "engine",
    )
    for key in ("gpu_python", "npu_python", "hub_cache"):
        if key in engine and (not isinstance(engine[key], str) or not engine[key]):
            raise ValueError(f"engine.{key} must be a path")
    limit = integer(
        engine.setdefault("max_model_len", 1024), "engine.max_model_len", 128, 4096
    )
    if v["input_tokens"] + v["output_tokens"] > limit:
        raise ValueError(
            "verify input_tokens + output_tokens exceeds engine.max_model_len"
        )
    instrumentation = config.setdefault("instrumentation", {})
    fields(instrumentation, {"rbln_events", "otlp_endpoint"}, set(), "instrumentation")
    if type(instrumentation.get("rbln_events", False)) is not bool:
        raise ValueError("instrumentation.rbln_events must be boolean")
    if "otlp_endpoint" in instrumentation and (
        not isinstance(instrumentation["otlp_endpoint"], str)
        or not instrumentation["otlp_endpoint"]
    ):
        raise ValueError("instrumentation.otlp_endpoint must be a nonempty address")
    if not isinstance(config["model"], str) or config["model"] not in engines.PRESETS:
        raise ValueError(
            f"unknown model preset {config['model']!r}; "
            f"supported: {', '.join(engines.PRESETS)}"
        )
    return config
