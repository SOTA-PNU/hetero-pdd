"""Render the official llm-d v1 processes and native file-discovery inventory."""

import json
import os
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROUTER_COMMIT = "b9393d1a9f1d5a72ecd86f8d2b643ab54c2a9b92"
LLMD_COMMIT = "378b42d18dabe0b9c4ced4e6f3670d93b52db08e"
ROUTE_HEADERS = ("target", "prefill", "path", "preload", "policy", "reason")
RESERVED = [
    "x-gateway-destination-endpoint",
    "x-envoy-original-dst-host",
    "x-data-parallel-host-port",
    "x-llm-d-pin-host-port",
    *(
        prefix + name
        for prefix in ("x-", "x-llm-d-")
        for name in (
            "prefiller-host-port",
            "encoder-hosts-ports",
            "kv-cache-source-host-port",
        )
    ),
    *("x-hetero-" + name for name in ROUTE_HEADERS),
    *(
        "x-hetero-expected-" + name
        for name in ("engine", "host", "port", "tp", "block-size", "num-blocks")
    ),
]


def binary_dir(bin_dir=None):
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return Path(
        bin_dir or os.environ.get("LLMD_BIN_DIR") or cache / "hetero-pdd/llmd-v1/bin"
    ).resolve()


def route_labels(endpoint):
    labels = dict(endpoint["labels"])
    for key, value in {
        "engine-port": endpoint["port"],
        "engine-id": endpoint.get("engine_id"),
        "kv-port": endpoint.get("kv_port"),
        "tp": endpoint.get("tp"),
        "block-size": endpoint.get("engine", {}).get("block_size"),
        "num-blocks": endpoint.get("engine", {}).get("num_blocks"),
    }.items():
        if value is not None:
            labels["pdd.sota.dev/" + key] = str(value)
    return labels


def _envoy_http(hcm):
    hcm["preserve_external_request_id"] = True
    hcm["always_set_request_id_in_response"] = True
    hcm["http_filters"].insert(
        0,
        {
            "name": "envoy.filters.http.header_mutation",
            "typed_config": {
                "@type": (
                    "type.googleapis.com/envoy.extensions.filters.http."
                    "header_mutation.v3.HeaderMutation"
                ),
                "mutations": {
                    "request_mutations": [{"remove": key} for key in RESERVED]
                },
            },
        },
    )
    virtual = hcm["route_config"]["virtual_hosts"][0]
    virtual["response_headers_to_add"] = [
        {
            "header": {
                "key": "x-hetero-" + name,
                "value": "%REQ(X-HETERO-" + name.upper() + ")%",
            },
            "append_action": "OVERWRITE_IF_EXISTS_OR_ADD",
        }
        for name in ROUTE_HEADERS
    ]
    hcm["access_log"] = [
        {
            "name": "envoy.access_loggers.stdout",
            "typed_config": {
                "@type": (
                    "type.googleapis.com/envoy.extensions.access_loggers."
                    "stream.v3.StdoutAccessLog"
                ),
                "log_format": {
                    "json_format": {
                        "event": "gateway_access",
                        "request_id": "%REQ(X-REQUEST-ID)%",
                        "http_status": "%RESPONSE_CODE%",
                        "destination": "%UPSTREAM_HOST%",
                        **{
                            key: "%REQ(X-HETERO-" + key.upper() + ")%"
                            for key in ROUTE_HEADERS
                        },
                    }
                },
            },
        }
    ]


def _envoy(ports, host):
    cfg = yaml.safe_load((HERE / "envoy.yaml").read_text())
    cfg["admin"]["address"]["socket_address"]["port_value"] = ports["admin"]
    listener = cfg["static_resources"]["listeners"][0]
    listener["address"]["socket_address"].update(
        address=host, port_value=ports["gateway"]
    )
    hcm = listener["filter_chains"][0]["filters"][0]["typed_config"]
    hcm["http_filters"][0]["typed_config"]["grpc_service"]["envoy_grpc"][
        "authority"
    ] = f"localhost:{ports['epp']}"
    _envoy_http(hcm)
    ext = cfg["static_resources"]["clusters"][1]
    target = ext["load_assignment"]["endpoints"][0]["lb_endpoints"][0]["endpoint"][
        "address"
    ]["socket_address"]
    target["port_value"] = ports["epp"]
    return cfg


def _trace_arg():
    enabled = bool(
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )
    return "--tracing=" + str(enabled).lower()


def _sidecar_args(endpoint, trace_arg="--tracing=false"):
    return [
        "--port",
        str(endpoint["sidecar_port"]),
        "--model-server-port",
        str(endpoint["port"]),
        "--kv-connector=nixlv2",
        "--prefill-max-retries=0",
        "--secure-serving=false",
        "--metrics-port",
        str(endpoint.get("sidecar_metrics_port") or 0),
        trace_arg,
    ]


def sidecar_container(endpoint, *, image):
    return {
        "name": "pd-sidecar",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "command": ["/pd-sidecar"],
        "args": _sidecar_args(endpoint),
        "env": [
            {"name": "HETERO_PDD_STRICT", "value": "1"},
            {"name": "HETERO_ENDPOINT_ID", "value": endpoint["id"]},
        ],
        "ports": [{"name": "pd", "containerPort": endpoint["sidecar_port"]}],
        "readinessProbe": {"httpGet": {"path": "/health", "port": "pd"}},
    }


def render(work, *, native, endpoints, ports, host="127.0.0.1", bin_dir=None):
    """Write resolved files and return ordered, foreground process specifications."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bins = binary_dir(bin_dir)
    cfg = deepcopy(native)
    discovery = {
        "type": "file-discovery",
        "name": "hetero-file-discovery",
        "parameters": {"path": str(work / "endpoints.yaml"), "watchFile": True},
    }
    cfg.setdefault("plugins", []).append(discovery)
    cfg.setdefault("dataLayer", {}).setdefault("discovery", {})["endpoints"] = {
        "pluginRef": discovery["name"]
    }
    inventory, specs = [], []
    for ep in endpoints:
        consumer = "decode" in ep["roles"]
        port = ep["sidecar_port"] if consumer else ep["port"]
        inventory.append(
            {
                "name": ep["id"],
                "address": ep["host"],
                "port": str(port),
                "labels": route_labels(ep),
            }
        )
        if consumer:
            specs.append(
                {
                    "name": "sidecar-" + ep["id"],
                    "argv": [
                        str(bins / "pd-sidecar"),
                        *_sidecar_args(ep, _trace_arg()),
                    ],
                    "env": {"HETERO_PDD_STRICT": "1", "HETERO_ENDPOINT_ID": ep["id"]},
                    "health": f"http://127.0.0.1:{port}/health",
                }
            )
    (work / "epp.yaml").write_text(json.dumps(cfg, indent=2))
    (work / "envoy.yaml").write_text(
        yaml.safe_dump(_envoy(ports, host), sort_keys=False)
    )
    # Go's YAML 1.1 reader treats valid endpoint IDs such as n and on as booleans.
    (work / "endpoints.yaml").write_text(json.dumps({"endpoints": inventory}, indent=2))
    specs.append(
        {
            "name": "epp",
            "argv": [
                str(bins / "epp"),
                "--config-file=" + str(work / "epp.yaml"),
                "--pool-name=file-discovery",
                "--pool-namespace=default",
                "--allow-experimental-plugins",
                f"--grpc-port={ports['epp']}",
                f"--grpc-health-port={ports['epp_health']}",
                f"--metrics-port={ports['epp_metrics']}",
                "--metrics-endpoint-auth=false",
                "--secure-serving=false",
                "--health-checking=true",
                "--zap-encoder=json",
                _trace_arg(),
            ],
            "env": {},
            "health": f"tcp://127.0.0.1:{ports['epp_health']}",
        }
    )
    specs.append(
        {
            "name": "gateway",
            "argv": [
                str(bins / "envoy"),
                "--config-path",
                str(work / "envoy.yaml"),
                "--concurrency",
                "2",
                "--disable-hot-restart",
            ],
            "env": {},
            "health": f"http://127.0.0.1:{ports['admin']}/ready",
        }
    )
    return specs


def kubernetes(native, *, namespace, name, model, ports, images, selector):
    """Specialize the checked-in render of the upstream standalone Helm chart.

    Cluster-local listener/health ports use the chart convention. Only the
    Service's external HTTP port is taken from the experiment.
    """
    import hashlib

    base = (HERE / "upstream/standalone.yaml").read_text()
    docs = [
        r
        for r in yaml.safe_load_all(
            base.replace("pdd-serving", name).replace("pdd-namespace", namespace)
        )
        if r
    ]
    native_yaml = json.dumps(native, indent=2)
    for resource in docs:
        kind = resource["kind"]
        if kind == "ConfigMap":
            if "envoy.yaml" in resource["data"]:
                resource["metadata"]["name"] = name + "-envoy"
                envoy = yaml.safe_load(resource["data"]["envoy.yaml"])
                for listener in envoy["static_resources"]["listeners"]:
                    for chain in listener.get("filter_chains", []):
                        for entry in chain["filters"]:
                            hcm = entry.get("typed_config", {})
                            if any(
                                f["name"] == "envoy.filters.http.ext_proc"
                                for f in hcm.get("http_filters", [])
                            ):
                                _envoy_http(hcm)
                resource["data"]["envoy.yaml"] = yaml.safe_dump(envoy, sort_keys=False)
            else:
                resource["data"] = {"epp.yaml": native_yaml}
        elif kind == "Service":
            for port in resource["spec"]["ports"]:
                if port["name"] == "http":
                    port["port"] = ports["gateway"]
        elif kind == "Deployment":
            template = resource["spec"]["template"]
            template["metadata"]["annotations"]["checksum/config"] = hashlib.sha256(
                native_yaml.encode()
            ).hexdigest()
            for container in template["spec"]["containers"]:
                if container["name"] == "envoy-proxy":
                    container["image"] = images["envoy"]
                    continue
                container["image"] = images["epp"]
                container["imagePullPolicy"] = "IfNotPresent"
                arg = container["args"].index("--endpoint-selector") + 1
                container["args"][arg] = ",".join(
                    f"{k}={v}" for k, v in sorted(selector.items())
                )
            for volume in template["spec"]["volumes"]:
                if volume["name"] == "config":
                    volume["configMap"]["name"] = name + "-envoy"
    return docs
