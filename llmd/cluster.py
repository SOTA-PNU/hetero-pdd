"""Render the standalone chart and one resource-accounted Pod per engine."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from . import config, engines, runtime, serving

ENVOY = (
    "docker.io/envoyproxy/envoy:distroless-v1.33.2@sha256:"
    "85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8"
)
IMAGES = {
    "engine": "hetero-pdd-engine:local",
    "epp": "hetero-pdd/epp:v1",
    "sidecar": "hetero-pdd/pd-sidecar:v1",
    "envoy": ENVOY,
}


def render(cfg, eps, work, namespace):
    site = cfg.get("cluster", {})
    config.fields(
        site,
        {
            "modelClaim",
            "cacheClaim",
            "runtimeClassName",
            "peerGpuVisibility",
            "nodeSelector",
            "cpu",
            "memory",
        },
        set(),
        "cluster",
    )
    placement = site.get("nodeSelector", {})
    if not isinstance(placement, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v
        for k, v in placement.items()
    ):
        raise ValueError("cluster.nodeSelector requires nonempty string labels")
    images = IMAGES | cfg.get("images", {})
    name = "pdd-" + eps[0]["engine_id"][:10]
    selector = {"pdd.sota.dev/pool": name}
    docs = serving.kubernetes(
        cfg["native"],
        namespace=namespace,
        name=name,
        model=cfg["model"]["name"],
        ports=cfg["ports"],
        images=images,
        selector=selector,
    )
    run_id = eps[0]["engine_id"].split("-", 1)[0]
    for doc in docs:
        doc["metadata"]["namespace"] = namespace
    for ep in eps:
        dec = "decode" in ep["roles"]
        ep.update(
            port=8001 if dec else 8000,
            sidecar_port=8000,
            kv_port=5557,
            sidecar_metrics_port=9110,
        )
        labels = {
            **selector,
            **serving.route_labels(ep),
            "pdd.sota.dev/node-group": ep["node"],
        }
        labels.pop("pdd.sota.dev/engine-id", None)
        labels["pdd.sota.dev/engine-id-source"] = "pod-name"
        ep["engine_id_source"] = "pod-name"
        eid = ep["id"]
        owned = {**selector, "pdd.sota.dev/id": eid}
        meta = {"name": name + "-" + eid, "namespace": namespace, "labels": owned}
        payload = dict(
            endpoint=ep, model=cfg["model"], instrumentation=cfg["instrumentation"]
        )
        docs.append(
            dict(
                apiVersion="v1",
                kind="ConfigMap",
                metadata=meta,
                data={"instance.json": json.dumps(payload)},
            )
        )
        required = {ep["resource"]: str(len(ep["devices"]))}
        cpu_mem = {k: site[k] for k in ("cpu", "memory") if k in site}
        cont = dict(
            name="engine",
            image=images.get(ep["backend"], images["engine"]),
            imagePullPolicy="IfNotPresent",
            command=[
                "/opt/pdd-decode/bin/python",
                "-m",
                "llmd.cluster",
                "/config/instance.json",
            ],
            resources={"requests": {**required, **cpu_mem}, "limits": required},
            ports=[{"name": "engine", "containerPort": ep["port"]}],
            env=[
                {
                    "name": "POD_NAME",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                },
                {
                    "name": "POD_IP",
                    "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                },
                {
                    "name": "POD_UID",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                },
                {
                    "name": "NODE_NAME",
                    "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
                },
            ],
            readinessProbe={
                "httpGet": {"path": "/health", "port": "engine"},
                "periodSeconds": 3,
            },
            volumeMounts=[
                {"name": "config", "mountPath": "/config", "readOnly": True},
                {"name": "models", "mountPath": "/models", "readOnly": True},
                {"name": "cache", "mountPath": "/cache"},
                {"name": "work", "mountPath": "/run/hetero"},
            ],
        )
        containers = [cont]
        if dec:
            containers.append(serving.sidecar_container(ep, image=images["sidecar"]))
        pod = dict(
            containers=containers,
            terminationGracePeriodSeconds=60,
            volumes=[
                {"name": "config", "configMap": {"name": meta["name"]}},
                {
                    "name": "models",
                    "persistentVolumeClaim": {
                        "claimName": site.get("modelClaim", "pdd-models")
                    },
                },
                {
                    "name": "cache",
                    "persistentVolumeClaim": {
                        "claimName": site.get("cacheClaim", "pdd-compiled")
                    },
                },
                {"name": "work", "emptyDir": {}},
            ],
        )
        if ep["node"] != "local":
            pod["nodeSelector"] = {"kubernetes.io/hostname": ep["node"]}
        endpoint_selector = ep.get("nodeSelector", {})
        if any(
            k in endpoint_selector and endpoint_selector[k] != v
            for k, v in pod.get("nodeSelector", {}).items()
        ):
            raise ValueError(
                f"{eid}: nodeSelector conflicts with the endpoint hostname"
            )
        pod.setdefault("nodeSelector", {}).update(endpoint_selector)
        if any(
            p["backend"] == "rbln" and p["node"] == ep["node"] and p.get("peers")
            for p in eps
        ):
            # CUDA-IPC needs a shared IPC namespace and a colocated producer.
            pod["hostIPC"] = True
            pod["affinity"] = {
                "podAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "labelSelector": {
                                "matchLabels": {
                                    **selector,
                                    "pdd.sota.dev/node-group": ep["node"],
                                }
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        }
                    ]
                }
            }
        if ep["backend"] == "rbln" and ep.get("peers"):
            visibility = site.get("peerGpuVisibility")
            if not visibility:
                raise ValueError(
                    "specify producer UUIDs or all in cluster.peerGpuVisibility"
                )
            cont["env"].extend(
                [
                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": visibility},
                    {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                ]
            )
        if site.get("runtimeClassName"):
            pod["runtimeClassName"] = site["runtimeClassName"]
        docs.append(
            dict(
                apiVersion="apps/v1",
                kind="Deployment",
                metadata=meta,
                spec=dict(
                    replicas=1,
                    strategy={"type": "Recreate"},
                    selector={"matchLabels": owned},
                    template={"metadata": {"labels": labels}, "spec": pod},
                ),
            )
        )
    for resource in docs:
        resource["metadata"].setdefault("labels", {})["pdd.sota.dev/run"] = run_id
        if resource["kind"] == "Deployment":
            spec = resource["spec"]["template"]["spec"]
            pod_selector = spec.get("nodeSelector", {})
            if any(
                k in pod_selector and pod_selector[k] != v for k, v in placement.items()
            ):
                raise ValueError(
                    "endpoint node selector conflicts with cluster.nodeSelector"
                )
            if placement:
                spec["nodeSelector"] = placement | pod_selector
            resource["spec"]["template"]["metadata"].setdefault("labels", {})[
                "pdd.sota.dev/run"
            ] = run_id
    out = work / "kubernetes.yaml"
    # Each document is JSON (valid YAML), including string labels like "y".
    out.write_text(
        "---\n" + "\n---\n".join(json.dumps(d, indent=2) for d in docs) + "\n"
    )
    info = {
        "run_id": run_id,
        "verify": cfg["verify"],
        "instrumentation": cfg["instrumentation"],
        "ports": cfg["ports"],
        "namespace": namespace,
        "name": name,
        "selector": selector,
        "manifest": str(out),
        "images": images,
        "endpoint": f"http://{name}-epp.{namespace}.svc:{cfg['ports']['gateway']}/v1/completions",
        "model": cfg["model"],
        "profile": cfg["native"],
        "instances": eps,
    }
    runtime.save(work / "deployment.json", info)
    return info


def run_pod(path):
    from .__main__ import launch_dirs

    cfg = json.loads(path.read_text())
    ep = cfg["endpoint"]
    ep["host"] = os.environ["POD_IP"]
    ep["node"] = os.environ["NODE_NAME"]
    count = len(ep["devices"])
    peers = []
    if ep["kind"] == "gpu" or ep.get("peers"):
        indexes = [
            int(v)
            for v in engines._checked(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
            ).splitlines()
        ]
        if ep["kind"] == "gpu":
            if len(indexes) != count:
                raise ValueError("GPU allocation differs from requested device count")
            ep["devices"] = indexes
        else:
            peers = indexes
    if ep["kind"] == "npu":
        raw = os.environ.get("RBLN_DEVICES")
        eps = (
            [int(i) for i in raw.split(",")]
            if raw
            else [
                d["npu"]
                for d in json.loads(engines._checked(["rbln-smi", "--json"]))["devices"]
            ]
        )
        if len(eps) != count:
            raise ValueError("RBLN allocation differs from requested device count")
        ep["devices"] = eps
    model = engines.model_info(cfg["model"], {"hub_cache": "/models"})
    hw = engines.preflight(model, [ep], cluster=True)[ep["id"]]
    work = Path("/run/hetero") / os.environ["POD_UID"]
    work.mkdir(parents=True, exist_ok=True)
    if "prefill" in ep["roles"] or "decode" in ep["roles"]:
        try:
            with (work / "nixl-incarnation").open("x") as marker:
                marker.write(os.environ["POD_NAME"])
        except FileExistsError as exc:
            raise RuntimeError(
                "NIXL process already started in this Pod. Replace "
                "the Pod to obtain a fresh engine identity."
            ) from exc
    ep["engine_id"] = os.environ["POD_NAME"]
    with engines.compilation_cache("/cache", model, ep, hw) as (cache, info, publish):
        spec = engines.engine_spec(
            model,
            ep,
            work=work,
            engine_id=ep["engine_id"],
            cache=cache,
            peer_devices=peers,
            instrumentation=cfg["instrumentation"],
        )
        # The official sidecar connects to localhost. Advertise only the Pod IP to NIXL.
        spec["argv"][spec["argv"].index("--host") + 1] = "0.0.0.0"
        launch_dirs(spec["env"])
        runtime.save(
            work / "engine.json",
            dict(
                model=model,
                runtime=hw,
                cache=info,
                argv=spec["argv"],
                pod=os.environ["POD_UID"],
                endpoint=ep,
            ),
        )
        proc = subprocess.Popen(spec["argv"], env=spec["env"])
        signal.signal(signal.SIGTERM, lambda *_: proc.terminate())
        try:
            runtime.wait_ready(
                {"engine": f"http://127.0.0.1:{ep['port']}/health"},
                {"engine": proc},
                2400,
            )
            publish()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    return proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=Path)
    raise SystemExit(run_pod(parser.parse_args().instance))
