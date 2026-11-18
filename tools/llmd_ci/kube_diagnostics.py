"""Read-only preflight and bounded evidence collection for a rendered Kubernetes run."""

import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from llmd import kube, runtime

MAX_BYTES = 256 * 1024
ERRORS = (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("time budget exhausted")
    return seconds


def preflight(work, context, timeout):
    work = Path(work)
    deploy = json.loads((work / "deployment.json").read_text())
    namespace = deploy["namespace"]
    deadline = time.monotonic() + timeout

    def get(*args):
        return json.loads(
            kube.command(
                context, namespace, ("get", *args, "-o", "json"), remaining(deadline)
            )
        )

    docs = kube.read_manifest(work, deploy)
    kube.expected_images(docs, strict=True)
    placements = {
        doc["spec"]["template"]["metadata"]["labels"].get(kube.ID_LABEL, ""): doc[
            "spec"
        ]["template"]["spec"]
        for doc in docs
        if doc["kind"] == "Deployment"
    }
    pods = list(placements.values())
    if not pods or any(not pod.get("nodeSelector") for pod in pods):
        raise ValueError("each Deployment needs an approved cluster.nodeSelector")
    for ep in deploy["instances"]:
        if ep.get("peers") and ep["backend"] == "rbln":
            engine = next(
                c for c in placements[ep["id"]]["containers"] if c["name"] == "engine"
            )
            visible = next(
                (
                    env.get("value", "")
                    for env in engine["env"]
                    if env["name"] == "NVIDIA_VISIBLE_DEVICES"
                ),
                "",
            )
            uuid = r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}"
            uuids = visible.split(",")
            if not all(re.fullmatch(uuid, value) for value in uuids) or len(
                uuids
            ) != len(set(uuids)):
                raise ValueError(
                    "cluster.peerGpuVisibility needs approved producer GPU UUIDs"
                )
    claims = sorted(
        {
            volume["persistentVolumeClaim"]["claimName"]
            for pod in pods
            for volume in pod.get("volumes", [])
            if "persistentVolumeClaim" in volume
        }
    )
    classes = sorted(
        {pod["runtimeClassName"] for pod in pods if pod.get("runtimeClassName")}
    )
    ns = get("namespace", namespace)
    if ns["status"]["phase"] != "Active" or ns["metadata"].get("deletionTimestamp"):
        raise ValueError(f"namespace {namespace} is not active")
    for claim in claims:
        pvc = get("pvc", claim)
        if pvc["status"].get("phase") != "Bound" or pvc["metadata"].get(
            "deletionTimestamp"
        ):
            raise ValueError(f"PVC {claim} must be Bound")
    for cls in classes:
        get("runtimeclass", cls)

    nodes = {}
    selectors = {
        ",".join(f"{k}={v}" for k, v in sorted(pod["nodeSelector"].items()))
        for pod in pods
    }
    selected_nodes = (
        node
        for selector in sorted(selectors)
        for node in get("nodes", "-l", selector)["items"]
    )
    for node in selected_nodes:
        if node["spec"].get("unschedulable") or node["metadata"].get(
            "deletionTimestamp"
        ):
            continue
        if any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in node["status"].get("conditions", [])
        ):
            nodes[node["metadata"]["name"]] = node
    # Only capacity published by the device plugin is counted. Kubernetes remains
    # authoritative for current availability, taints, affinity and final placement.
    capacities = {
        name: Counter(
            {r: int(n) for r, n in node["status"]["allocatable"].items() if "/" in r}
        )
        for name, node in nodes.items()
    }

    def candidates(selector):
        return {
            name
            for name, node in nodes.items()
            if all(node["metadata"]["labels"].get(k) == v for k, v in selector.items())
        }

    if not candidates(placements[""]["nodeSelector"]):
        raise ValueError("no Ready node matches the approved gateway placement")
    requirements = {}
    colocated = {
        ep["node"]
        for ep in deploy["instances"]
        if ep.get("peers") and ep["backend"] == "rbln"
    }
    for ep in deploy["instances"]:
        placement = placements[ep["id"]]["nodeSelector"]
        key = (
            ("group", ep["node"]) if ep["node"] in colocated else ("endpoint", ep["id"])
        )
        options, demand = requirements.setdefault(key, (set(nodes), Counter()))
        options.intersection_update(candidates(placement))
        demand[ep["resource"]] += len(ep["devices"])
    groups = sorted(requirements.values(), key=lambda pair: len(pair[0]))

    def fits(index):
        remaining(deadline)
        if index == len(groups):
            return True
        options, demand = groups[index]
        for name in sorted(options):
            free = capacities[name]
            if all(free[r] >= n for r, n in demand.items()):
                free.subtract(demand)
                if fits(index + 1):
                    return True
                free.update(demand)
        return False

    if not fits(0):
        raise ValueError(
            "Ready nodes lack requested device plugin capacity or placement"
        )
    return dict(
        context=context,
        namespace=namespace,
        namespace_uid=ns["metadata"]["uid"],
        claims=claims,
        runtime_classes=classes,
        eligible_nodes=sorted(set().union(*(g[0] for g in groups))),
    )


def collect(work, state, kubectl, deadline, reason):
    """Each call preserves earlier snapshots, including failures before rollback."""
    directory = work / "diagnostics"
    directory.mkdir(exist_ok=True)
    dest = Path(tempfile.mkdtemp(dir=directory, prefix=reason + "-"))
    report = dict(
        status="failed",
        run_id=state["run_id"],
        context=state["context"],
        namespace=state["namespace"],
        namespace_uid=state["namespace_uid"],
        snapshot=str(dest.relative_to(work)),
        reason=reason,
        missing=[],
        truncated=[],
    )
    unfinished = dict(
        report, missing=[dict(path="pods.json", reason="collection interrupted")]
    )
    runtime.save(dest / "index.json", unfinished)
    if not (directory / "index.json").is_file():
        runtime.save(directory / "index.json", unfinished)
    owner = f"{kube.RUN_LABEL}={state['run_id']}"
    endpoints = {e["id"]: e for e in state["experiment"]["instances"]}

    def save(name, *args):
        try:
            limit = MAX_BYTES + 1 if args[:2] == ("get", "--raw") else None
            data = kubectl(
                *args, budget=min(remaining(deadline), 5), limit=limit
            ).encode()
            if len(data) > MAX_BYTES:
                report["truncated"].append(
                    dict(path=name, reason=f"limited to {MAX_BYTES} bytes")
                )
                data = data[-MAX_BYTES:]
            path = dest / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(data)
        except ERRORS as exc:
            report["missing"].append(dict(path=name, reason=str(exc)[:1000]))

    def remote_file(pod, directory, local, remote):
        save(
            directory + "/" + local,
            "exec",
            pod,
            "-c",
            "engine",
            "--",
            "tail",
            "-c",
            str(MAX_BYTES + 1),
            remote,
        )

    try:
        pods = json.loads(
            kubectl(
                "get", "pods", "-l", owner, "-o", "json", budget=remaining(deadline)
            )
        )["items"]
        inventory = []
        for pod in pods:
            meta, spec, status = pod["metadata"], pod["spec"], pod["status"]
            if meta.get("labels", {}).get(kube.RUN_LABEL) != state["run_id"]:
                continue
            name, uid = meta["name"], meta["uid"]
            if not all(
                re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", v) for v in (name, uid)
            ):
                raise ValueError("invalid Kubernetes Pod identity")
            eid = meta["labels"].get(kube.ID_LABEL, "")
            expected = state.get("expected_images", {}).get(eid, {})
            _, images = kube.compare_images(pod, expected)
            inventory.append(
                dict(
                    name=name,
                    uid=uid,
                    node=spec.get("nodeName"),
                    ip=status.get("podIP"),
                    endpoint=eid,
                    phase=status.get("phase"),
                    images=images,
                    states={
                        s["name"]: s.get("state")
                        for s in status.get("containerStatuses", [])
                    },
                    conditions=[
                        {k: c.get(k) for k in ("type", "status", "reason", "message")}
                        for c in status.get("conditions", [])
                    ],
                )
            )
            ep = endpoints.get(eid)
            base = f"/run/hetero/{uid}"
            if ep:
                remote_file(name, name, "engine.json", base + "/engine.json")
                if ep["backend"] == "rbln" and state["experiment"][
                    "instrumentation"
                ].get("rbln_events"):
                    for rank in range(ep["tp"]):
                        file = f"events_{rank}.jsonl"
                        remote_file(name, name, file, f"{base}/events/{eid}/{file}")
                if ep.get("engine", {}).get("lookahead"):
                    remote_file(
                        name, name, "admission.jsonl", f"{base}/{eid}-admission.jsonl"
                    )
            restarted = {
                s["name"]
                for s in status.get("containerStatuses", [])
                if s.get("restartCount")
            }
            for container in spec["containers"]:
                cname = container["name"]
                if cname not in {"engine", "epp", "pd-sidecar", "envoy-proxy"}:
                    continue
                args = ("logs", name, "-c", cname, f"--limit-bytes={MAX_BYTES + 1}")
                save(f"{name}/{cname}.log", *args)
                if cname in restarted:
                    save(f"{name}/{cname}.previous.log", *args, "--previous")
            for container in spec["containers"]:
                cname = container["name"]
                ports = {
                    p["name"]: p["containerPort"]
                    for p in container.get("ports", [])
                    if "name" in p
                }
                port = {
                    "engine": ports.get("engine"),
                    "epp": ports.get("metrics"),
                    "envoy-proxy": ports.get("metrics-19001"),
                    "pd-sidecar": ep.get("sidecar_metrics_port") if ep else None,
                }.get(cname)
                if port:
                    route = "stats/prometheus" if cname == "envoy-proxy" else "metrics"
                    pods_url = f"/api/v1/namespaces/{state['namespace']}/pods"
                    url = f"{pods_url}/{name}:{port}/proxy/{route}"
                    save(f"{name}/{cname}.metrics", "get", "--raw", url)
        runtime.save(dest / "pods.json", inventory)
        if inventory:
            report["status"] = (
                "partial" if report["missing"] or report["truncated"] else "collected"
            )
        else:
            report["missing"].append(
                dict(path="pods.json", reason="no Pods for recorded run")
            )
    except ERRORS as exc:
        report["missing"].append(dict(path="pods.json", reason=str(exc)[:1000]))
    runtime.save(dest / "index.json", report)
    runtime.save(directory / "index.json", report)
    return report["status"]
