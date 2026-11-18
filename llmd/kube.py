"""Manage the standalone deployment in the active kubectl context."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import engines, runtime
from .verify import verify


def run(work, action, timeout, reference=None):
    deploy = json.loads((work / "deployment.json").read_text())
    ns, run_id = deploy["namespace"], deploy["run_id"]
    owner = "pdd.sota.dev/run=" + run_id
    path = work / "kube-run.json"

    def kubectl(*args):
        p = subprocess.run(
            ["kubectl", "-n", ns, *args],
            capture_output=True,
            text=True,
            timeout=timeout + 15,
        )
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or p.stdout.strip())
        return p.stdout

    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "schema_version": 2,
            "run_id": run_id,
            "experiment": deploy,
            "namespace": ns,
            "endpoint": deploy["endpoint"],
        }
    )
    if state["run_id"] != run_id:
        raise ValueError("deployment and saved run differ")
    kinds = "deployment,service,configmap,serviceaccount,role,rolebinding"

    def remove():
        return kubectl(
            "delete",
            kinds,
            "-l",
            owner,
            "--ignore-not-found",
            "--wait=true",
            f"--timeout={timeout}s",
        )

    if action == "start":
        if path.exists():
            raise ValueError("use a fresh render directory for each cluster run")
        if kubectl("get", kinds, "-l", owner, "-o", "name").strip():
            raise ValueError("objects for this run already exist")
        state["namespace_uid"] = json.loads(kubectl("get", "ns", ns, "-o", "json"))[
            "metadata"
        ]["uid"]
        state["status"] = "starting"
        runtime.save(path, state)
        try:
            kubectl("create", "-f", deploy["manifest"])
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            runtime.save(path, state)
            remove()
            raise
    if (
        state.get("namespace_uid")
        and state["namespace_uid"]
        != json.loads(kubectl("get", "ns", ns, "-o", "json"))["metadata"]["uid"]
    ):
        raise ValueError("deployment belongs to a different cluster namespace UID")
    if action == "stop":
        if not path.exists():
            raise ValueError("no start record exists")
        remove()
        state["status"] = "stopped"
        runtime.save(path, state)
        return state["status"]
    if not path.exists():
        raise ValueError("run start before readiness or verification")
    if state.get("status") == "stopped":
        raise ValueError("deployment has been stopped")
    try:
        kubectl(
            "wait",
            "--for=condition=Available",
            "deployment",
            "-l",
            owner,
            f"--timeout={timeout}s",
        )
        pods = json.loads(kubectl("get", "pods", "-l", owner, "-o", "json"))["items"]
        eps = {e["id"]: dict(e) for e in deploy["instances"]}
        engine_pods = {}
        gateway = None
        inventory = []
        for pod in pods:
            meta = pod["metadata"]
            if meta.get("deletionTimestamp"):
                continue
            statuses = pod["status"].get("containerStatuses", [])
            if not statuses or not all(s.get("ready") for s in statuses):
                continue
            entry = dict(
                name=meta["name"],
                uid=meta["uid"],
                ip=pod["status"]["podIP"],
                node=pod["spec"]["nodeName"],
                images={s["name"]: s["imageID"] for s in statuses},
            )
            inventory.append(entry)
            eid = meta["labels"].get("pdd.sota.dev/id")
            if eid in eps:
                info = json.loads(
                    kubectl(
                        "exec",
                        entry["name"],
                        "-c",
                        "engine",
                        "--",
                        "cat",
                        f"/run/hetero/{entry['uid']}/engine.json",
                    )
                )
                if eid in engine_pods:
                    raise RuntimeError(f"multiple ready Pods for {eid}")
                if entry["name"] != info["endpoint"]["engine_id"]:
                    raise ValueError(f"engine ID differs from manifest for {eid}")
                eps[eid].update(
                    model_identity={
                        k: info["model"].get(k) for k in engines.MODEL_IDENTITY
                    },
                    host=entry["ip"],
                    runtime=info["runtime"],
                    engine_id=info["endpoint"]["engine_id"],
                    node=entry["node"],
                    compile_cache=info["cache"],
                )
                engine_pods[eid] = entry
                if state.get("model") and any(
                    state["model"].get(k) != info["model"].get(k)
                    for k in engines.MODEL_IDENTITY
                ):
                    raise ValueError(
                        f"model/tokenizer content differs across Pods: {eid}"
                    )
                state["model"] = info["model"]
            elif "epp" in entry["images"]:
                gateway = entry
        if set(engine_pods) != set(eps) or not gateway:
            raise RuntimeError("ready Pod inventory incomplete")
        state.update(status="ready", pods=inventory, endpoints=list(eps.values()))
        runtime.save(path, state)
        if action == "verify":
            with ExitStack() as stack:
                deployed = dict(state)

                def forward(pod, port):
                    return stack.enter_context(port_forward(ns, pod, port))

                deployed["endpoint"] = (
                    forward(gateway["name"], 8081) + "/v1/completions"
                )
                local_model = engines.model_info(deploy["model"], {})
                if (
                    engines.tokenizer_identity(local_model["tokenizer_path"])
                    != state["model"]["tokenizer_sha256"]
                ):
                    raise ValueError(
                        "local tokenizer differs from the deployed tokenizer"
                    )
                deployed["model"] = dict(
                    state["model"], tokenizer_path=local_model["tokenizer_path"]
                )
                for eid, e in eps.items():
                    e["metrics"] = (
                        forward(engine_pods[eid]["name"], e["port"]) + "/metrics"
                    )
                    if e["backend"] == "rbln" and deploy["instrumentation"].get(
                        "rbln_events"
                    ):
                        e["events_dir"] = str(work / "events" / eid)

                def collect():
                    (work / "epp.log").write_text(
                        kubectl("logs", gateway["name"], "-c", "epp")
                    )
                    for eid, e in eps.items():
                        if not e.get("events_dir"):
                            continue
                        pod = engine_pods[eid]
                        dest = Path(e["events_dir"])
                        dest.mkdir(parents=True, exist_ok=True)
                        for i in range(e["tp"]):
                            filename = f"events_{i}.jsonl"
                            data = kubectl(
                                "exec",
                                pod["name"],
                                "-c",
                                "engine",
                                "--",
                                "cat",
                                f"/run/hetero/{pod['uid']}/events/{eid}/{filename}",
                            )
                            (dest / filename).write_text(data)

                verify(work, deployed, reference, collect=collect)
        return state["status"]
    except BaseException as exc:
        if action != "verify":
            state.update(status="failed", error=str(exc))
            runtime.save(path, state)
            if action == "start":
                remove()
        raise


@contextmanager
def port_forward(ns, pod, port):
    import selectors

    proc = subprocess.Popen(
        [
            "kubectl",
            "-n",
            ns,
            "port-forward",
            "--address=127.0.0.1",
            "pod/" + pod,
            f":{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            if not selector.select(20):
                raise RuntimeError("port-forward readiness timeout")
            line = proc.stdout.readline()
        m = re.search(r"Forwarding from 127\.0\.0\.1:(\d+)", line)
        if not m:
            raise RuntimeError(f"port-forward {pod}: {line}")
        yield f"http://127.0.0.1:{m[1]}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        proc.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "ready", "verify", "stop"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    try:
        with runtime.run_lock(args.work_dir):
            status = run(args.work_dir, args.action, args.timeout, args.reference)
        print(
            json.dumps({"status": status, "run": str(args.work_dir / "kube-run.json")})
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        sys.exit(1)
