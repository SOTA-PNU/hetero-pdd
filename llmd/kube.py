"""Manage a standalone deployment in the explicitly selected kubectl context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from . import engines, runtime
from .verify import verify

RUN_LABEL = "pdd.sota.dev/run"
ID_LABEL = "pdd.sota.dev/id"
KINDS = "deployment,service,configmap,serviceaccount,role,rolebinding"
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
PROOF_BYTES = 16 * 1024 * 1024


def limited_command(argv, timeout, limit):
    """Stop reading kubectl stdout at the artifact limit."""
    deadline = time.monotonic() + timeout
    data = bytearray()
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=errors) as proc,
    ):
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while len(data) < limit:
                    if not selector.select(max(0, deadline - time.monotonic())):
                        raise TimeoutError("kubectl output timed out")
                    block = os.read(proc.stdout.fileno(), min(65536, limit - len(data)))
                    if not block:
                        if proc.wait(timeout=max(0, deadline - time.monotonic())):
                            errors.seek(0)
                            raise RuntimeError(
                                errors.read(2000).decode(errors="replace")
                                or "kubectl output request failed"
                            )
                        break
                    data.extend(block)
        finally:
            if proc.poll() is None:
                proc.kill()
    return data.decode(errors="replace")


def command(context, namespace, args, timeout, *, limit=None):
    if not context or not namespace:
        raise ValueError("an explicit Kubernetes context and namespace are required")
    argv = ["kubectl", "--context", context, "-n", namespace, *args]
    if limit is not None:
        return limited_command(argv, timeout, limit)
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:2000] or f"kubectl {args[0]} failed")
    return result.stdout


def read_manifest(work, deploy):
    # Use the render beside deployment.json, including after an artifact relocation.
    docs = list(yaml.safe_load_all((work / "kubernetes.yaml").read_text()))
    if not docs or any(
        d["kind"].lower() not in KINDS.split(",")
        or d["metadata"].get("namespace") != deploy["namespace"]
        or d["metadata"].get("labels", {}).get(RUN_LABEL) != deploy["run_id"]
        for d in docs
    ):
        raise ValueError("rendered manifest contains resources outside this run")
    return docs


def expected_images(docs, strict):
    by_endpoint = {}
    for doc in docs:
        if doc["kind"] != "Deployment":
            continue
        tmpl = doc["spec"]["template"]
        key = tmpl["metadata"]["labels"].get(ID_LABEL, "")
        images = {c["name"]: c["image"] for c in tmpl["spec"]["containers"]}
        if key in by_endpoint:
            raise ValueError(f"duplicate rendered Pod group: {key}")
        if strict and any(
            "@" not in image or not DIGEST.fullmatch(image.rsplit("@", 1)[1])
            for image in images.values()
        ):
            raise ValueError(f"immutable per-platform image digests required: {images}")
        by_endpoint[key] = images
    return by_endpoint


def compare_images(pod, expected):
    declared = {c["name"]: c["image"] for c in pod["spec"]["containers"]}
    actual = {
        c["name"]: c.get("imageID", "")
        for c in pod["status"].get("containerStatuses", [])
    }
    matches = declared == expected and actual.keys() == expected.keys()
    records = {}
    for name in expected.keys() | actual.keys() | declared.keys():
        image = expected.get(name, "")
        reported = actual.get(name, "")
        image_id = (
            reported.removeprefix("docker-pullable://")
            .removeprefix("containerd://")
            .removeprefix("docker://")
        )
        digest = image_id.rsplit("@", 1)[-1]
        matched = bool(
            DIGEST.fullmatch(digest)
            and image.endswith("@" + digest)
            and declared.get(name) == image
        )
        matches &= matched
        records[name] = dict(
            expected=image,
            declared=declared.get(name),
            image_id=reported,
            matched=matched,
        )
    return matches, records


def run(
    work,
    action,
    timeout,
    reference=None,
    *,
    context=None,
    namespace_uid=None,
    require_image_digests=False,
    diagnostic_timeout=30,
    cleanup_timeout=60,
):
    deploy = json.loads((work / "deployment.json").read_text())
    ns, run_id = deploy["namespace"], deploy["run_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("invalid rendered run ID")
    owner = f"{RUN_LABEL}={run_id}"
    path = work / "kube-run.json"

    def kubectl(*args, budget=None, limit=None):
        return command(
            context, ns, args, timeout + 5 if budget is None else budget, limit=limit
        )

    if action == "start":
        if path.exists():
            raise ValueError("use a fresh render directory for each cluster run")
        if not context:
            raise ValueError("start requires an explicit --context")
        docs = read_manifest(work, deploy)
        state = dict(
            schema_version=3,
            run_id=run_id,
            experiment=deploy,
            namespace=ns,
            endpoint=deploy["endpoint"],
            context=context,
            status="starting",
            require_image_digests=require_image_digests,
            expected_images=expected_images(docs, require_image_digests),
            owned_resources=[
                d["kind"].lower() + "/" + d["metadata"]["name"] for d in docs
            ],
        )
    else:
        if not path.exists():
            raise ValueError("no start record exists")
        state = json.loads(path.read_text())
        if state["run_id"] != run_id or state["namespace"] != ns:
            raise ValueError("deployment and saved run differ")
        if context and state.get("context", context) != context:
            raise ValueError("supplied Kubernetes context differs from recorded run")
        context = context or state.get("context")
        if not state.get("namespace_uid") or not context:
            raise ValueError("recorded namespace UID and explicit context are required")
        if not state.get("context"):  # Legacy record. UID must match before access.
            state["context"] = context

    def check_namespace(budget=None):
        current = json.loads(kubectl("get", "ns", ns, "-o", "json", budget=budget))[
            "metadata"
        ]["uid"]
        if not current or any(
            uid and uid != current
            for uid in (namespace_uid, state.get("namespace_uid"))
        ):
            raise ValueError("deployment belongs to a different cluster namespace UID")
        return current

    def remove():
        budget = timeout if action == "stop" else cleanup_timeout
        deadline = time.monotonic() + budget
        check_namespace(budget)
        # Limit deletion by namespace, kind, exact rendered name and run label.
        resources = state.get("owned_resources") or [
            d["kind"].lower() + "/" + d["metadata"]["name"]
            for d in read_manifest(work, deploy)
        ]
        if any(
            r.partition("/")[0] not in KINDS.split(",")
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", r.partition("/")[2])
            for r in resources
        ):
            raise ValueError("invalid recorded resource for cleanup")
        remaining = list(resources)
        state["cleanup"] = dict(status="pending", remaining=remaining)
        runtime.save(path, state)
        errors = []
        for resource in resources:
            kind, name = resource.split("/", 1)
            budget = deadline - time.monotonic()
            if budget <= 0:
                errors.append("cleanup deadline exceeded")
                break
            try:
                kubectl(
                    "delete",
                    kind,
                    "-l",
                    owner,
                    "--field-selector",
                    f"metadata.name={name}",
                    "--ignore-not-found",
                    "--cascade=foreground",
                    "--wait=true",
                    f"--timeout={max(1, int(budget))}s",
                    budget=budget,
                )
                remaining.remove(resource)
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                errors.append(f"{resource}: {exc}")
        state["cleanup"].update(
            status="failed" if errors else "complete", errors=errors
        )
        runtime.save(path, state)
        if errors:
            raise RuntimeError("; ".join(errors))

    def diagnostics(reason):
        from tools.llmd_ci.kube_diagnostics import collect

        deadline = time.monotonic() + diagnostic_timeout
        check_namespace(diagnostic_timeout)
        return collect(work, state, kubectl, deadline, reason)

    if action == "start":
        state["namespace_uid"] = check_namespace()
        if kubectl("get", KINDS, "-l", owner, "-o", "name").strip():
            raise ValueError("objects for this run already exist")
        runtime.save(path, state)
    elif action == "collect":
        if (
            state.get("status") == "stopped"
            or state.get("cleanup", {}).get("status") == "complete"
        ):
            index = work / "diagnostics/index.json"
            if not index.is_file():
                raise ValueError(
                    "run already cleaned without a saved diagnostic snapshot"
                )
            status = json.loads(index.read_text())["status"]
        else:
            status = diagnostics("collect")
        if status == "failed":
            raise RuntimeError("collection failed; see diagnostics/index.json")
        return status
    elif action == "stop":
        if state.get("status") != "stopped":
            if state.get("cleanup", {}).get("status") != "complete":
                if not (work / "diagnostics/index.json").exists():
                    try:
                        diagnostics("stop")
                    except Exception as diag_error:
                        state["collection_error"] = str(diag_error)
                        runtime.save(path, state)
                remove()
            state["status"] = "stopped"
            runtime.save(path, state)
        return state["status"]
    else:
        check_namespace()
        if (
            state.get("status") == "stopped"
            or state.get("cleanup", {}).get("status") == "complete"
        ):
            raise ValueError("deployment has been stopped")

    try:
        if action == "start":
            kubectl("create", "-f", str(work / "kubernetes.yaml"))
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
            if meta["labels"].get(RUN_LABEL) != run_id:
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
            matches, entry["image_comparison"] = compare_images(
                pod, state.get("expected_images", {}).get(eid or "", {})
            )
            if state.get("require_image_digests") and not matches:
                state["pods"] = inventory
                runtime.save(path, state)
                raise ValueError(
                    f"{entry['name']}: Pod image or imageID differs from pinned render"
                )
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
                    return stack.enter_context(port_forward(context, ns, pod, port))

                deployed["endpoint"] = (
                    forward(gateway["name"], 8081) + "/v1/completions"
                )
                local_model = engines.model_info(deploy["model"], {})
                snapshot = Path(local_model["tokenizer_path"])
                identities = dict(
                    tokenizer=local_model["tokenizer"],
                    tokenizer_revision=local_model["tokenizer_revision"],
                    tokenizer_sha256=engines.tokenizer_identity(snapshot),
                    tokenizer_config_sha256=hashlib.sha256(
                        (snapshot / "tokenizer_config.json").read_bytes()
                    ).hexdigest(),
                )
                for key, value in identities.items():
                    if value != state["model"].get(key):
                        raise ValueError(f"local tokenizer differs from Pod: {key}")
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
                    deadline = time.monotonic() + diagnostic_timeout
                    limit = PROOF_BYTES + 1

                    def proof(destination, *args):
                        budget = deadline - time.monotonic()
                        if budget <= 0:
                            raise TimeoutError(
                                "verification evidence collection timed out"
                            )
                        data = kubectl(*args, budget=budget, limit=limit).encode()
                        destination.write_bytes(data[:PROOF_BYTES])
                        if len(data) > PROOF_BYTES:
                            raise ValueError(
                                f"verification evidence truncated: {destination.name} "
                                f"exceeds {PROOF_BYTES} bytes"
                            )

                    proof(
                        work / "epp.log",
                        "logs",
                        gateway["name"],
                        "-c",
                        "epp",
                        f"--limit-bytes={limit}",
                    )
                    for eid, e in eps.items():
                        if not e.get("events_dir"):
                            continue
                        pod = engine_pods[eid]
                        dest = Path(e["events_dir"])
                        dest.mkdir(parents=True, exist_ok=True)
                        for i in range(e["tp"]):
                            filename = f"events_{i}.jsonl"
                            proof(
                                dest / filename,
                                "exec",
                                pod["name"],
                                "-c",
                                "engine",
                                "--",
                                "tail",
                                "-c",
                                str(limit),
                                f"/run/hetero/{pod['uid']}/events/{eid}/{filename}",
                            )

                verify(work, deployed, reference, collect=collect)
        return state["status"]
    except BaseException as exc:
        state.update(
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        runtime.save(path, state)
        try:
            diagnostics(action)
        except Exception as diag_error:
            state["collection_error"] = str(diag_error)
            runtime.save(path, state)
        if action == "start":
            try:
                remove()
            except Exception as cleanup_error:
                state["cleanup_error"] = str(cleanup_error)
                runtime.save(path, state)
        raise


@contextmanager
def port_forward(context, ns, pod, port):
    proc = subprocess.Popen(
        [
            "kubectl",
            "--context",
            context,
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
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "ready", "verify", "collect", "stop")
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--context", help="explicit context (required for start)")
    parser.add_argument("--namespace-uid", help="expected UID from namespace preflight")
    parser.add_argument("--require-image-digests", action="store_true")
    parser.add_argument("--diagnostic-timeout", type=int, default=30)
    parser.add_argument("--cleanup-timeout", type=int, default=60)
    args = parser.parse_args()
    if min(args.timeout, args.diagnostic_timeout, args.cleanup_timeout) <= 0:
        parser.error("timeouts must be positive")

    def cancelled(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        with runtime.run_lock(args.work_dir):
            status = run(
                args.work_dir,
                args.action,
                args.timeout,
                args.reference,
                context=args.context,
                namespace_uid=args.namespace_uid,
                require_image_digests=args.require_image_digests,
                diagnostic_timeout=args.diagnostic_timeout,
                cleanup_timeout=args.cleanup_timeout,
            )
        print(
            json.dumps({"status": status, "run": str(args.work_dir / "kube-run.json")})
        )
    except KeyboardInterrupt as exc:
        print(json.dumps({"status": "cancelled", "error": str(exc)}), file=sys.stderr)
        sys.exit(1)
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        sys.exit(1)
