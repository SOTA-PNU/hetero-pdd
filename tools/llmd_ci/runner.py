"""Execute the two native llm-d lifecycles and journal ownership before allocation."""

import contextlib
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import urlopen

import yaml

from llmd import config, runtime

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("reference", "target")
PYTHON = [sys.executable, "-m"]
HELPER = PYTHON + ["tools.llmd_ci.runner"]

# Terminate an in-flight CLI if its CI supervisor disappears. The inherited directory
# lock outlives the supervisor until the CLI exits, preventing a concurrent replay.
EXEC_OWNED = """
import ctypes, os, signal, sys
if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
if os.getppid() != int(sys.argv[1]):
    sys.exit(1)
os.execvp(sys.argv[2], sys.argv[2:])
"""


class Blocked(RuntimeError):
    """Requested environment or pinned source inputs are unavailable."""


class Cancelled(RuntimeError):
    pass


def read(path):
    return json.loads(path.read_text())


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def prepare_configs(root, target, out):
    source = root / target["path"]
    raw = config.read_yaml(source)
    cfg = config.load(source)
    raw["model"] = cfg[
        "model"
    ]  # includes resolved local paths for inline and file models
    raw["profile"] = cfg["profile"]
    dest = out / "configs"
    dest.mkdir()
    write_yaml(dest / "target.yaml", raw)
    rule = target["reference"]
    candidates = [
        ep
        for ep in cfg["endpoints"]
        if ep["backend"] == "cuda"
        and (not rule.get("endpoint") or ep["id"] == rule["endpoint"])
    ]
    if len(candidates) != 1:
        raise Blocked(
            "reference requires one unambiguous CUDA endpoint; select it in the suite"
        )
    chosen = candidates[0]
    for key in ("dtype", "kv_dtype"):
        if {ep["engine"][key] for ep in cfg["endpoints"]} != {chosen["engine"][key]}:
            raise Blocked(
                f"one GPU full reference cannot reproduce target {key} conditions"
            )
    endpoint = dict(
        next(ep for ep in raw["endpoints"] if ep["id"] == chosen["id"]), roles=["full"]
    )
    for key in ("sidecar_port", "sidecar_metrics_port", "kv_port"):
        endpoint.pop(key, None)
    reference = dict(
        raw,
        endpoints=[endpoint],
        profile=str(root / rule["profile"]),
        verify=dict(raw["verify"], paths=["full"]),
    )
    write_yaml(dest / "reference.yaml", reference)
    config.load(dest / "reference.yaml")


@contextlib.contextmanager
def locked(out):
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another CI lifecycle command is active in {out}"
            ) from exc
        yield fd
    finally:
        os.close(fd)


@contextlib.contextmanager
def signals():
    received = []

    def remember(number, _frame):
        received.append(signal.Signals(number).name)

    old = {sig: signal.signal(sig, remember) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield received
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def check_time(end, received):
    if received:
        raise Cancelled(received[0])
    seconds = end - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("deadline exceeded")
    return max(1, math.ceil(seconds))


def tail_error(path):
    with path.open("rb") as log:
        log.seek(max(0, log.seek(0, os.SEEK_END) - 2048))
        lines = log.read().decode(errors="replace").rstrip().splitlines()
    if not lines:
        return "no diagnostic output"
    try:
        return str(json.loads(lines[-1])["error"])
    except (ValueError, KeyError, TypeError):
        return lines[-1][:1024]


class Lifecycle:
    def __init__(self, out, state, lock, received):
        self.out = out
        self.state = state
        self.lock = lock
        self.received = received
        self.root = ROOT

    def save(self):
        runtime.save(self.out / "lifecycle.json", self.state)

    def command(self, phase, action, argv, end, *, finalizing=False):
        received = [] if finalizing else self.received
        journal = self.state["phases"][phase]["commands"]
        rel = f"logs/{phase}-{len(journal):02}-{action}.log"
        entry = dict(action=action, argv=list(map(str, argv)), log=rel, returncode=None)
        journal.append(entry)
        self.save()
        try:
            check_time(end, received)
            with (self.out / rel).open("xb") as log:
                launch = [
                    sys.executable,
                    "-c",
                    EXEC_OWNED,
                    str(os.getpid()),
                    *entry["argv"],
                ]
                proc = subprocess.Popen(
                    launch,
                    cwd=self.root,
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    pass_fds=(self.lock,),
                )
                try:
                    while proc.poll() is None:
                        check_time(end, received)
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            proc.wait(
                                timeout=min(0.2, max(0.001, end - time.monotonic()))
                            )
                    check_time(end, received)
                finally:
                    if proc.poll() is None:
                        # Native servers use their own ownership token and sessions.
                        # Interrupt kubectl/port-forward and the attached native CLI.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    entry["returncode"] = proc.returncode
            if entry["returncode"]:
                reason = tail_error(self.out / rel)
                raise RuntimeError(f"{action} exited {entry['returncode']}: {reason}")
        except Exception as exc:
            entry["error"] = str(exc)
            raise
        finally:
            self.save()

    def work(self, phase):
        stage = self.state["phases"][phase]
        directory = self.out / phase
        info = directory.lstat()
        if (
            directory.is_symlink()
            or [info.st_dev, info.st_ino] != stage["directory_identity"]
        ):
            raise ValueError(f"{phase} work directory does not match this run")
        return directory

    def owner(self, phase):
        stage = self.state["phases"][phase]
        work = self.work(phase)
        kube = self.state["target"] == "kubernetes"
        path = work / ("kube-run.json" if kube else "run.json")
        if not path.exists():
            if stage.get("native_record_seen"):
                raise ValueError(f"missing recorded ownership file {path.name}")
            return None
        if path.is_symlink():
            raise ValueError(f"unexpected ownership file symlink {path}")
        native = read(path)
        run_id = native.get("run_id", "")
        if (
            not re.fullmatch("[a-f0-9]{32}", run_id)
            or stage.get("run_id", run_id) != run_id
        ):
            raise ValueError(f"{phase} native run ID changed")
        if kube:
            if (
                native.get("namespace_uid") != stage["namespace_uid"]
                or native.get("namespace") != self.state["namespace"]
                or native.get("context") != self.state["context"]
            ):
                raise ValueError(f"{phase} cluster ownership changed")
        elif native.get("work_dir") != str(work):
            raise ValueError(f"{phase} native work directory changed")
        stage.update(run_id=run_id, native_record_seen=True)
        self.save()
        return native

    def native(self, phase, action, end, *, finalizing=False):
        stage = self.state["phases"][phase]
        work = self.work(phase)
        kube = self.state["target"] == "kubernetes"
        argv = PYTHON + ["llmd.kube" if kube else "llmd", action]
        remaining = check_time(end, [] if finalizing else self.received)
        if kube:
            argv.extend(
                [
                    "--context",
                    self.state["context"],
                    "--timeout",
                    str(remaining),
                    "--diagnostic-timeout",
                    str(self.state["settings"]["collection_timeout"]),
                    "--cleanup-timeout",
                    str(self.state["settings"]["cleanup_timeout"]),
                ]
            )
            if action == "start":
                argv.extend(
                    [
                        "--require-image-digests",
                        "--namespace-uid",
                        stage["namespace_uid"],
                    ]
                )
        elif action == "start":
            argv.extend(
                [self.out / "configs" / f"{phase}.yaml", "--timeout", str(remaining)]
            )
        argv.extend(["--work-dir", work])
        if phase == "target" and action == "verify":
            argv.extend(["--reference", self.out / "reference/verification.json"])
        self.command(phase, action, argv, end, finalizing=finalizing)

    def prepare(self, plan, target, end):
        from .plan import verify_checkout

        self.root = verify_checkout(plan, root=self.root)
        prepare_configs(self.root, target, self.out)
        stage = self.state
        stage["source"] = plan["source"]
        stage["target"] = target["target"]
        stage["environment"] = target["environment"]
        settings = stage["settings"] = target["settings"]
        self.save()
        prep = self.out / "preflight.json"
        if stage["target"] == "local":
            self.command(
                "reference",
                "preflight",
                HELPER + ["local", self.out / "configs/target.yaml", prep],
                end,
            )
            stage["binaries"] = read(prep)["binaries"]
            return
        for key in ("context", "namespace"):
            name = settings[f"{key}_env"]
            value = os.environ.get(name)
            if not value:
                raise Blocked(f"missing {name} for approved Kubernetes {key}")
            stage[key] = value
        self.save()
        record_var = settings["image_record_env"]
        record_path = os.environ.get(record_var)
        if not record_path:
            raise Blocked(f"missing {record_var} for the image build record")
        site_path = self.root / target["site"]
        self.command(
            "reference",
            "preflight-images",
            HELPER
            + [
                "images",
                self.out / "configs/target.yaml",
                site_path,
                Path(record_path).resolve(),
                prep,
            ],
            end,
        )
        images = stage["images"] = read(prep)["images"]
        site = dict(config.read_yaml(site_path), images=images)
        approved_site = self.out / "configs/site.yaml"
        write_yaml(approved_site, site)
        for phase in PHASES:
            work = self.work(phase)
            row = stage["phases"][phase]
            self.command(
                phase,
                "render",
                PYTHON
                + [
                    "llmd",
                    "render",
                    self.out / "configs" / f"{phase}.yaml",
                    "--target",
                    "kubernetes",
                    "--namespace",
                    stage["namespace"],
                    "--site",
                    approved_site,
                    "--work-dir",
                    work,
                ],
                end,
            )
            row["run_id"] = read(work / "deployment.json")["run_id"]
            self.save()
            self.command(
                phase,
                "preflight-cluster",
                HELPER
                + [
                    "cluster",
                    work,
                    stage["context"],
                    str(check_time(end, self.received)),
                ],
                end,
            )
            uid = read(work / "preflight.json")["namespace_uid"]
            if (
                phase == "target"
                and uid != stage["phases"]["reference"]["namespace_uid"]
            ):
                raise Blocked("namespace UID changed between preflights")
            row["namespace_uid"] = uid
            self.save()

    def finish(self, phase, action):
        stage = self.state["phases"][phase]
        key = "collection" if action == "collect" else "cleanup"
        if not stage["start_attempted"]:
            return
        if stage["cleanup"]["status"] == "cleaned" and (
            action == "cleanup" or stage["collection"]["status"] != "failed"
        ):
            return
        detail = {}
        try:
            native = self.owner(phase)
            work = self.work(phase)
            kube = self.state["target"] == "kubernetes"
            if native is None:
                # Both native launchers publish ownership before creating any resource.
                detail = (
                    dict(
                        status="partial",
                        reason="start ended before native ownership was recorded",
                    )
                    if action == "collect"
                    else dict(status="cleaned", remaining=[])
                )
            else:
                end = time.monotonic() + self.state["settings"][key + "_timeout"]
                if action == "collect":
                    if kube:
                        self.native(phase, action, end, finalizing=True)
                        report = read(work / "diagnostics/index.json")
                    else:
                        self.command(
                            phase,
                            action,
                            HELPER + ["collect-local", work],
                            end,
                            finalizing=True,
                        )
                        report = read(work / "collection.json")
                    detail = dict(
                        status=report["status"],
                        missing=report.get("missing", []),
                        truncated=report.get("truncated", []),
                    )
                else:
                    self.native(phase, "stop", end, finalizing=True)
                    detail = dict(status="cleaned", remaining=[])
        except Exception as exc:
            detail = dict(status="failed", reason=str(exc))
            if action == "cleanup":
                # The native state lists the process/POD inventory and namespace UID.
                remaining = None
                try:
                    native = self.owner(phase)
                    if native:
                        if self.state["target"] == "kubernetes":
                            remaining = native.get("cleanup", {}).get(
                                "remaining", native.get("owned_resources")
                            )
                        else:
                            remaining = runtime.owned_processes(native["run_id"])
                except (OSError, ValueError, RuntimeError, KeyError):
                    pass
                detail["remaining"] = (
                    remaining
                    if remaining is not None
                    else [
                        dict(
                            run_id=stage.get("run_id"),
                            phase=phase,
                            reason="ownership or resource inventory unavailable",
                        )
                    ]
                )
        stage[key] = detail
        self.save()

    def phase(self, phase, end):
        stage = self.state["phases"][phase]
        stage["status"] = "running"
        try:
            check_time(end, self.received)
            stage["start_attempted"] = True
            self.save()
            self.native(phase, "start", end)
            native = self.owner(phase)
            if not native:
                raise RuntimeError("native start returned without ownership state")
            if self.state["target"] == "local":
                measured = native.get("binary_sha256", {})
                expected = self.state["binaries"]
                names = {Path(spec["argv"][0]).name for spec in native["control_plane"]}
                if not {"envoy", "epp"} <= names or any(
                    not expected.get(name) or expected[name] != measured.get(name)
                    for name in names
                ):
                    raise RuntimeError(
                        "running binary hashes differ from the verified build"
                    )
            self.native(phase, "ready", end)
            self.native(phase, "verify", end)
            report_path = self.work(phase) / "verification.json"
            report = read(report_path)
            if report.get("run_id") != native["run_id"]:
                raise ValueError("verifier output does not belong to this phase")
            if phase == "reference":
                from llmd.verify import reference_output

                reference_output(
                    report_path,
                    report["model"],
                    None,
                    [row["prompt_ids"] for row in report["requests"]],
                    report["conditions"],
                )
            stage["status"] = "completed"
        except Exception as exc:
            stage.update(
                status="cancelled" if isinstance(exc, Cancelled) else "failed",
                error=str(exc),
            )
            raise
        finally:
            try:
                self.finish(phase, "collect")
            finally:
                self.finish(phase, "cleanup")
            self.save()
        for key in ("collection", "cleanup"):
            if stage[key]["status"] == "failed":
                stage.update(
                    status="failed", error=f"{phase} {key} failed: {stage[key]}"
                )
                self.save()
                raise RuntimeError(stage["error"])
        check_time(end, self.received)


def cluster_blocker(out, state):
    """Classify a startup failure only with evidence from this phase's Pods."""
    if state.get("target") != "kubernetes":
        return None
    for phase, row in state["phases"].items():
        if row["status"] != "failed" or any(
            c["action"] == "verify" for c in row["commands"]
        ):
            continue
        for path in sorted((out / phase / "diagnostics").glob("*/pods.json")):
            try:
                pods = read(path)
                for pod in pods:
                    for value in pod["states"].values():
                        waiting = value.get("waiting", {})
                        if waiting.get("reason") in {
                            "ErrImagePull",
                            "ImagePullBackOff",
                            "InvalidImageName",
                        }:
                            message = waiting.get("message", "")
                            return f"{pod['name']}: {waiting['reason']} {message}"
                    for condition in pod["conditions"]:
                        if (
                            condition.get("reason") == "Unschedulable"
                            and condition.get("status") == "False"
                        ):
                            message = condition.get("message", "Unschedulable")
                            return f"{pod['name']}: {message}"
                    # Collection bounds each log and validates Pod names.
                    log = path.parent / pod["name"] / "engine.log"
                    if log.is_file():
                        for line in log.read_text(errors="replace").splitlines():
                            if any(
                                text in line
                                for text in (
                                    "missing pinned snapshot ",
                                    "missing pinned model file ",
                                    "missing tokenizer configuration ",
                                    "missing tokenizer vocabulary ",
                                )
                            ):
                                return line[:1000]
            except (KeyError, OSError, TypeError, ValueError):
                continue
    return None


def run(plan_path, experiment_id, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with locked(out) as lock, signals() as received:
        if any(out.iterdir()):
            raise ValueError(f"run requires a fresh output directory: {out}")
        state = dict(
            schema_version=1,
            run_id=uuid.uuid4().hex,
            experiment=experiment_id,
            status="running",
            source={},
            phases={},
        )
        (out / "logs").mkdir()
        for phase in PHASES:
            work = out / phase
            work.mkdir(mode=0o700)
            info = work.stat()
            state["phases"][phase] = dict(
                status="pending",
                start_attempted=False,
                work_dir=phase,
                directory_identity=[info.st_dev, info.st_ino],
                commands=[],
                collection=dict(status="not_run"),
                cleanup=dict(status="not_run"),
            )
        lifecycle = Lifecycle(out, state, lock, received)
        lifecycle.save()
        try:
            plan = read(Path(plan_path))
            runtime.save(out / "plan.json", plan)
            state["source"] = plan.get("source", {})
            target = next(
                (t for t in plan["targets"] if t["id"] == experiment_id), None
            )
            if target is None:
                blocked = next(
                    (
                        row
                        for row in plan.get("blocked", [])
                        if row.get("id") == experiment_id
                    ),
                    {},
                )
                reason = blocked.get("reason", "not selected")
                state["environment"] = blocked.get("environment")
                raise Blocked(f"{experiment_id} is not runnable in this plan: {reason}")
            state.update(
                environment=target["environment"],
                target=target["target"],
                settings=target["settings"],
            )
            end = time.monotonic() + target["settings"]["timeout"]
            try:
                lifecycle.prepare(plan, target, end)
            except (Cancelled, TimeoutError):
                raise
            except Exception as exc:
                raise Blocked(str(exc)) from exc
            for phase in PHASES:
                lifecycle.phase(phase, end)
            state["status"] = "completed"
        except Cancelled as exc:
            state.update(status="cancelled", error=str(exc))
        except Exception as exc:
            availability = cluster_blocker(out, state)
            status = "blocked" if isinstance(exc, Blocked) or availability else "failed"
            state.update(status=status, error=str(exc))
            if availability:
                state["error"] += f"; unavailable cluster input: {availability}"
                for row in state["phases"].values():
                    if row["status"] == "failed":
                        row.update(status="blocked", blocked_reason=availability)
        finally:
            if received:
                state["signal"] = received[0]
                state.update(status="cancelled", error=state.get("error", received[0]))
            lifecycle.save()
        return state


def replay(out, action):
    out = Path(out).resolve(strict=True)
    with locked(out) as lock:
        state = read(out / "lifecycle.json")
        lifecycle = Lifecycle(out, state, lock, [])
        if state["status"] == "running":
            state.update(
                status="failed", error="runner exited without completing the lifecycle"
            )
            lifecycle.save()
        for phase in PHASES:
            if action == "cleanup" and state["phases"][phase]["collection"][
                "status"
            ] in {"not_run", "failed"}:
                lifecycle.finish(phase, "collect")
            lifecycle.finish(phase, action)
        return state


def collect(out):
    return replay(out, "collect")


def cleanup(out):
    return replay(out, "cleanup")


def local_diagnostics(work):
    state = read(work / "run.json")
    report = dict(status="collected", missing=[], truncated=[])
    # Native local logs, events and admission files already live in this directory.
    # Confirm listener ownership before sampling ports which might have been recycled.
    try:
        if state["status"] != "ready":
            raise ValueError(f"native run is {state['status']}")
        runtime.confirm_listeners(state)
        targets = {"epp": state["metrics"]}
        for ep in state["endpoints"]:
            targets["engine-" + ep["id"]] = ep["metrics"]
            if ep.get("sidecar_metrics"):
                targets["sidecar-" + ep["id"]] = ep["sidecar_metrics"]
        ports = state["experiment"]["ports"]
        targets["envoy"] = f"http://127.0.0.1:{ports['admin']}/stats/prometheus"
        directory = work / "metrics"
        directory.mkdir(exist_ok=True)
        limit = 4 * 1024 * 1024
        for key, url in targets.items():
            rel = f"metrics/{key}.prom"
            try:
                with urlopen(url, timeout=3) as response:
                    payload = response.read(limit + 1)
                (work / rel).write_bytes(payload[:limit])
                if len(payload) > limit:
                    report["truncated"].append(
                        dict(path=rel, reason=f"exceeded {limit} bytes")
                    )
            except OSError as exc:
                report["missing"].append(dict(path=rel, reason=str(exc)))
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        report["missing"].append(dict(path="metrics", reason=str(exc)))
    if report["missing"] or report["truncated"]:
        report["status"] = "partial"
    runtime.save(work / "collection.json", report)


def prepare_inputs(mode, config_path, *args):
    import hashlib

    from llmd import engines, serving
    from llmd.cluster import IMAGES
    from llmd.verify import PromptTokenizer

    from .checks import verify_binaries, verify_image_record

    cfg = config.load(Path(config_path))
    model = engines.model_info(cfg["model"], cfg if mode == "local" else {})
    report = dict(tokenizer_sha256=engines.tokenizer_identity(model["tokenizer_path"]))
    PromptTokenizer(model["tokenizer_path"])
    if mode == "local":
        report["binaries"] = verify_binaries(ROOT, serving.binary_dir())
        report["hardware"] = engines.preflight(model, cfg["endpoints"])
    else:
        site_path, record_path, _ = map(Path, args)
        images = (
            IMAGES
            | cfg.get("images", {})
            | config.read_yaml(site_path).get("images", {})
        )
        required = {"envoy", "epp"}
        for ep in cfg["endpoints"]:
            required.add(ep["backend"] if ep["backend"] in images else "engine")
            if "decode" in ep["roles"]:
                required.add("sidecar")
        report["images"] = verify_image_record(
            ROOT, record_path, images, required=required
        )
        report["image_record_sha256"] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
    runtime.save(Path(args[-1]), report)


if __name__ == "__main__":
    # Isolate preflight I/O and raw-metrics sampling under the supervisor's deadline.
    mode, *inputs = sys.argv[1:]
    try:
        if mode == "cluster":
            from .kube_diagnostics import preflight

            work, context, timeout = inputs
            runtime.save(
                Path(work) / "preflight.json",
                preflight(Path(work), context, int(timeout)),
            )
        elif mode == "collect-local":
            local_diagnostics(Path(inputs[0]))
        elif mode in ("local", "images"):
            prepare_inputs(mode, *inputs)
        else:
            raise ValueError(f"unknown internal action {mode}")
    except Exception as exc:
        print(json.dumps(dict(error=str(exc))), file=sys.stderr)
        sys.exit(1)
