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
# SPDX-License-Identifier: Apache-2.0
"""Small, local lifecycle driver shared by the PDD shell entry points."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXPECTED_BRANCH = "0.11.1-2_native_dtype"
REQUIRED_COMMIT = "972ce20cf53b0d2d50c4155f9d44be6879ede966"
STEPS = (
    "prepare_run", "start_prefill", "start_decode", "start_proxy",
    "check_correctness", "warmup", "run_workload", "stop_pdd", "summarize_results",
)
ROLES = ("PREFILL", "DECODE", "PROXY")
TIMEOUTS = {"PREFILL": 900, "DECODE": 2400, "PROXY": 60}
WORKLOAD_DEFAULTS = {
    "WARMUP_REQUESTS": "3", "MEASURE_REQUESTS": "20", "CONCURRENCY": "1",
    "MAX_TOKENS": "128", "PROMPT_MODE": "long",
}
# These are the inherited runtime/search settings needed across independent shells.
# Do not serialize the entire environment (in particular CI credentials/tokens).
SAVED_ENV = (
    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME", "TMPDIR", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
)


class RunError(Exception):
    pass


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_text(path: Path | str, text: str) -> None:
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_env(path, values):
    assignments = (
        f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )
    atomic_text(path, "".join(assignments))


def read_env(path):
    """Parse assignments as data, never execute a saved shell fragment."""
    values = {}
    try:
        for assignment in shlex.split(Path(path).read_text(), comments=False):
            name, separator, value = assignment.partition("=")
            if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError("invalid assignment")
            if name in values:
                raise ValueError(f"duplicate key {name}")
            values[name] = value
    except (OSError, ValueError) as exc:
        raise RunError(f"cannot read {path}: {exc}") from exc
    return values


def pid_record_digest(values, role):
    """Bind one complete process identity to this run and its settings."""
    keys = ("RESULT_DIR", "RUN_ID", "RUNTIME_SHA256") + tuple(
        f"{role}_{field}" for field in ("PID", "PGID", "SID", "STARTTIME")
    )
    payload = json.dumps([role, *[values[key] for key in keys]], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_pids(result_dir, runtime, stages):
    """Recover independently verified roles without evaluating shell state.

    The canonical header may contain quoted newlines in RESULT_DIR. Remaining
    records contain only numeric identities and hex digests, so damaged lines
    can be rejected independently without losing another role's cleanup data.
    """
    path = result_dir / "pids.env"
    bindings = {
        "RESULT_DIR": str(result_dir),
        "RUN_ID": runtime.get("RUN_ID"),
        "RUNTIME_SHA256": stages.get("runtime_sha256"),
    }
    if not all(isinstance(value, str) and value for value in bindings.values()):
        return {}, ["pids.env has no trustworthy run/settings binding"]
    header = "".join(f"{key}={shlex.quote(value)}\n" for key, value in bindings.items())
    try:
        source = path.read_text()
    except (OSError, UnicodeError) as exc:
        return {}, [f"cannot read {path}: {exc}"]
    if not source.startswith(header):
        return {}, ["pids.env RESULT_DIR/RUN_ID/RUNTIME_SHA256 header mismatch"]

    values = dict(bindings)
    diagnostics = []
    damaged_roles = set()
    for number, line in enumerate(source[len(header):].splitlines(), 1):
        if not line.strip():
            continue
        marker = re.match(r"\s*(PREFILL|DECODE|PROXY)_", line)
        try:
            words = shlex.split(line, comments=False)
            if len(words) != 1:
                raise ValueError("expected one assignment")
            key, separator, value = words[0].partition("=")
            if key in bindings:
                return {}, [*diagnostics, f"pids.env duplicate binding {key}"]
            if not separator or not re.fullmatch(
                r"(PREFILL|DECODE|PROXY)_(PID|PGID|SID|STARTTIME|SHA256)", key
            ):
                raise ValueError("unknown process-state key")
            if key in values:
                raise ValueError(f"duplicate key {key}")
            values[key] = value
        except ValueError as exc:
            diagnostics.append(f"pids.env identity line {number}: {exc}")
            if marker:
                damaged_roles.add(marker[1])

    verified = dict(bindings)
    for role in ROLES:
        keys = [f"{role}_{field}" for field in ("PID", "PGID", "SID", "STARTTIME")]
        digest_key = f"{role}_SHA256"
        recorded = any(key in values for key in (*keys, digest_key))
        if not recorded and role not in damaged_roles:
            continue
        if role in damaged_roles or not all(
            re.fullmatch(r"[0-9]+", values.get(key, "")) for key in keys
        ):
            diagnostics.append(
                f"pids.env {role} identity is incomplete or damaged; excluded"
            )
            continue
        if values.get(digest_key) != pid_record_digest(values, role):
            diagnostics.append(f"pids.env {role} identity checksum mismatch; excluded")
            continue
        verified.update({key: values[key] for key in (*keys, digest_key)})
    return verified, diagnostics


@contextlib.contextmanager
def lock(path, directory=False):
    flags = os.O_RDONLY if directory else os.O_CREAT | os.O_RDWR
    fd = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunError(f"another PDD command holds lock: {path}") from exc
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def launch_lock(runtime):
    # Keep the original RESULT_ROOT directory lock, plus a per-user host lock
    # so different result roots cannot race each other on the same devices.
    global_lock = Path("/tmp") / f"vllm-rbln-pdd-{os.getuid()}.lock"
    with lock(global_lock), lock(runtime["RESULT_ROOT"], directory=True):
        yield


def capture(args, **kwargs):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT, **kwargs
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", str(exc))
        command = shlex.join(map(str, args))
        raise RunError(f"command failed: {command}: {detail}") from exc


def check_code(runtime, clean=True):
    repo = runtime["REPO"]
    env = execution_env(runtime)
    actual = capture(["git", "-C", repo, "rev-parse", "HEAD"], env=env)
    if actual != runtime["COMMIT"]:
        raise RunError("recorded commit does not match repository HEAD")
    branch = capture(["git", "-C", repo, "branch", "--show-current"], env=env)
    if branch and branch != EXPECTED_BRANCH:
        raise RunError(
            f"branch mismatch: expected {EXPECTED_BRANCH} or detached HEAD, "
            f"got {branch}"
        )
    if branch != runtime["BRANCH"]:
        raise RunError("recorded branch does not match repository branch")
    capture(
        ["git", "-C", repo, "merge-base", "--is-ancestor", REQUIRED_COMMIT, actual],
        env=env,
    )
    if clean and capture(
        ["git", "-C", repo, "status", "--porcelain=v1", "--untracked-files=all"],
        env=env,
    ):
        raise RunError("recorded repository working tree is not clean")


def validate_workload(env):
    for name in ("WARMUP_REQUESTS", "MEASURE_REQUESTS", "CONCURRENCY", "MAX_TOKENS"):
        value = env.get(name, "")
        if not re.fullmatch(r"[0-9]+", value) or (
            name != "WARMUP_REQUESTS" and int(value) < 1
        ):
            raise RunError(f"invalid {name}: {value!r}")
    if env.get("PROMPT_MODE") not in ("short", "long"):
        raise RunError("PROMPT_MODE must be short or long")


def execution_env(runtime):
    env = dict(os.environ)
    # Never accept cleanup authority or run metadata from a caller's shell.
    for key in list(env):
        if any(key.startswith(f"{role}_") for role in ROLES):
            del env[key]
    env.pop("PDD_STATE_ERRORS", None)
    for key in SAVED_ENV:
        env.pop(key, None)
        if f"ENV_{key}" in runtime:
            env[key] = runtime[f"ENV_{key}"]
    env.update({k: v for k, v in runtime.items() if not k.startswith("ENV_")})
    env["PDD_CONTROL_PY"] = sys.executable
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def proc_identity(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {
            "state": tail[0], "PGID": tail[2], "SID": tail[3], "STARTTIME": tail[19]
        }
    except (OSError, IndexError):
        return None


def role_identity(pids, role):
    fields = {
        key: pids.get(f"{role}_{key}", "")
        for key in ("PID", "PGID", "SID", "STARTTIME")
    }
    if not all(re.fullmatch(r"[0-9]+", value) for value in fields.values()):
        raise RunError(
            f"missing or invalid {role} process identity; "
            f"run start_{role.lower()}.sh first"
        )
    if (
        int(fields["PID"]) <= 1
        or fields["PID"] != fields["PGID"]
        or fields["PID"] != fields["SID"]
    ):
        raise RunError(f"{role} is not an isolated process group/session")
    actual = proc_identity(fields["PID"])
    if actual is None or actual["state"] in ("Z", "X"):
        raise RunError(f"{role} exited before readiness (PID {fields['PID']})")
    if any(actual[k] != fields[k] for k in ("PGID", "SID", "STARTTIME")):
        raise RunError(f"{role} process identity mismatch; refusing reuse")
    return fields


def owns_listener(runtime, pids, role):
    """A 200 from a competing listener must not make our server look ready."""
    port = int(runtime[f"{role}_PORT"])
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        return False
    sid = pids[f"{role}_SID"]
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = proc_identity(entry.name)
        if not identity or identity["SID"] != sid:
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def wait_ready(runtime, pids, role, timeout):
    port = runtime[f"{role}_PORT"]
    endpoint = "healthcheck" if role == "PROXY" else "health"
    url = f"http://127.0.0.1:{port}/{endpoint}"
    deadline = time.monotonic() + timeout
    while True:
        role_identity(pids, role)
        try:
            request_timeout = min(2, max(0.05, deadline - time.monotonic()))
            with urllib.request.urlopen(url, timeout=request_timeout) as response:
                if 200 <= response.status < 300 and owns_listener(runtime, pids, role):
                    role_identity(pids, role)
                    return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RunError(
                f"{role} readiness timed out after {timeout}s: {url}; "
                f"log: {runtime['RESULT_DIR']}/{role.lower()}.log"
            )
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def check_devices(runtime, role):
    env = execution_env(runtime)
    if role == "PREFILL":
        line = capture(
            ["nvidia-smi", "-i", runtime["CUDA_DEVICE"],
             "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
            env=env,
        )
        parts = line.split(",")
        if len(parts) < 3 or not parts[1].strip():
            raise RunError("CUDA device query returned an unexpected result")
        occupied = capture(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            env=env,
        )
        if parts[1].strip() in [value.strip() for value in occupied.splitlines()]:
            raise RunError(
                f"CUDA device {runtime['CUDA_DEVICE']} already has a compute process"
            )
    elif role == "DECODE":
        state = json.loads(capture(["rbln-smi", "--json"], env=env))
        device = runtime["RBLN_DEVICE"]
        selected = [d for d in state.get("devices", []) if str(d.get("npu")) == device]
        if len(selected) != 1 or selected[0].get("status") != "normal":
            raise RunError(f"RBLN device {device} is unavailable")
        if any(
            str(context.get("npu", context.get("device", ""))).lower()
            in (device, f"rbln{device}")
            for context in state.get("contexts", [])
        ):
            raise RunError(f"RBLN device {device} already has a context")


def check_ports(runtime, role):
    names = [f"{role}_PORT"]
    if role != "PROXY":
        names.append(f"{role}_SIDE_PORT")
    for name in names:
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", int(runtime[name])))
            except OSError as exc:
                raise RunError(
                    f"port conflict: {name}={runtime[name]}; "
                    "refusing to change saved ports"
                ) from exc


def terminate_spawned_session(process, identity):
    """Roll back only a session created by this command, including its children."""
    pid = str(process.pid)
    if identity is None or identity["PGID"] != pid or identity["SID"] != pid:
        raise RunError(f"cannot safely identify spawned session {pid} for cleanup")

    def active_members():
        current = proc_identity(pid)
        if current and any(
            current[key] != identity[key] for key in ("PGID", "SID", "STARTTIME")
        ):
            raise RunError(f"spawned PID {pid} identity changed; refusing cleanup")
        active = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            member = proc_identity(entry.name)
            if member is None or member["PGID"] != pid:
                continue
            if member["SID"] != pid:
                raise RunError(f"spawned group {pid} changed session; refusing cleanup")
            active |= member["state"] not in ("Z", "X")
        return active

    # Keep the leader unreaped until group cleanup finishes, reserving its PID.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not active_members():
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + 5
        while active_members() and time.monotonic() < deadline:
            time.sleep(0.05)
    if active_members():
        raise RunError(f"spawned session {pid} is still active after cleanup")
    process.wait(timeout=5)


def run_helper(args, env, logfile):
    # Drain output into both the console and the durable per-attempt log.
    with open(logfile, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            args, env=env, cwd=env["REPO"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        identity = None
        try:
            identity = proc_identity(process.pid)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            try:
                terminate_spawned_session(
                    process, identity or proc_identity(process.pid)
                )
            except (RunError, OSError, subprocess.TimeoutExpired) as cleanup_error:
                print(
                    f"ERROR: helper interruption cleanup: {cleanup_error}",
                    file=sys.stderr,
                )
            raise
        finally:
            process.stdout.close()
    if code:
        raise RunError(f"helper exited with code {code}; log: {logfile}")


def load_state(result_dir, tolerate=False):
    diagnostics = []
    runtime = {}
    stages = {"run_id": None, "steps": {}}
    pids = {}
    source = None
    source_bytes = None
    corrupt = False
    try:
        source_bytes = (result_dir / "stages.json").read_bytes()
        source = source_bytes.decode("utf-8")
        stages = json.loads(source)
        if not isinstance(stages, dict) or not isinstance(stages.get("steps"), dict):
            raise ValueError("invalid stages schema")
        stages.setdefault("run_id", None)
        saved_diagnostics = stages.get("diagnostics", [])
        if not isinstance(saved_diagnostics, list) or not all(
            isinstance(item, str) for item in saved_diagnostics
        ):
            diagnostics.append("invalid diagnostics record in stages.json")
            stages["diagnostics"] = []
            corrupt = True
        for name, record in list(stages["steps"].items()):
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("attempts", []), list)
                or not all(
                    isinstance(item, dict) for item in record.get("attempts", [])
                )
            ):
                diagnostics.append(f"invalid {name} step record in stages.json")
                del stages["steps"][name]
                corrupt = True
    except (OSError, ValueError) as exc:
        diagnostics.append(f"cannot read stages.json: {exc}")
        stages = {"run_id": None, "steps": {}}
        corrupt = source_bytes is not None
    if corrupt:
        backup = result_dir / f"stages.corrupt.{uuid.uuid4().hex}.json"
        try:
            if source is None:
                # Preserve the exact bytes if corruption is not valid UTF-8.
                backup.write_bytes(source_bytes)
            else:
                atomic_text(backup, source)
        except OSError as exc:
            # Do not overwrite the only copy of damaged evidence.
            raise RunError(f"cannot preserve corrupt stages.json: {exc}") from exc
        diagnostics.append(f"corrupt stages.json preserved at {backup}")
    try:
        runtime = read_env(result_dir / "runtime.env")
        if runtime.get("RESULT_DIR") != str(result_dir) or not runtime.get("RUN_ID"):
            raise RunError("recorded RESULT_DIR/RUN_ID mismatch")
        if runtime["RUN_ID"] != stages.get("run_id"):
            raise RunError("runtime/stages run ID mismatch")
        digest = hashlib.sha256((result_dir / "runtime.env").read_bytes()).hexdigest()
        if digest != stages.get("runtime_sha256"):
            raise RunError("runtime.env settings checksum mismatch")
    except RunError as exc:
        diagnostics.append(str(exc))
    pids, pid_diagnostics = load_pids(result_dir, runtime, stages)
    diagnostics.extend(pid_diagnostics)
    # Keep persistent state diagnostics visible on subsequent cleanup/summary calls.
    diagnostics = list(dict.fromkeys([*stages.get("diagnostics", []), *diagnostics]))
    if diagnostics and not tolerate:
        raise RunError("; ".join(diagnostics))
    return runtime, pids, stages, diagnostics


def prepare(result_dir, stages, logfile):
    env = dict(os.environ)
    env.update(
        RESULT_DIR=str(result_dir), RUN_ID=stages["run_id"],
        REPO=str(Path(env.get("REPO", REPO)).resolve()), PDD_CONTROL_PY=sys.executable,
    )
    root = Path(
        env.get("RESULT_ROOT", Path.home() / "vllm-rbln-profile-results")
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env["RESULT_ROOT"] = str(root)
    for key, default in WORKLOAD_DEFAULTS.items():
        env.setdefault(key, default)
    validate_workload(env)
    with launch_lock(env):
        run_helper(["bash", str(HERE / "legacy_steps.sh"), "prepare"], env, logfile)
        runtime = read_env(result_dir / "runtime.env")
        runtime.update({key: env[key] for key in WORKLOAD_DEFAULTS})
        runtime["PROMPT_ID"] = env.get("PROMPT_ID", "2")
        if runtime["PROMPT_ID"] not in ("2", "4"):
            raise RunError("PROMPT_ID must be 2 or 4")
        for key in SAVED_ENV:
            if key in env:
                runtime[f"ENV_{key}"] = env[key]
        runtime.update(
            MAX_MODEL_LEN="1024", MAX_NUM_SEQS="1", TENSOR_PARALLEL_SIZE="1",
            NUM_GPU_BLOCKS_OVERRIDE="32", SEED="0", ENABLE_CHUNKED_PREFILL="1",
            ENABLE_PREFIX_CACHING="0",
        )
        write_env(result_dir / "runtime.env", runtime)
        stages["runtime_sha256"] = hashlib.sha256(
            (result_dir / "runtime.env").read_bytes()
        ).hexdigest()
        write_env(result_dir / "pids.env", {
            "RESULT_DIR": str(result_dir), "RUN_ID": stages["run_id"],
            "RUNTIME_SHA256": stages["runtime_sha256"],
        })
    print(f"RESULT_DIR={result_dir}")


def start_role(result_dir, runtime, pids, stages, role):
    check_code(runtime)
    if stages["steps"].get("prepare_run", {}).get("status") != "SUCCESS":
        raise RunError("prepare_run must succeed before starting servers")
    if "stop_pdd" in stages["steps"]:
        raise RunError("this run has already entered cleanup; prepare a new run")
    with launch_lock(runtime):
        if any(key.startswith(f"{role}_") for key in pids):
            wait_ready(runtime, pids, role, 10)
            print(f"Reusing verified {role} PID {pids[role + '_PID']}")
            return
        predecessors = {
            "PREFILL": (), "DECODE": ("PREFILL",), "PROXY": ("PREFILL", "DECODE")
        }
        for predecessor in predecessors[role]:
            wait_ready(runtime, pids, predecessor, 10)
        check_devices(runtime, role)
        check_ports(runtime, role)
        logfile = result_dir / f"{role.lower()}.log"
        process, identity = None, None
        pending_signals = []
        previous_handlers = {}
        # Defer interruption only until the PID record is durable. Blocking the
        # signal mask instead would leave the server with blocked TERM signals.
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda received, frame: pending_signals.append(received))
        try:
            with open(logfile, "a") as log:
                process = subprocess.Popen(
                    ["bash", str(HERE / "legacy_steps.sh"), "commands", role],
                    env=execution_env(runtime), cwd=runtime["REPO"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, close_fds=True,
                )
            identity = proc_identity(process.pid)
            if (
                identity is None
                or identity["PGID"] != str(process.pid)
                or identity["SID"] != str(process.pid)
            ):
                raise RunError(
                    f"cannot inspect isolated {role} process {process.pid}; "
                    f"log: {logfile}"
                )
            updated = dict(pids)
            updated.update({
                f"{role}_PID": str(process.pid),
                **{
                    f"{role}_{key}": identity[key]
                    for key in ("PGID", "SID", "STARTTIME")
                },
            })
            updated[f"{role}_SHA256"] = pid_record_digest(updated, role)
            write_env(result_dir / "pids.env", updated)
            pids.update(updated)
        except BaseException:
            if process is not None:
                try:
                    terminate_spawned_session(
                        process, identity or proc_identity(process.pid)
                    )
                except (RunError, OSError, subprocess.TimeoutExpired) as cleanup_error:
                    print(
                        f"ERROR: unrecorded {role} cleanup: {cleanup_error}; "
                        f"log: {logfile}", file=sys.stderr,
                    )
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        if pending_signals:
            raise KeyboardInterrupt(
                "interrupted after preserving the server PID record"
            )
        try:
            wait_ready(runtime, pids, role, TIMEOUTS[role])
        except RunError as exc:
            process.poll()  # Reap an early exit while retaining the saved identity.
            raise RunError(
                f"{exc}; log: {logfile}; run stop_pdd.sh "
                f"--result-dir {shlex.quote(str(result_dir))}"
            ) from exc
        if role == "PROXY":
            for current in ROLES:
                wait_ready(runtime, pids, current, 10)
        print(f"{role}_PID={process.pid}\n{role}_PORT={runtime[role + '_PORT']}")


def select_directory(args, explicit):
    supplied = args.result_dir if explicit else (os.environ.get("RESULT_DIR") or None)
    if supplied is not None:
        if not supplied or not Path(supplied).is_absolute():
            raise RunError(
                "--result-dir / RESULT_DIR must be an absolute nonempty path"
            )
        return Path(supplied).resolve()
    root = Path(
        os.environ.get("RESULT_ROOT", Path.home() / "vllm-rbln-profile-results")
    ).resolve()
    if args.step in ("prepare_run", "start_pdd"):
        token = os.environ.get("PDD_RUN_TOKEN", "")
        if token and not re.fullmatch(r"[A-Za-z0-9._-]+", token):
            raise RunError("invalid PDD_RUN_TOKEN")
        suffix = f"-{token}" if token else ""
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return root / f"native-dtype-{stamp}-{uuid.uuid4().hex[:8]}{suffix}"
    if args.step in ("check_correctness", "run_workload") and args.phase is None:
        candidates = [p for p in root.glob("native-dtype-*") if p.is_dir()]
        if candidates:
            selected = max(candidates, key=lambda p: p.stat().st_mtime_ns)
            print(
                f"Legacy latest-result fallback: RESULT_DIR={selected}", file=sys.stderr
            )
            return selected
    raise RunError("--result-dir or RESULT_DIR is required; run prepare_run.sh first")


def execute(args, result_dir, explicit):
    step = args.step
    if step == "prepare_run":
        result_dir.mkdir(parents=True, exist_ok=True)
    if not result_dir.is_dir():
        raise RunError(f"RESULT_DIR does not exist: {result_dir}")
    with lock(result_dir / ".pdd.lock"):
        if step == "prepare_run":
            if any(p.name != ".pdd.lock" for p in result_dir.iterdir()):
                raise RunError(
                    f"refusing to overwrite existing run/results: {result_dir}"
                )
            runtime, pids, diagnostics = {}, {}, []
            stages = {"run_id": f"pdd-{uuid.uuid4()}", "steps": {}}
        else:
            runtime, pids, stages, diagnostics = load_state(result_dir, tolerate=True)
        stages["diagnostics"] = diagnostics
        previous = stages["steps"].get(step)
        repeated_workload = previous and step in ("warmup", "run_workload")
        attempts = (previous or {}).get("attempts", [])
        if previous:
            attempts = [
                *attempts, {k: v for k, v in previous.items() if k != "attempts"}
            ]
        logfile = result_dir / f"{step}.{len(attempts) + 1}.log"
        record = {
            "run_id": stages["run_id"], "stage": step, "status": "RUNNING",
            "started_at": utcnow(), "ended_at": None, "elapsed_seconds": None,
            "exit_code": None, "reason": None, "log_path": str(logfile),
            "attempts": attempts,
        }
        stages["steps"][step] = record
        atomic_json(result_dir / "stages.json", stages)
        logfile.touch()
        started = time.monotonic()
        code = 0
        try:
            if repeated_workload:
                raise RunError(
                    f"{step} was already attempted; "
                    "results are preserved, prepare a new run"
                )
            if diagnostics and step not in ("stop_pdd", "summarize_results"):
                raise RunError("; ".join(diagnostics))
            if step == "prepare_run":
                prepare(result_dir, stages, logfile)
            elif step.startswith("start_"):
                start_role(
                    result_dir, runtime, pids, stages,
                    step.removeprefix("start_").upper(),
                )
            elif step == "summarize_results":
                pass  # Generate after recording the completed script stage below.
            elif step == "stop_pdd":
                # A damaged runtime checksum cannot authorize any targets. Preserve
                # diagnostics and still generate a failed cleanup stage record.
                bound = (
                    runtime.get("RESULT_DIR") == str(result_dir)
                    and bool(runtime.get("RUN_ID"))
                    and runtime.get("RUN_ID") == stages.get("run_id")
                    and hashlib.sha256(
                        (result_dir / "runtime.env").read_bytes()
                    ).hexdigest() == stages.get("runtime_sha256")
                )
                if not bound:
                    raise RunError("unsafe cleanup metadata: " + "; ".join(diagnostics))
                try:
                    check_code(runtime, clean=False)
                except RunError as exc:
                    diagnostics.append(str(exc))
                env = execution_env(runtime)
                env.update(pids)
                env["PDD_STATE_ERRORS"] = "; ".join(diagnostics)
                with launch_lock(runtime):
                    run_helper(
                        ["bash", str(HERE / "legacy_steps.sh"), "stop"], env, logfile
                    )
            else:
                check_code(runtime)
                for role in ROLES:
                    wait_ready(runtime, pids, role, 10)
                env = execution_env(runtime)
                env.update(pids)
                if step == "check_correctness":
                    env["PROMPT_ID"] = os.environ.get(
                        "PROMPT_ID", runtime.get("PROMPT_ID", "2")
                    )
                    response_file = result_dir / (
                        f"correctness_{runtime['MODEL_DTYPE']}_"
                        f"prompt{env['PROMPT_ID']}_response.json"
                    )
                    if response_file.exists():
                        raise RunError(
                            f"correctness result already exists: {response_file}; "
                            "use a new run to repeat this prompt"
                        )
                    command = ["bash", str(HERE / "legacy_steps.sh"), "correctness"]
                else:
                    if not explicit and args.phase is None and step == "run_workload":
                        env.update({
                            key: os.environ[key]
                            for key in WORKLOAD_DEFAULTS if key in os.environ
                        })
                    validate_workload(env)
                    phase = "warmup" if step == "warmup" else (args.phase or "all")
                    command = [
                        runtime["CUDA_PY"], str(HERE / "workload.py"), "--phase", phase
                    ]
                run_helper(command, env, logfile)
        except (Exception, KeyboardInterrupt) as exc:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            record["reason"] = str(exc) or "interrupted"
            with open(logfile, "a") as log:
                log.write(f"ERROR: {record['reason']}\n")
            print(f"ERROR: {record['reason']}", file=sys.stderr)
        finally:
            record.update(
                status="SUCCESS" if code == 0 else "FAILED", ended_at=utcnow(),
                elapsed_seconds=time.monotonic() - started, exit_code=code,
            )
            atomic_json(result_dir / "stages.json", stages)
        if step == "summarize_results":
            try:
                from summary import generate
                generate(result_dir, runtime, stages, diagnostics)
                record.update(
                    ended_at=utcnow(), elapsed_seconds=time.monotonic() - started
                )
                atomic_json(result_dir / "stages.json", stages)
                # Publish the completed summary stage timing as well.
                generate(result_dir, runtime, stages, diagnostics)
                print(f"SUMMARY_FILE={result_dir / 'summary.md'}")
            except Exception as exc:
                record.update(
                    status="FAILED", exit_code=1, reason=str(exc),
                    ended_at=utcnow(), elapsed_seconds=time.monotonic() - started,
                )
                atomic_json(result_dir / "stages.json", stages)
                print(f"ERROR: summary generation failed: {exc}", file=sys.stderr)
                return 1
        return code


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    step = arguments.pop(0) if arguments else None
    parser = argparse.ArgumentParser(
        prog=f"{step}.sh" if step else "PDD runner",
        description=(
            "PDD CI procedure step; servers persist between shell invocations."
        ),
    )
    if step not in (*STEPS, "start_pdd"):
        parser.error("invoke one of the public PDD .sh entry points")
    parser.set_defaults(step=step, phase=None)
    parser.add_argument(
        "--result-dir",
        help="absolute run directory (takes precedence over RESULT_DIR)",
    )
    if step == "run_workload":
        parser.add_argument(
            "--phase", choices=("all", "measure"),
            help="default all keeps warmup + profiling pause + measure",
        )
    args = parser.parse_args(arguments)
    explicit = args.result_dir is not None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        result_dir = select_directory(args, explicit)
        if args.step == "start_pdd":
            for step in ("prepare_run", "start_prefill", "start_decode", "start_proxy"):
                # Invoke the same public entry points, with exactly one preparation.
                code = subprocess.call([
                    "bash", str(REPO / f"{step}.sh"), "--result-dir", str(result_dir)
                ])
                if code:
                    return code
            # prepare_run already printed the single RESULT_DIR line consumed
            # by the existing run_all_once.sh compatibility caller.
            runtime = read_env(result_dir / "runtime.env")
            print(f"COMPILER_VERSION={runtime['COMPILER_VERSION']}")
            print(f"COMMIT={runtime['COMMIT']}")
            print(f"LAYOUT_REORDER_ENABLED={runtime['VLLM_RBLN_PDD_LAYOUT_REORDER']}")
            return 0
        return execute(args, result_dir, explicit)
    except (RunError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
