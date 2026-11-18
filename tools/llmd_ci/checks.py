"""Build provenance and the existing CPU / official-process checks."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from llmd.cluster import ENVOY

BINARIES = ("epp", "pd-sidecar", "envoy")
BUILD_RECORD = "build-record.json"
DIGEST = re.compile(r"sha256:[a-f0-9]{64}")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(f.name, path)


def build_inputs(root):
    root = Path(root)
    epp = root / "llmd/epp"
    serving = root / "llmd/serving"
    inputs = [
        epp / "go.mod",
        epp / "go.sum",
        serving / "build.sh",
        serving / "images.sh",
    ]
    inputs.extend(epp.rglob("*.go"))
    inputs.extend(p for p in (serving / "patches").rglob("*") if p.is_file())
    files = {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(inputs)}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def image_inputs(root, role):
    if role in {"epp", "sidecar"}:
        return build_inputs(root)
    if role == "envoy":
        return hashlib.sha256(ENVOY.encode()).hexdigest()
    # The Dockerfile pins the external wheel and copies the checkout.
    # Embedded code and build scripts must agree with the trusted build job.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    ).split(b"\0")
    h = hashlib.sha256()
    for raw in sorted(set(filter(None, names))):
        # CI deployment settings are supplied after the engine build. The plan
        # hashes these separately, including any digest of this very image.
        if raw.startswith(b"ci/llmd/"):
            continue
        path = Path(root) / os.fsdecode(raw)
        if path.is_symlink():
            value = b"link:" + os.fsencode(os.readlink(path))
        elif path.is_file():
            mode = b"x" if os.access(path, os.X_OK) else b"-"
            value = mode + file_sha(path).encode()
        elif not path.exists():
            continue
        else:
            raise ValueError(f"unsupported image build input: {path}")
        h.update(raw + b"\0" + value + b"\0")
    return h.hexdigest()


def repository(ref):
    name = ref.split("@", 1)[0]
    return name.rsplit(":", 1)[0] if ":" in name.rsplit("/", 1)[-1] else name


def verify_image_record(root, record_path, images, required):
    record = json.loads(Path(record_path).read_text())
    if record.get("schema_version") != 1 or not isinstance(record.get("images"), dict):
        raise ValueError("invalid trusted image build record")
    resolved = {}
    for role in sorted(required):
        entry = record["images"].get(role, {})
        expected = images[role]
        image = entry.get("image", "")
        name, _, digest = image.partition("@")
        if not DIGEST.fullmatch(digest) or repository(name) != repository(expected):
            raise ValueError(
                f"{role}: expected repository and immutable platform manifest"
            )
        if entry.get("inputs_sha256") != image_inputs(root, role):
            raise ValueError(f"{role}: image build inputs differ from checkout")
        # A trusted resolver can bind a multi-platform index to its platform.
        pinned = expected.partition("@")[2]
        if pinned and pinned not in (digest, entry.get("index_digest")):
            raise ValueError(f"{role}: pinned image digest differs from build record")
        resolved[role] = repository(name) + "@" + digest
    return resolved


def verify_binaries(root, bin_dir):
    dest = Path(bin_dir)
    record = json.loads((dest / BUILD_RECORD).read_text())
    if (
        record.get("schema_version") != 1
        or record.get("platform") != platform.machine()
        or record.get("inputs_sha256") != build_inputs(root)
    ):
        raise ValueError("binary build record differs from checkout or platform")
    for name in BINARIES:
        path = dest / name
        if not os.access(path, os.X_OK) or file_sha(path) != record.get(
            "binaries", {}
        ).get(name):
            raise ValueError(f"binary digest or executable bit mismatch: {name}")
    return {name: record["binaries"][name] for name in BINARIES}


@contextlib.contextmanager
def binary_lock(dest):
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".ci-build.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def command(result, dest, label, args, cwd, env=None):
    log = dest / "logs" / (label + ".log")
    started = time.monotonic()
    argv = [str(x) for x in args]
    with log.open("wb") as f:
        try:
            rc = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
            ).returncode
        except OSError as exc:
            f.write(str(exc).encode())
            rc = 1
    result["commands"].append(
        dict(
            argv=argv,
            cwd=str(cwd),
            returncode=rc,
            log=str(log.relative_to(dest)),
            seconds=round(time.monotonic() - started, 3),
        )
    )
    return rc == 0


def prepare_binaries(root, dest, out, result, force=False):
    if (dest / BUILD_RECORD).is_file() and not force:
        return verify_binaries(root, dest)
    before = build_inputs(root)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".ci-build-") as tmp:
        if not command(
            result, out, "build", ["bash", root / "llmd/serving/build.sh", tmp], root
        ):
            raise ValueError("official binary build failed, see logs/build.log")
        if before != build_inputs(root):
            raise ValueError("inputs changed while building binaries")
        hashes = {name: file_sha(Path(tmp) / name) for name in BINARIES}
        for name in (*BINARIES, "binaries.sha256"):
            os.replace(Path(tmp) / name, dest / name)
        save(
            dest / BUILD_RECORD,
            dict(
                schema_version=1,
                inputs_sha256=before,
                binaries=hashes,
                platform=platform.machine(),
                head_sha=result["source"]["head_sha"],
            ),
        )
    return verify_binaries(root, dest)


def run_checks(root, kind, out, bin_dir=None, plan_path=None):
    from .plan import source_identity
    from .results import publish

    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"checks require an empty output directory: {out}")
    (out / "logs").mkdir()
    result = dict(
        schema_version=1,
        kind=kind,
        source=source_identity(root),
        commands=[],
        status="failed",
        hardware="not_requested",
        correctness="not_run",
        transfer="not_run",
        overlap="not_run",
    )
    try:
        if plan_path is not None:
            plan_data = json.loads(Path(plan_path).read_text())
            if (
                plan_data.get("schema_version") != 1
                or plan_data.get("head_sha") != result["source"]["head_sha"]
                or plan_data.get("source", {}).get("sha256")
                != result["source"]["sha256"]
            ):
                raise ValueError(
                    "checks checkout HEAD or runtime source differs from plan"
                )
            result["plan_sha256"] = file_sha(Path(plan_path))
        if kind == "unit":
            configs = sorted((root / "experiments").glob("*.yaml"))
            if not configs:
                raise ValueError("no experiment YAML files found")
            inputs = set(configs)
            from llmd.config import read_yaml

            for cfg in configs:
                raw = read_yaml(cfg)
                for key in ("model", "profile"):
                    if isinstance(raw.get(key), str):
                        inputs.add((cfg.parent / raw[key]).resolve())
            result["config_sha256"] = {
                str(p.relative_to(root)): file_sha(p) for p in sorted(inputs)
            }
            for cfg in configs:
                command(
                    result,
                    out,
                    "validate-" + cfg.stem,
                    [sys.executable, "-m", "llmd", "validate", cfg],
                    root,
                )
            command(
                result,
                out,
                "runtime-tests",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "llmd.test_engines",
                    "llmd.test_runtime",
                ],
                root,
            )
            command(
                result,
                out,
                "ci-tests",
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-v",
                    "-s",
                    "tests/llmd_ci",
                ],
                root,
            )
            go = os.environ.get("LLMD_GO") or shutil.which("go") or "go"
            epp = root / "llmd/epp"
            env = dict(os.environ, GOTOOLCHAIN="local", CGO_ENABLED="1")
            command(result, out, "go-version", [go, "version"], epp, env)
            version = re.search(r"(?m)^go (\S+)$", (epp / "go.mod").read_text()).group(
                1
            )
            if "go" + version + " " not in (out / "logs/go-version.log").read_text():
                raise ValueError(f"unit checks require Go {version} (set LLMD_GO)")
            command(
                result,
                out,
                "go-tests",
                [go, "test", "-mod=readonly", "./..."],
                epp,
                env,
            )
            command(
                result,
                out,
                "go-race",
                [go, "test", "-mod=readonly", "-race", "./plugins"],
                epp,
                env,
            )
        else:
            dest = Path(bin_dir).expanduser().resolve()
            with binary_lock(dest):
                result["binaries"] = prepare_binaries(
                    root, dest, out, result, force=kind == "build"
                )
                shutil.copyfile(dest / BUILD_RECORD, out / BUILD_RECORD)
                if kind == "process":
                    result["fixture"] = (
                        "Official Envoy, EPP and PD sidecar with simulated engines"
                    )
                    for label, module in [
                        ("protocol", "llmd.serving.protocol_test"),
                        ("policy", "llmd.epp.tests.policy_process"),
                    ]:
                        command(
                            result,
                            out,
                            label,
                            [
                                sys.executable,
                                "-m",
                                module,
                                "--bin-dir",
                                dest,
                                "--work",
                                out / label,
                            ],
                            root,
                        )
                    verify_binaries(root, dest)
        for name, expected in result.get("config_sha256", {}).items():
            if file_sha(root / name) != expected:
                raise ValueError(f"experiment inputs changed during checks: {name}")
        if source_identity(root)["sha256"] != result["source"]["sha256"]:
            raise ValueError("source changed during checks")
        if all(c["returncode"] == 0 for c in result["commands"]):
            result["status"] = "passed"
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    save(out / "result.json", result)
    publish(out, result)
    return result
