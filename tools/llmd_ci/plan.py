"""Select registered llm-d experiments from a Git comparison without probing devices."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
import subprocess
import tempfile
from pathlib import Path

import yaml

from llmd import config


def _git(root, *args):
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors="replace").strip() or "git failed")
    return proc.stdout


def _root(directory):
    return Path(
        os.fsdecode(_git(directory or ".", "rev-parse", "--show-toplevel")).rstrip("\n")
    )


def _commit(root, ref):
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ValueError(f"expected an explicit Git commit ref, got {ref!r}")
    try:
        return (
            _git(root, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}")
            .decode()
            .strip()
        )
    except ValueError as exc:
        raise ValueError(
            f"cannot resolve Git ref {ref!r}; fetch the required history"
        ) from exc


def _paths(raw):
    return [os.fsdecode(item) for item in raw.split(b"\0") if item]


def _changes(root, base, head):
    items = iter(
        _paths(_git(root, "diff", "--name-status", "-z", "-M", base, head, "--"))
    )
    result = []
    for status in items:
        path = next(items)
        if status.startswith(("C", "R")):
            result.append({"status": status, "old_path": path, "path": next(items)})
        else:
            result.append({"status": status, "path": path})
    return result


def _relative(name, directory="."):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError(f"expected a repository-relative file path, got {name!r}")
    path = posixpath.normpath(posixpath.join(directory, name))
    if path in (".", "..") or path.startswith(("../", ".git/")) or path == ".git":
        raise ValueError(f"path is outside the repository source: {name!r}")
    return path


def _file(root, rel):
    path = root / _relative(rel)
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or ".git" in resolved.relative_to(root).parts:
        raise ValueError(f"path resolves outside checkout: {rel!r}")
    if not path.is_file():
        raise ValueError(f"missing required file {rel!r}")
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(path):
    if path in {"pyproject.toml", "uv.lock", ".dockerignore"}:
        return True
    return path.startswith(
        (
            "llmd/",
            "tools/llmd_ci/",
            "tools/hetero/",
            "tools/pdd/",
            "tests/",
            "vllm_rbln/",
            "docker/",
        )
    ) and (
        Path(path).suffix
        in {
            ".py",
            ".go",
            ".mod",
            ".sum",
            ".patch",
            ".sh",
            ".cc",
            ".cu",
            ".h",
            ".yaml",
            ".yml",
            ".env",
        }
        or Path(path).name.startswith("Dockerfile")
    )


def source_identity(root=None, head_sha=None):
    """Identify this commit and the contents and modes of dirty runtime sources."""
    root = _root(root)
    actual = _commit(root, "HEAD")
    if head_sha is not None and head_sha != actual:
        raise ValueError(f"checkout HEAD {actual} differs from planned head {head_sha}")
    changed = set(
        _paths(_git(root, "diff", "--name-only", "--no-renames", "-z", "HEAD", "--"))
    )
    changed.update(
        _paths(_git(root, "ls-files", "-z", "--others", "--exclude-standard"))
    )
    dirty = {}
    for name in sorted(changed):
        if not _code(name):
            continue
        path = root / name
        if not path.exists() and not path.is_symlink():
            dirty[name] = None
            continue
        resolved = _file(root, name)
        dirty[name] = {
            "sha256": _digest(resolved),
            "executable": bool(resolved.stat().st_mode & stat.S_IXUSR),
        }
    return {
        "head_sha": actual,
        "sha256": config.fingerprint({"head_sha": actual, "dirty_source": dirty}),
        "dirty": bool(changed),
        "dirty_paths": dirty,
    }


def _suite_at(root, base, path):
    if not _git(root, "ls-tree", "-z", base, "--", ":(literal)" + path):
        return {}
    data = yaml.load(_git(root, "show", base + ":" + path), Loader=config.UniqueKeys)
    if not isinstance(data, dict):
        raise ValueError(f"invalid suite at {base}:{path}")
    return data


def _suite(root, path):
    suite = config.read_yaml(_file(root, path))
    config.fields(
        suite,
        {"version", "experiments", "environments", "reference"},
        {"version", "experiments", "environments", "reference"},
        "suite",
    )
    if type(suite["version"]) is not int or suite["version"] != 1:
        raise ValueError("suite version must be 1")
    for key in ("experiments", "environments", "reference"):
        if not isinstance(suite[key], dict):
            raise ValueError(f"suite.{key} requires a mapping")
    for key in ("experiments", "environments"):
        for name in suite[key]:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", name):
                raise ValueError(f"invalid suite {key} ID {name!r}")
    return suite


def _registration(entry):
    config.fields(
        entry,
        {"path", "environment", "enabled", "reference"},
        {"path", "environment"},
        "experiment registration",
    )
    if type(entry.get("enabled", True)) is not bool:
        raise ValueError("registration enabled must be boolean")
    if not isinstance(entry["environment"], str):
        raise ValueError("registration environment must be an ID")
    return _relative(entry["path"])


def _settings(suite, entry):
    eid = entry["environment"]
    if eid not in suite["environments"]:
        raise ValueError(f"unknown environment {eid!r}")
    env = suite["environments"][eid]
    required = {
        "target",
        "allowed",
        "timeout",
        "cleanup_timeout",
        "collection_timeout",
        "serving",
    }
    config.fields(
        env,
        required
        | {
            "site",
            "context_env",
            "namespace_env",
            "image_record_env",
            "transfer_evidence",
        },
        required,
        f"environment {eid}",
    )
    target = env["target"]
    if env.get("transfer_evidence", "per_request") not in ("per_request", "aggregate"):
        raise ValueError("transfer_evidence must be per_request or aggregate")
    if target not in ("local", "kubernetes"):
        raise ValueError(f"unsupported target {target!r}")
    if target == "kubernetes":
        _relative(env.get("site"))
        for key in ("context_env", "namespace_env", "image_record_env"):
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(env.get(key, ""))):
                raise ValueError(f"{eid} requires an explicit {key}")
    if target == "local" and env.get("site"):
        raise ValueError("a local target cannot apply a cluster site")
    if not isinstance(env["serving"], list):
        raise ValueError("serving must list repo-relative renderer inputs")
    for path in env["serving"]:
        _relative(path)
    for key in ("timeout", "cleanup_timeout", "collection_timeout"):
        config.integer(env[key], f"{eid}.{key}")
    return env


def _reference(suite, entry):
    overrides = entry.get("reference", {})
    config.fields(overrides, {"profile", "endpoint"}, set(), "reference override")
    ref = dict(suite["reference"]) | overrides
    config.fields(ref, {"profile", "endpoint"}, {"profile"}, "reference")
    ref["profile"] = _relative(ref["profile"])
    return ref


def _dependencies(path, raw, ref, env):
    directory = posixpath.dirname(path)
    deps = {path: ["experiment"]}

    def add(name, kind):
        deps.setdefault(name, []).append(kind)

    add(_relative(raw.get("profile"), directory), "profile")
    if isinstance(raw.get("model"), str):
        add(_relative(raw["model"], directory), "model")
    add(ref["profile"], "reference profile")
    for name in env["serving"]:
        add(_relative(name), "serving")
    if env.get("site"):
        add(_relative(env["site"]), "site")
    return deps


def _resources(cfg, env):
    allowed = env["allowed"]
    config.fields(
        allowed,
        {"gpu", "npu", "nodes", "gpu_devices", "npu_devices"},
        {"gpu", "npu", "nodes"},
        "allowed resources",
    )
    nodes = allowed["nodes"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or not all(isinstance(node, str) for node in nodes)
    ):
        raise ValueError("allowed.nodes must list permitted node groups")
    eps = cfg["endpoints"]
    required = {"nodes": sorted({e["node"] for e in eps})}
    for kind in ("gpu", "npu"):
        config.integer(allowed[kind], f"allowed.{kind}", low=0)
        devices = [
            (e["node"], device)
            for e in eps
            if e["kind"] == kind
            for device in e["devices"]
        ]
        required[kind] = len(devices)
        if len(devices) > allowed[kind]:
            raise ValueError(
                f"requires {len(devices)} {kind}, environment permits {allowed[kind]}"
            )
        key = kind + "_devices"
        if key in allowed:
            if not isinstance(allowed[key], list) or not all(
                type(d) is int and d >= 0 for d in allowed[key]
            ):
                raise ValueError(f"allowed.{key} requires nonnegative device indices")
            if extra := sorted({d for _, d in devices} - set(allowed[key])):
                raise ValueError(
                    f"{kind} device indices outside allowed placement: {extra}"
                )
    if extra := set(required["nodes"]) - set(nodes):
        raise ValueError(f"node groups outside allowed placement: {sorted(extra)}")
    if env["target"] == "local" and len(required["nodes"]) > 1:
        raise ValueError("local runner requires a single node")
    return required


def _resolve_reference(cfg, ref):
    engines = cfg["endpoints"]
    if any(
        len({e["engine"][key] for e in engines}) != 1 for key in ("dtype", "kv_dtype")
    ):
        raise ValueError(
            "GPU reference cannot reproduce multiple target dtype or KV dtype values"
        )
    candidates = [e["id"] for e in engines if e["backend"] == "cuda"]
    explicit = ref.get("endpoint")
    if explicit is not None:
        if explicit not in candidates:
            raise ValueError(
                f"reference endpoint {explicit!r} is not a target CUDA endpoint"
            )
    elif len(candidates) == 1:
        explicit = candidates[0]
    else:
        raise ValueError(f"specify a reference CUDA endpoint from {candidates}")
    return {**ref, "endpoint": explicit}


def _target(root, name, entry, path, cfg, ref, env, deps, reasons):
    required = _resources(cfg, env)
    reference = _resolve_reference(cfg, ref)
    inputs = {p: _digest(_file(root, p)) for p in sorted(deps)}
    native = config.read_yaml(_file(root, reference["profile"]))
    if (
        native.get("apiVersion") != "llm-d.ai/v1"
        or native.get("kind") != "EndpointPickerConfig"
    ):
        raise ValueError(
            "reference profile must be a native llm-d EndpointPickerConfig"
        )
    site = _relative(env["site"]) if env.get("site") else None
    if site is not None:
        site_data = config.read_yaml(_file(root, site))
        config.fields(site_data, {"images", "cluster"}, set(), "cluster site")
        cluster = site_data.get("cluster", cfg.get("cluster", {}))
        if not isinstance(cluster, dict):
            raise ValueError("site cluster must be a mapping")
        selector = cluster.get("nodeSelector")
        if (
            not isinstance(selector, dict)
            or not selector
            or not all(isinstance(value, str) for value in selector.values())
        ):
            raise ValueError("cluster site requires an approved nodeSelector")
        if any(ep["backend"] == "rbln" and ep.get("peers") for ep in cfg["endpoints"]):
            visibility = cluster.get("peerGpuVisibility")
            if not isinstance(visibility, str) or visibility.strip() in ("", "all"):
                raise ValueError(
                    "cluster site requires producer GPU IDs for RBLN peers"
                )
    return {
        "id": name,
        "path": path,
        "sha256": inputs[path],
        "inputs": inputs,
        "environment": entry["environment"],
        "target": env["target"],
        "site": site,
        "reference": reference,
        "requirements": required,
        "settings": {
            key: env[key]
            for key in (
                "context_env",
                "namespace_env",
                "image_record_env",
                "timeout",
                "cleanup_timeout",
                "collection_timeout",
                "transfer_evidence",
            )
            if key in env
        },
        "reasons": reasons,
    }


def _load_experiment(root, path, deps):
    # config.load follows model/profile references itself. Check containment first.
    for dep, roles in deps.items():
        if {"model", "profile"}.intersection(roles):
            _file(root, dep)
    return config.load(root / path)


def _registration_reasons(previous, entry, name, suite):
    old = previous.get("experiments", {}).get(name)
    if old is None:
        return ["suite registration added"]
    reasons = []
    if entry != old:
        reasons.append("suite registration changed")
    env = entry["environment"]
    if previous.get("environments", {}).get(env) != suite["environments"].get(env):
        reasons.append(f"suite environment {env} changed")
    if _reference(suite, entry) != _reference(previous, old):
        reasons.append("suite reference changed")
    return reasons


def plan(
    base, head, suite, manual=False, experiment=None, comparison="direct", *, root=None
):
    root = _root(root)
    head = _commit(root, head)
    source = source_identity(root, head)
    suite_path = Path(suite)
    if suite_path.is_absolute():
        try:
            suite_path = suite_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("suite must be inside checkout") from exc
    suite_path = _relative(str(suite_path))
    data = _suite(root, suite_path)
    if manual != (experiment is not None):
        raise ValueError("manual selection requires --manual and --experiment together")
    if manual and experiment not in data["experiments"]:
        raise ValueError(f"unknown experiment {experiment!r}")
    if comparison not in ("direct", "merge-base"):
        raise ValueError("comparison requires direct or merge-base")
    if base is None and not manual:
        raise ValueError(
            "automatic planning requires a base commit; fetch and pass the base SHA"
        )
    requested_base = _commit(root, base) if base is not None else None
    base = requested_base
    if base and comparison == "merge-base":
        try:
            base = _git(root, "merge-base", base, head).decode().strip()
        except ValueError as exc:
            raise ValueError(
                "cannot resolve merge-base; fetch the branch histories"
            ) from exc
    changes = _changes(root, base, head) if base else []
    paths = {p for row in changes for p in (row["path"], row.get("old_path")) if p}
    deleted = {
        row.get("old_path", row["path"])
        for row in changes
        if row["status"].startswith(("D", "R"))
    }
    suite_changed = suite_path in paths
    previous = _suite_at(root, base, suite_path) if suite_changed else {}
    targets, excluded, blocked = [], [], []
    requested = False
    checks = (
        manual
        or suite_changed
        or any(
            _code(p)
            or p.startswith("experiments/models/")
            and Path(p).suffix in {".yaml", ".yml"}
            for p in paths
        )
    )
    registered = set()

    for name, entry in data["experiments"].items():
        reasons = []
        try:
            path = _registration(entry)
            registered.add(path)
            if manual and name != experiment:
                continue
            if manual:
                reasons.append("manual selection")
            if not entry.get("enabled", True):
                if manual:
                    raise ValueError("requested registration is disabled")
                excluded.append(
                    {"id": name, "path": path, "reason": "registration disabled"}
                )
                continue
            if path in deleted and not (root / path).exists() and not manual:
                excluded.append(
                    {
                        "id": name,
                        "path": path,
                        "reason": "experiment deleted or renamed",
                    }
                )
                checks = True
                continue
            if suite_changed:
                reasons.extend(_registration_reasons(previous, entry, name, data))
            if path in paths:
                reasons.append(f"experiment changed: {path}")
            ref = _reference(data, entry)
            env = _settings(data, entry)
            raw = config.read_yaml(_file(root, path))
            deps = _dependencies(path, raw, ref, env)
            for dep in sorted(paths.intersection(deps)):
                reasons.extend(
                    f"{role} changed: {dep}"
                    for role in deps[dep]
                    if role != "experiment"
                )
            # Validate all enabled head experiments, including a surviving dangling
            # model/profile reference after a deletion in this diff.
            cfg = _load_experiment(root, path, deps)
            if reasons:
                targets.append(
                    _target(root, name, entry, path, cfg, ref, env, deps, reasons)
                )
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            blocked.append(
                {
                    "id": name,
                    "path": entry.get("path") if isinstance(entry, dict) else None,
                    "environment": entry.get("environment")
                    if isinstance(entry, dict)
                    else None,
                    "reason": str(exc).replace(str(root) + "/", ""),
                    "reasons": reasons,
                }
            )
            checks = True
        requested |= bool(reasons)

    for name, old in previous.get("experiments", {}).items():
        if name not in data["experiments"]:
            path = _registration(old)
            excluded.append(
                {
                    "id": name,
                    "path": path,
                    "reason": "experiment deleted"
                    if path in deleted
                    else "registration removed",
                }
            )
            registered.add(path)
    for path in sorted(paths - registered):
        if not re.fullmatch(r"experiments/[^/]+\.ya?ml", path):
            continue
        excluded.append(
            {
                "id": None,
                "path": path,
                "reason": "experiment deleted"
                if path in deleted
                else "experiment not registered",
            }
        )
        checks = True
    return {
        "schema_version": 1,
        "base_sha": base,
        "requested_base_sha": requested_base,
        "head_sha": head,
        "comparison": comparison if base else "manual",
        "source": source,
        "changes": changes,
        "suite": suite_path,
        "suite_sha256": _digest(root / suite_path),
        "checks_required": bool(checks or requested),
        "hardware_requested": requested,
        "targets": targets,
        "excluded": excluded,
        "blocked": blocked,
        "matrix": {
            "include": [
                {"experiment": t["id"], "environment": t["environment"]}
                for t in targets
            ]
        },
    }


def verify_checkout(plan, root=None):
    root = _root(root)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported plan schema")
    source = source_identity(root, plan["head_sha"])
    if source["sha256"] != plan["source"]["sha256"]:
        raise ValueError("runtime source hash differs from plan")
    path = _file(root, plan["suite"])
    if _digest(path) != plan["suite_sha256"]:
        raise ValueError("suite hash differs from plan")
    suite = _suite(root, plan["suite"])
    seen = set()
    for target in plan["targets"]:
        name = target["id"]
        if name in seen or name not in suite["experiments"]:
            raise ValueError(f"duplicate or unknown planned experiment {name!r}")
        seen.add(name)
        entry = suite["experiments"][name]
        exp_path = _registration(entry)
        if not entry.get("enabled", True):
            raise ValueError(f"{name}: registration disabled")
        ref, env = _reference(suite, entry), _settings(suite, entry)
        raw = config.read_yaml(_file(root, exp_path))
        deps = _dependencies(exp_path, raw, ref, env)
        cfg = _load_experiment(root, exp_path, deps)
        expected = _target(
            root, name, entry, exp_path, cfg, ref, env, deps, target["reasons"]
        )
        if target != expected:
            raise ValueError(
                f"{name}: execution configuration or input hash differs from plan"
            )
    return root


def write_outputs(plan, output, github_output=None):
    path = Path(output).absolute()
    if github_output and any(c in str(path) for c in "\r\n"):
        raise ValueError("GitHub output path must fit on one line")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(plan, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    if github_output:
        with Path(github_output).open("a") as f:
            for key in ("checks_required", "hardware_requested", "matrix"):
                print(key + "=" + json.dumps(plan[key], separators=(",", ":")), file=f)
            print("has_runnable=" + json.dumps(bool(plan["targets"])), file=f)
            print("blocked=" + json.dumps(bool(plan["blocked"])), file=f)
            print("plan=" + str(path), file=f)
