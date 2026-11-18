"""Summarize existing verifier evidence and package approved diagnostic files."""

from __future__ import annotations

import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

from llmd.verify import reference_output

from .checks import save


def _json(path):
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _verify_exit(phase, report):
    return bool(
        report.get("run_id")
        and phase.get("run_id") == report["run_id"]
        and any(
            c.get("action") == "verify" and c.get("returncode") == 0
            for c in phase.get("commands", [])
        )
    )


def summarize(out, journal):
    phases = journal.get("phases", {})
    ref_phase, target_phase = (phases.get(p, {}) for p in ("reference", "target"))
    ref_path = out / "reference/verification.json"
    ref, target = _json(ref_path), _json(out / "target/verification.json")
    original = journal.get("status", "failed")
    state = dict(
        schema_version=1,
        kind="hardware",
        experiment=journal.get("experiment"),
        environment=journal.get("environment"),
        source=journal.get("source"),
        hardware="not_run",
        status=original,
        reference="not_run",
        smoke="not_run",
        transfer="not_run",
        correctness="not_run",
        overlap="not_run",
        collection={k: v.get("collection", {}) for k, v in phases.items()},
        cleanup={k: v.get("cleanup", {}) for k, v in phases.items()},
    )
    errors = []
    if journal.get("error"):
        errors.append(str(journal["error"]))
    if any(p.get("start_attempted") for p in phases.values()):
        state["hardware"] = "attempted"
    valid_reference = False
    if ref or ref_phase.get("start_attempted"):
        state["reference"] = "failed"
        if _verify_exit(ref_phase, ref):
            try:
                reference_output(
                    ref_path,
                    ref["model"],
                    target.get("run_id"),
                    [r["prompt_ids"] for r in ref["requests"]],
                    ref["conditions"],
                )
                valid_reference = True
                state["reference"] = "passed"
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
    rows = target.get("requests", [])
    if target:
        state.update(
            smoke=target.get("smoke", "not_run"),
            transfer=target.get("transfer", "not_run"),
            correctness=target.get("correctness", "not_run"),
        )
        state["observed_paths"] = target.get("observed_paths", [])
    compared = False
    if (
        valid_reference
        and _verify_exit(target_phase, target)
        and target.get("status") == "verified"
    ):
        try:
            reference_output(
                ref_path,
                target["model"],
                target["run_id"],
                [r["prompt_ids"] for r in rows],
                target["conditions"],
            )
            compared = bool(
                rows
                and target.get("reference")
                and Path(target["reference"]).parts[-2:]
                == ("reference", "verification.json")
                and any(
                    c.get("action") == "verify"
                    and ["--reference", target["reference"]] == c.get("argv", [])[-2:]
                    for c in target_phase.get("commands", [])
                )
                and all(
                    isinstance(r.get("token_ids"), list)
                    and r["token_ids"]
                    and not r.get("error")
                    for r in rows
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if state["correctness"] == "passed" and not compared:
        state["correctness"] = "unconfirmed"
    routes = {r.get("path") for r in rows}
    plan = _json(out / "plan.json")
    entry = next(
        (t for t in plan.get("targets", []) if t["id"] == state["experiment"]), {}
    )
    aggregate = entry.get("settings", {}).get("transfer_evidence") == "aggregate"
    if "pdd" in routes:
        transfer_ok = state["transfer"] == "passed" and all(
            r.get("worker_import") for r in rows if r["path"] == "pdd"
        )
        transfer_ok |= aggregate and state["transfer"] == "aggregate_observed"
    else:
        transfer_ok = routes == {"full"} and state["transfer"] == "not_applicable"
    if state["transfer"] == "passed" and not transfer_ok:
        state["transfer"] = "unconfirmed"
    phase_ok = all(
        p.get("status") == "completed"
        and p.get("cleanup", {}).get("status") == "cleaned"
        and p.get("collection", {}).get("status") in ("collected", "partial")
        for p in (ref_phase, target_phase)
    )
    if original == "completed":
        if (
            phase_ok
            and compared
            and state["correctness"] == state["smoke"] == "passed"
            and transfer_ok
        ):
            state.update(status="passed", hardware="verified")
        else:
            state["status"] = "failed"
            errors.append(
                "Missing required verifier, transfer, collection or cleanup evidence"
            )
    elif original not in ("cancelled", "blocked", "failed"):
        state["status"] = "incomplete"
    probe = _json(out / "target/overlap-verification.json")
    if (
        probe.get("run_id")
        and probe["run_id"] == target.get("run_id")
        and probe.get("proof")
    ):
        state["overlap"] = probe.get("status", "not_run")
    state["errors"] = errors
    return state


ROOT_FILES = {
    "plan.json",
    "lifecycle.json",
    "result.json",
    "summary.md",
    "build-record.json",
    "preflight.json",
}
NATIVE_FILES = {
    "verification.json",
    "run.json",
    "kube-run.json",
    "deployment.json",
    "engine.json",
    "collection.json",
    "endpoints.yaml",
    "kubernetes.yaml",
    "epp.yaml",
    "envoy.yaml",
    "overlap-verification.json",
    "protocol-results.json",
    "policy-results.json",
    "requests.json",
    "preflight.json",
}


def _candidates(out):
    for name in sorted(ROOT_FILES):
        yield out / name
    yield from (out / "logs").glob("*.log")
    for area in ("reference", "target", "policy", "protocol"):
        base = out / area
        dirs = [base]
        if area == "policy" and base.is_dir():
            dirs += [p for p in base.iterdir() if p.is_dir() and not p.is_symlink()]
        for folder in dirs:
            for p in folder.glob("*"):
                if (
                    p.name in NATIVE_FILES
                    or p.suffix in (".log", ".prom")
                    or p.name.endswith("-admission.jsonl")
                ):
                    yield p
            yield from folder.glob("events/*/events_*.jsonl")
            yield from folder.glob("metrics/*.prom")
            for pattern in (
                "index.json",
                "*/index.json",
                "*/pods.json",
                "*/*/engine.json",
                "*/*/events_*.jsonl",
                "*/*/admission.jsonl",
                "*/*/*.log",
                "*/*/*.metrics",
            ):
                yield from (folder / "diagnostics").glob(pattern)
    for name in ("target.yaml", "reference.yaml", "site.yaml"):
        yield out / "configs" / name


PRIVATE = re.compile(
    r"(?:password|credential|authorization|api[_-]?key|access[_-]?token|(?:^|[_-])token$|secret|[_-]headers$)",
    re.I,
)
TOKEN = re.compile(
    rb"(?i)(bearer\s+)[\w.+/=-]+|\b(?:gh[pousr]_[a-zA-Z0-9_]+|github_pat_[a-zA-Z0-9_]+|hf_[a-zA-Z0-9]{16,})\b"
)
LIMIT = 16 * 1024 * 1024


def _scrub(obj):
    if isinstance(obj, dict):
        if (
            isinstance(obj.get("name"), str)
            and PRIVATE.search(obj["name"])
            and "value" in obj
        ):
            obj = {**obj, "value": "[redacted]"}
        return {
            k: ("[redacted]" if PRIVATE.search(k) else _scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def publish(out, result):
    """Package bounded diagnostics for upload."""
    lines = ["# llm-d " + result["kind"], "", f"**{result['status']}**", ""]
    for key in (
        "experiment",
        "fixture",
        "hardware",
        "reference",
        "smoke",
        "transfer",
        "correctness",
        "overlap",
        "collection",
        "cleanup",
    ):
        if key in result:
            value = result[key]
            if isinstance(value, dict):
                value = ", ".join(
                    f"{k}={v.get('status', 'not_run')}" for k, v in value.items()
                )
            lines.append(f"- {key} · {value}")
    source = result.get("source") or {}
    lines.append(
        f"- checkout · {source.get('head_sha', 'unknown')} "
        f"(dirty={source.get('dirty', 'unknown')})"
    )
    for error in result.get("errors", []) + (
        [result["error"]] if result.get("error") else []
    ):
        lines.append(f"- {error}")
    for c in result.get("commands", []):
        lines.append(f"- {c['log']} · exit {c['returncode']}")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    expected = []
    if result["kind"] == "hardware":
        expected = [f"{phase}/verification.json" for phase in ("reference", "target")]
    elif result["kind"] == "process":
        expected = ["protocol/protocol-results.json", "policy/policy-results.json"]
    files, missing = [], []
    with tempfile.NamedTemporaryFile(dir=out) as tmp:
        with tarfile.open(fileobj=tmp, mode="w:gz", compresslevel=1) as tar:

            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            for path in sorted(set(_candidates(out))):
                name = path.relative_to(out).as_posix()
                if (
                    not path.is_file()
                    or path.resolve() != path.absolute()
                    or PRIVATE.search(path.name)
                    or path.name.startswith(".")
                ):
                    continue
                size = path.stat().st_size
                if size > LIMIT and path.suffix not in (
                    ".jsonl",
                    ".log",
                    ".prom",
                    ".txt",
                ):
                    missing.append(
                        dict(path=name, reason="exceeds artifact size limit")
                    )
                    continue
                with path.open("rb") as f:
                    if size > LIMIT:
                        f.seek(-LIMIT, os.SEEK_END)
                    data = f.read(LIMIT)
                if path.suffix == ".json":
                    try:
                        data = (
                            json.dumps(_scrub(json.loads(data)), indent=2) + "\n"
                        ).encode()
                    except (ValueError, UnicodeError):
                        missing.append(dict(path=name, reason="unreadable JSON"))
                        continue
                data = TOKEN.sub(lambda m: (m.group(1) or b"") + b"[redacted]", data)
                add(name, data)
                files.append(dict(path=name, bytes=len(data), truncated=size > LIMIT))
            seen = {f["path"] for f in files + missing}
            for name in expected:
                if name not in seen:
                    missing.append(
                        dict(
                            path=name,
                            reason="not produced; see lifecycle or command log",
                        )
                    )
            inventory = dict(schema_version=1, files=files, missing=missing)
            save(out / "artifacts.json", inventory)
            add("artifacts.json", (out / "artifacts.json").read_bytes())
        tmp.flush()
        # Keep the old complete archive visible until the new archive is complete.
        staged = str(tmp.name) + ".ready"
        os.link(tmp.name, staged)
        os.replace(staged, out / "artifacts.tar.gz")


def write_results(out):
    out = Path(out).resolve()
    journal = json.loads((out / "lifecycle.json").read_text())
    result = summarize(out, journal)
    save(out / "result.json", result)
    publish(out, result)
    return result
