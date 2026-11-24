"""Check an llm-d experiment against a GPU answer key and profile it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

import yaml

from llmd import config, engines
from tools.hetero.profiler import analysis
from tools.hetero.profiler.common import live_rows
from tools.hetero.profiler.pdd_summary import folded, key_results, metric_table, table

ROOT = Path(__file__).resolve().parents[3]
REFERENCES = (
    ROOT / "experiments/qwen3-baseline.yaml",
    ROOT / "experiments/qwen2-baseline.yaml",
)
PERFORMANCE_BASELINE = ROOT / "experiments/baseline-hetero.yaml"
SEED = 20260924

METRICS = (
    ("service_ttft_mean_seconds", "Mean TTFT", "ms"),
    ("service_ttft_p95_seconds", "P95 TTFT", "ms"),
    ("tpot_mean_seconds", "Mean TPOT", "ms"),
    ("service_mean_seconds", "Mean E2E", "ms"),
    ("service_p95_seconds", "P95 E2E", "ms"),
    ("output_tokens_per_second", "Output throughput", "tokens/s"),
    ("completed_requests_per_second", "Request throughput", "requests/s"),
)


def reference_yaml(cfg: dict, destination: Path) -> Path | None:
    source = next(
        (path for path in REFERENCES if config.load(path)["model"] == cfg["model"]),
        None,
    )
    if source is None:
        return None
    raw = config.read_yaml(source)
    raw["model"] = cfg["model"]
    raw["profile"] = str((source.parent / raw["profile"]).resolve())
    if "benchmark" in cfg:
        gpu = next(e for e in cfg["endpoints"] if e["backend"] == "cuda")
        raw["endpoints"] = [
            {k: gpu[k] for k in ("id", "backend", "devices", "port", "host", "engine")}
            | {"roles": ["full"]}
        ]
        for key in ("hub_cache", "cache_root", "ports", "host"):
            if key in cfg:
                raw[key] = cfg[key]
    raw["verify"] = {**raw["verify"], **cfg["verify"], "paths": ["full"]}
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    config.load(destination)
    return source


def performance_baseline_yaml(cfg: dict, destination: Path) -> bool:
    if config.load(PERFORMANCE_BASELINE)["model"] != cfg["model"]:
        return False
    raw = config.read_yaml(PERFORMANCE_BASELINE)
    raw["model"] = str((PERFORMANCE_BASELINE.parent / raw["model"]).resolve())
    raw["profile"] = str((PERFORMANCE_BASELINE.parent / raw["profile"]).resolve())
    raw["verify"] = {**raw["verify"], **cfg["verify"], "paths": ["full"]}
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    config.load(destination)
    return True


def download_model(cfg: dict) -> None:
    from huggingface_hub import snapshot_download

    model = cfg["model"]
    cache_dir = cfg.get("hub_cache") or os.environ.get("HF_HUB_CACHE")
    resolved = engines.model_info(model, cfg)
    seen = set()
    for path_key, name_key, revision_key in (
        ("path", "name", "revision"),
        ("tokenizer_path", "tokenizer", "tokenizer_revision"),
    ):
        identity = (model[name_key], model[revision_key])
        if Path(resolved[path_key]).is_dir() or identity in seen:
            continue
        seen.add(identity)
        snapshot_download(identity[0], revision=identity[1], cache_dir=cache_dir)


class StepError(RuntimeError):
    pass


class Steps(list):
    """Record each CI step with its status, duration and failure reason."""

    def run(self, name, fn, *args):
        entry = {"step": name, "status": "RUNNING", "seconds": None, "reason": None}
        self.append(entry)
        started = time.monotonic()
        try:
            result = fn(*args)
        except BaseException as exc:
            entry["status"] = "FAILED"
            entry["reason"] = str(exc) or type(exc).__name__
            raise
        finally:
            entry["seconds"] = round(time.monotonic() - started, 1)
        entry["status"] = "PASS"
        return result


def failure_reason(lines) -> str | None:
    for line in reversed(lines):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    return next((line for line in reversed(lines) if line.strip()), None)


def command(*args: str) -> None:
    tail = deque(maxlen=20)
    with subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, stderr=subprocess.PIPE, text=True
    ) as proc:
        for line in proc.stderr:
            sys.stderr.write(line)
            tail.append(line.rstrip())
    if proc.returncode:
        reason = failure_reason(tail) or f"exit code {proc.returncode}"
        raise StepError(reason[:300])


def workload_config(cfg: dict) -> dict:
    if "benchmark" in cfg:
        return {
            k: v
            for k, v in cfg["benchmark"].items()
            if k not in {"comparisons", "require"}
        }
    verify = cfg["verify"]
    max_len = min(ep["engine"]["max_model_len"] for ep in cfg["endpoints"])
    output_tokens = min(
        max(32, verify["output_tokens"]), max_len - verify["input_tokens"]
    )
    if output_tokens < 2:
        raise ValueError("profiling needs room for at least two output tokens")
    return {
        "input_tokens": verify["input_tokens"],
        "output_tokens": output_tokens,
        "requests": max(20, verify["requests"]),
        "concurrency": verify["concurrency"],
    }


def benchmark_args(cfg: dict, work: Path, name: str, workload: dict) -> list[str]:
    model = engines.model_info(cfg["model"], cfg)
    endpoint = f"http://{cfg['host']}:{cfg['ports']['gateway']}"
    return [
        "-m",
        "tools.hetero.loadgen",
        "--endpoint",
        endpoint,
        "--target",
        name,
        "--concurrency",
        str(workload["concurrency"]),
        "--model",
        model["name"],
        "--tokenizer",
        model["tokenizer_path"],
        "--input-tokens",
        str(workload["input_tokens"]),
        "--output-tokens",
        str(workload["output_tokens"]),
        "--requests",
        str(workload["requests"]),
        "--warmup-waves" if "warmup" in workload else "--warmup-per-endpoint",
        str(workload.get("warmup", 2)),
        "--seed",
        str(workload.get("seed", SEED)),
        "--workload",
        "experiment-comparison",
        "--run-dir",
        str(work / "profile"),
    ]


def run_service(
    experiment: Path,
    root: Path,
    name: str,
    reference: Path | None,
    steps: Steps,
    workload: dict | None = None,
) -> None:
    cfg = config.load(experiment)
    work = root / name
    work.mkdir()
    failed = True
    try:
        steps.run(
            f"{name}: start",
            command,
            "-m",
            "llmd",
            "start",
            str(experiment),
            "--work-dir",
            str(work),
        )
        verify = ["-m", "llmd", "verify", "--work-dir", str(work)]
        if reference:
            verify += ["--reference", str(reference)]
        steps.run(f"{name}: verify", command, *verify)
        if workload is not None:
            steps.run(
                f"{name}: load", command, *benchmark_args(cfg, work, name, workload)
            )
        failed = False
    finally:
        if (work / "run.json").exists():
            try:
                steps.run(
                    f"{name}: stop",
                    command,
                    "-m",
                    "llmd",
                    "stop",
                    "--work-dir",
                    str(work),
                )
            except StepError:
                if not failed:
                    raise
            else:
                offload = work / "offload"
                if offload.is_dir():
                    shutil.rmtree(offload)


def profile(work: Path, benchmark=False) -> dict:
    folder = work / "profile"
    rows = list(live_rows(folder / "requests.jsonl"))
    workload = json.loads((folder / "config.json").read_text())
    if len(rows) != workload["requests"]:
        raise RuntimeError(f"{work.name}: missing profiling requests")
    state = json.loads((work / "run.json").read_text())
    measured = analysis.client_metrics(
        rows, sum(len(e["devices"]) for e in state["endpoints"])
    )
    (folder / "client_metrics.json").write_text(json.dumps(measured, indent=2) + "\n")
    result = measured["summary"]
    if result["success_count"]["value"] != workload["requests"]:
        raise RuntimeError(f"{work.name}: profiling requests failed")
    metrics = {}
    for key, _, _ in METRICS:
        value = result[key]["value"]
        if value is None or value <= 0:
            raise RuntimeError(f"{work.name}: missing profiling metric {key}")
        metrics[key] = value
    data = {"workload": workload, "metrics": metrics}
    if benchmark:
        rows.sort(key=lambda r: r["request_index"])
        ids = [r["streamed_token_ids"] or [] for r in rows]
        if (
            [r["request_index"] for r in rows] != list(range(workload["requests"]))
            or any(r["actual_input_tokens"] != workload["input_tokens"] for r in rows)
            or any(len(tokens) != workload["output_tokens"] for tokens in ids)
        ):
            raise StepError(
                f"{work.name}: incorrect measured token count or request IDs"
            )
        data["output_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return data


def arm_yaml(raw, cfg, root, name, profile_path):
    arm = dict(raw, model=cfg["model"], profile=str(profile_path))
    arm.pop("benchmark", None)
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(arm, sort_keys=False))
    config.load(path)
    return path


def fixed_pdd_checks(experiment, cfg, root, answers, steps, result):
    raw = config.read_yaml(experiment)
    raw["verify"] = dict(cfg["verify"], paths=["pdd"])
    endpoints = raw["endpoints"]
    decoders = [
        e["id"] for e in endpoints if e["backend"] == "rbln" and "decode" in e["roles"]
    ]
    if len(decoders) != 2:
        raise ValueError("fixed PDD checks require exactly two RBLN decode endpoints")
    rows = []
    for targets in ([decoders[0]], [decoders[1]], decoders):
        name = "fixed-" + "-".join(targets)
        raw["endpoints"] = [
            e for e in endpoints if e["id"] not in decoders or e["id"] in targets
        ]
        native_file = ROOT / "llmd/epp/profiles/fixed-pdd.yaml"
        if len(targets) == 2:
            # trace-allocation invokes fixedPair with the request's pinned consumer.
            native = config.read_yaml(ROOT / "llmd/epp/profiles/trace-allocation.yaml")
            handler = next(
                p for p in native["plugins"] if p["type"] == "trace-allocation-handler"
            )
            handler["parameters"] = {
                "allocation": {str(i): target for i, target in enumerate(targets)},
                "pddTargets": targets,
            }
            native_file = root / "fixed-pairs-profile.yaml"
            native_file.write_text(yaml.safe_dump(native, sort_keys=False))
        path = arm_yaml(raw, cfg, root, name, native_file)
        run_service(path, root, name, answers, steps)
        work = root / name
        verification = json.loads((work / "verification.json").read_text())
        executed = Counter(
            r["target"]
            for r in verification["requests"]
            if r["path"] == "pdd" and r.get("worker_import")
        )
        contexts = json.loads((work / "rbln-smi-ready.json").read_text())["contexts"]
        active = {int(c["npu"]) for c in contexts}
        for ep in raw["endpoints"]:
            if ep["id"] in targets:
                rows.append(
                    (
                        name,
                        ep["id"],
                        ep["devices"],
                        executed[ep["id"]],
                        verification["correctness"],
                    )
                )
        result["lines"] = ["", "## Fixed PDD verification"] + table(
            ("Run", "Decoder", "NPU", "Imported requests", "Correctness"), rows
        )
        if (
            set(executed) != set(targets)
            or verification["transfer"] != "passed"
            or verification["correctness"] != "passed"
            or any(
                not set(e["devices"]) <= active
                for e in raw["endpoints"]
                if e["id"] in targets
            )
        ):
            raise StepError(
                f"{name}: missing PDD, physical device context or GPU token agreement"
            )


def require_routes(evidence, required, endpoints):
    unknown = evidence["executed_paths"].get("unverified", 0)
    if unknown:
        raise StepError(f"route evidence missing for {unknown} measured request(s)")
    if not required:
        return
    decoders = [
        e["id"] for e in endpoints if e["backend"] == "rbln" and "decode" in e["roles"]
    ]
    if not decoders:
        raise StepError(f"{required} verification has no RBLN decode endpoints")
    for decoder in decoders:
        row = evidence["per_endpoint"].get(decoder, {})
        if not all(
            row.get(k, {}).get("pdd") for k in ("requested_paths", "executed_paths")
        ):
            raise StepError(f"{decoder}: required PDD selection or execution missing")
        if required == "overlap" and not all(
            row.get(k)
            for k in ("preload_selected", "preload_approved", "actual_overlap")
        ):
            raise StepError(
                f"{decoder}: required preload selection, approval or overlap missing"
            )


def benchmark(experiment, cfg, root, answers, steps, workload, stage, result):
    from tools.hetero.profiler.route_evidence import analyze_routes

    raw = config.read_yaml(experiment)
    raw["verify"].pop("paths", None)
    required = cfg["benchmark"].get("require")
    arms = dict(cfg["benchmark"]["comparisons"])
    candidate = cfg["profile"]
    if stage in {"smoke", "overlap"}:
        native = config.read_yaml(candidate)
        if not any(
            p["type"] == "cost-route-handler" and p["parameters"].get("preload")
            for p in native["plugins"]
        ):
            raise ValueError("overlap stage requires a preload-enabled cost profile")
        native["schedulingProfiles"] = [
            p for p in native["schedulingProfiles"] if p["name"] != "full"
        ]
        candidate = root / "pdd-only-overlap.yaml"
        candidate.write_text(yaml.safe_dump(native, sort_keys=False))
        arms, required = {}, "overlap"
        workload["repeats"] = 1
        if stage == "smoke":
            workload["requests"] = min(
                workload["requests"], max(16, workload["concurrency"])
            )
    arms["candidate"] = candidate
    paths = {name: arm_yaml(raw, cfg, root, name, path) for name, path in arms.items()}
    # Reuse the same GPU reference engine for measured token agreement when
    # the experiment doesn't already define a matched GPU performance arm.
    if "gpu-only" not in paths:
        paths = {"gpu-only": root / "answer-reference.yaml", **paths}
    runs = []
    data = {
        "commit": os.environ.get("GITHUB_SHA"),
        "stage": stage,
        "require": required,
        "runs": runs,
    }
    summary = root / "benchmark-summary.json"
    for repeat in range(workload["repeats"]):
        directory = root / f"repeat-{repeat + 1:02d}"
        directory.mkdir()
        names = list(paths)
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            work = directory / name
            tag = str(work.relative_to(root))
            run = {"arm": name, "repeat": repeat + 1}
            runs.append(run)
            try:
                run_service(paths[name], directory, name, answers, steps, workload)
                run["route_evidence"] = steps.run(
                    f"{tag}: route evidence", analyze_routes, work
                )
                run.update(steps.run(f"{tag}: measurement", profile, work, True))
                steps.run(
                    f"{tag}: required routes",
                    require_routes,
                    run["route_evidence"],
                    required if name == "candidate" else None,
                    cfg["endpoints"],
                )
            except (OSError, ValueError, RuntimeError) as exc:
                run["error"] = str(exc)
            summary.write_text(json.dumps(data, indent=2) + "\n")
    data["improvement_percent"] = benchmark_results(runs, paths, result["lines"])
    summary.write_text(json.dumps(data, indent=2) + "\n")
    if errors := [
        f"{r['arm']} repeat {r['repeat']}: {r['error']}" for r in runs if "error" in r
    ]:
        raise StepError("; ".join(errors))


def benchmark_results(runs, arms, lines):
    # Match measured outputs to the actual GPU-only arm at the same repeat.
    # Every arm also verified its shorter deterministic prompts against the GPU.
    references = {
        r["repeat"]: r for r in runs if r["arm"] == "gpu-only" and "error" not in r
    }
    valid = {name: {} for name in arms}
    routes = []
    for r in runs:
        name, repeat = r["arm"], r["repeat"]
        for ep, stats in r.get("route_evidence", {}).get("per_endpoint", {}).items():
            actual = stats["executed_paths"]
            count = sum(actual.values())
            share = 100 * actual.get("pdd", 0) / count if count else 0
            routes.append(
                (
                    repeat,
                    name,
                    ep,
                    actual.get("gpu_full", 0),
                    actual.get("npu_full", 0),
                    stats["requested_paths"].get("pdd", 0),
                    actual.get("pdd", 0),
                    f"{share:.1f}%",
                    stats["preload_selected"],
                    stats["preload_approved"],
                    stats["actual_overlap"],
                )
            )
        if "error" in r:
            continue
        ref = references.get(repeat)
        if not ref or ref["route_evidence"]["executed_paths"] != {
            "gpu_full": ref["workload"]["requests"]
        }:
            r["error"] = "missing verified GPU-only reference for measured tokens"
            continue
        if (
            ref["output_ids_sha256"] != r["output_ids_sha256"]
            or ref["workload"]["prompts_sha256"] != r["workload"]["prompts_sha256"]
        ):
            r["error"] = "correctness: measured tokens differ from GPU reference"
            continue
        valid[name][repeat] = r["metrics"]
    lines += ["", "## Route evidence by decoder and repeat"] + table(
        (
            "Repeat",
            "Profile",
            "Endpoint",
            "GPU full",
            "NPU full",
            "PDD selected",
            "PDD executed",
            "Endpoint PDD share",
            "Preload selected",
            "Approved",
            "Overlap",
        ),
        routes,
    )
    metrics = []
    gains = {}
    for name, arm in valid.items():
        if not arm:
            continue
        for key, label, unit in METRICS:
            values = [r[key] * (1000 if unit == "ms" else 1) for r in arm.values()]
            metrics.append(
                (
                    name,
                    label,
                    unit,
                    ", ".join(f"{v:.2f}" for v in values),
                    f"{statistics.median(values):.2f}",
                )
            )
            pairs = sorted(arm.keys() & valid["candidate"].keys())
            if name in ("selective", "baseline-hetero") and len(pairs) >= 3:
                sign = -1 if unit == "ms" else 1
                gain = statistics.median(
                    sign * 100 * (valid["candidate"][n][key] / arm[n][key] - 1)
                    for n in pairs
                )
                gains.setdefault(name, {})[label] = gain
    lines += ["", "## Valid measurements"] + table(
        ("Profile", "Metric", "Unit", "Runs", "Median"), metrics
    )
    if gains:
        lines += ["", "## Median paired improvement"] + table(
            ("Control", "Paired repeats", "Metric", "Candidate improvement"),
            [
                (
                    arm,
                    len(valid[arm].keys() & valid["candidate"].keys()),
                    k,
                    f"{v:+.2f}%",
                )
                for arm, values in gains.items()
                for k, v in values.items()
            ],
        )
    return gains


def report(
    root: Path, experiment: Path, reference: Path | None, has_comparison: bool
) -> dict:
    candidate = profile(root / "candidate")
    verification = json.loads((root / "candidate/verification.json").read_text())
    expected = "passed" if reference else "not_evaluated"
    if verification["correctness"] != expected:
        raise RuntimeError("candidate output did not match GPU answer reference")
    baseline = profile(root / "performance-baseline") if has_comparison else None
    if baseline:
        same = (
            "model",
            "tokenizer",
            "input_tokens",
            "output_tokens",
            "requests",
            "seed",
        )
        if any(
            baseline["workload"][key] != candidate["workload"][key] for key in same
        ) or len(baseline["workload"]["slots"]) != len(candidate["workload"]["slots"]):
            raise RuntimeError("performance baseline and candidate workloads differ")
    comparison = []
    for key, label, unit in METRICS:
        current = candidate["metrics"][key]
        factor = 1000 if unit == "ms" else 1
        base = gain = None
        if baseline:
            base = baseline["metrics"][key]
            gain = (
                current / base - 1
                if key.endswith("_per_second")
                else 1 - current / base
            ) * 100
            base *= factor
        comparison.append(
            {
                "metric": label,
                "baseline": base,
                "candidate": current * factor,
                "improvement_percent": gain,
                "unit": unit,
            }
        )
    result = {
        "experiment": str(experiment),
        "answer_reference": str(reference) if reference else None,
        "correctness": verification["correctness"],
        "performance_baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    (root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def client_data(work: Path) -> tuple[dict, dict] | None:
    folder = work / "profile"
    try:
        workload = json.loads((folder / "config.json").read_text())
        saved = folder / "client_metrics.json"
        if saved.exists():
            return json.loads(saved.read_text()), workload
        rows = list(live_rows(folder / "requests.jsonl"))
        return (analysis.client_metrics(rows), workload) if rows else None
    except (OSError, ValueError, KeyError):
        return None


def comparison_lines(result: dict, experiment: Path) -> list[str]:
    rows = result["comparison"]
    if result["performance_baseline"]:
        lines = [
            "",
            "## Comparison with baseline-hetero",
            "",
            "Positive improvement means the algorithm is faster.",
            "",
            "| Metric | Baseline-hetero | Algorithm | Improvement |",
            "| --- | ---: | ---: | ---: |",
        ]
        return lines + [
            f"| {r['metric']} | {r['baseline']:.2f} {r['unit']} | "
            f"{r['candidate']:.2f} {r['unit']} | {r['improvement_percent']:+.1f}% |"
            for r in rows
        ]
    reason = (
        "This experiment is the performance baseline."
        if experiment == PERFORMANCE_BASELINE
        else "No performance baseline exists for this model."
    )
    lines = ["", "## Algorithm results", "", reason, "", "| Metric | Algorithm |"]
    lines.append("| --- | ---: |")
    return lines + [
        f"| {r['metric']} | {r['candidate']:.2f} {r['unit']} |" for r in rows
    ]


def profile_lines(root: Path, name: str, title: str) -> list[str]:
    data = client_data(root / name)
    if data is None:
        return []
    measured, workload = data
    settings = {
        "CONCURRENCY": len(workload.get("slots", [])) or "unknown",
        "MEASURE_REQUESTS": workload.get("requests", "unknown"),
    }
    lines = key_results({"client": measured}, settings)
    lines[lines.index("## Key results")] = f"## Key results: {title}"
    return lines + folded(
        f"{title}: client request details", metric_table(measured["requests"])
    )


def configuration_lines(
    experiment: Path, cfg: dict, workload: dict | None
) -> list[str]:
    def shown(path) -> str:
        path = Path(path)
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    rows = [
        ("Experiment", shown(experiment)),
        ("Model", f"{cfg['model']['name']}@{cfg['model']['revision'][:12]}"),
        ("EPP profile", shown(cfg["profile"])),
    ]
    rows += [
        (
            f"Endpoint {ep['id']}",
            f"{ep['backend']} {'/'.join(ep['roles'])}, devices {ep['devices']}",
        )
        for ep in cfg["endpoints"]
    ]
    if workload:
        rows += [(f"Workload {key}", value) for key, value in workload.items()]
        if "seed" not in workload:
            rows.append(("Workload seed", str(SEED)))
    return ["", "## Saved configuration"] + table(("Setting", "Value"), rows)


def write_summary(
    root: Path,
    experiment: Path,
    cfg: dict,
    steps: Steps,
    reference: Path | None,
    result: dict | None = None,
    workload: dict | None = None,
    error: BaseException | None = None,
) -> str:
    is_reference = experiment in REFERENCES
    if error is not None:
        status = "FAILED"
    elif is_reference:
        status = "PASSED (GPU answer reference smoke test)"
    elif reference is None:
        status = "CORRECTNESS NOT EVALUATED"
    else:
        status = "PASSED"
    lines = [f"# Experiment summary: {experiment.name}", "", f"Result: **{status}**"]
    if error is not None:
        lines += ["", f"Failure: `{type(error).__name__}: {error}`"]
    if is_reference:
        answer = "this experiment"
    elif reference:
        answer = f"`{reference.name}`"
    else:
        answer = f"none for `{cfg['model']['name']}`"
    lines += ["", f"- Answer reference: {answer}"]
    if os.environ.get("GITHUB_SHA"):
        lines.append(f"- Commit: `{os.environ['GITHUB_SHA']}`")
    if result and "lines" in result:
        lines += result["lines"]
    elif result:
        lines += comparison_lines(result, experiment)
    lines += ["", "## Run steps"] + table(
        ("Step", "Status", "Seconds", "Failure reason"),
        [(s["step"], s["status"], s["seconds"], s["reason"] or "") for s in steps],
    )
    lines += profile_lines(root, "candidate", f"Algorithm (`{experiment.name}`)")
    lines += profile_lines(
        root, "performance-baseline", "Performance baseline (`baseline-hetero.yaml`)"
    )
    lines += configuration_lines(experiment, cfg, workload)
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument(
        "--stage",
        choices=("configured", "pdd", "overlap", "smoke"),
        default="configured",
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    root = args.result_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = config.load(experiment)
    steps = Steps()
    reference_source = result = workload = error = None
    try:
        if len({ep["node"] for ep in cfg["endpoints"]}) > 1:
            raise ValueError("multi-node experiment needs Kubernetes profiling")
        if args.stage != "configured" and "benchmark" not in cfg:
            raise ValueError("functional stages require a benchmark experiment")
        if args.stage in {"pdd", "smoke"}:
            cfg["verify"].update(concurrency=2, requests=2)
        steps.run("download model", download_model, cfg)
        if experiment in REFERENCES:
            run_service(experiment, root, "reference", None, steps)
            return
        reference_source = reference_yaml(cfg, root / "answer-reference.yaml")
        answers = root / "reference/verification.json" if reference_source else None
        workload = workload_config(cfg)
        if reference_source:
            run_service(root / "answer-reference.yaml", root, "reference", None, steps)
        if "benchmark" in cfg:
            if not answers:
                raise StepError("no matching GPU answer reference")
            result = {"lines": []}
            if args.stage in {"pdd", "smoke"}:
                fixed_pdd_checks(experiment, cfg, root, answers, steps, result)
            if args.stage != "pdd":
                benchmark(
                    experiment, cfg, root, answers, steps, workload, args.stage, result
                )
            return
        performance_file = root / "performance-baseline.yaml"
        has_comparison = (
            experiment != PERFORMANCE_BASELINE
            and performance_baseline_yaml(cfg, performance_file)
        )
        if has_comparison:
            run_service(
                performance_file, root, "performance-baseline", answers, steps, workload
            )
        run_service(experiment, root, "candidate", answers, steps, workload)
        result = steps.run(
            "compare results",
            report,
            root,
            experiment,
            reference_source,
            has_comparison,
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        print(
            write_summary(
                root, experiment, cfg, steps, reference_source, result, workload, error
            )
        )
    if reference_source is None:
        sys.exit(
            f"{experiment.name}: no GPU answer reference for model "
            f"{cfg['model']['name']}; output correctness was not evaluated"
        )


if __name__ == "__main__":
    main()
