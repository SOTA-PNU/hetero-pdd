"""Check an llm-d experiment against a GPU answer key and profile it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from llmd import config, engines
from tools.hetero.profiler import analysis
from tools.hetero.profiler.common import live_rows

ROOT = Path(__file__).resolve().parents[3]
REFERENCES = (ROOT / "experiments/qwen3-baseline.yaml", ROOT / "experiments/qwen2-baseline.yaml")
PERFORMANCE_BASELINE = ROOT / "experiments/baseline-hetero.yaml"
SEED = 20260924

METRICS = (
    ("service_ttft_mean_seconds", "Mean TTFT", "ms"),
    ("service_ttft_p95_seconds", "P95 TTFT", "ms"),
    ("tpot_mean_seconds", "Mean TPOT", "ms"),
    ("service_mean_seconds", "Mean E2E", "ms"),
    ("service_p95_seconds", "P95 E2E", "ms"),
    ("output_tokens_per_second", "Output throughput", "tokens/s"),
)


def reference_yaml(cfg: dict, destination: Path) -> Path | None:
    source = next(
        (path for path in REFERENCES if config.load(path)["model"] == cfg["model"]),
        None,
    )
    if source is None:
        return None
    raw = config.read_yaml(source)
    raw["model"] = str((source.parent / raw["model"]).resolve())
    raw["profile"] = str((source.parent / raw["profile"]).resolve())
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


def command(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def workload_config(cfg: dict) -> dict:
    verify = cfg["verify"]
    max_len = min(ep["engine"]["max_model_len"] for ep in cfg["endpoints"])
    output_tokens = min(max(32, verify["output_tokens"]), max_len - verify["input_tokens"])
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
        "-m", "tools.hetero.loadgen",
        "--endpoint", endpoint,
        "--target", name,
        "--concurrency", str(workload["concurrency"]),
        "--model", model["name"],
        "--tokenizer", model["tokenizer_path"],
        "--input-tokens", str(workload["input_tokens"]),
        "--output-tokens", str(workload["output_tokens"]),
        "--requests", str(workload["requests"]),
        "--warmup-per-endpoint", "2",
        "--seed", str(SEED),
        "--workload", "experiment-comparison",
        "--run-dir", str(work / "profile"),
    ]


def run_service(
    experiment: Path,
    root: Path,
    name: str,
    reference: Path | None,
    workload: dict | None = None,
) -> None:
    cfg = config.load(experiment)
    work = root / name
    work.mkdir()
    try:
        command("-m", "llmd", "start", str(experiment), "--work-dir", str(work))
        verify = ["-m", "llmd", "verify", "--work-dir", str(work)]
        if reference:
            verify += ["--reference", str(reference)]
        command(*verify)
        if workload is not None:
            command(*benchmark_args(cfg, work, name, workload))
    finally:
        if (work / "run.json").exists():
            stop = subprocess.run(
                [sys.executable, "-m", "llmd", "stop", "--work-dir", str(work)],
                check=False,
                cwd=ROOT,
            )
            if stop.returncode and sys.exc_info()[0] is None:
                raise RuntimeError(f"failed to stop {name} experiment")


def profile(work: Path) -> dict:
    folder = work / "profile"
    rows = list(live_rows(folder / "requests.jsonl"))
    workload = json.loads((folder / "config.json").read_text())
    if len(rows) != workload["requests"]:
        raise RuntimeError(f"{work.name}: missing profiling requests")
    measured = analysis.client_metrics(rows)
    (folder / "client_metrics.json").write_text(
        json.dumps(measured, indent=2) + "\n"
    )
    result = measured["summary"]
    if result["success_count"]["value"] != workload["requests"]:
        raise RuntimeError(f"{work.name}: profiling requests failed")
    metrics = {}
    for key, _, _ in METRICS:
        value = result[key]["value"]
        if value is None or value <= 0:
            raise RuntimeError(f"{work.name}: missing profiling metric {key}")
        metrics[key] = value
    return {"workload": workload, "metrics": metrics}


def report(
    root: Path, experiment: Path, reference: Path | None, has_comparison: bool
) -> None:
    candidate = profile(root / "candidate")
    verification = json.loads((root / "candidate/verification.json").read_text())
    expected = "passed" if reference else "not_evaluated"
    if verification["correctness"] != expected:
        raise RuntimeError("candidate output did not match GPU answer reference")
    baseline = profile(root / "performance-baseline") if has_comparison else None
    if baseline:
        same = ("model", "tokenizer", "input_tokens", "output_tokens", "requests", "seed")
        if (
            any(baseline["workload"][key] != candidate["workload"][key] for key in same)
            or len(baseline["workload"]["slots"])
            != len(candidate["workload"]["slots"])
        ):
            raise RuntimeError("performance baseline and candidate workloads differ")
    rows = []
    for key, label, unit in METRICS:
        current = candidate["metrics"][key]
        factor = 1000 if unit == "ms" else 1
        if baseline:
            base = baseline["metrics"][key]
            gain = (
                current / base - 1
                if key == "output_tokens_per_second"
                else 1 - current / base
            ) * 100
            rows.append((label, base * factor, current * factor, gain, unit))
        else:
            rows.append((label, None, current * factor, None, unit))
    result = {
        "experiment": str(experiment),
        "answer_reference": str(reference) if reference else None,
        "correctness": verification["correctness"],
        "performance_baseline": baseline,
        "candidate": candidate,
        "comparison": [
            {
                "metric": label,
                "baseline": base,
                "candidate": current,
                "improvement_percent": gain,
                "unit": unit,
            }
            for label, base, current, gain, unit in rows
        ],
    }
    (root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if reference:
        answer = f"Answer reference: `{reference.name}`; output correctness: **passed**."
    else:
        answer = (
            f"Answer reference: none for `{candidate['workload']['model']}`; "
            "output correctness: **not evaluated** (smoke test passed)."
        )
    lines = [
        f"## {experiment.name}",
        "",
        answer,
        f"Algorithm profile: {candidate['workload']['requests']} requests, "
        f"{candidate['workload']['input_tokens']} input tokens, "
        f"{candidate['workload']['output_tokens']} output tokens, "
        f"concurrency {len(candidate['workload']['slots'])}.",
        "",
    ]
    if baseline:
        lines += [
            "Performance baseline: `baseline-hetero.yaml`. "
            "Positive improvement means the algorithm is faster.",
            "",
            "| Metric | Baseline-hetero | Algorithm | Improvement |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {label} | {base:.2f} {unit} | {current:.2f} {unit} | {gain:+.1f}% |"
            for label, base, current, gain, unit in rows
        ]
    else:
        reason = (
            "This experiment is the performance baseline."
            if experiment == PERFORMANCE_BASELINE
            else "No performance baseline exists for this model."
        )
        lines += [
            reason,
            "",
            "| Metric | Algorithm |",
            "| --- | ---: |",
        ]
        lines += [
            f"| {label} | {current:.2f} {unit} |"
            for label, _, current, _, unit in rows
        ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    root = args.result_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = config.load(experiment)
    if len({ep["node"] for ep in cfg["endpoints"]}) > 1:
        raise ValueError("multi-node experiment needs Kubernetes profiling")
    download_model(cfg)
    if experiment in REFERENCES:
        run_service(experiment, root, "reference", None)
        (root / "summary.md").write_text(
            f"## {experiment.name}\n\nGPU answer reference: output smoke test passed.\n"
        )
        print((root / "summary.md").read_text())
        return
    reference_file = root / "answer-reference.yaml"
    reference_source = reference_yaml(cfg, reference_file)
    answers = root / "reference/verification.json" if reference_source else None
    workload = workload_config(cfg)
    if reference_source:
        run_service(reference_file, root, "reference", None)
    performance_file = root / "performance-baseline.yaml"
    has_comparison = experiment != PERFORMANCE_BASELINE and performance_baseline_yaml(
        cfg, performance_file
    )
    if has_comparison:
        run_service(
            performance_file,
            root,
            "performance-baseline",
            answers,
            workload,
        )
    run_service(experiment, root, "candidate", answers, workload)
    report(root, experiment, reference_source, has_comparison)
    print((root / "summary.md").read_text())
    if reference_source is None:
        sys.exit(
            f"{experiment.name}: no GPU answer reference for model "
            f"{cfg['model']['name']}; output correctness was not evaluated"
        )


if __name__ == "__main__":
    main()
