"""Measure YAML-defined native profiles and engine options on local accelerators."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import statistics
import sys
import tempfile
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path

import yaml

from llmd import config
from llmd.verify import PromptTokenizer, verify
from tools.hetero import loadgen
from tools.hetero.profiler import benchmark_answers
from tools.hetero.profiler import experiment_ci as ci
from tools.hetero.profiler.resident import gateway

WORKLOAD_KEYS = {
    "input_tokens",
    "output_tokens",
    "requests",
    "concurrency",
    "seed",
    "warmup",
    "repeats",
    "expected_paths",
}


def named(mapping, field):
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", name) for name in mapping)
    ):
        raise ValueError(f"{field} requires nonempty named entries")
    return mapping


def override_engines(raw, overrides):
    eps = {ep["id"]: ep for ep in raw["endpoints"]}
    config.fields(overrides, eps, set(), "endpoint engine overrides")
    for key, values in overrides.items():
        config.fields(values, config.ENGINE_KEYS, set(), f"{key}.engine")
        eps[key].setdefault("engine", {}).update(deepcopy(values))


def materialize(suite, directory):
    source = config.read_yaml(suite)
    config.fields(
        source,
        {"apiVersion", "experiment", "gpu_reference", "engines", "arms", "workloads"},
        {"apiVersion", "experiment", "gpu_reference", "workloads"},
        "service suite",
    )
    if source["apiVersion"] != "hetero-pdd/service-suite-v1":
        raise ValueError("expected hetero-pdd/service-suite-v1")
    base = (suite.parent / source["experiment"]).resolve()
    raw = config.read_yaml(base)
    normalized = config.load(base)
    raw["model"] = normalized["model"]
    benchmark = normalized.get("benchmark")
    if not benchmark:
        raise ValueError("suite experiment requires an explicit benchmark")
    base_profile = normalized["profile"]
    defaults = {n: {"profile": p} for n, p in benchmark["comparisons"].items()}
    defaults["candidate"] = {}
    arms = named(source.get("arms", defaults), "arms")
    if "gpu-reference" in arms:
        raise ValueError("gpu-reference is reserved for unmeasured answers")
    override_engines(raw, source.get("engines", {}))
    cells = {}
    directory.mkdir(parents=True, exist_ok=True)
    for cell, overrides in named(source["workloads"], "workloads").items():
        config.fields(overrides, WORKLOAD_KEYS, set(), f"{cell}.workload")
        names = {}
        reference = dict(
            next(iter(arms.values())),
            profile=source["gpu_reference"],
            expected_paths=["gpu_full"],
        )
        for name, options in {"gpu-reference": reference, **arms}.items():
            config.fields(
                options, {"profile", "engines", "expected_paths"}, set(), f"{name} arm"
            )
            item = deepcopy(raw)
            profile_file = options.get("profile", base_profile)
            item["profile"] = str((suite.parent / profile_file).resolve())
            item["benchmark"] = {
                **benchmark,
                **overrides,
                "comparisons": {},
                "ci": False,
            }
            if "expected_paths" in options:
                item["benchmark"]["expected_paths"] = options["expected_paths"]
            # Explicit verification path requirements belong to the base profile.
            if item["profile"] != base_profile:
                item["verify"].pop("paths", None)
                item["verify"].pop("overlap_probe", None)
            override_engines(item, options.get("engines", {}))
            path = directory / f"{cell}--{name}.yaml"
            path.write_text(yaml.safe_dump(item, sort_keys=False))
            names[name] = (path, config.load(path))
        cells[cell] = names
    return cells


@contextmanager
def managed_engine(experiment, work):
    try:
        ci.command("-m", "llmd", "start", str(experiment), "--work-dir", str(work))
        yield work
    finally:
        if (work / "run.json").is_file():
            ci.command("-m", "llmd", "stop", "--work-dir", str(work))


def measure(live, cfg, work, label, tokenizer):
    workload = ci.workload_config(cfg)
    with gateway(live, cfg, work) as (state, refresh):
        verify(work, state, None)

        def execute(*args):
            loadgen.main(args[2:], tokenizer=tokenizer)

        ci.run_benchmark(
            state["experiment"], work, label, workload, execute=execute, refresh=refresh
        )
    data = ci.profile(work)
    routes = json.loads((work / "profile/route_evidence.json").read_text())["requests"]
    groups = defaultdict(list)
    for request in routes:
        transfer = request.get("transfer")
        if transfer is None:
            continue
        runs = 0
        for block_group in transfer["local_block_ids"]:
            blocks = sorted(block_group)
            runs += bool(blocks) + sum(b != a + 1 for a, b in zip(blocks, blocks[1:]))
        groups[(runs, transfer["copy_calls"])].append(transfer)
    data["transfers_by_physical_runs_and_calls"] = [
        {
            "contiguous_runs": runs,
            "copy_calls": calls,
            "requests": len(recs),
            "median_receive_ms": statistics.median(r["receive_ms"] for r in recs),
            "median_import_envelope_ms": statistics.median(
                r["import_ms"] for r in recs
            ),
            "median_bytes": statistics.median(r["bytes"] for r in recs),
        }
        for (runs, calls), recs in groups.items()
    ]
    return data


def service_lines(runs):
    return [
        "",
        "## Service measurements",
        "",
        "Each burst excludes GPU reference generation, verification and warmup.",
        "",
        *ci.table(
            (
                "Repeat",
                "Arm",
                "Burst seconds",
                "GPU output agreement",
                "Valid comparison",
            ),
            [
                (
                    r["repeat"],
                    r["arm"],
                    f"{r['profile']['burst_seconds']:.3f}",
                    r["gpu_reference_correct"],
                    r["comparison_valid"],
                )
                for r in runs
            ],
        ),
        "",
        "Transfer medians by contiguous block runs and actual copy calls.",
        "",
        *ci.table(
            (
                "Repeat",
                "Arm",
                "Runs",
                "Calls",
                "Requests",
                "MiB",
                "Receive ms",
                "Import envelope ms",
            ),
            [
                (
                    r["repeat"],
                    r["arm"],
                    g["contiguous_runs"],
                    g["copy_calls"],
                    g["requests"],
                    f"{g['median_bytes'] / (1024**2):.2f}",
                    f"{g['median_receive_ms']:.2f}",
                    f"{g['median_import_envelope_ms']:.2f}",
                )
                for r in runs
                for g in r["profile"]["transfers_by_physical_runs_and_calls"]
            ],
        ),
    ]


def execute_suite(path, root, resident=None):
    cells = materialize(path, root / "config")
    shutil.copyfile(path, root / "suite.yaml")
    runs = {cell: [] for cell in cells}
    arms = [k for k in next(iter(cells.values())) if k != "gpu-reference"]
    signature = None
    live = None
    boots = 0
    with ExitStack() as deployment:

        def engines(experiment, cfg):
            nonlocal signature, live, boots
            key = config.fingerprint(
                {
                    "endpoints": cfg["endpoints"],
                    "model": cfg["model"],
                    "instrumentation": cfg["instrumentation"],
                }
            )
            if resident:
                return resident
            if key != signature:
                deployment.close()
                boots += 1
                live = deployment.enter_context(
                    managed_engine(experiment, root / f"engines-{boots:02d}")
                )
                signature = key
            return live

        for cell, configs in cells.items():
            experiment, cfg = configs["gpu-reference"]
            source = engines(experiment, cfg)
            state = json.loads((source / "run.json").read_text())
            tokenizer = PromptTokenizer(state["model"]["tokenizer_path"])
            work = root / cell / "gpu-reference"
            measure(source, cfg, work, "gpu-reference", tokenizer)
            # Refuse to use an incomplete or non-GPU answer key.
            benchmark_answers.gpu_answer_key(work)

        repeats = {
            cell: configs[arms[0]][1]["benchmark"]["repeats"]
            for cell, configs in cells.items()
        }
        for i in range(max(repeats.values())):
            for j in range(len(arms)):
                arm = arms[(j - i) % len(arms)]
                for cell, configs in cells.items():
                    if i >= repeats[cell]:
                        continue
                    experiment, cfg = configs[arm]
                    source = engines(experiment, cfg)
                    work = root / cell / f"repeat-{i + 1:02d}" / arm
                    print(f"Measuring {cell} / {arm} / repeat {i + 1}", flush=True)
                    data = measure(source, cfg, work, arm, tokenizer)
                    entry = dict(repeat=i + 1, arm=arm, work=str(work), profile=data)
                    if "expected_paths" in cfg["benchmark"]:
                        entry["expected_paths"] = cfg["benchmark"]["expected_paths"]
                    runs[cell].append(entry)
                    (root / cell / "partial.json").write_text(
                        json.dumps(runs[cell], indent=2) + "\n"
                    )
    errors = []
    for cell, rows in runs.items():
        folder = root / cell
        result = ci.summarize_runs(folder, rows, gpu_reference=folder / "gpu-reference")
        lines = ci.benchmark_lines(result) + service_lines(result["runs"])
        (folder / "summary.md").write_text("\n".join(lines) + "\n")
        errors.extend(f"{cell}: {s}" for s in result["errors"])
    return errors


def main():
    def interrupted(sig, _frame):
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument(
        "--resident-dir",
        type=Path,
        help="Reuse this ready llmd deployment; engine settings must match every arm",
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        with tempfile.TemporaryDirectory() as tmp:
            cells = materialize(args.suite.resolve(), Path(tmp))
            print(json.dumps({k: list(v) for k, v in cells.items()}))
        return
    if not args.result_dir:
        parser.error("--result-dir is required to measure")
    root = args.result_dir.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("--result-dir must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    if errors := execute_suite(args.suite.resolve(), root, args.resident_dir):
        sys.exit("\n".join(errors))


if __name__ == "__main__":
    main()
