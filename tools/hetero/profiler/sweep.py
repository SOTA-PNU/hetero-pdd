# SPDX-License-Identifier: Apache-2.0
"""Run paired scenarios and resume successful runs without overwriting attempts."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[3])

from tools.hetero.profiler import run
from tools.hetero.profiler.log import log_line

PAIR_FIELDS = ("requests", "input_tokens", "output_tokens", "seed")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def outside_repo(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(run.REPO):
        raise ValueError(f"{path}: output must be outside repository")
    return path


def validate_pairs(scenarios):
    groups, names = {}, set()
    for scenario in scenarios:
        name = scenario["name"]
        if not run.NAME.fullmatch(name) or name in names:
            raise ValueError(f"invalid or repeated scenario name: {name}")
        names.add(name)
        comparison = scenario["comparison"]
        if comparison not in ("capacity", "equal_concurrency"):
            raise ValueError(f"{name}: unknown comparison {comparison}")
        run.validate_workloads(scenario["workloads"], scenario["configuration"])
        signature = {}
        for work in scenario["workloads"]:
            values = tuple(work[k] for k in PAIR_FIELDS)
            if comparison == "equal_concurrency":
                slots = (
                    sum(work["slots"].values())
                    if "slots" in work
                    else work["concurrency"]
                )
                values += (slots,)
            signature[work["name"]] = values
        key = scenario["pair_id"]
        expected = groups.setdefault(key, (comparison, signature))
        if expected != (comparison, signature):
            raise ValueError(
                f"{key}/{name}: workload counts, tokens, seed or concurrency differ"
            )


def cache_key(scenario):
    # TODO(verify: H7): compilation compatibility of shared service caches.
    args = {k.replace("-", "_"): v for k, v in scenario["service_args"].items()}
    args.setdefault("layout_reorder", 0)
    args = {
        k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
        for k, v in args.items()
    }
    conditions = dict(model=scenario["model"], configuration=scenario["configuration"])
    conditions.update(service_args=args, profiler=scenario["mode"] == "deep")
    return hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()[
        :20
    ]


def successful(root, scenario, attempt):
    states = [
        read_json(root / attempt["path"] / w["name"] / "profile/summary.json")
        for w in scenario["workloads"]
    ]
    return attempt.get("returncode") == 0 and all(
        s.get("run_status") == s.get("analysis_status") == "PASS" for s in states
    )


def execute(spec, root, models, cache_root=None, only=()):
    validate_pairs(spec["scenarios"])
    if set(only) - {s["name"] for s in spec["scenarios"]}:
        raise ValueError("--only contains an unknown scenario name")
    root = outside_repo(root)
    cache_root = outside_repo(cache_root) if cache_root else None
    for scenario in spec["scenarios"]:
        if (not only or scenario["name"] in only) and scenario["model"] not in models:
            raise ValueError(
                f"{scenario['name']}: missing --model {scenario['model']}=PATH"
            )
    manifest = root / "scenarios.json"
    saved = read_json(manifest)
    if saved:
        original = dict(
            saved,
            scenarios=[
                {k: v for k, v in s.items() if k != "attempts"}
                for s in saved["scenarios"]
            ],
        )
        if original != spec:
            raise ValueError(
                f"{manifest}: scenario definitions changed; use a new sweep-dir"
            )
        spec = saved
    root.mkdir(parents=True, exist_ok=True)

    def save():
        manifest.write_text(json.dumps(spec, indent=2) + "\n")

    failed = False
    for scenario in spec["scenarios"]:
        if only and scenario["name"] not in only:
            continue
        attempts = scenario.setdefault("attempts", [])
        if attempts and successful(root, scenario, attempts[-1]):
            continue
        name, suffix = scenario["name"], 0
        folder = root / "runs" / name
        while folder.exists():
            suffix += 1
            folder = root / "runs" / f"{name}_{suffix}"
        folder.mkdir(parents=True)
        attempt = dict(path=str(folder.relative_to(root)), returncode=None, error=None)
        attempts.append(attempt)
        save()
        inputs = root / "inputs"
        inputs.mkdir(exist_ok=True)
        workfile = inputs / f"{folder.name}.json"
        workfile.write_text(json.dumps(scenario["workloads"], indent=2) + "\n")
        options = {k.replace("-", "_"): v for k, v in scenario["service_args"].items()}
        options.update(configuration=scenario["configuration"], mode=scenario["mode"])
        options.update(
            run_dir=folder, workloads=workfile, model_path=models[scenario["model"]]
        )
        options.setdefault("model_name", scenario["model"])
        if cache_root:
            options["cache_dir"] = cache_root / cache_key(scenario)
        options = {
            k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            for k, v in options.items()
        }
        argv = [sys.executable, str(Path(run.__file__)), *run.flags(**options)]
        try:
            with (inputs / f"{folder.name}.log").open("w") as output:
                attempt["returncode"] = subprocess.run(
                    argv, stdout=output, stderr=subprocess.STDOUT
                ).returncode
            if not successful(root, scenario, attempt):
                attempt["error"] = (
                    f"{folder}: exit {attempt['returncode']} or incomplete summary"
                )
        except Exception as exc:
            attempt.update(returncode=1, error=f"{folder}: {type(exc).__name__}: {exc}")
        failed |= bool(attempt["error"])
        if attempt["error"]:
            log_line("sweep", attempt["error"])
        save()
    from tools.hetero.finalize_results import finalize

    return int(bool(finalize(root) or failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="KEY=PATH")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--only", action="append", default=[], metavar="SCENARIO")
    args = parser.parse_args(argv)
    models = {
        key: str(Path(value).expanduser().resolve())
        for key, value in (s.split("=", 1) for s in args.model)
    }
    return execute(
        read_json(args.scenarios), args.sweep_dir, models, args.cache_root, args.only
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log_line("sweep", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
