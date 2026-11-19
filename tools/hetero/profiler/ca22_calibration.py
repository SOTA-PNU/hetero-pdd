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
"""Measure isolated prefill, decode, receive and import before policy evaluation."""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

from llmd import config
from llmd.runtime import fetch
from llmd.verify import PromptTokenizer
from tools.hetero import loadgen


def histogram(url):
    """Get both the count and sum so unrelated samples cannot silently skew a cell."""
    data = fetch(url)[0].decode()
    readings = {}
    for field in ("sum", "count"):
        pattern = (
            rf"^vllm:request_prefill_time_seconds_{field}(?:\{{[^}}]*\}})?\s+(\S+)"
        )
        readings[field] = sum(float(v) for v in re.findall(pattern, data, re.M))
    return readings


def measure(state, route, settings, folder, tokenizer):
    eps = {e["backend"]: e for e in state["endpoints"]}
    # A split request's prefilling engine is CUDA.
    engine = eps["rbln" if route == "rbln" else "cuda"]
    seed = settings["seed"]
    slot = loadgen.Slot(0, route, state["endpoint"].rsplit("/v1/", 1)[0])
    path = folder / "requests.jsonl"
    lengths = settings["lengths"]
    rows = []
    with path.open("x", buffering=1) as out:
        for repeat in range(-settings["warmup"], settings["repeats"]):
            for i, length in enumerate(lengths):
                idx = (repeat + settings["warmup"]) * len(lengths) + i
                prompt = loadgen.make_prompt(tokenizer, length, idx, seed)
                before = histogram(engine["metrics"])
                row = loadgen.issue_request(
                    slot=slot,
                    request_index=idx,
                    prompt_ids=prompt,
                    scheduled_arrival=time.monotonic(),
                    model=state["model"]["name"],
                    output_tokens=settings["output_tokens"],
                    seed=seed,
                    timeout=60,
                    workload=f"cal-{route}-{state['run_id'][:8]}",
                )
                after = histogram(engine["metrics"])
                deadline = time.monotonic() + 5
                while after["count"] == before["count"] and time.monotonic() < deadline:
                    time.sleep(0.04)
                    after = histogram(engine["metrics"])
                row.update(
                    route=route,
                    repeat=repeat,
                    prompt_ids=prompt,
                    prefill_hist_before=before,
                    prefill_hist_after=after,
                )
                out.write(json.dumps(row) + "\n")
                if not loadgen.request_succeeded(row):
                    raise RuntimeError(
                        f"failed calibration request {row['request_id']}"
                    )
                if (
                    after["count"] - before["count"] != 1
                    or after["sum"] <= before["sum"]
                ):
                    raise RuntimeError(
                        f"unexpected prefill observation {before}, {after}"
                    )
                rows.append(row)
    return rows


def read_events(path):
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            yield json.loads(line)


def collect(folder, state, route, rows):
    access = {
        e["request_id"]: e
        for e in read_events(folder / "gateway.log")
        if e.get("event") == "gateway_access"
    }
    eps = {e["backend"]: e for e in state["endpoints"]}
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    if route == "pdd":
        for path in sorted(Path(eps["rbln"]["events_dir"]).glob("events_*.jsonl")):
            for ev in read_events(path):
                if ev["kind"] in ("kv_xfer_start", "kv_xfer_done", "kv_h2d"):
                    by_id[ev["req_id"]][ev["kind"]] = ev
    samples = []
    for row in rows:
        rid = row["request_id"]
        a = access[rid]
        target = eps["cuda" if route == "cuda" else "rbln"]
        port = target.get("sidecar_port") or target["port"]
        if (
            a["http_status"] != 200
            or a["target"] != target["id"]
            or a["destination"] != f"{target['host']}:{port}"
            or a["path"] != ("pdd" if route == "pdd" else "full")
        ):
            raise RuntimeError(f"unexpected execution for {rid}: {a}")
        if route == "pdd" and (a["prefill"] != eps["cuda"]["id"] or a["preload"]):
            raise RuntimeError(f"expected ordinary PDD from CUDA for {rid}")
        if row["actual_input_tokens"] != row["requested_input_tokens"]:
            raise RuntimeError(f"prompt length mismatch for {rid}")
        first = row["first_output_monotonic"]
        rec = dict(
            route=route,
            request_id=rid,
            length=row["actual_input_tokens"],
            repeat=row["repeat"],
            target=a["target"],
            prefill_ms=(
                row["prefill_hist_after"]["sum"] - row["prefill_hist_before"]["sum"]
            )
            * 1000,
            ttft_ms=(first - row["actual_send_monotonic"]) * 1000,
            tpot_ms=(row["last_output_monotonic"] - first)
            * 1000
            / (row["actual_output_tokens"] - 1),
        )
        if route == "pdd":
            (ev,) = [v for k, v in by_id.items() if k.startswith(f"cmpl-{rid}-")]
            imp = ev["kv_h2d"]
            rec.update(
                receive_ms=(
                    ev["kv_xfer_done"]["monotonic_ns"]
                    - ev["kv_xfer_start"]["monotonic_ns"]
                )
                / 1e6,
                import_ms=(imp["end_ns"] - imp["start_ns"]) / 1e6,
                blocks=imp["num_blocks"],
                bytes=imp["bytes_est"],
                copy_calls=imp["copy_calls"],
            )
        samples.append(rec)
    return samples


def run(path, mode, folder):
    from tools.hetero.profiler.resident import gateway

    spec = config.read_yaml(path)
    keys = {
        "experiment",
        "profiles",
        "lengths",
        "output_tokens",
        "warmup",
        "repeats",
        "seed",
        "engine_modes",
    }
    config.fields(spec, keys, keys, "calibration")
    config.fields(
        spec["profiles"],
        {"cuda", "rbln", "pdd"},
        {"cuda", "rbln", "pdd"},
        "calibration.profiles",
    )
    for key in ("output_tokens", "repeats", "warmup", "seed"):
        config.integer(
            spec[key], key, low={"seed": 0, "warmup": 0, "output_tokens": 2}.get(key, 1)
        )
    if not isinstance(spec["lengths"], list) or not spec["lengths"]:
        raise ValueError("lengths must be a nonempty list")
    for length in spec["lengths"]:
        config.integer(length, "lengths")
    if len(set(spec["lengths"])) != len(spec["lengths"]):
        raise ValueError("duplicate lengths")
    if mode not in spec["engine_modes"]:
        raise ValueError(f"unknown calibration mode {mode}")
    base = (path.parent / spec["experiment"]).resolve()
    raw = config.read_yaml(base)
    raw["model"] = str((base.parent / raw["model"]).resolve())
    raw.pop("benchmark", None)
    (endpoint,) = [e for e in raw["endpoints"] if e["backend"] == "rbln"]
    endpoint.setdefault("engine", {}).update(spec["engine_modes"][mode])
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "settings.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    sources = {}
    for route, profile in spec["profiles"].items():
        raw["profile"] = str((path.parent / profile).resolve())
        source = folder / f"{route}.yaml"
        source.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = config.load(source)
        if max(spec["lengths"]) + spec["output_tokens"] > min(
            e["engine"]["max_model_len"] for e in cfg["endpoints"]
        ):
            raise ValueError("calibration workload exceeds context")
        sources[route] = source
    resident = folder / "engine"

    def llmd(action, *args):
        subprocess.run(
            [sys.executable, "-m", "llmd", action, *args, "--work-dir", str(resident)],
            check=True,
        )

    samples = []
    try:
        llmd("start", str(sources["pdd"]))
        model = json.loads((resident / "run.json").read_text())["model"]
        tokenizer = PromptTokenizer(model["tokenizer_path"])
        for route, source in sources.items():
            arm = folder / route
            with gateway(resident, config.load(source), arm) as (state, refresh):
                rows = measure(state, route, spec, arm, tokenizer)
                refresh()
            samples.extend(collect(arm, state, route, rows))
            (folder / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    finally:
        if (resident / "run.json").exists():
            llmd("stop")
    finish(folder)


def finish(folder):
    samples = json.loads((folder / "samples.json").read_text())
    spec = config.read_yaml(folder / "settings.yaml")
    # Calibration arms use the same resident model and the same deterministic
    # request index, length and seed on all three native profiles.
    gold = {
        r["request_index"]: r for r in read_events(folder / "cuda" / "requests.jsonl")
    }
    comparisons = []
    for route in ("rbln", "pdd"):
        for row in read_events(folder / route / "requests.jsonl"):
            if row["repeat"] < 0:
                continue
            ref = gold[row["request_index"]]
            if ref["prompt_ids"] != row["prompt_ids"]:
                raise ValueError("reference prompts differ")
            x, y = ref["streamed_token_ids"], row["streamed_token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(x, y)) if a != b), None)
            if first is None and len(x) != len(y):
                first = min(len(x), len(y))
            comparisons.append(
                dict(
                    route=route,
                    request_id=row["request_id"],
                    gpu_request_id=ref["request_id"],
                    first_difference=first,
                    first_gpu_token=x[first]
                    if first is not None and first < len(x)
                    else None,
                    first_observed_token=y[first]
                    if first is not None and first < len(y)
                    else None,
                    request_index=row["request_index"],
                    length=row["actual_input_tokens"],
                    correct=first is None,
                )
            )
    (folder / "reference.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    failed = {r["request_id"] for r in comparisons if not r["correct"]}
    status = {
        r: all(j["correct"] for j in comparisons if j["route"] == r)
        for r in ("rbln", "pdd")
    }
    (folder / "reference-summary.json").write_text(
        json.dumps(
            dict(
                route_correct=status,
                mismatches=[r for r in comparisons if not r["correct"]],
            ),
            indent=2,
        )
        + "\n"
    )
    for name, eligible in (
        ("medians", samples),
        ("matching-medians", [r for r in samples if r["request_id"] not in failed]),
    ):
        cells: dict[str, dict[int, dict[str, float]]] = {}
        for route in spec["profiles"]:
            for length in spec["lengths"]:
                rows = [
                    r
                    for r in eligible
                    if r["route"] == route
                    and r["length"] == length
                    and r["repeat"] >= 0
                ]
                metrics = ["prefill_ms", "ttft_ms", "tpot_ms"]
                if route == "pdd":
                    metrics += ["receive_ms", "import_ms", "blocks", "copy_calls"]
                cells.setdefault(route, {})[length] = {
                    k: statistics.median(r[k] for r in rows) for k in metrics
                } | {"count": len(rows)}
        (folder / f"{name}.json").write_text(json.dumps(cells, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    opts = parser.parse_args()
    run(opts.config.resolve(), opts.mode, opts.result_dir.resolve())
