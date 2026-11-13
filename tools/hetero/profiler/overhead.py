# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[3])

from tools.hetero.profiler import analysis, log, run
from tools.hetero.profiler.common import live_rows

# NIST t(0.95, df), indexed by df; published to three decimal places.
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
T95 = (0, 6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812)
METRICS = ("output_tokens_per_second", "e2e_mean_seconds", "e2e_p95_seconds")


def order(block):
    return ("off", "on") if block == 0 else ("off", "on", "on", "off")


def cpu_ticks(stat):
    return sum(map(int, stat.rsplit(")", 1)[1].split()[11:15]))


def cpu_snapshot(processes):
    ticks = {
        name: cpu_ticks((Path(os.sep) / "proc" / str(p.pid) / "stat").read_text())
        for name, p in processes.items()
    }
    return dict(
        mono_ns=time.monotonic_ns(),
        ticks=ticks,
        pids={k: p.pid for k, p in processes.items()},
    )


def cpu_usage(before, after, hz, cores):
    elapsed = (after["mono_ns"] - before["mono_ns"]) / 1e9
    single = {
        k: (after["ticks"][k] - v) / hz / elapsed * 100
        for k, v in before["ticks"].items()
    }
    return {
        k: dict(single_core_pct=v, all_cores_pct=v / cores) for k, v in single.items()
    }


def block_metrics(runs):
    off, on = [
        [mean(r[k] for r in runs if r["mode"] == mode) for k in ("tp", "lat")]
        for mode in ("off", "on")
    ]
    return dict(d_tp=(off[0] - on[0]) / off[0], d_lat=(on[1] - off[1]) / off[1])


def decide(blocks):
    stats = {}
    for key in ("d_tp", "d_lat"):
        values = [b[key] for b in blocks]
        avg = mean(values)
        margin = T95[len(values) - 1] * stdev(values) / len(values) ** 0.5
        stats[key] = dict(mean=avg, low=avg - margin, high=avg + margin)
    crossing = any(s["low"] <= 0.05 < s["high"] for s in stats.values())
    if len(blocks) == 6:
        verdict = "PASS" if all(s["mean"] <= 0.05 for s in stats.values()) else "FAIL"
    elif any(s["low"] > 0.05 for s in stats.values()):
        verdict = "FAIL"
    else:
        verdict = "PASS" if all(s["high"] <= 0.05 for s in stats.values()) else "ADD"
    return dict(
        blocks=len(blocks),
        verdict=verdict,
        stats=stats,
        uncertainty="불확실성 큼" if len(blocks) == 6 and crossing else None,
    )


def validate(folder, work, config, on):
    rows = list(live_rows(folder / "requests.jsonl"))
    validity = analysis.validate_raw(folder, rows, dict(requests=work["requests"]))
    if on and rows:
        window = (
            min(r["scheduled_arrival_monotonic"] for r in rows),
            max(r["completed_monotonic"] for r in rows),
        )
        samples, edges = [
            list(live_rows(folder / "monitor" / f"{k}.jsonl"))
            for k in ("metrics", "boundaries")
        ]
        checked = analysis.validate_measurement(
            samples, edges, config["metrics_endpoints"], window
        )
        validity["measurement"] = checked
        validity["failures"] += checked["failures"]
    summary = analysis.client_metrics(rows)["summary"]
    return dict(
        requests=rows,
        validity=validity,
        **{
            k: summary.get(name, {}).get("value")
            for k, name in zip(("tp", "lat", "p95"), METRICS)
        },
    )


def trial(folder, work, config, on):
    (folder / "monitor").mkdir(parents=True)
    data = dict(path=folder.name, mode="on" if on else "off", steps=[], cpu={})
    call, cli = run.stage_functions(folder, data)
    monitors, endpoints = run.Monitors(folder), dict(config["metrics_endpoints"])
    try:
        if on:
            endpoints["all-smi"] = call("allsmi_start", monitors.allsmi)
        mode = "monitor" if on else "none"
        with run.workload_session(
            work, config, folder, mode, monitors, call, cli, endpoints
        ) as argv:
            if on:
                data["cpu"]["before"] = cpu_snapshot(monitors.processes)
            try:
                cli(work["name"] + ":measure", argv)
            finally:
                if on:
                    data["cpu"]["after"] = cpu_snapshot(monitors.processes)
                    data["cpu"]["usage"] = cpu_usage(
                        data["cpu"]["before"],
                        data["cpu"]["after"],
                        os.sysconf("SC_CLK_TCK"),
                        os.cpu_count(),
                    )
    except Exception as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for name in ("scrape", "all-smi"):
            if name in monitors.processes:
                call(name + "_stop", monitors.stop, name, optional=True)
    data.update(validate(folder, work, config, on))
    data["validity"]["failures"] += [s["error"] for s in data["steps"] if not s["ok"]]
    data["valid"] = not (
        data.get("error") or data["validity"]["failures"] or data.get("record_error")
    )
    return data


def measure_block(number, measure, data):
    for attempt in range(2):
        prefix = (f"{number:02d}" if number else "warmup") + (
            "_retry1" if attempt else ""
        )
        runs = []
        for seq, mode in enumerate(order(number)):
            runs.append(measure(f"{prefix}_{seq}_{mode}", mode == "on"))
            data["runs"].append(runs[-1])
        valid = all(r["valid"] for r in runs)
        data["attempts"].append(
            dict(
                block=number,
                attempt=attempt,
                valid=valid,
                included=bool(number and valid),
                runs=[r["path"] for r in runs],
            )
        )
        if valid:
            return dict(block=number, **block_metrics(runs)) if number else None
    reasons = [
        r.get("error") or "; ".join(r["validity"]["failures"])
        for r in runs
        if not r["valid"]
    ]
    raise RuntimeError(f"block {number}: retry invalid: {'; '.join(reasons)}")


def save(root, data):
    (root / "overhead.json").write_text(json.dumps(data, indent=2) + "\n")
    final = data["decisions"][-1] if data["decisions"] else {}
    # PASS is a measurement verdict, not a password or credential.
    label = {"PASS": "합격", "FAIL": "불합격", "ABORTED": "측정 중단"}.get(  # nosec B105
        data["verdict"], "측정 중"
    )
    lines = [
        f"판정: {label} ({data['verdict']}) {final.get('uncertainty') or ''}",
        "",
        "| 블록 | d_tp | d_lat |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {b['block']} | {b['d_tp']:.6%} | {b['d_lat']:.6%} |" for b in data["blocks"]
    ]
    lines.append("")
    lines += [
        f"- {k}: 평균 {v['mean']:.6%}, 90% CI [{v['low']:.6%}, {v['high']:.6%}]"
        for k, v in final.get("stats", {}).items()
    ]
    included = {p for a in data["attempts"] if a["included"] for p in a["runs"]}
    lines += [
        "",
        "| 실행 | p95 E2E(s) | CPU(%: 단일 코어/전체 코어) |",
        "|---|---:|---|",
    ]
    lines += [
        f"| {r['path']} | {r['p95']} | {r['cpu'].get('usage', {})} |"
        for r in data["runs"]
        if r["path"] in included
    ]
    if data["verdict"] == "ABORTED":
        reason = (
            data.get("error")
            or data.get("record_error")
            or "; ".join(s["error"] for s in data["steps"] if not s["ok"])
        )
        lines += ["", f"중단 사유: {reason}", "실패한 실행 (path / failures):"]
        lines += [
            f"- {r['path']}: {r['validity']['failures']}"
            for r in data["runs"]
            if not r["valid"]
        ]
    settings = {
        k: data["settings"][k] for k in ("configuration", "workload", "service_args")
    }
    lines += ["", "실행 설정: " + json.dumps(settings, ensure_ascii=False)]
    (root / "overhead.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args, forwarded, service, workloads, argv = run.parse_args(argv, single=True)
    root, work = args.run_dir, workloads[0]
    if any(service.role_args_json.values()) or any(
        v.get(k) == "1"
        for v in service.role_env_json.values()
        for k in ("VLLM_RBLN_METRICS", "RBLN_PROFILER")
    ):
        raise ValueError("PER-001 requires tracing and RBLN instrumentation disabled")
    out = root / "overhead"
    out.mkdir(parents=True)
    data = dict(
        mode="none",
        steps=[],
        verdict="PENDING",
        runs=[],
        attempts=[],
        blocks=[],
        decisions=[],
    )
    data["settings"] = dict(
        argv=argv,
        configuration=args.configuration,
        workload=work,
        service_args=forwarded,
        limit=0.05,
        confidence=0.9,
        initial_blocks=4,
        extra_blocks=2,
        retries=1,
        order=order(1),
        warmup_order=order(0),
        clock_ticks=os.sysconf("SC_CLK_TCK"),
        cpu_count=os.cpu_count(),
        t95=T95,
    )
    save(out, data)
    try:
        with run.service_session(args, forwarded, service, data) as session:
            config, _, call, _, _ = session
            data["settings"]["service"] = config
            call("monitor_tools", run.check_tools, "monitor")
            measure = lambda name, on: trial(out / name, work, config, on)
            for number in range(7):
                block = measure_block(number, measure, data)
                if block:
                    data["blocks"].append(block)
                if number in (4, 6):
                    decision = decide(data["blocks"])
                    data["decisions"].append(decision)
                    data["verdict"] = decision["verdict"]
                save(out, data)
                if data["verdict"] in ("PASS", "FAIL"):
                    break
    except (Exception, KeyboardInterrupt) as exc:
        data.update(verdict="ABORTED", error=f"{type(exc).__name__}: {exc}")
    if data.get("record_error") or any(not s["ok"] for s in data["steps"]):
        data["verdict"] = "ABORTED"
    save(out, data)
    return int(data["verdict"] == "ABORTED")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.log_line("overhead", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
