# SPDX-License-Identifier: Apache-2.0
"""Summarize offline operator traces; run with the tools venv."""

import argparse
import csv
import gzip
import io
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# H6b (2026-11-10), ATOM: device thread names and categories agree.
NPU_DEVICE_TRACKS = ("Neural Engine Clusters", "Neural DMA")
NPU_SEQUENCE = re.compile(r"rbln_\d{8}_\d{9}_(\d+)\.pb")
# vLLM 0.22.0 gpu_worker.annotate_profile; CUDA correlation checked in H6b.
STEP = re.compile(r"execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)")
IDENTITY = ("device", "phase", "model")
FIELDS = IDENTITY + ("category", "name", "calls", "total_ms", "mean_us", "ratio")
FIELDS += ("status", "reason", "scope")
TRACE_ERROR = "RBLN Profiler trace 무결성 문제(종료<시작 이벤트)"
# First match wins; DMA categories stay separate from compute categories.
TYPE_RULES = (
    ("gpu", "matmul", r"gemm|wmma|gemv"),
    ("gpu", "attention", r"flash|attention"),
    ("gpu", "norm", r"norm"),
    ("gpu", "elementwise", r"elementwise|silu|gelu|activation"),
    ("npu", "DMA", r"dma"),
    ("npu", "linear", r"linear|matmul|gemm"),
    ("npu", "attention", r"attention|flash|attn"),
    ("npu", "norm", r"norm"),
    ("npu", "elementwise", r"elementwise|silu|gelu|activation"),
)


def aggregate_types(rows, device):
    typed = []
    for row in rows:
        name = re.sub(r"\d+", "", row["name"]) if device == "npu" else row["name"]
        target = name + " " + row["category"]
        kind = next(
            (
                k
                for d, k, pattern in TYPE_RULES
                if d == device and re.search(pattern, target, re.I)
            ),
            "기타",
        )
        typed.append(dict(row, name=kind))
    result = aggregate(typed, top=len(TYPE_RULES) + 1)
    for row in result:
        row["operator_type"] = row.pop("name")
    return result


def chrome_rows(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        events = [e for e in json.load(stream)["traceEvents"] if e.get("ph") == "X"]
    steps = [(e, STEP.fullmatch(e["name"])) for e in events]
    steps = [(e, m) for e, m in steps if m]
    launches = {
        e.get("args", {}).get("correlation"): e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    }
    rows = []
    for e in events:
        if e.get("cat") not in ("kernel", "cpu_op"):
            continue
        host = (
            launches.get(e.get("args", {}).get("correlation"), {})
            if e["cat"] == "kernel"
            else e
        )
        step = next(
            (
                i
                for i, (s, _) in enumerate(steps)
                if host.get("pid") == s.get("pid")
                and host.get("tid") == s.get("tid")
                and s["ts"] <= host.get("ts", -1) < s["ts"] + s["dur"]
            ),
            None,
        )
        ctx, gen = steps[step][1].groups() if step is not None else ("1", "0")
        row = dict(name=e["name"], category=e["cat"], start_ns=e["ts"] * 1000)
        row.update(
            duration_ns=e["dur"] * 1000, step=step, decode=ctx == "0" and int(gen) > 0
        )
        rows.append(row)
    return rows


def pb_rows(path, health=None):
    sql = """SELECT s.name, s.category, s.ts AS start_ns, s.dur AS duration_ns,
             COALESCE(t.name, th.name) AS track FROM slice s
             JOIN track t ON s.track_id=t.id
             LEFT JOIN thread_track tt ON tt.id=t.id
             LEFT JOIN thread th ON th.utid=tt.utid"""
    result = subprocess.run(
        [os.environ["PERFETTO_TP_BIN"], "query", str(path), sql],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        dict(r, start_ns=int(r["start_ns"]), duration_ns=int(r["duration_ns"]))
        for r in csv.DictReader(io.StringIO(result.stdout))
    ]
    if health is not None:
        health.update(
            errors={
                k: int(v) for k, v in re.findall(r"    (\w+): (\d+) \|", result.stderr)
            },
            unfinished=dict(Counter(r["track"] for r in rows if r["duration_ns"] < 0)),
        )
    return rows


def aggregate(rows, top=20):
    groups, totals = defaultdict(list), defaultdict(float)
    for r in rows:
        if r["duration_ns"] < 0:
            raise ValueError(f"{r['name']}: unfinished slice")
        groups[r["category"], r["name"]].append(r["duration_ns"])
        totals[r["category"]] += r["duration_ns"]
    result = []
    for category in sorted(totals):
        ranked = [(name, ds) for (cat, name), ds in groups.items() if cat == category]
        for name, ds in sorted(ranked, key=lambda p: (-sum(p[1]), p[0]))[:top]:
            row = dict(category=category, name=name, calls=len(ds))
            row.update(total_ms=sum(ds) / 1e6, mean_us=sum(ds) / len(ds) / 1000)
            row["ratio"] = sum(ds) / totals[category] if totals[category] else 0
            result.append(row)
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root, out = args.input.expanduser().resolve(), args.out.expanduser().resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError(f"{out}: output must be outside repository")
    meta = json.loads((root / "meta.json").read_text())
    device, phase = meta["device"], meta["phase"]
    pattern = "*.pb" if device == "npu" else "*.trace.json*"
    files = sorted(root.rglob(pattern))
    if device == "npu":
        # Full decode exceeds Perfetto's string pool; use the final measured module.
        # opprofile performs no model execution after its measured generate.
        sequences = {
            int(m[1]): p for p in files if (m := NPU_SEQUENCE.fullmatch(p.name))
        }
        files = [sequences[max(sequences)]]
    rows, steps, tracks = [], [], set()
    health = {}
    for path in files:
        parsed = pb_rows(path, health) if device == "npu" else chrome_rows(path)
        tracks.update((r.get("track", ""), r["category"]) for r in parsed)
        rows.extend(
            r
            for r in parsed
            if device != "npu"
            or (r.get("track") in NPU_DEVICE_TRACKS and r["category"] == r["track"])
        )
        grouped = defaultdict(float)
        for r in parsed:
            if r["category"] == "kernel" and r.get("decode"):
                grouped[r["step"]] += r["duration_ns"] / 1e6
        steps.extend(
            dict(trace=str(path.relative_to(root)), step=i, kernel_ms=ms)
            for i, ms in sorted(grouped.items())
        )
    invalid = any(any(counts.values()) for counts in health.values())
    identity = dict(
        {k: meta[k] for k in IDENTITY},
        status="not_collected" if invalid else "measured",
        reason=TRACE_ERROR if invalid else None,
        scope=f"last_module:{files[0].name}" if device == "npu" else "whole_generate",
    )
    result = [dict(identity, **r) for r in ([] if invalid else aggregate(rows))]
    if invalid:
        result = [dict.fromkeys(FIELDS) | identity]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "operators.csv", result, FIELDS)
    type_fields = tuple("operator_type" if k == "name" else k for k in FIELDS)
    typed = [
        dict(
            identity,
            **r,
            input_tokens=meta.get("input_tokens"),
            output_tokens=meta.get("output_tokens"),
        )
        for r in ([] if invalid else aggregate_types(rows, device))
    ]
    if invalid:
        typed = [dict.fromkeys(type_fields) | identity]
    write_csv(
        out / "operators_by_type.csv",
        typed,
        type_fields + ("input_tokens", "output_tokens"),
    )
    write_csv(out / "kernel_steps.csv", steps, ("trace", "step", "kernel_ms"))
    reason = "step별 KV 문맥/가중치 접근량 미확인(H6b)"
    unavailable = dict(
        value=None, status="not_collected", reason=reason, source="kernel_steps.csv"
    )
    memory = dict(
        decode_mem_bw_gbps_est=dict(unavailable, unit="GB/s"),
        mem_bw_efficiency_est=dict(unavailable, unit="ratio"),
    )
    payloads = dict(memory_bw=memory, tracks=sorted(tracks), diagnostics=health)
    for name, data in payloads.items():
        (out / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    unlinked = sum(r["category"] == "kernel" and r.get("step") is None for r in rows)
    note = (
        f"대상: 마지막 본측정 module {files[0].name}; warm-up 제외. "
        "전체 generate 또는 평균 step의 대표값이 아님; PB 상대 시간축."
        if device == "npu"
        else "whole-generate 연산 합계; decode 입력에도 초기 prefill 포함"
    )
    text = f"""# {device} / {phase} operators

{note}
상태: {identity["status"]}; 사유: {identity["reason"]}; NPU Perfetto 진단: {health}
비율 분모: 상위20 선정 전 카테고리별 inclusive 시간 합.
GPU decode kernel step 수: {len(steps)} (annotation+CUDA correlation).
미연결 커널 {unlinked}개는 step 집계에서 제외(연산자 합계에는 포함).
순수 decode는 context=0, generation>0.
T06·T07: not_collected — {reason}

| category | name | calls | total ms | mean us | ratio | status | reason | scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""
    for r in result:
        cells = [str(r[k]).replace("|", "/") for k in FIELDS[3:]]
        text += "| " + " | ".join(cells) + " |\n"
    (out / "operators.md").write_text(text)
    if (
        invalid
        or not result
        or (device == "gpu" and not any(r["category"] == "kernel" for r in rows))
    ):
        raise ValueError(
            f"{root}: {identity['reason'] or 'no device operators'}; {note}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"operator_stats: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
