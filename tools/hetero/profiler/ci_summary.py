# SPDX-License-Identifier: Apache-2.0
"""Render stored profiler results as a CI Markdown summary; no metric recomputation."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def cell(value):
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.4g}"
    except (ValueError, TypeError):
        return str(value).replace("|", "\\|").replace("\n", " ")


def missing(value):
    if isinstance(value, dict):
        if value.get("status") == "not_collected":
            yield value.get("reason") or "사유 없음"
        else:
            for child in value.values():
                yield from missing(child)
    elif isinstance(value, list):
        for child in value:
            yield from missing(child)


def render(root, overhead=None):
    rows = read_csv(root / "results.csv")
    lines = ["# 프로파일러 CI 요약", "", "## 실행 정보", ""]
    for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "PROFILE_SCENARIOS"):
        lines.append(f"- {key}: {cell(os.getenv(key))}")

    def table(title, headers, values):
        lines.extend(["", f"## {title}", "", "| " + " | ".join(headers) + " |"])
        lines.append("|" + "---|" * len(headers))
        lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in values)

    identity = ["scenario", "run_id", "workload"]
    states = identity + ["run_status", "analysis_status", "error"]
    table(
        "시나리오 상태",
        ["시나리오", "실행", "workload", "run", "analysis", "오류"],
        ([r.get(k) for k in states] for r in rows),
    )
    metric_keys = ["output_tokens_per_second", "makespan_seconds"]
    metric_keys += [f"{name}_mean_seconds" for name in ("ttft", "tpot", "e2e")]
    metrics, warnings = [], []
    for row in rows:
        source = row.get("source_summary")
        path = root / source if source else None
        summary = json.loads(path.read_text()) if path and path.is_file() else {}
        energy = (
            summary.get("energy", {})
            .get("total", {})
            .get("energy_per_output_token_j_est", {})
        )
        metrics.append(
            [row.get(k) for k in identity + metric_keys]
            + [energy.get("value"), energy.get("status")]
        )
        reasons = Counter(missing(summary))
        label = f"{row.get('scenario')}/{row.get('run_id')}/{row.get('workload')}"
        warnings.append(f"- {label}: 미수집 {sum(reasons.values())}개; {dict(reasons)}")
        if not summary:
            warnings.append(f"- {label}: summary.json 없음; 미수집 지표 수 확인 불가")
        duration = row.get("makespan_seconds")
        target = 120 if row.get("pair_id", "").startswith("s6") else 30
        if duration and float(duration) < target:
            warnings.append(f"- {label}: 목표 {target}s 미달 ({cell(duration)}s)")
    headers = "시나리오 실행 workload tokens/s makespan(s) TTFT평균(s) "
    headers += "TPOT평균(s) E2E평균(s) J/token 에너지상태"
    table("핵심 지표", headers.split(), metrics)
    comparisons = read_csv(root / "comparison.csv")
    keys = (
        "pair_id",
        "workload",
        "baseline",
        "candidate",
        "throughput_improvement_percent_vs_b0",
        "makespan_reduction_percent_vs_b0",
    )
    table(
        "B0 대 PDD",
        ["쌍", "workload", "B0", "PDD", "처리량 개선(%)", "makespan 감소(%)"],
        ([r.get(k) for k in keys] for r in comparisons),
    )
    lines.extend(["", "## 경고", "", *warnings])
    if not rows:
        lines.append("- results.csv 없음 또는 비어 있음: 시나리오 결과 확인 불가")
    if overhead:
        path = overhead / "overhead.json"
        data = json.loads(path.read_text()) if path.is_file() else {}
        lines.extend(
            ["", "## PER-001", "", f"판정: {data.get('verdict', 'INCOMPLETE')}"]
        )
        decisions = data.get("decisions") or [{}]
        for key, stats in decisions[-1].get("stats", {}).items():
            lines.append(
                f"- {key}: 평균 {cell(stats.get('mean'))}, "
                f"90% CI [{cell(stats.get('low'))}, {cell(stats.get('high'))}] (비율)"
            )
        if data.get("error"):
            lines.append(f"- 중단 사유: {cell(data['error'])}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Writes <sweep-dir>/summary.md. Optional GITHUB_SHA, GITHUB_RUN_ID and "
            "PROFILE_SCENARIOS label the run. Shell entry point requires CUDA_PY."
        ),
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--overhead-dir", type=Path, help="Directory containing overhead.json"
    )
    args = parser.parse_args()
    render(args.sweep_dir, args.overhead_dir)


if __name__ == "__main__":
    main()
