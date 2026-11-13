# SPDX-License-Identifier: Apache-2.0
"""Create report figures from exported CSVs (run in the tools environment)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value.startswith(("{", "[")):
                row[key] = json.loads(value)
    return rows


def num(value):
    return float(value) if value not in (None, "") else float("nan")


def panel(ax, title, records, kind="bar"):
    if kind == "text":
        ax.axis("off")
        ax.text(0.02, 0.2, "\n\n".join(records), fontsize=11)
        return
    labels = list(dict.fromkeys(x for x, _, _ in records))
    groups = list(dict.fromkeys(g for _, g, _ in records))
    base = [0.0] * len(labels)
    for i, group in enumerate(groups):
        points = [(x, num(y)) for x, g, y in records if g == group]
        if kind == "line":
            ax.plot(
                *zip(*points), label=group, marker="o" if len(points) < 10 else None
            )
            continue
        lookup = dict(points)
        values = [lookup.get(x, 0 if kind == "stack" else float("nan")) for x in labels]
        width = 0.8 / max(1, len(groups))
        shift = 0 if kind == "stack" else i * width
        ys = [j + shift for j in range(len(labels))]
        height = 0.8 if kind == "stack" else width
        ax.barh(ys, values, height=height, left=base, label=group)
        if kind == "stack":
            base = [a + b for a, b in zip(base, values)]
    if kind != "line":
        offset = 0 if kind == "stack" else (len(groups) - 1) * 0.4 / max(1, len(groups))
        ax.set(yticks=[j + offset for j in range(len(labels))], yticklabels=labels)
        ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if groups:
        if kind == "stack":
            ax.legend(
                fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4
            )
        else:
            ax.legend(fontsize=7, loc="best")


def generate(root, operators, overhead, out):
    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})
    rows, pairs = read_csv(root / "results.csv"), read_csv(root / "comparison.csv")
    latest = {}
    for row in rows:
        if (
            row["run_status"] == row["analysis_status"] == "PASS"
            and row["returncode"] == "0"
        ):
            latest[row["scenario"], row["workload"]] = row
    rows = list(latest.values())
    series = read_csv(root / "dataset/resource_timeseries.csv")
    types = [r for p in operators.rglob("operators_by_type.csv") for r in read_csv(p)]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def emit(key, title, panels, source, status):
        cols = min(3, len(panels))
        height = max(
            [
                len(set(x for x, _, _ in p[1])) * 0.48
                for p in panels
                if p[2] not in ("line", "text")
            ]
            + [4]
        )
        grid_rows = (len(panels) + cols - 1) // cols
        size = (7 * cols, height * grid_rows)
        fig, axes = plt.subplots(grid_rows, cols, figsize=size, squeeze=False)
        for ax, spec in zip(axes.flat, panels):
            panel(ax, *spec)
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        fig.text(0.02, 0.01, f"출처: {source}\n{status}", fontsize=9)
        fig.savefig(out / f"{key}.png", dpi=150, facecolor="white")
        plt.close(fig)
        manifest[key] = dict(title=title, source=source, status=status)

    design = [
        "B0: loadgen → GPU·NPU0·NPU1 (각자 prefill+decode)",
        "PDD: loadgen → GPU prefill → KV → NPU0·NPU1 decode",
        "warm-up → before → 본측정·수집 → after",
    ]
    emit(
        "01_design",
        "구성과 측정 경계",
        [("구성", design, "text")],
        "PLAN 5.3·5.4",
        "설계도(측정값 아님)",
    )
    key = lambda r: (r["run_id"], r["workload"])
    lookup = {key(r): r for r in rows}
    measures = [
        ("output_tokens_per_second", "출력 token/s"),
        ("makespan_seconds", "makespan (s)"),
        ("ttft_mean_seconds", "평균 TTFT (s)"),
        ("tpot_mean_seconds", "평균 TPOT (s)"),
        ("e2e_mean_seconds", "평균 E2E (s)"),
    ]
    panels = []
    for metric, title in measures:
        records = []
        for p in pairs:
            label = f"{p['pair_id']}/{p['workload']}\n{p['comparison']}"
            for side, name in [("baseline", "B0"), ("candidate", "PDD")]:
                records.append((label, name, lookup[p[side], p["workload"]][metric]))
        panels.append((title, records, "bar"))
    emit(
        "02_comparison",
        "B0 대 PDD 성능",
        panels,
        "results.csv·comparison.csv; 표의 pair/workload",
        "measured; 단일 측정, TPOT 미정의는 공란",
    )
    panels = []
    for prefix in ("s3a", "s3b", "s4"):
        selected = [r for r in rows if r["scenario"].startswith(prefix)]
        for metric, unit in (measures[0], measures[-1]):
            records = []
            for r in selected:
                setting = r["pair_id"].split("_")[-1][1:]
                x = r["model"] if prefix == "s4" else int(setting)
                records.append((x, r["configuration"], r[metric]))
            kind = "bar" if prefix == "s4" else "line"
            panels.append((prefix + ": " + unit, sorted(records), kind))
    emit(
        "03_scaling",
        "배치·동시성·모델 크기",
        panels,
        "results.csv; S3a·S3b·S4",
        "measured; S3a 배치/capacity, S3b 총 동시성/equal_concurrency, S4 모델명",
    )
    phase = [r for r in rows if r["scenario"].startswith("s5")]
    shares, active = [], []
    for r in phase:
        label = f"{r['scenario']}/{r['workload']} ({r['mode']})"
        shares += [
            (label, k, v * 100)
            for k, v in (r["request_phase_share"].get("value") or {}).items()
        ]
        for d, stages in r["device_phase_active_ratio"].items():
            active += [
                (label + "/" + d, k, 100 * num(m["value"])) for k, m in stages.items()
            ]
    emit(
        "04_phases",
        "S5·S5t 단계 비율",
        [("M14a (%)", shares, "stack"), ("M14b (%)", active, "bar")],
        "results.csv; S5·S5t",
        "monitor: estimated 근사; trace: measured. M14b는 단계 간 겹침 허용",
    )
    selected = [
        r
        for r in rows
        if r["scenario"] in ("s2_b0", "s2_pdd", "s5t_pdd")
        and r["workload"] == "i4096_o256"
    ]
    panels, cumulative, idle, transfer, relative = [], [], [], [], []
    for r in selected:
        tag = r["scenario"]
        points = [p for p in series if (p["run_id"], p["workload"]) == key(r)]
        origin = min((num(p["mono_seconds"]) for p in points), default=0)
        for metric, unit in [("utilization", "%"), ("power", "W")]:
            records = [
                (num(p["mono_seconds"]) - origin, p["device"], p["value"])
                for p in points
                if p["metric"] == metric and p["device"] != "system"
            ]
            panels.append((tag + " " + unit + " / 상대 초", records, "line"))
        samples = r["phase_resources"]["samples"]
        coverage = [
            (s["device"] + "/" + s["state"], "coverage", 100 * num(s["coverage"]))
            for s in samples
            if s["metric"] == "power" and s["state"] != "all"
        ]
        panels.append((tag + " coverage (%)", coverage, "bar"))
        for d, m in r["transfer"].items():
            v = m.get("nixl_gbps", {}).get("value")
            if v is not None:
                transfer.append((tag + "/" + d, "T01", v))
                percent = 100 * v / m["link_theoretical"]["value"]
                reference = f"T01/T04({m['link_theoretical']['value']:g} GB/s)"
                relative.append((tag + "/" + d, reference, percent))
        if tag.startswith("s2"):
            for device in ("gpu", "npu0", "npu1"):
                ps = sorted(
                    (num(p["mono_seconds"]), num(p["value"]))
                    for p in points
                    if p["device"] == device and p["metric"] == "power"
                )
                energy = 0
                for a, b in zip(ps, ps[1:]):
                    energy += (b[0] - a[0]) * (a[1] + b[1]) / 2
                    cumulative.append((b[0] - ps[0][0], tag + "/" + device, energy))
            idle += [
                (tag + "/" + s["device"], "idle 평균 W", s["value"]["mean"])
                for s in samples
                if s["metric"] == "power" and s["state"] == "idle" and s["value"]
            ]
    emit(
        "05_resources",
        "활용률·전력·단계 coverage",
        panels,
        "dataset/resource_timeseries.csv; results.csv phase_resources; "
        "S2·S5t i4096_o256",
        "시계열 measured; 단계 귀속 estimated; 실행별 첫 표본 기준 상대 시각",
    )
    emit(
        "06_transfer",
        "KV 전송과 링크 규격",
        [("T01 GB/s", transfer, "bar"), ("이론치 대비 %", relative, "bar")],
        "results.csv transfer; S2·S5t i4096_o256; link_theoretical",
        "T01 measured; 비율은 참고 비교이며 T05·PCIe 실측 효율 아님",
    )
    energy = []
    for p in pairs:
        for side, name in [("baseline", "B0"), ("candidate", "PDD")]:
            m = lookup[p[side], p["workload"]]["energy"]["total"]
            label = p["pair_id"] + "/" + p["workload"]
            value = m["energy_per_output_token_j_est"]["value"]
            energy.append((label, name, value))
    emit(
        "07_energy",
        "에너지 효율과 유휴 전력",
        [
            ("J/token", energy, "bar"),
            ("S2 표본 범위 누적 J / 초", cumulative, "line"),
            ("S2 idle 평균 W", idle, "bar"),
        ],
        "results.csv energy·phase_resources; "
        "dataset/resource_timeseries.csv; S2 idle/누적",
        "estimated; idle 전력을 총량에서 빼지 않음, 누적선은 관측 표본 범위",
    )
    records = []
    for r in types:
        if r["status"] != "not_collected" and r["category"] != "cpu_op":
            label = "/".join(r[k] for k in ("device", "phase", "category"))
            records.append((label, r["operator_type"], 100 * num(r["ratio"])))
    emit(
        "08_types",
        "이름 기반 연산 유형 분포",
        [("카테고리별 시간 비율 (%)", records, "stack")],
        "h6b/*/operators_by_type.csv",
        "시간 measured; 이름 규칙 분류. GPU 전체 generate / "
        "NPU prefill 마지막 모듈; NPU decode 미수집",
    )
    data = json.loads(overhead.read_text())
    blocks = [
        (b["block"], k, b[k] * 100) for b in data["blocks"] for k in ("d_tp", "d_lat")
    ]
    emit(
        "09_overhead",
        "PER-001 1차: " + data["verdict"],
        [("블록별 변화 (%)", blocks, "line")],
        "per001_r3/overhead/overhead.json; blocks·decisions",
        "measured; 판정은 4블록 평균·90% t CI, 기준 5%; 최종 측정은 코드 확정 후",
    )
    (out / "captions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("sweep-dir", "opprofile-dir", "overhead", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        generate(args.sweep_dir, args.opprofile_dir, args.overhead, args.out)
    except Exception as exc:
        print(f"report_figures: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
