# SPDX-License-Identifier: Apache-2.0
"""Render existing PDD analysis metrics without recomputing them."""

import re

LABELS = {
    "prefill": "Prefill server (GPU)",
    "decode0": "Decode server (NPU0)",
    "summary.queue": "Queue (mean, s)",
    "summary.prefill": "Prefill compute (mean, s)",
    "summary.decode": "Decode (token generation) (mean, s)",
    "summary.kv_cache_usage": "KV cache usage (time-avg)",
    "summary.avg_running_requests": "Running requests (time-avg)",
    "summary.waiting_requests": "Waiting requests (time-avg)",
    "decode": "Decode (NPU)",
    "kv_transfer": "KV transfer",
    "layout": "KV load (H2D + transform)",
    "p_prefill": "Prefill (GPU)",
    "d_queue": "Decode server queue",
    "p_queue": "Prefill server queue",
    "other": "Other (proxy, gaps)",
}


def label(name):
    return re.sub(
        r"(?:cmpl-)?[\w-]*-req(-?\d+)(?:-0(?:-[0-9a-f]{8})?)?",
        r"req \1",
        LABELS.get(name, name),
    )


def gib(value):
    return f"{value / 2**30:.2f}" if value is not None else None


def metrics(value, prefix=""):
    if isinstance(value, dict):
        if "status" in value and "unit" in value:
            if "device" in value:
                prefix += "." + ".".join(
                    str(value[k]) for k in ("device", "state", "metric")
                )
            yield prefix, value
        else:
            for key, child in value.items():
                yield from metrics(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from metrics(child, f"{prefix}[{index}]")


def display(value):
    return (
        f"{value:.4g}"
        if isinstance(value, (int, float))
        else "unavailable"
        if value is None
        else str(value)
    ).replace("|", "\\|")


def table(headers, rows):
    return [
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ] + ["| " + " | ".join(map(display, row)) + " |" for row in rows]


def folded(title, content):
    return ["", f"<details><summary>{title}</summary>", "", *content, "", "</details>"]


def key_results(data, config):
    values = data.get("client", {}).get("summary", {})
    get = lambda key: values.get(key, {}).get("value")
    rows = lambda keys: [
        (label, *(get(f"{key}_{s}_seconds") for s in ("mean", "p50", "p90")))
        for key, label in keys
    ]
    lines = [
        "",
        "## Key results",
        "",
        "Send-based service timing (client queue excluded); seconds.",
    ]
    lines += table(
        ("Metric", "Mean", "p50", "p90"),
        rows(
            [
                ("service_ttft", "Service TTFT"),
                ("service", "Service E2E"),
                ("tpot", "TPOT"),
            ]
        ),
    )
    lines += table(
        ("Metric", "Value"),
        [
            (k, get(k))
            for k in (
                "output_tokens_per_second",
                "makespan_seconds",
                "success_count",
                "failure_count",
                "incomplete_count",
            )
        ],
    )
    lines += [
        "",
        "Including client-side queueing "
        f"(concurrency {config.get('CONCURRENCY', 'unknown')}, "
        f"{config.get('MEASURE_REQUESTS', 'unknown')} requests arriving at once)",
    ]
    lines += table(
        ("Metric (s)", "Mean", "p50", "p90"),
        rows([("ttft", "Client TTFT"), ("e2e", "Client E2E")]),
    )
    return lines + [""]


def visible(data):
    active = data.get("active_devices")
    for name, node in metrics(data):
        if name.startswith("transfer.prefill."):
            continue
        if name.startswith("energy."):
            _, device, key = name.split(".", 2)
            if (
                active is not None
                and device not in [*active, "total"]
                and key != "energy_j"
            ):
                continue
        yield name, node


def metric_table(values):
    if isinstance(values, list):
        values = {
            r.get("request_id", str(i)): r.get("metrics", r)
            for i, r in enumerate(values)
        }
    rows = []
    for name, node in metrics(values):
        if node["status"] == "not_collected":
            continue
        value = node.get("value")
        for suffix, item in value.items() if isinstance(value, dict) else [("", value)]:
            rows.append(
                (
                    label(name) + ("." + LABELS.get(suffix, suffix) if suffix else ""),
                    item,
                    node["unit"],
                    node["status"],
                )
            )
    return table(("Metric", "Value", "Unit", "Status"), rows)


def observed(data):
    lines = [
        "",
        "## Observed requests",
        "",
    ]
    for title, key in (
        ("Server stages (OTEL)", "server"),
        ("KV Transfer / KV Load (H2D+transform)", "transfer"),
    ):
        lines += [
            "",
            "### " + title,
            "",
            "Server durations come from OTEL spans; gauges are time averages."
            if key == "server"
            else "Decoder transfer counters and host KV copy events.",
        ]
        for role, values in data.get(key, {}).items():
            if key == "transfer" and role == "prefill":
                continue
            lines += ["", "#### " + label(role)] + metric_table(
                {k: v for k, v in values.items() if k not in ("requests", "boundaries")}
            )
            for field in ("requests", "boundaries"):
                if values.get(field):
                    heading = (
                        "Prometheus metrics: change over the measurement window"
                        if field == "boundaries"
                        else label(role) + " request details"
                    )
                    lines += folded(heading, metric_table(values[field]))
    lines += folded(
        "Client request details",
        metric_table(data.get("client", {}).get("requests", [])),
    )
    shares = data.get("request_phase_share", {}).get("summary", {}).get("value") or {}
    lines += [
        "",
        "### Request time breakdown (share of service time)",
        "",
        "Non-overlapping request intervals between client send and completion.",
    ] + table(
        ("Phase", "Share (%)"), [(LABELS.get(k, k), v * 100) for k, v in shares.items()]
    )
    lines += folded(
        "Request time breakdown details",
        metric_table(data.get("request_phase_share", {}).get("requests", {})),
    )
    resources = data.get("phase_resources", {})
    rows = [
        (
            r["device"],
            r["state"],
            r["metric"],
            gib(r["value"].get("mean"))
            if r["unit"] == "bytes"
            else r["value"].get("mean"),
            gib(r["value"].get("max"))
            if r["unit"] == "bytes"
            else r["value"].get("max"),
            "GiB" if r["unit"] == "bytes" else r["unit"],
        )
        for r in resources.get("samples", [])
        if r["status"] != "not_collected"
    ]
    devices = sorted({r[0] for r in rows} - {"system"})
    means = {(d, m): avg for d, phase, m, avg, _, _ in rows if phase == "all"}
    overview = [
        (
            d,
            "yes" if d in data.get("active_devices", []) else "no",
            *(means.get((d, m)) for m in ("utilization", "power", "memory")),
        )
        for d in devices
    ]
    lines += (
        [
            "",
            "### Device resources",
            "",
            "All-window means from sampled device counters.",
        ]
        + table(
            ("device", "active", "utilization (%)", "power (W)", "memory (GiB)"),
            overview,
        )
        + folded(
            "Device resource details",
            table(("device", "phase", "metric", "mean", "max", "unit"), rows)
            + metric_table(resources.get("device_phase_active_ratio", {})),
        )
    )
    lines += [
        "",
        "### Energy",
        "",
        "Energy is integrated from sampled power over the measurement window. "
        "Inactive devices: energy only; token efficiency applies to active devices.",
    ]
    energy = {
        name.removeprefix("energy."): node
        for name, node in visible(data)
        if name.startswith("energy.")
    }
    return lines + metric_table(energy)


REASONS = {
    "device 시간 없음: 호스트 관측 구간 기준 하한 추정": (
        "device time unavailable; lower bound estimated from host interval"
    ),
    "kv_dtype_or_context_missing(H4)": "KV data type or context length is missing",
    "WP9에서 산출": "derived offline only",
    "copy_ 동기 여부 미확인(H4)": (
        "copy_ synchronization not verified; H2D bandwidth not reported"
    ),
    "전송 count 단위 미확인(H5)": "transfer counter granularity not verified",
    "device_time_missing_or_exceeds_window(H4)": "device time missing or out of range",
    "미구현(후속)": "this metric is not implemented",
    "not_applicable": "not applicable to this configuration",
    "zero_count": "no observations in the measurement interval",
    "missing_raw": "required raw data is missing",
    "이론치 미설정": "theoretical bandwidth is not configured",
}


def uncollected(data):
    return [
        f"- {label(name)}: {english_reason(node.get('reason'))}"
        for name, node in visible(data)
        if node["status"] == "not_collected"
    ]


def english_reason(reason):
    if reason in REASONS.values():
        return reason
    return REASONS.get(
        reason, reason.replace("_", " ") if reason else "missing required data"
    )


def english_reasons(value):
    if isinstance(value, dict):
        return {
            k: english_reason(v)
            if k == "reason" and isinstance(v, str)
            else english_reasons(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [english_reasons(v) for v in value]
    return value
