# SPDX-License-Identifier: Apache-2.0
"""Render existing PDD analysis metrics without recomputing them."""

import json


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


def observed(data):
    lines = [
        "",
        "## Observed requests",
        "",
        "Streaming synthetic tokens; not comparable to the legacy fixed prompt.",
    ]
    sections = {
        "Client": data.get("client", {}).get("summary", {}),
        "Server stages (OTEL)": data.get("server", {}),
        "KV Transfer / KV Load (H2D+transform)": data.get("transfer", {}),
        "M14a": data.get("request_phase_share", {}).get("summary", {}),
        "Device resources": data.get("phase_resources", {}),
        "Energy": data.get("energy", {}),
    }
    lines += [f"Active devices (energy total): {data.get('active_devices', [])}"]
    for title, values in sections.items():
        lines += [
            "",
            f"### {title}",
            "",
            "| Metric | Value | Unit | Status |",
            "| --- | --- | --- | --- |",
        ]
        for name, node in metrics(values):
            value = json.dumps(node.get("value"), ensure_ascii=False).replace(
                "|", "\\|"
            )
            lines.append(f"| {name} | {value} | {node['unit']} | {node['status']} |")
    return lines


def uncollected(data):
    return [
        f"- {name}: {node.get('reason') or 'missing_raw'}"
        for name, node in metrics(data)
        if node["status"] == "not_collected"
    ]
