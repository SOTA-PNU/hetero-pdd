# SPDX-License-Identifier: Apache-2.0
import json
from statistics import mean

from tools.hetero.profiler import expected, server
from tools.hetero.profiler.common import (
    active_ratio,
    attribute_samples,
    interval_segments,
    metric,
)
from tools.hetero.profiler.common import union as union

DEVICES = ("gpu", "npu0", "npu1")
DEVICE_METRICS = dict(
    utilization="gpu_utilization",
    memory="gpu_memory_used_bytes",
    power="gpu_power_consumption_watts",
)
ROLE_DEVICE = dict(prefill="gpu", decode0="npu0", decode1="npu1")


COLUMN_DOCS = {
    "request_phase_share": """Trace: clip to [send, completed]; assign each
instant once: decode > layout (H2D+reorder) > kv_transfer > prefill > server queues.
B0: queue/prefill/decode; PDD: p_queue/p_prefill/kv_transfer/layout/d_queue/decode.
Queue ties: D before P. Other is the uncovered service interval; shares sum to 1.
Monitor: first-token/after-first-token approximation. Summary: request mean.
M06-M10 retain their original overlapping elapsed times."""
}


def device_states(phases, window):
    intervals = (ab for group in phases.values() for ab in group)
    result = {}
    for a, b in interval_segments(intervals, window):
        active = {
            p for p, spans in phases.items() if any(x < b and y > a for x, y in spans)
        }
        state = "mixed" if len(active) > 1 else next(iter(active), "idle")
        result.setdefault(state, []).append((a, b))
    return result


def classify_devices(text, links):
    gpu_uuid = links["nvidia_smi"].split(",")[0].strip()
    devices = json.loads(links["rbln_smi"])["devices"]
    pci = {d["pci"]["bus_id"].lower(): f"npu{d['npu']}" for d in devices}
    result = {}
    for sample in server.samples(text):
        labels = sample.labels
        if sample.name == "all_smi_gpu_info":
            uuid = labels["gpu_uuid"]
            role = pci.get(labels.get("pci_bus_id", "").lower())
            role = "gpu" if uuid == gpu_uuid else role
            if role:
                result[uuid] = role
    return result


def read_samples(collected, links, window):
    series = {(d, k): [] for d in DEVICES for k in DEVICE_METRICS}
    series.update({("system", k): [] for k in ("cpu", "memory")})
    names = {"all_smi_" + v: k for k, v in DEVICE_METRICS.items()}
    names.update(all_smi_cpu_utilization="cpu", all_smi_memory_used_bytes="memory")
    for row in collected:
        t = row["mono_ns"] / 1e9
        if row["endpoint"] != "all-smi" or not window[0] <= t <= window[1]:
            continue
        values = {}
        if row["status"] == 200:
            roles = {None: "system", **classify_devices(row["text"], links)}
            for sample in server.samples(row["text"]):
                role = roles.get(sample.labels.get("gpu_uuid"))
                key = role, names.get(sample.name)
                if key in series:
                    values[key] = sample.value
        for key, points in series.items():
            points.append((t, values.get(key)))
    return series


def mono(ns, anchors):
    anchor = min(anchors, key=lambda x: abs(x["time_ns"] - ns))
    return (ns - anchor["time_ns"] + anchor["monotonic_ns"]) / 1e9


def phase_intervals(rows, ids, spans, events, anchors, trace):
    phases = {d: {} for d in DEVICES}

    def add(role, phase, a, b):
        phases[ROLE_DEVICE.get(role, role)].setdefault(phase, []).append((a, b))

    for role, names in ids.items():
        for row in rows:
            if row["request_id"] not in names:
                continue
            rid = expected.server_request_id(row["request_id"])
            send, first, end = (
                row[k + "_monotonic"]
                for k in ("actual_send", "first_output", "completed")
            )
            if trace:
                span = spans.get(role, {}).get(rid)
                if span is None:
                    continue
                # vLLM 0.22 output_processor.py:730-747: span arrival + TTFT.
                first = (
                    mono(span["start"], anchors)
                    + span["gen_ai.latency.time_to_first_token"]
                )
                send = first - span["gen_ai.latency.time_in_model_prefill"]
                end = first + span["gen_ai.latency.time_in_model_decode"]
            if not role.startswith("decode"):
                add(role, "prefill", send, first)
            if role != "prefill":
                add(role, "decode", first, end)
        starts = {
            e["req_id"]: e["monotonic_ns"]
            for e in events.get(role, [])
            if e["kind"] == "kv_xfer_start"
        }
        for event in events.get(role, []):
            if event["kind"] == "kv_xfer_done" and event["req_id"] in starts:
                begin = starts[event["req_id"]] / 1e9
                add(role, "kv_transfer", begin, event["monotonic_ns"] / 1e9)
            if event["kind"] == "kv_h2d":
                add(role, "layout", event["start_ns"] / 1e9, event["end_ns"] / 1e9)
    return phases


def exclusive_phase_shares(parts, window):
    order = ("decode", "layout", "kv_transfer", "p_prefill", "prefill")
    order += ("d_queue", "p_queue", "queue")
    parts = {key: parts[key] for key in order if key in parts}
    seconds = dict.fromkeys([*parts, "other"], 0.0)
    for a, b in interval_segments(parts.values(), window):
        phase = next((p for p, (x, y) in parts.items() if x <= a < b <= y), "other")
        seconds[phase] += b - a
    return {p: s / (window[1] - window[0]) for p, s in seconds.items()}


def request_phase_share(rows, ids, spans, events, anchors, trace):
    requests = {}
    source = "spans+events" if trace else "requests.jsonl"
    status = "measured" if trace else "estimated"
    for row in filter(expected.request_succeeded, rows):
        rid = expected.server_request_id(row["request_id"])
        duration = row["completed_monotonic"] - row["actual_send_monotonic"]
        first = row["first_output_monotonic"] - row["actual_send_monotonic"]
        parts = dict(first_token=first, after_first_token=duration - first)
        roles = [r for r, names in ids.items() if row["request_id"] in names]
        ratios = {k: v / duration for k, v in parts.items()}
        if trace:
            selected = {
                r: [
                    e
                    for e in events.get(r, [])
                    if expected.engine_request_base(e.get("req_id")) == rid
                ]
                for r in roles
            }
            phases = phase_intervals([row], ids, spans, selected, anchors, True)
            parts = {p: ab[0] for device in phases.values() for p, ab in device.items()}
            pdd = "prefill" in roles
            if pdd:
                parts["p_prefill"] = parts.pop("prefill", None)
                for key in ("layout", "kv_transfer"):
                    parts.setdefault(key, None)
            for role in roles:
                key = "p_queue" if role == "prefill" else "d_queue" if pdd else "queue"
                span = spans.get(role, {}).get(rid, {})
                queue = span.get("gen_ai.latency.time_in_queue")
                start = mono(span["start"], anchors) if span else 0
                parts[key] = (start, start + queue) if queue is not None else None
            window = row["actual_send_monotonic"], row["completed_monotonic"]
            ratios = (
                exclusive_phase_shares(parts, window)
                if None not in parts.values()
                else None
            )
        requests[row["request_id"]] = metric(ratios, "ratio", source, status=status)
    values = [node["value"] for node in requests.values()]
    summary = None
    if values and None not in values:
        summary = {k: mean(v[k] for v in values) for k in values[0]}
    return dict(
        definition=COLUMN_DOCS["request_phase_share"],
        requests=requests,
        summary=metric(summary, "ratio", source, status=status),
    )


def phase_resources(series, phases, window, trace):
    system, ratios, rows = {}, {}, []
    status = "measured" if trace else "estimated"
    source = "spans+events" if trace else "requests.jsonl"
    for device, parts in phases.items():
        ratios[device] = {}
        for phase, intervals in parts.items():
            system.setdefault(phase, []).extend(intervals)
            ratios[device][phase] = metric(
                active_ratio(intervals, window), "ratio", source, status=status
            )
    states = {
        d: device_states(p, window) for d, p in dict(phases, system=system).items()
    }
    for (device, name), points in series.items():
        unit = "W" if name == "power" else "bytes" if name == "memory" else "%"
        for state, intervals in dict(states[device], all=[window]).items():
            avg, maximum, coverage, _ = attribute_samples(points, intervals)
            value = dict(mean=avg, max=maximum) if avg is not None else None
            node = metric(
                value, unit, server.SERIES, status="estimated", coverage=coverage
            )
            rows.append(dict(device=device, state=state, metric=name, **node))
    return dict(samples=rows, device_phase_active_ratio=ratios)


def energy_efficiency(power, window, tokens):
    result, valid = {}, []
    for device in (*DEVICES, "total"):
        coverage, reason = 0, "missing_device"
        if device != "total":
            points = [
                (t, v) for t, v in power.get(device, []) if window[0] <= t <= window[1]
            ]
            _, _, coverage, energy = attribute_samples(points, [window], linear=True)
            reason = "coverage_below_80_percent"
            if sum(v is not None for _, v in points) < 2:
                reason = "insufficient_power_samples"
            energy = energy if coverage >= 0.8 and len(points) >= 2 else None
            if energy is not None:
                valid.append((energy, coverage))
        else:
            energy = sum(e for e, _ in valid) if len(valid) == 3 else None
            coverage = min((c for _, c in valid), default=0)
        per_token = energy / tokens if energy is not None and tokens else None
        efficiency = tokens / energy if energy else None
        values = dict(
            energy_j=(energy, "J"),
            energy_per_output_token_j_est=(per_token, "J/token"),
            tok_per_s_per_w_est=(efficiency, "tokens/s/W"),
        )
        result[device] = {
            k: metric(v, u, server.SERIES, reason, "estimated", coverage=coverage)
            for k, (v, u) in values.items()
        }
    return result
