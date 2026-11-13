# SPDX-License-Identifier: Apache-2.0
from statistics import mean

from prometheus_client.parser import text_string_to_metric_families

from tools.hetero.profiler.common import attribute_samples, metric

BOUNDARY = "monitor/boundaries.jsonl"
TRACE = "otel/traces.jsonl"
SERIES = "monitor/metrics.jsonl"

# H0 verified (2026-11-06): gen_ai.latency.*; H5: Prometheus family names.
LATENCIES = dict(
    queue="time_in_queue",
    prefill="time_in_model_prefill",
    decode="time_in_model_decode",
)
GAUGES = dict(
    kv_cache_usage="kv_cache_usage_perc",
    avg_running_requests="num_requests_running",
    waiting_requests="num_requests_waiting",
)


def samples(text):
    return [s for f in text_string_to_metric_families(text) for s in f.samples]


def boundary_deltas(before, after):
    snapshots = []
    for text in (before, after):
        families = {}
        for family in text_string_to_metric_families(text):
            if family.type in ("histogram", "counter", "summary"):
                families[family.name] = {
                    (s.name, tuple(sorted(s.labels.items()))): s.value
                    for s in family.samples
                    if s.name.endswith(("_sum", "_count", "_total"))
                }
        snapshots.append(families)
    result = {}
    for name in snapshots[0].keys() | snapshots[1].keys():
        a, b = (snapshot.get(name, {}) for snapshot in snapshots)
        delta = {key: b[key] - a[key] for key in a.keys() & b.keys()}
        reason = "missing_boundary_series" if a.keys() != b.keys() else None
        if any(v < 0 for v in delta.values()):
            reason = "counter_reset"
        total = sum(v for (k, _), v in delta.items() if k.endswith(("_sum", "_total")))
        counter = any(k.endswith("_total") for k, _ in delta)
        count = sum(v for (k, _), v in delta.items() if k.endswith("_count"))
        if not counter and count == 0:
            reason = reason or "zero_count"
        value = None if reason else total if counter else total / count
        unit = "bytes" if "bytes" in name else "s" if "seconds" in name else "count"
        result[name] = metric(value, unit, BOUNDARY, reason, sum=total, count=count)
    return result


def server_metrics(boundaries, collected, spans, trace, window, roles):
    result = {}
    source = TRACE if trace else BOUNDARY
    for role in roles:
        edges = {}
        for row in boundaries:
            if row["endpoint"] == role and row["status"] == 200:
                edges[row["label"]] = row["text"]
        deltas = boundary_deltas(edges.get("before", ""), edges.get("after", ""))
        requests, summary = {}, {}
        for rid, span in spans.get(role, {}).items():
            requests[rid] = {
                k: metric(span.get("gen_ai.latency." + a), "s", TRACE)
                for k, a in LATENCIES.items()
            }
        for key in LATENCIES:
            values = [r[key]["value"] for r in requests.values()]
            values = [v for v in values if v is not None]
            summary[key] = deltas.get(f"vllm:request_{key}_time_seconds")
            if trace or summary[key] is None:
                summary[key] = metric(mean(values) if values else None, "s", source)
        timeline = []
        for row in collected:
            t = row["mono_ns"] / 1e9
            if row["endpoint"] == role and window[0] <= t <= window[1]:
                values = samples(row["text"]) if row["status"] == 200 else []
                timeline.append((t, {s.name: s.value for s in values}))
        for key, name in GAUGES.items():
            points = [(t, values.get("vllm:" + name)) for t, values in timeline]
            value, _, coverage, _ = attribute_samples(points, [window])
            unit = "ratio" if key == "kv_cache_usage" else "requests"
            summary[key] = metric(
                value, unit, SERIES, status="estimated", coverage=coverage
            )
        result[role] = dict(summary=summary, requests=requests, boundaries=deltas)
    return result
