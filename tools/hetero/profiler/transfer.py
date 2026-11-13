# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from statistics import mean

from tools.hetero.profiler import expected, resources, server
from tools.hetero.profiler.common import active_ratio, metric

COPY_SYNC_CONFIRMED = False  # TODO(verify: H4): copy_ completion semantics.
SOURCE = "rbln_metrics/events_*.jsonl"
COPY_REASON = "copy_ 동기 여부 미확인(H4)"
DEVICE_REASON = "device_time_missing_or_exceeds_window(H4)"
THEORY_REASON = "이론치 미설정"
MEMORY_FIELDS = ("decode_mem_bw_gbps_est", "mem_bw_efficiency_est")


def device_time_ratio(steps, window):
    values, raw = [s["device"] for s in steps], None
    if values and None not in values:
        raw = sum(values) / 1e6 / (window[1] - window[0])
    value = raw if raw is not None and raw <= 1 else None
    return metric(value, "ratio", SOURCE, DEVICE_REASON, raw_value=raw)


def h2d_bandwidth(size, seconds):
    value = size / seconds / 1e9 if size is not None and seconds else None
    value = value if COPY_SYNC_CONFIRMED else None
    return metric(value, "GB/s", SOURCE, COPY_REASON, "estimated")


def memory_bandwidth(config, steps, running, rows):
    path = Path(config.get("model_path", ""))
    weights = list(path.glob("*.safetensors"))
    steps = [s for s in steps if s["phase"] == "decode"]
    times = [s["device"] for s in steps]
    host_timing = None in times
    if host_timing:
        times = [
            (s["span_end_ns"] - s["span_start_ns"]) / 1e3
            if s.get("span_end_ns") is not None and s.get("span_start_ns") is not None
            else None
            for s in steps
        ]
    value, reason = None, "model_config_or_weights_missing"
    if (path / "config.json").is_file() and weights:
        model = json.loads((path / "config.json").read_text())
        width = model["hidden_size"] // model["num_attention_heads"]
        width = model.get("head_dim", width)
        dtype = config.get("kv_cache_dtype", config.get("dtype"))
        size = {"float16": 2, "bfloat16": 2, "float32": 4, "fp8": 1}.get(dtype)
        context = [
            r["actual_input_tokens"] + (r["actual_output_tokens"] - 1) / 2 for r in rows
        ]
        fallback = running * mean(context) if running is not None and context else None
        contexts = [s.get("context_len_sum") for s in steps]
        contexts = [fallback if c is None else c for c in contexts]
        valid = times and None not in times and sum(times) > 0
        reason = "kv_dtype_or_context_missing(H4)" if valid else DEVICE_REASON
        if size and valid and None not in contexts:
            kv = 2 * width * size * model["num_hidden_layers"]
            kv *= model["num_key_value_heads"]
            weight_bytes = sum(f.stat().st_size for f in weights) * len(times)
            value = (weight_bytes + sum(contexts) * kv) / (sum(times) * 1e3)
    source = "model/config.json+safetensors+steps"
    bandwidth = metric(value, "GB/s", source, reason, "estimated")
    exact = all(s.get("context_len_sum") is not None for s in steps)
    bandwidth["context_source"] = "step" if exact else "running*mean_context_est"
    theoretical = float(os.environ.get("NPU_MEM_BW_GBPS") or 0)
    efficiency = value / theoretical if value is not None and theoretical else None
    result = dict(decode_mem_bw_gbps_est=bandwidth)
    reason = reason if theoretical else THEORY_REASON
    result["mem_bw_efficiency_est"] = metric(
        efficiency, "ratio", "NPU_MEM_BW_GBPS", reason, "estimated"
    )
    if host_timing:
        for node in result.values():
            if node["value"] is not None:
                node["reason"] = "device 시간 없음: 호스트 관측 구간 기준 하한 추정"
    return result


def transfer_metrics(config, rows, events, servers, links, window):
    gpu_memory = {}
    for key, unit in zip(MEMORY_FIELDS, ("GB/s", "ratio")):
        gpu_memory[key] = metric(None, unit, "WP9", "WP9에서 산출")
    result, memory = {}, dict(gpu=gpu_memory)
    gpu = links["nvidia_smi"].strip().split(",")
    specs = {"gpu": (int(gpu[-2]), int(gpu[-1]))}
    pdd = config["configuration"] == "pdd"
    missing = "missing_nixl" if pdd else "not_applicable"
    # TODO(verify: H5): request-count equality does not establish transfer units.
    speeds = {"8.0GT/s": 3, "16.0GT/s": 4, "32.0GT/s": 5}
    for d in json.loads(links["rbln_smi"])["devices"]:
        pci = d.get("pci", {})
        gen = speeds.get(
            pci.get("link_speed"), int(os.environ.get("NPU_PCIE_GEN") or 0)
        )
        width = int(pci.get("link_width") or os.environ.get("NPU_PCIE_WIDTH") or 0)
        specs[f"npu{d['npu']}"] = gen, width
    for role, data in servers.items():
        device = resources.ROLE_DEVICE.get(role, role)
        gen, width = specs[device]
        link = {3: 0.985, 4: 1.969, 5: 3.938}.get(gen, 0) * width
        targets = (
            [data]
            if role != "prefill"
            else [s for r, s in servers.items() if r.startswith("decode")]
        )
        pairs = [
            [d["boundaries"].get("vllm:nixl_" + k, {}) for d in targets]
            for k in ("xfer_time_seconds", "bytes_transferred")
        ]
        valid = all(d.get("status") == "measured" for pair in pairs for d in pair)
        seconds, size = (sum(d["sum"] for d in pair) if valid else 0 for pair in pairs)
        speed = size / seconds / 1e9 if seconds else None
        xfer = data["boundaries"].get("vllm:nixl_xfer_time_seconds")
        requests, grouped = {}, {}
        for e in events.get(role, []):
            if e.get("req_id"):
                grouped.setdefault(expected.engine_request_base(e["req_id"]), {})[
                    e["kind"]
                ] = e
        for rid, group in grouped.items():
            a, b, h = [
                group.get(k, {}) for k in ("kv_xfer_start", "kv_xfer_done", "kv_h2d")
            ]
            seconds = (b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 if a and b else None
            values = dict(kv=seconds, h2d=h.get("h2d_ns"), reorder=h.get("reorder_ns"))
            requests[rid] = {
                k: metric(v / 1e9 if k != "kv" and v is not None else v, "s", SOURCE)
                for k, v in values.items()
            }
            requests[rid]["h2d"]["measurement"] = "host_call"
            requests[rid]["bytes_per_request_est"] = metric(
                h.get("bytes_est"), "bytes", SOURCE, status="estimated"
            )
        steps = [e for e in events.get(role, []) if e["kind"] == "step"]
        times = [s["device"] / 1e6 for s in steps if s["device"] is not None]
        h2d = [e for e in events.get(role, []) if e["kind"] == "kv_h2d"]
        bandwidth = h2d_bandwidth(
            sum(e["bytes_est"] for e in h2d), sum(e["h2d_ns"] for e in h2d) / 1e9
        )
        rate = speed if device == "gpu" else bandwidth["value"]
        data = dict(requests=requests, h2d_gbps_est=bandwidth)
        if not pdd or device == "gpu":
            data["h2d_gbps_est"] = metric(None, "GB/s", SOURCE, "not_applicable")
        data["nixl_gbps"] = metric(speed, "GB/s", server.BOUNDARY, missing)
        data["bytes_per_transfer"] = metric(
            None,
            "bytes",
            server.BOUNDARY,
            "전송 count 단위 미확인(H5)" if pdd else "not_applicable",
        )
        data["link_theoretical"] = metric(
            link or None, "GB/s", "links.json", THEORY_REASON
        )
        efficiency = rate / link if rate is not None and link else None
        reason = (
            (COPY_REASON if device != "gpu" else "missing_nixl")
            if pdd
            else "not_applicable"
        )
        reason = reason if link else THEORY_REASON
        data["pcie_efficiency"] = metric(
            efficiency, "ratio", "transfer/links.json", reason, "estimated"
        )
        data["kv_transfer_seconds"] = xfer or metric(
            None, "s", server.BOUNDARY, missing
        )
        result[role] = data
        if device == "gpu":
            continue
        value = mean(times) if times and len(times) == len(steps) else None
        data["device_seconds"] = metric(value, "s", SOURCE, DEVICE_REASON)
        data["npu_device_time_ratio"] = device_time_ratio(steps, window)
        intervals = [(s["span_start_ns"] / 1e9, s["span_end_ns"] / 1e9) for s in steps]
        ratio = active_ratio(intervals, window) if steps else None
        result[role]["npu_step_window_ratio"] = metric(
            ratio, "ratio", SOURCE, status="estimated"
        )
        running = servers[role]["summary"]["avg_running_requests"]["value"]
        memory[device] = memory_bandwidth(config, steps, running, rows)
    return result, memory
