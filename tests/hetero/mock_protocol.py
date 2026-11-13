# SPDX-License-Identifier: Apache-2.0
"""Unverified wire/metric assumptions live here, never in production parsers."""

from __future__ import annotations

import json

# TODO(verify: H0/H1): Check deployed vLLM SSE token_ids, final-token finish_reason,
# usage-only chunk and [DONE]. Original loadgen and real chunks support this shape,
# but do not preserve the complete wire stream. KV metadata below is synthetic.
PREFILL_KV = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}
DECODE_KV = {
    **PREFILL_KV,
    "do_remote_decode": False,
    "do_remote_prefill": True,
    "remote_engine_id": "mock-prefill",
    "remote_block_ids": [0],
}


def completion_payload(body, tokens, finish_reason, *, usage=False):
    choice = {
        "text": "".join(f"t{token} " for token in tokens),
        "finish_reason": finish_reason,
    }
    if body.get("return_token_ids"):
        choice["token_ids"] = tokens
    payload: dict[str, object] = {"choices": [choice]}
    if usage:
        prompt, output = len(body["prompt"]), body["max_tokens"]
        payload["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
        }
    return payload


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


# TODO(verify: H0/H5): Confirm vLLM names/labels and NIXL histogram count units.
# Values and buckets are synthetic; they do not assert HW units or device mapping.
VLLM_NAMES = {
    "success": "vllm:request_success_total",
    "latency": "vllm:e2e_request_latency_seconds",
    "running": "vllm:num_requests_running",
    "xfer": "vllm:nixl_xfer_time_seconds",
    "bytes": "vllm:nixl_bytes_transferred",
}
MODEL_LABELS = 'model_name="mock"'
SUCCESS_LABELS = MODEL_LABELS + ',finished_reason="length"'
# PC4 confirmed (2026-11-06), all-smi 0.28.1: names and labels below.
# Real service binds 0.0.0.0; /-/ready returns 503 before collection, then 200.
# NPU gpu_info carries pci_bus_id; NVIDIA gpu_info did not expose a bus ID.
# TODO(verify: H2): Verify device mapping stability and readings under load.
ALLSMI_NAMES = (
    "all_smi_cpu_utilization",
    "all_smi_memory_used_bytes",
    "all_smi_gpu_utilization",
    "all_smi_gpu_memory_used_bytes",
    "all_smi_gpu_power_consumption_watts",
)
DEVICES = ("Mock GPU", "RBLN-CA22", "RBLN-CA22")
ALLSMI_HOST_LABELS = 'instance="mock",hostname="mock",index="0"'


def histogram(name, samples):
    return [
        f"# TYPE {name} histogram",
        f'{name}_bucket{{{MODEL_LABELS},le="+Inf"}} {len(samples)}',
        f"{name}_sum{{{MODEL_LABELS}}} {sum(samples)}",
        f"{name}_count{{{MODEL_LABELS}}} {len(samples)}",
    ]


def metrics_text(role, durations, running):
    if role == "all-smi":
        lines = [f"# TYPE {name} gauge" for name in ALLSMI_NAMES]
        lines.extend(
            (
                f'{ALLSMI_NAMES[0]}{{cpu_model="Mock CPU",{ALLSMI_HOST_LABELS}}} 12.5',
                f"{ALLSMI_NAMES[1]}{{{ALLSMI_HOST_LABELS}}} 1073741824",
                "# TYPE all_smi_gpu_info gauge",
            )
        )
        for index, device in enumerate(DEVICES):
            labels = (
                f'gpu="{device}",instance="mock",gpu_uuid="mock-{index}",'
                f'gpu_index="{index}"'
            )
            for name, value in zip(ALLSMI_NAMES[2:], (25, 536870912, 45)):
                lines.append(f"{name}{{{labels}}} {value + index}")
            info = (
                'type="GPU"'
                if index == 0
                else (
                    f'type="NPU",device_path="rbln{index - 1}",'
                    f'pci_bus_id="0000:{index:02x}:00.0"'
                )
            )
            lines.append(f"all_smi_gpu_info{{{labels},{info}}} 1")
    else:
        success, gauge = VLLM_NAMES["success"], VLLM_NAMES["running"]
        lines = [
            f"# TYPE {success} counter",
            f"{success}{{{SUCCESS_LABELS}}} {len(durations)}",
            f"# TYPE {gauge} gauge",
            f"{gauge}{{{MODEL_LABELS}}} {running}",
        ]
        lines += histogram(VLLM_NAMES["latency"], durations)
        if role.startswith("decode"):
            lines += histogram(VLLM_NAMES["xfer"], [0.01] * len(durations))
            lines += histogram(VLLM_NAMES["bytes"], [1024] * len(durations))
    return ("\n".join(lines) + "\n").encode()
