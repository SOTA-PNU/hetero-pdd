# SPDX-License-Identifier: Apache-2.0
"""Collect one offline, single-device operator profile in the device's venv."""

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[3])

from tools.hetero.orchestrate import clean_base_env
from tools.hetero.profiler.log import log_line

REPO = Path(__file__).resolve().parents[3]


def configure(device, out):
    env = clean_base_env()
    env.update(
        PYTHONPATH=str(REPO),
        VLLM_CACHE_ROOT=str(out / "cache"),
        CUDA_VISIBLE_DEVICES="0" if device == "gpu" else "",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
    )
    if device == "gpu":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    else:
        # Keep aligned with orchestrate's standalone NPU role (no NIXL).
        env.update(
            RBLN_DEVICES="0",
            RBLN_PROFILER="1",
            VLLM_RBLN_USE_VLLM_MODEL="1",
            VLLM_RBLN_USE_DEVICE_TENSOR="1",
            VLLM_RBLN_COMPILE_MODEL="1",
            VLLM_RBLN_COMPILE_STRICT_MODE="1",
            VLLM_RBLN_SAMPLER="0",
            RBLN_USE_CUSTOM_KERNEL="0",
            RBLN_ROOT_IP="127.0.0.1",
            RBLN_LOCAL_IP="127.0.0.1",
        )
    os.environ.clear()
    os.environ.update(env)
    os.chdir(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "npu"), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    for flag, default in (
        ("max-model-len", 4352),
        ("max-num-seqs", 4),
        ("max-num-batched-tokens", 128),
        ("block-size", 128),
        ("num-blocks", 160),
    ):
        parser.add_argument("--" + flag, type=int, default=default)
    args = parser.parse_args(argv)
    out, model = args.out.expanduser().resolve(), args.model_path.expanduser().resolve()
    if out.is_relative_to(REPO) or (out.exists() and any(out.iterdir())):
        raise ValueError(f"{out}: output must be empty and outside repository")
    if args.max_model_len < 2:
        raise ValueError("max-model-len must allow input and output tokens")
    inputs = min(4096 if args.phase == "prefill" else 32, args.max_model_len - 1)
    outputs = min(
        1 if args.phase == "prefill" else args.decode_tokens,
        args.max_model_len - inputs,
    )
    out.mkdir(parents=True, exist_ok=True)
    configure(args.device, out)
    settings = dict(
        model=str(model),
        served_model_name=args.model_name,
        model_impl="vllm",
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
        num_gpu_blocks_override=args.num_blocks,
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        seed=0,
        generation_config="vllm",
        profiler_config=dict(profiler="torch", torch_profiler_dir=str(out)),
    )
    meta = dict(
        device=args.device,
        model=args.model_name,
        phase=args.phase,
        settings=settings,
        input_tokens=inputs,
        output_tokens=outputs,
        requests=4,
        started_ns=time.time_ns(),
        status="RUNNING",
    )

    def save():
        meta["files"] = [
            str(p.relative_to(out))
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and "cache" not in p.relative_to(out).parts
            and p.name != "meta.json"
        ]
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Register before SDK import so its exit-time exports precede the final inventory.
    atexit.register(save)
    llm = None
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(**settings)
        token = llm.get_tokenizer().encode("hello", add_special_tokens=False)[0]
        prompts = [dict(prompt_token_ids=[token] * inputs) for _ in range(4)]
        sampling = SamplingParams(temperature=0, max_tokens=outputs, ignore_eos=True)
        llm.generate(prompts, sampling, use_tqdm=False)
        llm.start_profile()
        try:
            meta["measurement_start"] = dict(
                time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()
            )
            result = llm.generate(prompts, sampling, use_tqdm=False)
            meta["measurement_end"] = dict(
                time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns()
            )
            meta["output_token_counts"] = [len(r.outputs[0].token_ids) for r in result]
        finally:
            llm.stop_profile()
        meta["status"] = "PASS"
    except Exception as exc:
        meta.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        except Exception as exc:
            meta.update(status="FAILED", error=f"shutdown: {exc}")
            raise
        finally:
            meta["ended_ns"] = time.time_ns()
            save()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_line("opprofile", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
