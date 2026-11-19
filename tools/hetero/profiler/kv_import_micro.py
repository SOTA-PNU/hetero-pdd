"""Measure completed CA22 transfers on flat buffers and real HND allocation shapes."""

import argparse
import ctypes
import inspect
import json
import math
import os
import re
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch_rbln  # noqa: F401  (registers the installed RBLN backend)
from rebel._C.vmem import ensure_synced_on_device
from rebel.kv_cache import aligned_tensor

from llmd.config import fields, integer, read_yaml
from vllm_rbln.v1.worker.kv_copy import import_thread_scope


def read_config(path):
    cfg = read_yaml(path)
    keys = {
        "physical_npu",
        "uuid",
        "layers",
        "layer_shape",
        "warmup",
        "samples",
        "official_mib",
        "conditions",
    }
    options = {"capture_physical_layout", "use_worker_cpu_setup"}
    fields(cfg, keys | options, keys, "microbenchmark")
    for option in options:
        if type(cfg.get(option, False)) is not bool:
            raise ValueError(f"{option} must be boolean")
    integer(cfg["physical_npu"], "physical_npu", low=0)
    for key in ["layers", "warmup", "samples"]:
        integer(cfg[key], key)
    shape = cfg["layer_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 6
        or shape[0] != 2
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError(
            "layer_shape must be [2, blocks, heads, groups, tokens, channels]"
        )
    if not isinstance(cfg["conditions"], dict) or not cfg["conditions"]:
        raise ValueError("conditions must be a nonempty map")
    if not isinstance(cfg["official_mib"], list):
        raise ValueError("official_mib must be a list")
    for size in cfg["official_mib"]:
        integer(size, "official_mib")
    for name, case in cfg["conditions"].items():
        if not isinstance(case, dict):
            raise ValueError(f"{name}: expected a condition mapping")
        if "kv_import_threads" in case:
            integer(case["kv_import_threads"], "kv_import_threads")
        if case.get("layout") == "flat":
            keys = {"layout", "mib", "calls"}
            fields(case, keys | {"kv_import_threads"}, keys, name)
            size = integer(case["mib"], "mib") << 20
            calls = integer(case["calls"], "calls")
            if size % (2 * calls):
                raise ValueError(
                    "flat byte count must divide into complete FP16 ranges"
                )
        else:
            keys = {"layout", "ids", "max_run"}
            fields(case, keys | {"kv_import_threads"}, keys, name)
            ids = case["ids"]
            if (
                case["layout"] not in ("kv", "flat_layers")
                or not isinstance(ids, list)
                or not ids
                or any(type(i) is not int or not 0 <= i < shape[1] for i in ids)
                or ids != sorted(set(ids))
            ):
                raise ValueError(
                    f"{name}: expected ascending unique physical block IDs"
                )
            integer(case["max_run"], "max_run")
    return cfg


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def text_command(args):
    return subprocess.run(
        args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
    ).stdout


def vendor_test(cfg, out):
    device = str(cfg["physical_npu"])
    binary = "rblnBandwidthLatencyTest"
    (out / "vendor-help.txt").write_text(text_command([binary, "--help"]))
    (out / "vendor-version.txt").write_text(text_command([binary, "--version"]))
    (out / "vendor-test-list.txt").write_text(text_command([binary, "--test_list"]))
    results = []
    for size in cfg["official_mib"]:
        for phase, count in (("warmup", cfg["warmup"]), ("measure", cfg["samples"])):
            args = [
                binary,
                "--tc=3",
                f"--device_name={device}",
                f"--npu_id_list={device}",
                f"--test_size={size}M",
                f"--iteration_cnt={count}",
                "--skip_test_env_info",
            ]
            if phase == "measure":
                args.append("--warmup")
            raw = text_command(args)
            (out / f"vendor-{size}MiB-{phase}.log").write_text(raw)
            match = re.search(
                r"\[H2D\] HOST -> rbln"
                + device
                + r" : MAX ([\d.]+), MIN ([\d.]+), AVG ([\d.]+)",
                raw,
            )
            size_line = re.search(r"Measure Buffer Size : (0x[0-9a-fA-F]+)", raw)
            if (
                not match
                or not size_line
                or int(size_line[1], 16) != size << 20
                or "PASS" not in raw
            ):
                raise ValueError(f"unexpected official tool result for {size} MiB")
            results.append(
                dict(
                    phase=phase,
                    bytes=size << 20,
                    iterations=count,
                    argv=args,
                    h2d_gb_s=dict(
                        zip(("max", "min", "mean"), map(float, match.groups()))
                    ),
                )
            )
    return results


def ranges(ids, maximum):
    runs = []
    for i in ids:
        if runs and i == runs[-1][1] and i - runs[-1][0] < maximum:
            runs[-1][1] += 1
        else:
            runs.append([i, i + 1])
    return runs


def describe(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        offset_elements=tensor.storage_offset(),
        contiguous=tensor.is_contiguous(),
    )


def host_pages(tensor):
    """Read flags for the CPU VMA containing the aligned tensor's first byte."""
    address = tensor.data_ptr()
    lines = (Path("/proc") / "self" / "smaps").read_text().splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
        if match and int(match[1], 16) <= address < int(match[2], 16):
            data = {}
            for item in lines[index + 1 :]:
                k, _, v = item.partition(":")
                if k in ("KernelPageSize", "AnonHugePages", "Locked", "VmFlags"):
                    data[k] = v.strip()
                if k == "VmFlags":
                    break
            return dict(
                pointer_mod_4096=address % 4096,
                vma=data,
                rbln_pinned=tensor.is_pinned("rbln"),
            )
    raise RuntimeError("host tensor mapping missing")


def allocate(shape, layers, device, materialize):
    inputs, outputs, expected = [], [], []
    prototype = torch.zeros(shape, dtype=torch.float16, device=device)
    for layer in range(layers):
        cpu = aligned_tensor(math.prod(shape)).view(shape)
        cpu.fill_(2)
        if len(shape) > 1:
            for kv in range(shape[0]):
                for block in range(shape[1]):
                    marker = cpu[kv, block].view(-1)
                    # Small integers stay exact in IEEE FP16 and CA22 DLFloat16.
                    marker.fill_(2 + (block % 31))
                    marker[::5] = 2 + (layer % 31)
                    marker[::11] = 2 + kv
        cpu.view(-1)[::7].neg_()
        inputs.append(cpu)
        outputs.append(materialize(prototype))
        expected.append(torch.ones_like(cpu))
    return inputs, outputs, expected


def make_views(case, allocations):
    inputs, outputs, _ = allocations
    if case["layout"] == "flat":
        count = case["calls"]
        elements = (case["mib"] << 19) // count
        # One guard page on either side, 4 KiB aligned source offset.
        for i in range(count):
            select = slice(2048 + i * elements, 2048 + (i + 1) * elements)
            yield inputs[0][select], outputs[0][select], 0
        return
    for layer, (cpu, npu) in enumerate(zip(inputs, outputs)):
        for kv in range(cpu.shape[0]):
            source, target = cpu[kv], npu[kv]
            per_block = source[0].numel()
            for a, b in ranges(case["ids"], case["max_run"]):
                if case["layout"] == "flat_layers":
                    span = slice(a * per_block, b * per_block)
                    yield source.view(-1)[span], target.view(-1)[span], layer
                else:
                    index = a if b == a + 1 else slice(a, b)
                    yield source[index], target[index], layer


def sample(case, buffers, sync):
    threads = case.get("kv_import_threads")
    sync()
    enter = time.perf_counter_ns()
    call_ns = []
    with import_thread_scope(threads) as (previous, applied):
        if threads is not None and applied != threads:
            raise RuntimeError(f"requested {threads} import threads, applied {applied}")
        for source, target, _ in make_views(case, buffers):
            begin = time.perf_counter_ns()
            target.copy_(source)
            call_ns.append(time.perf_counter_ns() - begin)
    returned = time.perf_counter_ns()
    sync()
    completed = time.perf_counter_ns()
    return dict(
        previous_thread_limit=previous,
        applied_thread_limit=applied,
        outer_host_ns=returned - enter,
        batch_complete_ns=completed - enter,
        post_batch_sync_ns=completed - returned,
        copy_call_sum_ns=sum(call_ns),
        call_ns=call_ns,
        calls=len(call_ns),
        reorder_ns=0,
    )


def validate(case, buffers, sync, materialize):
    inputs, outputs, expected = buffers
    if case["layout"] == "flat":
        slices = [slice(2048, 2048 + (case["mib"] << 19))]
    else:
        slices = [(slice(None), i) for i in case["ids"]]
    for cpu, want in zip(inputs, expected):
        for sub in slices:
            # Change each destination's expected bytes between conditions:
            # a skipped copy cannot pass by reusing the previous case's data.
            cpu[sub].neg_()
            want[sub].copy_(cpu[sub])
    with torch.rbln.explain() as trace:
        sample(case, buffers, sync)
    # Observe residual lazy H2D separately; synchronous writes must have
    # reached the device before an engine can import/consume the blocks.
    with torch.rbln.explain() as readiness:
        for tensor in outputs:
            ensure_synced_on_device(tensor.data_ptr())
        sync()
    pushed = readiness.dump()["runtime_residency"]["real_host_sync_h2d"]
    if pushed != {"bytes": 0, "count": 0}:
        raise RuntimeError(f"extra H2D required after reported completion: {pushed}")
    # A compiled addition consumes each entire allocation. This includes the
    # untouched blocks, neighbouring guards and holes of fragmented maps.
    for npu, want in zip(outputs, expected):
        result = materialize(npu).cpu()
        target = want + 1
        if not torch.equal(result, target):
            first = (result != target).flatten().nonzero()[0].item()
            raise ValueError(
                f"device consumer/guard mismatch for {case} at element {first}: "
                f"got {result.flatten()[first]}, expected {target.flatten()[first]}"
            )
    report = trace.dump()
    report["readiness_extra_h2d"] = pushed
    counter = report["runtime_residency"].get("real_host_sync_h2d")
    views = list(make_views(case, buffers))
    expected_push = {"bytes": sum(s.nbytes for s, _, _ in views), "count": len(views)}
    if counter != expected_push:
        raise RuntimeError(f"expected real H2D {expected_push}, observed {counter}")
    return report


def stats(values):
    return dict(
        n=len(values),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        variance=float(np.var(values, ddof=1)) if len(values) > 1 else None,
    )


def physical_layout(tensor):
    # The supported allocation-info API returns C++ vectors by reference.
    # Compile against the matching, installed SDK instead of guessing its ABI.
    import rebel
    import tvm

    with tempfile.TemporaryDirectory() as directory:
        lib = Path(directory) / "layout.so"
        source = Path(__file__).with_name("ca22_memory_info.cpp")
        text_command(
            [
                "c++",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-I",
                str(Path(rebel.__file__).parent / "include"),
                str(source),
                str(Path(tvm.__file__).parent / "librbln.so"),
                "-o",
                str(lib),
            ]
        )
        get = ctypes.CDLL(str(lib)).ca22_memory_info
        get.argtypes = [ctypes.c_uint64]
        get.restype = ctypes.c_char_p
        raw = get(tensor.data_ptr())
        if raw is None:
            raise RuntimeError("rbln_get_memory_info could not query live KV tensor")
        return json.loads(raw)


def configure_host(use_worker):
    if not use_worker:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return None

    import numba
    from vllm.config import ParallelConfig

    from vllm_rbln.v1.worker.utils import (
        get_rbln_planned_affinity_cpu_count,
        set_omp_num_threads,
    )

    # Match the TP1/DP1 engine worker's ordering. Its NUMA split determines
    # intra-op threads before compilation, then it binds after warmup.
    parallel = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1)
    cpus = get_rbln_planned_affinity_cpu_count(0, 0, parallel)
    set_omp_num_threads(0, 0, max(2, cpus // 2))
    numba.set_num_threads(torch.get_num_threads())
    torch.set_num_threads(numba.get_num_threads())
    return parallel


def run(cfg, out):
    device_num = cfg["physical_npu"]
    if (
        os.environ.get("RBLN_DEVICES") != str(device_num)
        or torch.rbln.device_count() != 1
    ):
        raise RuntimeError("expose exactly the YAML physical_npu through RBLN_DEVICES")
    devices = text_command(["rbln-smi", "-q"])
    (out / "device.txt").write_text(devices)
    line = next((r for r in devices.splitlines() if cfg["uuid"] in r.split()), "")
    if not line or line.split()[0] != str(device_num) or "RBLN-CA22" not in line:
        raise ValueError("physical ordinal/UUID/model mismatch")
    inventory = dict(
        physical_npu=device_num,
        uuid=cfg["uuid"],
        torch=torch.__version__,
        torch_rbln=version("torch-rbln"),
        rebel_compiler=version("rebel-compiler"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch_sync_source=inspect.getsource(torch.rbln.synchronize),
        aligned_source=inspect.getsource(aligned_tensor),
        host_nixl_registration=False,
        initial_torch_threads=torch.get_num_threads(),
        initial_thread_env={
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OMP_PLACES",
                "OMP_PROC_BIND",
                "RBLN_NUM_THREADS",
                "VLLM_RBLN_NUMA",
            )
        },
    )
    save(out / "vendor.json", vendor_test(cfg, out))
    device = torch.device("rbln:0")

    def sync():
        torch.rbln.synchronize(device)

    parallel = configure_host(cfg.get("use_worker_cpu_setup", False))
    materialize = torch.compile(lambda x: x + 1, backend="rbln", dynamic=False)
    kv = allocate(tuple(cfg["layer_shape"]), cfg["layers"], device, materialize)
    flat = {
        m: allocate(((m << 19) + 4096,), 1, device, materialize)
        for m in {c["mib"] for c in cfg["conditions"].values() if c["layout"] == "flat"}
    }
    sync()
    if parallel:
        from vllm_rbln.v1.worker.utils import set_cpu_affinity

        set_cpu_affinity(0, 0, parallel)
    with import_thread_scope(None) as (inherited, _):
        inventory.update(
            measured_cpu_affinity=sorted(os.sched_getaffinity(0)),
            measured_torch_threads=torch.get_num_threads(),
            measured_torch_interop_threads=torch.get_num_interop_threads(),
            inherited_rbln_omp_limit=inherited,
            measured_rbln_num_threads_env=os.environ.get("RBLN_NUM_THREADS"),
        )
    actual = torch.rbln.memory_allocated(device)
    needed = sum(t.nbytes for t in kv[1]) + sum(
        t.nbytes for f in flat.values() for t in f[1]
    )
    if actual < needed:
        raise RuntimeError(
            f"lazy allocation: {actual} device bytes for {needed} payload bytes"
        )
    context = text_command(["rbln-stat"])
    (out / "context.txt").write_text(context)
    matches = [
        line
        for line in context.splitlines()
        if re.search(rf"\|\s*{os.getpid()}\s*\|", line)
    ]
    if not matches or any(int(line.split("|")[1]) != device_num for line in matches):
        raise RuntimeError(f"unverified physical device context: {matches}")
    inventory.update(
        allocated_device_bytes=actual,
        required_device_bytes=needed,
        host_allocations=[host_pages(t) for t in kv[0]],
        host_allocation="one rebel.kv_cache.aligned_tensor per layer",
        dtype="torch.float16",
        equal_block_mapping=True,
    )
    if cfg.get("capture_physical_layout"):
        inventory["live_kv_allocation"] = physical_layout(kv[1][0])
    save(out / "environment.json", inventory)
    names = list(cfg["conditions"])
    buffers = {}
    plans = {}
    for name in names:
        case = cfg["conditions"][name]
        buf = buffers[name] = flat[case["mib"]] if case["layout"] == "flat" else kv
        views = [
            dict(
                source=describe(s),
                destination=describe(t),
                allocation=layer,
                bytes=s.nbytes,
            )
            for s, t, layer in make_views(case, buf)
        ]
        plans[name] = dict(
            calls=len(views),
            bytes=sum(v["bytes"] for v in views),
            views=views,
            physical_ids=case.get("ids"),
            contiguous_runs=ranges(case["ids"], len(case["ids"]))
            if "ids" in case
            else None,
            completion="torch.rbln.synchronize(rbln:0) after batch",
            kv_import_threads=case.get("kv_import_threads"),
            condition=case,
            allocation_shape=list(buf[0][0].shape),
            host_buffer=host_pages(buf[0][0]),
            guard_validation_elements=sum(t.numel() for t in buf[1]),
            thread_runtime_trace=validate(case, buf, sync, materialize),
        )
    save(out / "plans.json", plans)
    results = {name: [] for name in names}
    blank = []
    with (out / "samples.jsonl").open("x", buffering=1) as stream:
        for repeat in range(-cfg["warmup"], cfg["samples"]):
            # Each case covers each position, including both AB/BA orders.
            order = names[repeat % len(names) :] + names[: repeat % len(names)]
            if (repeat // len(names)) % 2:
                order = list(reversed(order))
            sync()
            start = time.perf_counter_ns()
            sync()
            empty_ns = time.perf_counter_ns() - start
            if repeat >= 0:
                blank.append(empty_ns)
            for pos, name in enumerate(order):
                row = sample(cfg["conditions"][name], buffers[name], sync)
                if row["calls"] != plans[name]["calls"]:
                    raise RuntimeError("copy plan differs from executed calls")
                row.update(
                    condition=name,
                    repeat=repeat,
                    position=pos,
                    bytes=plans[name]["bytes"],
                    physical_ids=plans[name]["physical_ids"],
                    ranges=plans[name]["contiguous_runs"],
                    completion=plans[name]["completion"],
                    empty_sync_ns=empty_ns,
                    geometry=f"plans.json#{name}",
                )
                stream.write(json.dumps(row) + "\n")
                if repeat >= 0:
                    results[name].append(row)
            if repeat in (-1, 0) or repeat == cfg["samples"] - 1:
                print(
                    "repeat",
                    repeat,
                    {
                        k: round(results[k][-1]["batch_complete_ns"] / 1e6, 2)
                        for k in names
                        if results[k]
                    },
                    flush=True,
                )
    summary = dict(empty_sync_ms=stats([n / 1e6 for n in blank]), cells={})
    for name, rows in results.items():
        summary["cells"][name] = {
            field.removesuffix("_ns") + "_ms": stats([r[field] / 1e6 for r in rows])
            for field in (
                "outer_host_ns",
                "batch_complete_ns",
                "copy_call_sum_ns",
                "post_batch_sync_ns",
                "reorder_ns",
            )
        }
        summary["cells"][name].update(bytes=rows[0]["bytes"], calls=rows[0]["calls"])
    factors = [
        k
        for k in names
        if cfg["conditions"][k]["layout"] == "kv"
        and "kv_import_threads" not in cfg["conditions"][k]
        and len(plans[k]["contiguous_runs"]) == 1
    ]
    if len(factors) >= 3:
        x = [[1, plans[k]["calls"], plans[k]["bytes"] / (1 << 20)] for k in factors]
        y = [summary["cells"][k]["batch_complete_ms"]["median"] for k in factors]
        coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank == 3:
            summary["linear_fit_ms"] = dict(
                intercept=coef[0],
                per_call=coef[1],
                per_mib=coef[2],
                residual=dict(zip(factors, y - np.asarray(x) @ coef)),
            )
    save(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    args.result_dir.mkdir(parents=True, exist_ok=False)
    save(args.result_dir / "config.json", cfg)
    run(cfg, args.result_dir)


if __name__ == "__main__":
    main()
