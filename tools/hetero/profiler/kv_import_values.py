# SPDX-License-Identifier: Apache-2.0
"""Measure matched CA22 transfers of integer markers and fractional FP16 inputs."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import time
from importlib.util import find_spec
from pathlib import Path

from llmd.config import fields, integer, read_yaml


def settings(path):
    cfg = read_yaml(path)
    keys = {
        "apiVersion",
        "physical_npu",
        "uuid",
        "layers",
        "layer_shape",
        "first_block",
        "block_counts",
        "import_threads",
        "seed",
        "warmup",
        "samples",
        "cpu_call_kib",
        "cpu_samples",
        "canonical_implementation",
    }
    fields(cfg, keys, keys, "CA22 value benchmark")
    if cfg["apiVersion"] != "hetero-pdd/ca22-value-import-v1":
        raise ValueError("unsupported value benchmark apiVersion")
    if not isinstance(cfg["uuid"], str) or not cfg["uuid"]:
        raise ValueError("expected physical device UUID")
    integer(cfg["physical_npu"], "physical_npu", low=0)
    for key in (
        "layers",
        "first_block",
        "import_threads",
        "warmup",
        "samples",
        "cpu_samples",
    ):
        integer(cfg[key], key)
    integer(cfg["seed"], "seed", low=0)
    shape = cfg["layer_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 6
        or shape[0] != 2
        or any(type(i) is not int or i < 1 for i in shape)
    ):
        raise ValueError("layer_shape must be a two-plane HND layer shape")
    for key in ("block_counts", "cpu_call_kib"):
        entries = cfg[key]
        if (
            not isinstance(entries, list)
            or not entries
            or any(type(n) is not int or n < 1 for n in entries)
            or entries != sorted(set(entries))
        ):
            raise ValueError(f"expected sorted unique positive {key}")
    if cfg["first_block"] + max(cfg["block_counts"]) >= shape[1]:
        raise ValueError("leave an untouched guard before and after chosen blocks")
    if math.prod(shape[2:]) * min(cfg["block_counts"]) < 65536:
        raise ValueError("smallest view must exercise every FP16 encoding")
    if cfg["canonical_implementation"] not in ("diagnostic", "worker"):
        raise ValueError("canonical_implementation must be diagnostic or worker")
    return cfg


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def order(names, repeat):
    start = repeat % len(names)
    rotated = names[start:] + names[:start]
    return list(reversed(rotated)) if (repeat // len(names)) % 2 else rotated


class Codec:
    def __init__(self, directory, implementation):
        sdk = find_spec("tvm")
        if sdk is None or not sdk.origin:
            raise RuntimeError("installed RBLN SDK tvm/librbln.so is required")
        path = Path(sdk.origin).parent
        library = directory / "value-codec.so"
        argv = [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-O3",
            "-march=native",
            "-fPIC",
            "-shared",
            str(Path(__file__).with_name("ca22_value_codec.cpp")),
            str(path / "librbln.so"),
            f"-Wl,-rpath,{path}",
            "-o",
            str(library),
        ]
        subprocess.run(argv, check=True)
        save(directory / "codec-build.json", argv)
        loaded = ctypes.CDLL(str(library.resolve()))
        for name in ("canonical", "lanes", "decode", "batch"):
            fn = getattr(loaded, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = None
            setattr(self, name, fn)
        if implementation == "worker":
            from vllm_rbln.v1.worker.kv_canonical import _library

            shim = _library()
            if shim is None:
                raise RuntimeError(
                    "build the supported RBLN canonical worker shim first"
                )
            # Verify precisely the transform that this worker will use, including
            # encodings where ordinary floating-point equality collapses NaNs/zeros.
            self.canonical = shim.ca22_canonical_v1_prepare

    @staticmethod
    def array(fn, src, dst):
        fn(src.ctypes.data, dst.ctypes.data, src.size)

    def proof(self):
        import numpy as np

        src = np.arange(65536, dtype=np.uint16)
        prepared, expected, observed, bulk = (np.empty_like(src) for _ in range(4))
        self.array(self.canonical, src, prepared)
        self.array(self.lanes, src, expected)
        self.array(self.lanes, prepared, observed)
        self.array(self.batch, prepared, bulk)
        if not np.array_equal(expected, observed) or not np.array_equal(observed, bulk):
            raise RuntimeError(
                "canonical values differ in vendor scalar or batch conversion"
            )
        return dict(
            verified_fp16_encodings=src.size,
            changed=int(np.count_nonzero(src != prepared)),
            canonical_lut_sha256=hashlib.sha256(prepared).hexdigest(),
        )

    def reference(self, source):
        import torch

        encoded, decoded = torch.empty_like(source), torch.empty_like(source)
        self.lanes(source.data_ptr(), encoded.data_ptr(), source.numel())
        self.decode(encoded.data_ptr(), decoded.data_ptr(), source.numel())
        return decoded


def cpu_profile(cfg, codec, out):
    import numpy as np

    rng = np.random.default_rng(cfg["seed"])
    records = {}
    for kib in cfg["cpu_call_kib"]:
        n = kib * 512
        dense = rng.standard_normal(n).astype(np.float16).view(np.uint16)
        integer_input = np.resize(np.array([2, -2, 5, -5], dtype=np.float16), n).view(
            np.uint16
        )
        dest, tmp, reference = (np.empty_like(dense) for _ in range(3))
        codec.array(codec.lanes, dense, reference)
        names = ["integer", "normal", "normal_with_prep", "prep_only"]
        trials = {name: [] for name in names}
        for repeat in range(-cfg["warmup"], cfg["cpu_samples"]):
            for name in order(names, repeat):
                start = time.perf_counter_ns()
                if name in ("normal_with_prep", "prep_only"):
                    codec.array(codec.canonical, dense, tmp)
                if name != "prep_only":
                    source = {
                        "integer": integer_input,
                        "normal": dense,
                        "normal_with_prep": tmp,
                    }[name]
                    codec.array(codec.batch, source, dest)
                elapsed = time.perf_counter_ns() - start
                if name in ("normal", "normal_with_prep") and not np.array_equal(
                    dest, reference
                ):
                    raise RuntimeError("bulk conversion differs from scalar lanes")
                if repeat >= 0:
                    trials[name].append(elapsed)
        records[kib] = dict(
            samples_ns=trials,
            median_us={k: float(np.median(v)) / 1000 for k, v in trials.items()},
            normal_mantissa_lsb_one=float(np.mean(dense & 1)),
        )
    save(
        out / "value-cpu.json",
        dict(
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            sizes_kib=records,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--result-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    cfg = settings(args.config)
    if args.validate:
        print(json.dumps(cfg))
        return
    if not args.result_dir:
        parser.error("--result-dir must be a new directory")
    out = args.result_dir
    out.mkdir(parents=True, exist_ok=False)
    save(out / "config.json", cfg)
    if not args.cpu_only:
        # Load the backend before linking the codec to its matching SDK.
        from tools.hetero.profiler.kv_value_device import measure

    codec = Codec(out, cfg["canonical_implementation"])
    save(out / "encoding-proof.json", codec.proof())
    if args.cpu_only:
        cpu_profile(cfg, codec, out)
    else:
        measure(cfg, codec, out)


if __name__ == "__main__":
    main()
