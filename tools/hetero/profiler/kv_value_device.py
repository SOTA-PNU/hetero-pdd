# SPDX-License-Identifier: Apache-2.0
"""Physical-device portion of the CA22 value diagnostic."""

import hashlib
import json
import math
import os
import time
from importlib.metadata import version

import numpy as np
import torch
import torch_rbln  # noqa: F401 (register the installed device backend)
from rebel._C.vmem import ensure_synced_on_device
from rebel.kv_cache import aligned_tensor

from tools.hetero.profiler.kv_import_micro import (
    configure_host,
    describe,
    host_pages,
    physical_layout,
    stats,
    text_command,
)
from tools.hetero.profiler.kv_import_values import order, save
from vllm_rbln.v1.worker.kv_copy import import_thread_scope
from vllm_rbln.v1.worker.utils import set_cpu_affinity

MODES = ("integer", "normal", "canonical")


def pushes(trace):
    return trace.dump()["runtime_residency"]["real_host_sync_h2d"]


class ValueTrial:
    def __init__(self, cfg, codec, out):
        self.cfg, self.codec, self.out = cfg, codec, out
        self.device = torch.device("rbln:0")
        self.shape = tuple(cfg["layer_shape"])
        self.layers = cfg["layers"]
        self.first = cfg["first_block"]
        self.rng = np.random.default_rng(cfg["seed"])
        physical = str(cfg["physical_npu"])
        if (
            os.environ.get("RBLN_DEVICES") != physical
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or torch.rbln.device_count() != 1
        ):
            raise RuntimeError(
                "expose exactly the configured physical NPU through RBLN_DEVICES"
            )
        inventory = text_command(["rbln-smi", "-q"])
        (out / "device.txt").write_text(inventory)
        if not any(
            line.split()[:2] == [physical, "RBLN-CA22"] and cfg["uuid"] in line.split()
            for line in inventory.splitlines()
        ):
            raise RuntimeError("device ordinal, CA22 model and UUID did not match")

        parallel = configure_host(True)
        materialize = torch.compile(lambda x: x + 1, backend="rbln", dynamic=False)
        self.negate = torch.compile(lambda x: -x, backend="rbln", dynamic=False)
        zeros = torch.zeros(self.shape, device=self.device, dtype=torch.float16)
        self.targets = {
            mode: [materialize(zeros) for _ in range(self.layers)] for mode in MODES
        }
        self.sync()
        all_targets = [t for group in self.targets.values() for t in group]
        allocated = torch.rbln.memory_allocated(self.device)
        required = sum(t.nbytes for t in all_targets)
        if (
            len({t.data_ptr() for t in all_targets}) != len(all_targets)
            or allocated < required
        ):
            raise RuntimeError(
                "expected independent materialized destinations for all cases"
            )
        context = text_command(["rbln-stat"])
        (out / "context.txt").write_text(context)
        lines = [
            line
            for line in context.splitlines()
            if str(os.getpid()) in [field.strip() for field in line.split("|")]
        ]
        if not lines or any(line.split("|")[1].strip() != physical for line in lines):
            raise RuntimeError(
                "could not map this process to the claimed physical device"
            )
        set_cpu_affinity(0, 0, parallel)
        layouts = {k: physical_layout(v[0]) for k, v in self.targets.items()}
        if not all(
            m["is_user_fp16"]
            and m["is_physical_custom_fp16"]
            and m["user_shape"] == m["physical_shape"] == list(self.shape)
            for m in layouts.values()
        ):
            raise RuntimeError(
                "expected shape-preserving FP16 to CustomFloat16 allocations"
            )
        self.sources = {
            mode: [
                aligned_tensor(math.prod(self.shape)).reshape(self.shape)
                for _ in range(self.layers)
            ]
            for mode in ("integer", "normal")
        }
        self.scratch = (
            aligned_tensor(max(cfg["block_counts"]) * math.prod(self.shape[2:]))
            if cfg["canonical_implementation"] == "diagnostic"
            else None
        )
        self.worker_copy = None
        if cfg["canonical_implementation"] == "worker":
            from vllm_rbln.v1.worker.kv_canonical import copy_if_supported

            self.worker_copy = copy_if_supported
        with import_thread_scope(None) as (prior, _):
            save(
                out / "environment.json",
                dict(
                    physical_npu=cfg["physical_npu"],
                    uuid=cfg["uuid"],
                    torch=torch.__version__,
                    torch_rbln=version("torch-rbln"),
                    rebel_compiler=version("rebel-compiler"),
                    required_device_bytes=required,
                    allocated_device_bytes=allocated,
                    logical_physical=layouts,
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    torch_threads=torch.get_num_threads(),
                    inherited_rbln_omp_limit=prior,
                    host_buffer=host_pages(self.sources["normal"][0]),
                    host_nixl_registration=False,
                ),
            )
        self.checks = {}

    def sync(self):
        torch.rbln.synchronize(self.device)

    def fill(self, exhaustive):
        n = max(self.cfg["block_counts"])
        for layer in range(self.layers):
            for kv in range(2):
                select = (kv, slice(self.first, self.first + n))
                normal = self.sources["normal"][layer][select].view(-1)
                if exhaustive:
                    normal.numpy().view(np.uint16)[:] = np.resize(
                        np.arange(65536, dtype=np.uint16), normal.numel()
                    )
                else:
                    normal.numpy()[:] = self.rng.standard_normal(
                        normal.numel(), dtype=np.float32
                    ).astype(np.float16)
                marker = self.sources["integer"][layer][select].view(-1)
                marker.fill_(2 + layer % 23 + kv)
                marker[::7].neg_()

    def views(self, mode, blocks):
        src = self.sources["integer" if mode == "integer" else "normal"]
        for layer, (host, destination) in enumerate(zip(src, self.targets[mode])):
            for kv in range(2):
                select = kv, slice(self.first, self.first + blocks)
                yield layer, host[select], destination[select]

    def sample(self, mode, blocks):
        calls = prep_ns = api_ns = bytes_moved = scratch_bytes = 0
        self.sync()
        begin = time.perf_counter_ns()
        with import_thread_scope(self.cfg["import_threads"]) as (previous, applied):
            if applied != self.cfg["import_threads"]:
                raise RuntimeError("requested import thread limit was unavailable")
            for _, src, dst in self.views(mode, blocks):
                if mode == "canonical" and self.worker_copy is not None:
                    # Includes the production predicates and runtime H2D counter check.
                    start = time.perf_counter_ns()
                    amount, scratch, copied, prep = self.worker_copy(
                        src, dst, profile=True
                    )
                    api_ns += time.perf_counter_ns() - start - prep
                    if amount != src.nbytes or copied != 1:
                        raise RuntimeError(
                            "worker canonical preparation rejected a measured view"
                        )
                    prep_ns += prep
                    scratch_bytes = max(scratch_bytes, scratch)
                else:
                    if mode == "canonical":
                        start = time.perf_counter_ns()
                        tmp = self.scratch[: src.numel()].view_as(src)
                        self.codec.canonical(
                            src.data_ptr(), tmp.data_ptr(), src.numel()
                        )
                        src = tmp
                        prep_ns += time.perf_counter_ns() - start
                        scratch_bytes = self.scratch.nbytes
                    start = time.perf_counter_ns()
                    dst.copy_(src)
                    api_ns += time.perf_counter_ns() - start
                calls += 1
                bytes_moved += dst.nbytes
        host = time.perf_counter_ns()
        self.sync()
        done = time.perf_counter_ns()
        return dict(
            host_ns=host - begin,
            completed_ns=done - begin,
            sync_ns=done - host,
            prep_ns=prep_ns,
            copy_and_metadata_ns=api_ns,
            calls=calls,
            bytes=bytes_moved,
            scratch_bytes=scratch_bytes,
            previous_limit=previous,
            applied_limit=applied,
        )

    def check_targets(self, mode, blocks):
        direct_digest, consumer_digest = hashlib.sha256(), hashlib.sha256()
        fields = (
            "direct_ref_mismatches",
            "consumer_ref_mismatches",
            "zero_negation_sign_differences",
            "unexpected_consumer_mismatches",
            "untouched_direct_mismatches",
            "untouched_consumer_mismatches",
            "reconsumer_changed_bits",
            "readback_additional_pushes",
        )
        counts = dict.fromkeys(fields, 0)
        host = self.sources["integer" if mode == "integer" else "normal"]
        for src, target in zip(host, self.targets[mode]):
            expected = torch.ones_like(src)
            expected_neg = -torch.ones_like(src)
            selected = slice(self.first, self.first + blocks)
            for kv in range(2):
                decoded = self.codec.reference(src[kv, selected])
                expected[kv, selected] = decoded
                expected_neg[kv, selected] = self.codec.reference(-decoded)
            observed = self.negate(target).cpu().clone()
            direct = target.cpu().clone()
            with torch.rbln.explain() as ready:
                ensure_synced_on_device(target.data_ptr())
                self.sync()
            counts["readback_additional_pushes"] += pushes(ready)["count"]
            recalled = self.negate(target).cpu().clone()
            counts["reconsumer_changed_bits"] += int(
                torch.count_nonzero(
                    recalled.view(torch.int16) != observed.view(torch.int16)
                )
            )
            direct_digest.update(direct.numpy())
            consumer_digest.update(observed.numpy())
            # Compare raw half words: NaN payloads and signed zeros are observable.
            dm = direct.view(torch.int16) != expected.view(torch.int16)
            nm = observed.view(torch.int16) != expected_neg.view(torch.int16)
            counts["direct_ref_mismatches"] += int(dm.sum())
            counts["consumer_ref_mismatches"] += int(nm.sum())
            if nm.any():
                # This device negate produces +0 for +0; the IEEE unary reference
                # produces -0. Require that exact triple, and then also require the
                # candidate and ordinary device hashes to agree without exceptions.
                input_raw = expected.view(torch.int16)[nm]
                ref_raw = expected_neg.view(torch.int16)[nm]
                actual_raw = observed.view(torch.int16)[nm]
                known = (input_raw == 0) & (ref_raw == -32768) & (actual_raw == 0)
                counts["zero_negation_sign_differences"] += int(known.sum())
                counts["unexpected_consumer_mismatches"] += int((~known).sum())
            for guard in (slice(None, self.first), slice(self.first + blocks, None)):
                counts["untouched_direct_mismatches"] += int(dm[:, guard].sum())
                counts["untouched_consumer_mismatches"] += int(nm[:, guard].sum())
        return dict(
            **counts,
            direct_sha256=direct_digest.hexdigest(),
            consumer_sha256=consumer_digest.hexdigest(),
            verified_elements=self.layers * math.prod(self.shape),
        )

    def verify(self, label, blocks, modes):
        reports = {}
        for mode in modes:
            with torch.rbln.explain() as trace:
                measured = self.sample(mode, blocks)
            with torch.rbln.explain() as ready:
                for target in self.targets[mode]:
                    ensure_synced_on_device(target.data_ptr())
                self.sync()
            actual, extra = pushes(trace), pushes(ready)
            report = self.check_targets(mode, blocks)
            reports[mode] = dict(
                **report,
                real_h2d=actual,
                readiness_extra_h2d=extra,
                copy_counters_exact=actual
                == dict(count=measured["calls"], bytes=measured["bytes"]),
                expected_bytes=measured["bytes"],
                expected_calls=measured["calls"],
            )
        a, b = reports["normal"], reports["canonical"]
        direct = a["direct_sha256"] == b["direct_sha256"]
        consumer = a["consumer_sha256"] == b["consumer_sha256"]
        passed = (
            direct
            and consumer
            and all(
                r["copy_counters_exact"]
                and r["readiness_extra_h2d"] == dict(count=0, bytes=0)
                and not any(
                    r[k]
                    for k in (
                        "reconsumer_changed_bits",
                        "readback_additional_pushes",
                        "direct_ref_mismatches",
                        "unexpected_consumer_mismatches",
                        "untouched_direct_mismatches",
                        "untouched_consumer_mismatches",
                    )
                )
                for r in reports.values()
            )
        )
        self.checks[label] = dict(
            passed=passed,
            baseline_canonical_direct_exact=direct,
            baseline_canonical_consumer_exact=consumer,
            modes=reports,
        )
        save(self.out / "checks.json", self.checks)
        return passed


def measure(cfg, codec, out):
    trial = ValueTrial(cfg, codec, out)
    geometries = {}
    for blocks in cfg["block_counts"]:
        geometries[blocks] = [
            dict(
                layer=layer,
                source=describe(src),
                target=describe(dst),
                bytes=src.nbytes,
            )
            for layer, src, dst in trial.views("normal", blocks)
        ]
    save(out / "geometry.json", geometries)
    trial.fill(True)
    success = [
        trial.verify(
            "all-65536-encodings", min(cfg["block_counts"]), ("normal", "canonical")
        )
    ]
    trial.fill(False)
    for blocks in cfg["block_counts"]:
        success.append(trial.verify(f"{blocks}-blocks", blocks, MODES))
    if not all(success):
        raise RuntimeError(
            "value/guard or transfer-count verification failed; see checks.json"
        )
    cases = [(mode, blocks) for blocks in cfg["block_counts"] for mode in MODES]
    by_condition = {}
    with (out / "samples.jsonl").open("x", buffering=1) as raw:
        for repeat in range(-cfg["warmup"], cfg["samples"]):
            for pos, (mode, blocks) in enumerate(order(cases, repeat)):
                row = trial.sample(mode, blocks)
                row.update(
                    repeat=repeat,
                    position=pos,
                    mode=mode,
                    blocks=blocks,
                    physical_ids=list(range(trial.first, trial.first + blocks)),
                )
                raw.write(json.dumps(row) + "\n")
                if repeat >= 0:
                    by_condition.setdefault(f"{blocks}-{mode}", []).append(row)
    save(
        out / "summary.json",
        {
            name: {
                f.removesuffix("_ns") + "_ms": stats([r[f] / 1e6 for r in rows])
                for f in (
                    "host_ns",
                    "completed_ns",
                    "sync_ns",
                    "prep_ns",
                    "copy_and_metadata_ns",
                )
            }
            for name, rows in by_condition.items()
        },
    )
