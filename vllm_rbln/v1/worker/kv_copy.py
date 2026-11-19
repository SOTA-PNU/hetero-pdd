# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Copies of explicitly allocated blocks between the host buffer and RBLN KV."""

import ctypes
import importlib.util
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker import metrics_v2

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _runtime_openmp() -> ctypes.CDLL | None:
    # Resolve the public OpenMP ABI from the *loaded* RBLN runtime. Other
    # libraries in the worker can link a different OpenMP implementation.
    tvm = importlib.util.find_spec("tvm")
    if tvm is None or tvm.origin is None:
        return None
    library_dir = Path(tvm.origin).parent
    for name in ("librbln.so", "librbln_runtime.so"):
        try:
            lib = ctypes.CDLL(str(library_dir / name), mode=os.RTLD_NOLOAD)
            lib.omp_get_max_threads.argtypes = []
            lib.omp_get_max_threads.restype = ctypes.c_int
            lib.omp_get_thread_limit.argtypes = []
            lib.omp_get_thread_limit.restype = ctypes.c_int
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            lib.omp_set_num_threads.restype = None
            return lib
        except (AttributeError, OSError):
            continue
    return None


@lru_cache(maxsize=1)
def _warn_no_openmp() -> None:
    logger.warning("RBLN OpenMP API unavailable; using the default KV copy threads")


@contextmanager
def import_thread_scope(threads: int | None) -> Iterator[tuple[int | None, int | None]]:
    """Temporarily size the RBLN runtime's OpenMP team on this calling task."""
    lib = _runtime_openmp()
    if lib is None:
        if threads is not None:
            _warn_no_openmp()
        yield None, None
        return
    previous = lib.omp_get_max_threads()
    if threads is None:
        yield previous, previous
        return
    if threads < 1:
        raise ValueError("kv_import_threads must be positive")
    applied = min(threads, len(os.sched_getaffinity(0)), lib.omp_get_thread_limit())
    if applied == previous:
        yield previous, previous
        return
    # OpenMP specifies nthreads-var per generating task. Setting it here
    # avoids changing PyTorch's process-wide intra-op configuration or the
    # worker's decode thread settings outside this synchronous import.
    lib.omp_set_num_threads(applied)
    try:
        yield previous, applied
    finally:
        lib.omp_set_num_threads(previous)


def block_runs(ids: list[int]) -> list[int | slice]:
    """Preserve wire order, duplicates and singleton indexing at run boundaries."""
    result: list[int | slice] = []
    i = 0
    while i < len(ids):
        j = i + 1
        while j < len(ids) and ids[j] == ids[j - 1] + 1:
            j += 1
        result.append(ids[i] if j == i + 1 else slice(ids[i], ids[j - 1] + 1))
        i = j
    return result


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
    *,
    reorder: bool,
    coalesce: bool,
    import_threads: int | None = None,
    canonicalize: bool = False,
) -> None:
    args = (src_kv_caches, dst_kv_caches, src_block_ids, dst_block_ids, direction)
    if direction == "h2d" and (import_threads is not None or metrics_v2.EVENTS_ENABLED):
        with import_thread_scope(import_threads) as limits:
            if metrics_v2.EVENTS_ENABLED:
                metrics_v2.record_copy_threads(import_threads, *limits)
            _copy_kv_blocks(
                *args, reorder=reorder, coalesce=coalesce, canonicalize=canonicalize
            )
    else:
        _copy_kv_blocks(
            *args, reorder=reorder, coalesce=coalesce, canonicalize=canonicalize
        )


def _copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
    *,
    reorder: bool,
    coalesce: bool,
    canonicalize: bool,
) -> None:
    if not src_kv_caches or not dst_kv_caches or not src_block_ids or not dst_block_ids:
        return
    assert src_block_ids == dst_block_ids, (
        f"KV block maps differ: {src_block_ids=} {dst_block_ids=}"
    )
    h2d = direction == "h2d"
    reorder = h2d and reorder
    profile = h2d and metrics_v2.EVENTS_ENABLED
    use_runs = coalesce and h2d and not reorder and len(src_block_ids) > 1
    if use_runs:
        # Host and device each use the same block ID. For unique IDs each
        # write has a distinct destination in a separate address space, so a
        # recycled, descending allocation can be copied in physical order.
        ids = src_block_ids
        if len(set(ids)) == len(ids):
            ids = sorted(ids)
        runs = block_runs(ids)
        lowest, highest = min(src_block_ids), max(src_block_ids)
    canonical = canonicalize and h2d and not reorder
    if canonical:
        from vllm_rbln.v1.worker import kv_canonical
    accepted = accepted_bytes = rejected = scratch_capacity = canonical_ns = prep_ns = 0
    copy_ns = reorder_ns = calls = native_copy_ns = 0
    for name, dst_cache in dst_kv_caches.items():
        src_cache = src_kv_caches[name]
        layer = kv_canonical.CanonicalLayer(dst_cache) if canonical else None
        if reorder:
            assert src_cache.shape[0] == dst_cache.shape[0] == 2
        for kv in range(dst_cache.shape[0]):
            src, dst = src_cache[kv], dst_cache[kv]
            indices: Sequence[int | slice] = src_block_ids
            # A strided device slice can trigger an expensive fallback. Group
            # only directly DMA-able views; ordinary integer indexing preserves
            # errors for out-of-range block IDs (a slice would silently clamp).
            if (
                use_runs
                and src.is_contiguous()
                and dst.is_contiguous()
                and src.dtype == dst.dtype
                and src.shape == dst.shape
                and lowest >= 0
                and highest < dst.shape[0]
            ):
                indices = runs
            if profile:
                calls += len(indices)
            for index in indices:
                source, target = src[index], dst[index]
                if reorder:
                    assert source.is_contiguous() and source.dtype == target.dtype
                    assert source.element_size() == target.element_size() == 2
                    assert (
                        source.numel() == target.numel()
                        and source.ndim == target.ndim == 4
                    )
                    h, singleton, t, d = target.shape
                    assert singleton == 1
                    source = source.view(-1).view(t, h, d)
                    if profile:
                        started = time.monotonic_ns()
                    source = source.permute(1, 0, 2).unsqueeze(1).contiguous()
                    if profile:
                        reorder_ns += time.monotonic_ns() - started
                    assert source.shape == target.shape and source.dtype == target.dtype
                if profile:
                    started = time.monotonic_ns()
                if canonical:
                    candidate_start = time.monotonic_ns() if profile else 0
                    size, scratch, candidate_calls, prepared_ns = (
                        kv_canonical.copy_if_supported(
                            source, target, profile=profile, layer=layer
                        )
                    )
                    if profile:
                        prep_ns += prepared_ns
                        canonical_ns += time.monotonic_ns() - candidate_start
                        scratch_capacity = max(scratch_capacity, scratch)
                    if size:
                        accepted += 1
                        accepted_bytes += size
                    else:
                        rejected += 1
                        assert layer is not None
                        if candidate_calls:
                            layer.rejected = True
                        else:
                            layer.verified = False
                        calls += candidate_calls
                        target.copy_(source)
                else:
                    target.copy_(source)
                if profile:
                    copy_ns += time.monotonic_ns() - started
        if profile and layer is not None:
            native_copy_ns += layer.copy_ns
    if profile:
        # Report the host-call envelope without introducing a device barrier.
        try:
            metrics_v2.add_copy_stats(copy_ns, reorder_ns, len(src_block_ids), calls)
            if canonicalize:
                metrics_v2.record_canonical_copy(
                    accepted,
                    accepted_bytes,
                    rejected if canonical else calls,
                    scratch_capacity,
                    canonical_ns,
                    prep_ns,
                    native_copy_ns,
                )
        except Exception as exc:
            metrics_v2.disable_events(exc)
