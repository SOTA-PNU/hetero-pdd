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

import time
from typing import Literal

import torch

from vllm_rbln.v1.worker import metrics_v2


def block_runs(ids: list[int]) -> list[int | slice]:
    """Preserve wire order, duplicates and singleton indexing at run boundaries."""
    result = []
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
        runs = block_runs(src_block_ids)
        lowest, highest = min(src_block_ids), max(src_block_ids)
    copy_ns = reorder_ns = calls = 0
    for name, dst_cache in dst_kv_caches.items():
        src_cache = src_kv_caches[name]
        if reorder:
            assert src_cache.shape[0] == dst_cache.shape[0] == 2
        for kv in range(dst_cache.shape[0]):
            src, dst = src_cache[kv], dst_cache[kv]
            indices = src_block_ids
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
                target.copy_(source)
                if profile:
                    copy_ns += time.monotonic_ns() - started
    if profile:
        # Report the host-call envelope without introducing a device barrier.
        try:
            metrics_v2.add_copy_stats(copy_ns, reorder_ns, len(src_block_ids), calls)
        except Exception as exc:
            metrics_v2.disable_events(exc)
