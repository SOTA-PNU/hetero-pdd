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
import pytest
import torch

from vllm_rbln.v1.worker import kv_copy


@pytest.mark.parametrize(
    "direction,coalesce,strided,calls",
    [
        ("h2d", False, False, 10),
        ("h2d", True, False, 6),
        ("h2d", True, True, 10),
        ("d2h", True, False, 0),
    ],
)
def test_copy_only_explicit_blocks(monkeypatch, direction, coalesce, strided, calls):
    monkeypatch.setattr(kv_copy.metrics_v2, "EVENTS_ENABLED", True)
    stats = []
    monkeypatch.setattr(
        kv_copy.metrics_v2, "add_copy_stats", lambda *a: stats.append(a)
    )
    src = torch.arange(2 * 10 * 6).reshape(2, 10, 6)
    if strided:
        src = src[..., ::2]
    dst = torch.full_like(src, -1)
    blocks = [4, 5, 1, 1, 2]
    kv_copy.copy_kv_blocks(
        {"a": src},
        {"a": dst},
        blocks,
        blocks,
        direction,
        coalesce=coalesce,
        reorder=False,
    )
    expected = torch.full_like(dst, -1)
    expected[:, blocks] = src[:, blocks]
    assert torch.equal(dst, expected)
    assert [sample[2:] for sample in stats] == ([(len(blocks), calls)] if calls else [])


def test_reorder_stays_per_block_and_index_errors_are_preserved():
    src = torch.arange(2 * 6 * 2 * 4 * 8, dtype=torch.float16).view(2, 6, 2, 1, 4, 8)
    a, b = torch.zeros_like(src), torch.zeros_like(src)
    for dest, coalesce in ((a, True), (b, False)):
        kv_copy.copy_kv_blocks(
            {"layer": src},
            {"layer": dest},
            [2, 3],
            [2, 3],
            "h2d",
            reorder=True,
            coalesce=coalesce,
        )
    assert torch.equal(a, b)
    with pytest.raises(IndexError):
        kv_copy.copy_kv_blocks(
            {"layer": src},
            {"layer": a},
            [5, 6],
            [5, 6],
            "h2d",
            reorder=False,
            coalesce=True,
        )


def test_recycled_descending_ids_merge_only_requested_physical_neighbors(monkeypatch):
    src = torch.arange(2 * 16 * 4).reshape(2, 16, 4)
    dst = torch.full_like(src, -1)
    blocks = [3, 10, 9, 8]
    monkeypatch.setattr(kv_copy.metrics_v2, "EVENTS_ENABLED", True)
    stats = []
    monkeypatch.setattr(
        kv_copy.metrics_v2, "add_copy_stats", lambda *a: stats.append(a)
    )
    kv_copy.copy_kv_blocks(
        {"layer": src},
        {"layer": dst},
        blocks,
        blocks,
        "h2d",
        reorder=False,
        coalesce=True,
    )
    expected = torch.full_like(dst, -1)
    expected[:, blocks] = src[:, blocks]
    assert torch.equal(dst, expected)
    assert stats[0][2:] == (4, 4)  # two runs across K and V


def test_import_threads_restore_on_failed_copy_and_skip_device_to_host(monkeypatch):
    class OpenMP:
        limit = 2
        changes = []

        def omp_get_max_threads(self):
            return self.limit

        def omp_get_thread_limit(self):
            return 3

        def omp_set_num_threads(self, limit):
            self.limit = limit
            self.changes.append(limit)

    omp = OpenMP()
    monkeypatch.setattr(kv_copy, "_runtime_openmp", lambda: omp)
    monkeypatch.setattr(kv_copy.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    monkeypatch.setattr(kv_copy.metrics_v2, "EVENTS_ENABLED", True)
    src = torch.arange(12).reshape(2, 3, 2)
    dst = torch.full_like(src, -1)
    kwargs = dict(coalesce=True, reorder=False, import_threads=8)
    kv_copy.metrics_v2.reset_copy_stats()
    kv_copy.copy_kv_blocks({"a": src}, {"a": dst}, [1], [1], "h2d", **kwargs)
    assert torch.equal(dst[:, 1], src[:, 1])
    assert (dst[:, [0, 2]] == -1).all()
    assert omp.changes == [3, 2]
    assert kv_copy.metrics_v2.copy_thread_fields() == {
        "import_threads_requested": 8,
        "import_threads_previous": 2,
        "import_threads_applied": 3,
    }
    with pytest.raises(IndexError):
        kv_copy.copy_kv_blocks({"a": src}, {"a": dst}, [3], [3], "h2d", **kwargs)
    assert omp.changes == [3, 2, 3, 2]
    kv_copy.copy_kv_blocks({"a": src}, {"a": dst}, [0], [0], "d2h", **kwargs)
    assert omp.changes == [3, 2, 3, 2]
    with kv_copy.import_thread_scope(None) as limits:
        assert limits == (2, 2)
    assert omp.changes == [3, 2, 3, 2]


def test_unavailable_runtime_still_copies_complete_blocks(monkeypatch):
    monkeypatch.setattr(kv_copy, "_runtime_openmp", lambda: None)
    monkeypatch.setattr(kv_copy.metrics_v2, "EVENTS_ENABLED", True)
    src = torch.arange(20).reshape(2, 5, 2)
    dst = torch.full_like(src, -1)
    kv_copy.copy_kv_blocks(
        {"a": src},
        {"a": dst},
        [3, 1],
        [3, 1],
        "h2d",
        coalesce=True,
        reorder=False,
        import_threads=8,
    )
    assert torch.equal(dst[:, [3, 1]], src[:, [3, 1]])
    assert (dst[:, [0, 2, 4]] == -1).all()
    assert kv_copy.metrics_v2.copy_thread_fields()["import_threads_applied"] is None
