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
