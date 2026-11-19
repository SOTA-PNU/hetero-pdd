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
from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.v1.worker import build_kv_canonical, kv_canonical, kv_copy


def test_exhaustive_validated_mapping_when_helper_is_installed():
    lib = kv_canonical._library()
    if lib is None:
        pytest.skip("optional CA22 shim absent or unvalidated")
    assert build_kv_canonical.verified_mapping(lib)
    src = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    original = src.clone()
    dst = torch.empty_like(src)
    lib.ca22_canonical_v1_prepare(src.data_ptr(), dst.data_ptr(), src.numel())
    assert torch.equal(src, original)
    bits = src.int() & 0xFFFF
    special = ((bits & 0x7C00) == 0) | ((bits & 0x7C00) == 0x7C00)
    special |= (bits & 0x7FFF) == 0x7BFF
    assert torch.equal(dst[special], src[special])
    assert torch.equal(dst[~special], (src + (src & 1))[~special])


def test_residency_rejects_host_pointer_shard_and_unknown_error():
    def absent(_):
        raise RuntimeError("Failed to get single device address")

    buf = SimpleNamespace(size=128)
    api = SimpleNamespace(get_host_ptr=absent, get_device_buffers=lambda _: [buf])
    assert kv_canonical._resident(api, 256, 128)
    api.get_host_ptr = lambda _: 64
    assert not kv_canonical._resident(api, 256, 128)
    assert not kv_canonical._resident(
        SimpleNamespace(get_device_buffers=lambda _: [buf, buf]), 256, 128
    )
    assert not kv_canonical._resident(
        SimpleNamespace(get_device_buffers=lambda _: [buf]), 256, 32
    )

    def unexpected(_):
        raise RuntimeError("device unavailable")

    api.get_host_ptr = unexpected
    assert not kv_canonical._resident(api, 256, 128)


@pytest.mark.parametrize("candidate_calls", [0, 1])
def test_original_replayed_and_counted_on_rejection(monkeypatch, candidate_calls):
    src = torch.arange(32, dtype=torch.int16).view(torch.float16).view(2, 4, 4)
    before = src.clone()
    dst = torch.full_like(src, -1)
    attempted = []

    def rejected(s, d, *, profile):
        assert profile
        attempted.append(s.nbytes)
        if candidate_calls:
            d.fill_(99)
        return 0, 128, candidate_calls, 100

    monkeypatch.setattr(kv_canonical, "copy_if_supported", rejected)
    monkeypatch.setattr(kv_copy.metrics_v2, "EVENTS_ENABLED", True)
    kv_copy.metrics_v2.reset_copy_stats()
    kv_copy.copy_kv_blocks(
        {"a": src},
        {"a": dst},
        [1],
        [1],
        "h2d",
        reorder=False,
        coalesce=True,
        canonicalize=True,
    )
    fields = kv_copy.metrics_v2.copy_thread_fields()
    assert fields["canonical_copy_calls"] == fields["canonical_bytes"] == 0
    assert fields["canonical_fallback_calls"] == 2
    assert fields["canonical_prep_ns"] == 200
    assert fields["canonical_scratch_bytes"] == 128
    assert kv_copy.metrics_v2.take_copy_stats()[-1] == 2 * (1 + candidate_calls)
    assert len(attempted) == 2
    assert torch.equal(src, before)
    assert torch.equal(dst[:, 1].view(torch.int16), src[:, 1].view(torch.int16))
    assert (dst[:, [0, 2, 3]] == -1).all()


def test_cpu_or_unvalidated_library_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_RBLN_KV_CANONICAL_LIBRARY", str(tmp_path / "absent.so"))
    kv_canonical._library.cache_clear()
    try:
        assert kv_canonical._library() is None
        cpu = torch.zeros(8, dtype=torch.float16)
        assert kv_canonical.copy_if_supported(cpu, cpu.clone()) == (0, 0, 0, 0)
    finally:
        kv_canonical._library.cache_clear()


def test_source_sample_emits_bounded_counts_once(monkeypatch):
    monkeypatch.setattr(kv_canonical._scratch, "sampled", False, raising=False)
    events = []
    monkeypatch.setattr(
        kv_canonical.metrics_v2,
        "record_event",
        lambda kind, **data: events.append((kind, data)),
    )
    bits = torch.zeros(4100, dtype=torch.int16)
    bits[1:4096:2] = 0x3C01
    bits[4096:] = 0x3C01
    canonical = bits + (bits & 1)
    canonical[1] = bits[1]
    for _ in range(2):
        kv_canonical._sample_once(
            bits.view(torch.float16), canonical.view(torch.float16)
        )
    assert events == [
        (
            "kv_canonical_source_sample",
            {
                "sampled_elements": 4096,
                "changed_elements": 2047,
                "mantissa_lsb_one": 2048,
            },
        )
    ]
