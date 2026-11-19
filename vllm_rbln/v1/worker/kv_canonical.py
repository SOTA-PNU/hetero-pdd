# SPDX-License-Identifier: Apache-2.0
"""Optional IEEE FP16 preparation for one validated CA22 runtime build."""

import ctypes
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import torch

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.worker.build_kv_canonical import default_library, verified_mapping

_LIMIT = 1 << 20  # Maximum live scratch per importing thread.
_scratch = threading.local()
_log = init_logger(__name__)
_reported: set[str] = set()


@lru_cache(maxsize=1)
def _library():
    location = Path(os.environ.get("VLLM_RBLN_KV_CANONICAL_LIBRARY", default_library()))
    if not location.is_absolute():
        return None
    try:
        lib = ctypes.CDLL(str(location))
        u64 = ctypes.c_uint64
        lib.ca22_canonical_v1_supported.argtypes = []
        lib.ca22_canonical_v1_supported.restype = ctypes.c_bool
        lib.ca22_canonical_v1_target.argtypes = [u64, u64, u64, u64, ctypes.c_uint32]
        lib.ca22_canonical_v1_target.restype = ctypes.c_bool
        lib.ca22_canonical_v1_describe.argtypes = [u64]
        lib.ca22_canonical_v1_describe.restype = ctypes.c_char_p
        lib.ca22_canonical_v1_prepare.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.ca22_canonical_v1_prepare.restype = None
        lib.ca22_canonical_v1_pushes.argtypes = [
            ctypes.POINTER(u64),
            ctypes.POINTER(u64),
        ]
        lib.ca22_canonical_v1_pushes.restype = None
        return (
            lib if lib.ca22_canonical_v1_supported() and verified_mapping(lib) else None
        )
    except (AttributeError, OSError):
        return None


@lru_cache(maxsize=1)
def _one_ca22() -> bool:
    import rebel

    physical = os.environ.get("RBLN_DEVICES", "")
    return physical.isdecimal() and rebel.get_npu_name(int(physical)) == "RBLN-CA22"


def _rejection(reason, source, target, lib=None):
    if reason not in _reported:
        _reported.add(reason)
        detail = lib.ca22_canonical_v1_describe(target.data_ptr()) if lib else None
        _log.warning(
            "Canonical KV import fallback %s; dtype=%s/%s shape=%s stride=%s "
            "offset=%s source_bytes=%d SDK=%s",
            reason,
            source.dtype,
            target.dtype,
            tuple(target.shape),
            target.stride(),
            target.storage_offset(),
            source.nbytes,
            detail,
        )


def _no_single_host_view(vmem, base: int) -> bool:
    # On the identified SDK this getter is const. Live, materialized CA22 KV
    # has no exposed user pointer, including after device readback. A
    # host-backed allocation exposes a pointer or lacks a device buffer.
    try:
        vmem.get_host_ptr(base)
    except RuntimeError as exc:
        return str(exc) == "Failed to get single device address"
    return False


def _resident(vmem, base: int, allocation: int) -> bool:
    try:
        buffers = vmem.get_device_buffers(base)
        return (
            len(buffers) == 1
            and buffers[0].size == allocation
            and _no_single_host_view(vmem, base)
        )
    except (AttributeError, RuntimeError):
        return False


def _pushes(lib) -> tuple[int, int]:
    count, size = ctypes.c_uint64(), ctypes.c_uint64()
    lib.ca22_canonical_v1_pushes(ctypes.byref(count), ctypes.byref(size))
    return count.value, size.value


def copy_if_supported(
    source: torch.Tensor, target: torch.Tensor, *, profile: bool = False
) -> tuple[int, int, int, int]:
    """Return accepted bytes, scratch capacity, copy calls, and preparation time.

    The caller replays the untouched original for every rejected candidate.
    """
    if (
        source.device.type != "cpu"
        or target.device.type != "rbln"
        or source.dtype != torch.float16
        or target.dtype != torch.float16
        or source.shape != target.shape
        or not source.is_contiguous()
        or not target.is_contiguous()
        or not 0 < source.nbytes <= _LIMIT
        or source.requires_grad
        or target.requires_grad
    ):
        _rejection("ineligible tensor", source, target)
        return 0, 0, 0, 0
    lib = _library()
    if lib is None:
        _rejection("missing or unvalidated helper", source, target)
        return 0, 0, 0, 0
    try:
        from rebel._C import vmem

        if not _one_ca22() or target.device.index != 0:
            _rejection("requires exactly one visible CA22", source, target)
            return 0, 0, 0, 0
        storage = target.untyped_storage()
        base, allocation = storage.data_ptr(), storage.nbytes()
        if not lib.ca22_canonical_v1_target(
            target.data_ptr(), base, target.nbytes, allocation, target.device.index
        ):
            _rejection("unsupported SDK metadata or extent", source, target, lib)
            return 0, 0, 0, 0
        if not _resident(vmem, base, allocation):
            _rejection("nonresident, host-backed or sharded", source, target, lib)
            return 0, 0, 0, 0
        scratch = getattr(_scratch, "buffer", None)
        if scratch is None:
            scratch = torch.empty(_LIMIT // 2, dtype=torch.float16, device="cpu")
            _scratch.buffer = scratch
    except (ImportError, AttributeError, RuntimeError, MemoryError) as exc:
        _rejection(f"SDK or scratch unavailable ({type(exc).__name__})", source, target)
        return 0, 0, 0, 0
    prepared = scratch[: source.numel()].view(source.shape)
    started = time.monotonic_ns() if profile else 0
    lib.ca22_canonical_v1_prepare(
        source.data_ptr(), prepared.data_ptr(), source.numel()
    )
    prep_ns = time.monotonic_ns() - started if profile else 0
    before_count, before_bytes = _pushes(lib)
    # Retain native tensor-copy dispatch and synchronous buffer lifetime.
    target.copy_(prepared)
    count, size = _pushes(lib)
    if (
        count - before_count != 1
        or size - before_bytes != source.nbytes
        or not _no_single_host_view(vmem, base)
    ):
        _rejection(
            "destination or push evidence changed; replay original", source, target, lib
        )
        return 0, scratch.nbytes, 1, prep_ns
    return source.nbytes, scratch.nbytes, 1, prep_ns
