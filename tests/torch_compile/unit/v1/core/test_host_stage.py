"""Exercise the actual vLLM 0.22 get_finished path with a simulated NIXL handle."""

import queue
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    RemoteMeta,
    ReqMeta,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker import (
    RblnNixlConnectorWorker,
)


@pytest.fixture
def worker(monkeypatch):
    w = object.__new__(RblnNixlConnectorWorker)
    w._host_stage_enabled = True
    w._receive_timeout_s = 2
    w._hold_deadlines = {}
    w._host_ready = set()
    w._failed_stages = set()
    w._import_allowed = set()
    w._discard = set()
    w._stop_polls = Event()
    w._poll_executor = ThreadPoolExecutor(max_workers=1)
    w._polls = {}
    w._recving_transfers = {}
    w._recving_metadata = {}
    w._reqs_to_send = {}
    w._invalid_block_ids = queue.SimpleQueue()
    w._failed_recv_reqs = queue.SimpleQueue()
    w.xfer_stats = Mock()
    w.nixl_wrapper = Mock()
    w.nixl_wrapper.check_xfer_state.return_value = "DONE"
    w.transfer_topo = Mock()
    w.transfer_topo.block_size_ratio.return_value = 1
    w._get_new_notifs = set
    w.use_host_buffer = True
    w.use_mla = w.enable_permute_local_kv = False
    w.enable_heterogeneous_attn_post_process = w._is_hma_required = False
    w.sync_recved_kv_to_device = Mock()
    w.tp_rank = 0
    w.engine_id = "consumer"
    monkeypatch.setattr(NixlConnectorWorker, "start_load_kv", lambda self, meta: None)

    def post(self, rid, meta):
        self._recving_transfers[rid] = [17]
        self._recving_metadata[rid] = meta

    monkeypatch.setattr(NixlConnectorWorker, "_read_blocks_for_req", post)
    yield w
    w.shutdown()


def control(w, hold=(), allow=(), cancel=()):
    meta = NixlConnectorMetadata()
    meta.hetero_hold, meta.hetero_import, meta.hetero_discard = hold, allow, cancel
    w.start_load_kv(meta)


def post(w):
    control(w, hold=("B",))
    meta = ReqMeta([[8]], [[8]], 1, RemoteMeta([[7]], "host", 8888, "P", "producer-B"))
    w._read_blocks_for_req("B", meta)
    return w._polls["B"][1]


def test_completed_read_stays_in_host_until_permitted(worker):
    assert post(worker).result(timeout=2)[0] == "DONE"
    assert worker.get_finished() == (set(), set())
    assert worker._host_ready == {"B"}
    worker.sync_recved_kv_to_device.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)
    control(worker, allow=("B",))
    assert worker.get_finished() == (set(), {"B"})
    worker.sync_recved_kv_to_device.assert_called_once()
    assert worker._hold_deadlines == {} and not worker._host_ready
    assert worker.get_finished() == (set(), set())


def test_cancel_during_receive_discards_after_quiescence(worker):
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    future = post(worker)
    control(worker, cancel=("B",))
    assert worker.get_finished() == (set(), set())
    assert "B" in worker._hold_deadlines
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    future.result(timeout=2)
    assert worker.get_finished() == (set(), {"B"})
    worker.sync_recved_kv_to_device.assert_not_called()
    assert not worker._recving_metadata and not worker._host_ready


def test_timeout_marks_reserved_blocks_invalid_after_canceling_native_handle(worker):
    worker._receive_timeout_s = 0.001
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    future = post(worker)
    assert future.result(timeout=2)[0] == "timeout"
    assert worker.get_finished() == (set(), {"B"})
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)
    assert worker._invalid_block_ids.get_nowait() == {8}
    worker.sync_recved_kv_to_device.assert_not_called()


def test_timeout_with_uncancelable_dma_refuses_to_return_a_completion(worker):
    worker._receive_timeout_s = 0.001
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    worker.nixl_wrapper.release_xfer_handle.side_effect = RuntimeError("NIXL busy")
    post(worker).result(timeout=2)
    with pytest.raises(RuntimeError, match="NIXL busy"):
        worker.get_finished()
    worker.sync_recved_kv_to_device.assert_not_called()
    assert "B" in worker._recving_metadata and "B" in worker._hold_deadlines


def test_producer_reclaims_expired_lease_after_consumer_exit(worker):
    worker._reqs_to_process = {"P-owner"}
    worker._reqs_to_send = {"P-owner": 0}
    worker.consumer_notification_counts_by_req = {}
    assert worker.get_finished() == ({"P-owner"}, set())
    assert not worker._reqs_to_process and not worker._reqs_to_send
    worker.xfer_stats.record_kv_expired_req.assert_called_once()


def test_shutdown_returns_quiesced_poll_handles_to_native_deregistration(
    worker, monkeypatch
):
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    future = post(worker)
    released = []

    def native_shutdown(self):
        assert future.done()
        released.extend(self._recving_transfers.pop("B", []))

    monkeypatch.setattr(NixlConnectorWorker, "shutdown", native_shutdown)
    worker.shutdown()
    assert released == [17]
    assert not worker._polls
    worker.sync_recved_kv_to_device.assert_not_called()
