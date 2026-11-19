"""CPU scheduler regressions; no accelerator runtime/model execution."""

import json
import sys
from functools import partial

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from .test_pd_disaggregation import _simulate_kv_transfer_completion
from .utils import (
    MockKVConfig,
    advance_to_decode,
    create_requests,
    create_runner_output,
    create_scheduler,
)


@pytest.fixture(scope="module")
def model_directory(tmp_path_factory):
    from transformers import OPTConfig

    directory = tmp_path_factory.mktemp("lookahead-opt")
    config = OPTConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=2)
    config.architectures = ["OPTForCausalLM"]
    config.save_pretrained(directory)
    return str(directory)


@pytest.fixture(autouse=True)
def offline_scheduler_config(monkeypatch, model_directory):
    monkeypatch.setattr(
        sys.modules[__name__],
        "create_scheduler",
        partial(create_scheduler, model=model_directory, skip_tokenizer_init=True),
    )


def make_scheduler(
    active=1, depth=1, blocks=128, budget=2**30, events=None, matched=64
):
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        num_blocks=blocks,
        block_size=16,
        use_kv_connector=MockKVConfig(matched_tokens=matched, is_async=True),
    )
    scheduler.recompute_kv_load_failures = False
    scheduler.vllm_config.kv_transfer_config.kv_buffer_device = "cpu"
    scheduler._configure_lookahead(
        dict(
            active_decode_limit=active,
            lookahead_depth=depth,
            prefetch_byte_budget=budget,
            events_path=str(events) if events else None,
        )
    )
    return scheduler


def request(name, remote=True, tokens=64, output=16):
    req = create_requests(
        1,
        num_tokens=tokens,
        max_tokens=output,
        block_size=16,
        req_ids=[name],
        ignore_eos=True,
    )[0]
    if remote:
        req.kv_transfer_params = dict(
            do_remote_prefill=True,
            hetero_preload=True,
            remote_request_id="producer-" + name,
            remote_engine_id="gpu",
        )
    return req


def add_pending(scheduler, name):
    req = request(name)
    scheduler.add_request(req)
    return req


def test_depth_zero_prepares_empty_slot_and_promotes_full_prefix():
    scheduler = make_scheduler(depth=0)
    req = add_pending(scheduler, "r0")
    output = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert output.total_num_scheduled_tokens == 0
    assert not output.scheduled_new_reqs and not scheduler.running
    assert len(scheduler._lookahead["pending"]) == 1
    _simulate_kv_transfer_completion(scheduler, output, req.request_id)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {req.request_id: 1}
    assert req.num_computed_tokens == req.num_prompt_tokens
    assert not scheduler._lookahead["pending"]
    assert scheduler.scheduler_config.max_num_seqs == 4


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_active_full_prepare_bound_and_compute_batch_unchanged(depth):
    scheduler = make_scheduler(depth=depth)
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    pending = [add_pending(scheduler, "r" + str(i)) for i in range(4)]
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {running.request_id}
    assert len(scheduler._lookahead["pending"]) == depth
    assert [
        r.request_id
        for r in pending
        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    ] == ["r" + str(i) for i in range(depth)]
    assert all(r.request_id not in output.num_scheduled_tokens for r in pending)
    assert scheduler._lookahead["max_extra_prepare"] == depth


def test_token_budget_zero_does_not_prevent_receive_preparation():
    scheduler = make_scheduler()
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    scheduler.max_num_scheduled_tokens = 1
    pending = add_pending(scheduler, "r")
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 1
    assert pending.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert pending.request_id not in output.num_scheduled_tokens


def test_no_running_token_budget_zero_still_prepares():
    scheduler = make_scheduler()
    scheduler.max_num_scheduled_tokens = 0
    req = add_pending(scheduler, "r")
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_finished_recv_observed_while_full_then_individual_admission(tmp_path):
    events = tmp_path / "events.jsonl"
    scheduler = make_scheduler(depth=2, events=events)
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    r0 = add_pending(scheduler, "r0")
    r1 = add_pending(scheduler, "r1")
    output = scheduler.schedule()
    _simulate_kv_transfer_completion(scheduler, output, r0.request_id)
    output = scheduler.schedule()
    assert r0.status == RequestStatus.WAITING
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert r0.request_id not in output.num_scheduled_tokens
    scheduler.update_from_output(output, create_runner_output(output, 1))
    scheduler.finish_requests([running.request_id], RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    assert r0.request_id in output.num_scheduled_tokens
    assert r1.request_id not in output.num_scheduled_tokens
    rows = [json.loads(s) for s in events.read_text().splitlines()]
    ready = [r for r in rows if r["event"] == "FINISHED_RECV_OBSERVED"]
    assert len(ready) == 1 and ready[0]["remote_request_id"] == "producer-r0"
    assert ready[0]["device_ready_independently_observed"] is False
    eligible = [
        r for r in rows if r["event"] == "IMPORT_GRANTED" and r["request_id"] == "r0"
    ]
    admit = [
        r for r in rows if r["event"] == "DECODE_ADMITTED" and r["request_id"] == "r0"
    ]
    assert eligible and eligible[-1]["monotonic_ns"] < admit[0]["monotonic_ns"]


def test_cancel_receiving_quarantines_until_actual_late_completion():
    scheduler = make_scheduler(depth=0)
    req = add_pending(scheduler, "cancel")
    output = scheduler.schedule()
    original_blocks = scheduler.kv_cache_manager.get_block_ids(req.request_id)
    scheduler.finish_requests([req.request_id], RequestStatus.FINISHED_ABORTED)
    assert req.request_id in scheduler.requests
    assert req.request_id in scheduler._lookahead["pending"]
    assert scheduler.kv_cache_manager.get_block_ids(req.request_id) == original_blocks
    following = add_pending(scheduler, "following")
    assert scheduler.schedule().total_num_scheduled_tokens == 0
    assert following.status == RequestStatus.WAITING
    _simulate_kv_transfer_completion(scheduler, output, req.request_id)
    assert req.request_id not in scheduler.requests
    assert req.request_id not in scheduler._lookahead["pending"]
    scheduler.schedule()
    assert following.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_cancel_after_native_import_releases_immediately():
    scheduler = make_scheduler()
    req = add_pending(scheduler, "cancel")
    output = scheduler.schedule()
    _simulate_kv_transfer_completion(scheduler, output, req.request_id)
    scheduler._lookahead_observe_ready()
    scheduler.finish_requests([req.request_id], RequestStatus.FINISHED_ABORTED)
    assert req.request_id not in scheduler.requests
    assert req.request_id not in scheduler._lookahead["pending"]


def test_transfer_failure_is_error_not_hidden_recompute_and_safe_release():
    scheduler = make_scheduler()
    req = add_pending(scheduler, "failed")
    output = scheduler.schedule()
    block_id = scheduler.kv_cache_manager.get_block_ids(req.request_id)[0][0]
    result = create_runner_output(output, 1)
    result.kv_connector_output = KVConnectorOutput(
        invalid_block_ids={block_id}, finished_recving={req.request_id}
    )
    scheduler.update_from_output(output, result)
    assert req.status == RequestStatus.FINISHED_ERROR
    assert req.request_id not in scheduler.requests
    assert req.request_id not in scheduler._lookahead["pending"]
    assert req.num_output_tokens == 0


def test_early_eos_releases_active_slot_for_ready_request():
    scheduler = make_scheduler()
    first = add_pending(scheduler, "first")
    second = add_pending(scheduler, "second")
    output = scheduler.schedule()
    scheduler.finished_recving_kv_req_ids.update({first.request_id, second.request_id})
    output = scheduler.schedule()
    assert first.status == RequestStatus.RUNNING
    scheduler.update_from_output(output, create_runner_output(output, 1))
    scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_STOPPED)
    output = scheduler.schedule()
    assert second.status == RequestStatus.RUNNING
    assert list(output.num_scheduled_tokens) == [second.request_id]


def test_ready_credit_return_can_reuse_request_id_without_stale_state():
    scheduler = make_scheduler(depth=0)
    old = add_pending(scheduler, "same")
    output = scheduler.schedule()
    _simulate_kv_transfer_completion(scheduler, output, old.request_id)
    scheduler._lookahead_observe_ready()
    scheduler.finish_requests([old.request_id], RequestStatus.FINISHED_ABORTED)
    new = add_pending(scheduler, "same")
    output = scheduler.schedule()
    assert new.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler._lookahead["pending"][new.request_id]["ready_ns"] is None
    assert not output.num_scheduled_tokens


def test_memory_growth_reserve_blocks_prepare_without_allocating_future():
    # 1 null + running full future 5 + prepared full future 5 > 10 blocks.
    scheduler = make_scheduler(blocks=10)
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    pending = add_pending(scheduler, "pending")
    scheduler.schedule()
    assert pending.status == RequestStatus.WAITING
    assert not scheduler._lookahead["pending"]
    assert not scheduler.kv_cache_manager.get_block_ids(pending.request_id)[0]
    assert scheduler.kv_cache_manager.get_block_ids(running.request_id)[0]


def test_oversized_preload_waits_for_native_in_slot_receive():
    scheduler = make_scheduler(budget=1)
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    req = add_pending(scheduler, "large")
    scheduler.schedule()
    assert req.status == RequestStatus.WAITING
    scheduler.finish_requests([running.request_id], RequestStatus.FINISHED_STOPPED)
    scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler._lookahead["ordinary"] == {req.request_id}
    assert not scheduler._lookahead["pending"]


def test_future_reservation_also_protects_against_direct_admission():
    scheduler = make_scheduler(active=2, depth=1, blocks=10)
    pending = add_pending(scheduler, "pending")
    scheduler.schedule()
    direct = request("direct", remote=False)
    scheduler.add_request(direct)
    output = scheduler.schedule()
    assert direct.request_id not in output.num_scheduled_tokens
    assert direct.status == RequestStatus.WAITING
    assert pending.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("active,depth", [(4, 1), (4, 2), (4, 4)])
def test_four_active_plus_prepare_stay_outside_dispatch(active, depth):
    scheduler = make_scheduler(active=active, depth=depth)
    initial = [add_pending(scheduler, "a" + str(i)) for i in range(active)]
    output = scheduler.schedule()
    for req in initial:
        scheduler.finished_recving_kv_req_ids.add(req.request_id)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == active
    scheduler.update_from_output(output, create_runner_output(output, 1))
    extra = [add_pending(scheduler, "p" + str(i)) for i in range(depth + 1)]
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == active
    assert len(scheduler._lookahead["pending"]) == depth
    assert all(req.request_id not in output.num_scheduled_tokens for req in extra)
    assert scheduler.scheduler_config.max_num_seqs == 4


def test_partial_or_synchronous_remote_match_fails_closed():
    scheduler = make_scheduler(matched=32)
    add_pending(scheduler, "r")
    with pytest.raises(ValueError, match="complete async"):
        scheduler.schedule()
    assert not scheduler._lookahead["pending"]


def test_disabled_has_no_active_override_and_original_pd_lifecycle():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        use_kv_connector=MockKVConfig(matched_tokens=64, is_async=True),
    )
    assert scheduler._lookahead is None and scheduler.max_num_running_reqs == 4
    req = add_pending(scheduler, "r")
    output = scheduler.schedule()
    _simulate_kv_transfer_completion(scheduler, output, req.request_id)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {req.request_id: 1}


@pytest.mark.parametrize(
    "updates",
    [
        dict(active_decode_limit=5),
        dict(lookahead_depth=-1),
        dict(prefetch_byte_budget=0),
        dict(active_decode_limit=True),
        dict(unknown=1),
        dict(events_path="relative"),
    ],
)
def test_invalid_limits_fail_closed(updates):
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        use_kv_connector=MockKVConfig(matched_tokens=64, is_async=True),
    )
    scheduler.recompute_kv_load_failures = False
    settings = dict(active_decode_limit=1, lookahead_depth=1, prefetch_byte_budget=1000)
    settings.update(updates)
    with pytest.raises(ValueError):
        scheduler._configure_lookahead(settings)


def test_only_tagged_receives_can_preload_and_older_import_keeps_vacancy():
    scheduler = make_scheduler()
    running = request("running", remote=False)
    advance_to_decode(scheduler, running)
    ordinary = add_pending(scheduler, "ordinary")
    del ordinary.kv_transfer_params["hetero_preload"]
    tagged = add_pending(scheduler, "tagged")
    out = scheduler.schedule()
    assert ordinary.status == RequestStatus.WAITING
    assert tagged.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert out.kv_connector_metadata.hetero_import == ()
    scheduler.finish_requests([running.request_id], RequestStatus.FINISHED_STOPPED)
    out = scheduler.schedule()
    assert out.kv_connector_metadata.hetero_import == ("tagged",)
    assert ordinary.status == RequestStatus.WAITING
    scheduler.finish_requests([tagged.request_id], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    assert out.kv_connector_metadata.hetero_discard == ("tagged",)
