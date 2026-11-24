"""RBLN scheduler admission for bounded, host-staged NIXL reads.

Based on the local research scheduler at 8cdcb24b. The official sidecar tags
eligible transfers. Only the RBLN worker and allocator own the actual buffers.
"""

import json
import math
import time
from functools import lru_cache
from pathlib import Path

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import RequestStatus

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def _prepare_metrics():
    from prometheus_client import Gauge

    descriptions = {
        "prepare_slots_free": "Available speculative host receive slots",
        "prepare_bytes_free": "Available speculative host receive bytes",
        "running_requests": "Current RBLN scheduler running requests",
        "waiting_requests": "Current RBLN waiting and skipped requests",
        "kv_usage": "Fraction of RBLN KV blocks in use",
    }
    return {
        name: Gauge("hetero_pdd_" + name, detail, multiprocess_mode="livemostrecent")
        for name, detail in descriptions.items()
    }


class RemoteKVLookahead:
    """Mixin for RBLNScheduler; disabled unless the nested config is present."""

    def _init_lookahead(self):
        self._lookahead = None
        transfer = self.vllm_config.kv_transfer_config
        config = (
            transfer.kv_connector_extra_config.get("rbln_lookahead")
            if transfer is not None
            else None
        )
        if config is not None:
            self._configure_lookahead(config)

    def _configure_lookahead(self, config):
        if self._lookahead is not None:
            raise ValueError("lookahead is already configured")
        allowed = {
            "active_decode_limit",
            "lookahead_depth",
            "prefetch_byte_budget",
            "events_path",
            "receive_timeout_s",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("invalid rbln_lookahead configuration")
        values = [
            config.get(k)
            for k in ("active_decode_limit", "lookahead_depth", "prefetch_byte_budget")
        ]
        if any(type(v) is not int for v in values):
            raise ValueError("lookahead limits must be integers")
        active, depth, budget = values
        if not 1 <= active <= self.scheduler_config.max_num_seqs:
            raise ValueError(
                "active limit must fit the unchanged compiled max_num_seqs"
            )
        if depth < 0 or budget <= 0:
            raise ValueError("depth must be nonnegative and byte budget positive")
        timeout = config.get("receive_timeout_s", 30)
        if (
            type(timeout) not in (float, int)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("receive_timeout_s must be finite and positive")
        if self.vllm_config.kv_transfer_config.kv_buffer_device != "cpu":
            raise ValueError("rbln_lookahead requires registered CPU host buffers")
        groups = self.kv_cache_config.kv_cache_groups
        if (
            self.connector is None
            or self.cache_config.enable_prefix_caching
            or len(groups) != 1
            or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or self.num_lookahead_tokens
            or self.lora_config
            or self.is_encoder_decoder
            or self.recompute_kv_load_failures
            or self.policy != SchedulingPolicy.FCFS
            or self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
            or self.dcp_world_size != 1
            or self.pcp_world_size != 1
        ):
            raise ValueError(
                "lookahead requires TP/PP1 full-attention, prefix-off, "
                "FCFS, no LoRA/speculation, and KV failure policy fail"
            )
        tensors = self.kv_cache_config.kv_cache_tensors
        if tensors:
            total_bytes = sum(t.size for t in tensors)
            if total_bytes % self.kv_cache_config.num_blocks:
                raise ValueError("physical cache size is not a uniform block multiple")
            block_bytes = total_bytes // self.kv_cache_config.num_blocks
            byte_source = "allocated_kv_cache_tensor_sizes / num_blocks"
        else:
            # Empty tensor lists are used by CPU-only scheduler tests.
            block_bytes = groups[0].kv_cache_spec.page_size_bytes * len(
                groups[0].layer_names
            )
            byte_source = "full_attention_spec_page_size_times_layers"
        if block_bytes <= 0:
            raise ValueError("unknown KV block bytes")
        event_path = config.get("events_path")
        if event_path and not Path(event_path).is_absolute():
            raise ValueError("events_path must be absolute")
        self._lookahead = dict(
            active=active,
            depth=depth,
            budget=budget,
            block_bytes=block_bytes,
            byte_source=byte_source,
            events_path=event_path,
            pending={},
            ordinary=set(),
            last_counts=None,
            max_active=0,
            max_total=0,
            max_extra_prepare=0,
            max_pending_bytes=0,
        )
        # Do not alter scheduler_config.max_num_seqs or any compiled input shape.
        self.max_num_running_reqs = active
        self._lookahead_counts()
        self._lookahead_event(
            "CONFIG",
            compiled_max_num_seqs=self.scheduler_config.max_num_seqs,
            active_decode_limit=active,
            lookahead_depth=depth,
            prefetch_byte_budget=budget,
            block_bytes=block_bytes,
            block_bytes_source=byte_source,
            future_growth_reservation="virtual_prompt_plus_max_tokens",
        )

    def _lookahead_event(self, event, request=None, **fields):
        state = self._lookahead
        if state is None or not state["events_path"]:
            return
        transfer = self.vllm_config.kv_transfer_config
        params = getattr(request, "kv_transfer_params", None) or {}
        row = dict(
            event=event,
            monotonic_ns=time.monotonic_ns(),
            engine_id=transfer.engine_id,
            request_id=getattr(request, "request_id", None),
            remote_request_id=params.get("remote_request_id"),
            remote_engine_id=params.get("remote_engine_id"),
            **fields,
        )
        try:
            with open(state["events_path"], "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("RBLN scheduler trace disabled: %s", exc)
            state["events_path"] = None

    def _lookahead_blocks(self, request):
        return len(self.kv_cache_manager.get_block_ids(request.request_id)[0])

    def _lookahead_future_blocks(self, request):
        if request.is_finished():
            return self._lookahead_blocks(request)
        return math.ceil(
            min(request.num_prompt_tokens + request.max_tokens, self.max_model_len)
            / self.block_size
        )

    def _lookahead_memory_fits(self, candidate):
        """Protect full requested growth of every active and prepared request."""
        if self._lookahead is None:
            return True
        reserved = {r.request_id: r for r in self.running}
        reserved.update(
            {
                rid: self.requests[rid]
                for rid in (*self._lookahead["pending"], *self._lookahead["ordinary"])
            }
        )
        reserved[candidate.request_id] = candidate
        missing = sum(
            max(0, self._lookahead_future_blocks(r) - self._lookahead_blocks(r))
            for r in reserved.values()
        )
        return missing <= self.kv_cache_manager.block_pool.get_num_free_blocks()

    def _lookahead_fresh_remote(self, request):
        params = request.kv_transfer_params or {}
        return (
            self._lookahead is not None
            and request.num_computed_tokens == 0
            and params.get("do_remote_prefill") is True
            and params.get("hetero_preload") is True
            and not request.resumable
            and not request.has_encoder_inputs
            and math.ceil(request.num_tokens / self.block_size)
            * self._lookahead["block_bytes"]
            <= self._lookahead["budget"]
            and request.request_id not in self._lookahead["pending"]
        )

    def _lookahead_counts(self):
        state = self._lookahead
        if state is None:
            return
        active = len(self.running)
        pending = state["pending"]
        pending_bytes = sum(p["allocated_bytes"] for p in pending.values())
        ordinary = len(state["ordinary"])
        extra = max(0, len(pending) - max(0, state["active"] - active - ordinary))
        total = active + ordinary + len(pending)
        assert active <= state["active"]
        assert total <= state["active"] + state["depth"]
        assert extra <= state["depth"]
        assert pending_bytes <= state["budget"]
        state["max_active"] = max(state["max_active"], active)
        state["max_total"] = max(state["max_total"], total)
        state["max_extra_prepare"] = max(state["max_extra_prepare"], extra)
        state["max_pending_bytes"] = max(state["max_pending_bytes"], pending_bytes)
        ready = sum(p["ready_ns"] is not None for p in pending.values())
        ready_bytes = sum(
            p["allocated_bytes"] for p in pending.values() if p["ready_ns"] is not None
        )
        quarantined = sum(self.requests[rid].is_finished() for rid in pending)
        reserved = {r.request_id: r for r in self.running}
        reserved.update(
            {rid: self.requests[rid] for rid in (*pending, *state["ordinary"])}
        )
        growth_blocks = sum(
            max(0, self._lookahead_future_blocks(req) - self._lookahead_blocks(req))
            for req in reserved.values()
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        waiting = len(self.waiting) + len(self.skipped_waiting)
        counts = (
            active,
            waiting,
            ordinary,
            len(pending),
            extra,
            total,
            pending_bytes,
            ready,
            quarantined,
            growth_blocks,
            free_blocks,
        )
        if counts != state["last_counts"]:
            gauges = _prepare_metrics()
            # Native mostrecent vLLM gauges can have no samples on a pristine
            # multiprocess engine. Publish actual state even before first use.
            gauges["running_requests"].set(active)
            gauges["waiting_requests"].set(waiting)
            gauges["kv_usage"].set(self.kv_cache_manager.usage)
            gauges["prepare_slots_free"].set(max(0, state["depth"] - extra))
            gauges["prepare_bytes_free"].set(
                min(
                    state["budget"] - pending_bytes,
                    max(0, free_blocks - growth_blocks) * state["block_bytes"],
                )
            )
            self._lookahead_event(
                "STATE",
                active=active,
                pending=len(pending),
                extra_prepare=extra,
                total_credits=total,
                pending_bytes=pending_bytes,
                imported_count=ready,
                quarantined_count=quarantined,
                imported_bytes=ready_bytes,
                virtual_growth_reserved_blocks=growth_blocks,
                kv_pool_free_blocks=free_blocks,
                max_active=state["max_active"],
                max_total=state["max_total"],
                max_extra_prepare=state["max_extra_prepare"],
                max_pending_bytes=state["max_pending_bytes"],
            )
            state["last_counts"] = counts

    def _lookahead_observe_ready(self):
        if self._lookahead is None:
            return
        for rid in list(self._lookahead["pending"]):
            request = self.requests[rid]
            if (
                request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and rid in self.finished_recving_kv_req_ids
            ):
                self._try_promote_blocked_waiting_request(request)

    def _lookahead_promoted(self, request):
        state = self._lookahead
        if state is None or request.request_id not in state["pending"]:
            return
        pending = state["pending"][request.request_id]
        if pending["ready_ns"] is None:
            pending["ready_ns"] = time.monotonic_ns()
            self._lookahead_event(
                "FINISHED_RECV_OBSERVED",
                request,
                signal="existing_worker_finished_recving_after_host_copy_callback",
                device_ready_independently_observed=False,
                num_computed_tokens=request.num_computed_tokens,
            )

    def _prepare_lookahead_requests(self):
        state = self._lookahead
        if state is None or self._pause_state != PauseState.UNPAUSED:
            return
        for request in list(self.skipped_waiting) + list(self.waiting):
            if not self._lookahead_fresh_remote(request):
                continue
            if (
                request.status != RequestStatus.WAITING
                or request.resumable
                or request.has_encoder_inputs
            ):
                raise ValueError(
                    "lookahead supports new non-streaming decoder requests only"
                )
            if (
                len(self.running) + len(state["ordinary"]) + len(state["pending"])
                >= state["active"] + state["depth"]
            ):
                break
            block_count = math.ceil(request.num_tokens / self.block_size)
            allocated_bytes = block_count * state["block_bytes"]
            used = sum(p["allocated_bytes"] for p in state["pending"].values())
            reason = (
                "byte_budget"
                if used + allocated_bytes > state["budget"]
                else "future_growth_reserve"
                if not self._lookahead_memory_fits(request)
                else None
            )
            if reason:
                # Do not acquire credits before these gates. The oldest blocked
                # request cannot leak a credit or starve existing decode growth.
                if getattr(request, "_lookahead_blocked_reason", None) != reason:
                    self._lookahead_event("PREPARE_BLOCKED", request, reason=reason)
                    request._lookahead_blocked_reason = reason
                break
            matched, load_async = self.connector.get_num_new_matched_tokens(request, 0)
            if matched is None:
                continue
            if matched != request.num_tokens or not load_async:
                raise ValueError(
                    "lookahead requires a complete async remote prompt match"
                )
            blocks = self.kv_cache_manager.allocate_slots(
                request,
                0,
                num_external_computed_tokens=matched,
                num_lookahead_tokens=0,
                delay_cache_blocks=True,
            )
            if blocks is None:
                self._lookahead_event(
                    "PREPARE_BLOCKED", request, reason="allocation_failed"
                )
                break
            # Single-threaded scheduler owns this transition; register before
            # connector publication. A publication exception is fatal, not retry.
            state["pending"][request.request_id] = {
                "allocated_bytes": self._lookahead_blocks(request)
                * state["block_bytes"],
                "reserved_ns": time.monotonic_ns(),
                "ready_ns": None,
            }
            request.num_external_computed_tokens = matched
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            request.num_computed_tokens = matched
            self.connector.update_state_after_alloc(
                request, self.kv_cache_manager.get_blocks(request.request_id), matched
            )
            if request.prefill_stats is not None:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=0,
                    num_external_cached_tokens=matched,
                )
            if self.connector_prefix_cache_stats is not None:
                self.connector_prefix_cache_stats.record(
                    num_tokens=request.num_tokens, num_hits=matched, preempted=False
                )
            self._lookahead_event(
                "PREPARE_ALLOCATED_RECV_QUEUED",
                request,
                allocated_blocks=block_count,
                allocated_bytes=allocated_bytes,
                local_block_ids=self.kv_cache_manager.get_block_ids(request.request_id),
                prompt_token_count=request.num_tokens,
                virtual_full_output_blocks=self._lookahead_future_blocks(request),
                active_count=len(self.running),
                token_compute_scheduled=False,
            )
        self._lookahead_counts()

    def _lookahead_admission_fits(self, request):
        state = self._lookahead
        if state is None:
            return True
        # An ordinary async receive claims one real execution vacancy until it
        # runs or finishes. Only the tagged receive pass can spend extra slots.
        if (
            request.request_id not in state["pending"]
            and request.request_id not in state["ordinary"]
        ):
            imports = sum(
                entry.get("allow_import", False) for entry in state["pending"].values()
            )
            if len(self.running) + len(state["ordinary"]) + imports >= state["active"]:
                return False
        return self._lookahead_memory_fits(request)

    def _lookahead_admitted(self, request):
        state = self._lookahead
        if state is None:
            return
        state["ordinary"].discard(request.request_id)
        pending = state["pending"].pop(request.request_id, None)
        self._lookahead_event(
            "DECODE_ADMITTED" if pending else "DIRECT_ADMITTED",
            request,
            imported_dwell_ns=(
                time.monotonic_ns() - pending["ready_ns"]
                if pending and pending["ready_ns"] is not None
                else None
            ),
        )

    def _lookahead_worker_control(self, meta):
        """Keep import permission sticky until actual admission or terminal release."""
        state = self._lookahead
        if state is None:
            return
        pending = state["pending"]
        meta.hetero_hold = tuple(
            rid for rid in getattr(meta, "reqs_to_recv", ()) if rid in pending
        )
        self._lookahead_grant_import()
        live = {rid for rid in pending if not self.requests[rid].is_finished()}
        meta.hetero_import = tuple(
            rid for rid in live if pending[rid].get("allow_import")
        )
        meta.hetero_discard = tuple(pending.keys() - live)

    def _lookahead_grant_import(self):
        state = self._lookahead
        if state is None or self._pause_state != PauseState.UNPAUSED:
            return
        live = [
            (rid, p)
            for rid, p in state["pending"].items()
            if not self.requests[rid].is_finished()
        ]
        granted = sum(p.get("allow_import", False) for _, p in live)
        vacancies = (
            state["active"] - len(self.running) - len(state["ordinary"]) - granted
        )
        for rid, entry in live:
            if vacancies <= 0:
                break
            if not entry.get("allow_import"):
                entry["allow_import"] = True
                vacancies -= 1
                self._lookahead_event("IMPORT_GRANTED", self.requests[rid])

    def _lookahead_free_blocks(self, request, released_block_ids):
        state = self._lookahead
        if state is None:
            return
        state["ordinary"].discard(request.request_id)
        pending = state["pending"].pop(request.request_id, None)
        self._lookahead_event(
            "BLOCKS_RELEASED",
            request,
            actual_super_free_completed=True,
            released_block_ids=released_block_ids,
            pending_credit_released=pending is not None,
            released_pending_bytes=pending["allocated_bytes"] if pending else 0,
        )
