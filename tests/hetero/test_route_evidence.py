"""Regression tests for the measured overlap evidence boundary."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from llmd.overlap import _gpu_full_route
from tools.hetero.profiler.route_evidence import (
    _overlap,
    _transfer,
    _worker,
    analyze_routes,
)


class RouteEvidenceTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.rows = []
        self.sched = []
        self.anchor, self.follow = "cmpl-a-0-deadbeef", "cmpl-b-0-12345678"
        self.events_dir = self.work / "events/npu"
        self.events_dir.mkdir(parents=True)
        self.row = dict(request_id="b", actual_input_tokens=128, response_headers={})
        self.ep = dict(
            id="npu",
            kind="npu",
            engine_id="n",
            port=1,
            sidecar_port=2,
            roles=["full", "decode"],
            engine=dict(block_size=128, lookahead=dict(active_decode_limit=1)),
        )
        self.producer = dict(
            id="gpu", kind="gpu", port=3, roles=["full"], engine_id="g"
        )
        self.event(
            "kv_xfer_start",
            req_id=self.follow,
            engine_id="n",
            remote_engine_id="g",
            remote_request_id="cmpl-b-0-cafe1234",
            local_block_ids=[[2]],
            remote_block_ids=[[4]],
            monotonic_ns=200,
        )
        self.event(
            "model_call",
            request_ids=[self.anchor],
            phase="decode",
            span_start_ns=150,
            span_end_ns=155,
        )
        self.event("step", phase="decode", span_start_ns=149, span_end_ns=280)
        self.event("kv_xfer_done", req_id=self.follow, monotonic_ns=260)
        self.event(
            "kv_host_ready", req_id=self.follow, monotonic_ns=260, last_proc_ns=250
        )
        self.event(
            "kv_h2d",
            req_id=self.follow,
            num_blocks=1,
            bytes_est=128,
            start_ns=340,
            end_ns=410,
        )
        self.event(
            "model_call",
            request_ids=[self.follow],
            phase="decode",
            span_start_ns=450,
            span_end_ns=455,
        )
        self.event("step", phase="decode", span_start_ns=448, span_end_ns=490)
        self.admit("REQUEST_TERMINAL", self.anchor, 300)
        self.admit(
            "PREPARE_ALLOCATED_RECV_QUEUED",
            self.follow,
            199,
            active_count=1,
            token_compute_scheduled=False,
            allocated_bytes=128,
            local_block_ids=[[2]],
        )
        self.admit("IMPORT_GRANTED", self.follow, 320)

    def event(self, kind, **fields):
        if kind == "step":
            fields["sampled"] = True
        self.rows.append(dict(rank="0", kind=kind, **fields))

    def admit(self, event, rid, time, **fields):
        self.sched.append(
            dict(event=event, request_id=rid, monotonic_ns=time, **fields)
        )

    def save_rows(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def proof(self):
        self.save_rows(self.events_dir / "events_0.jsonl", self.rows)
        self.save_rows(self.work / "npu-admission.jsonl", self.sched)
        worker = _worker(self.work, "npu")
        transfer = _transfer(
            self.row, self.ep, self.producer, worker[0][self.follow], self.follow
        )
        return _overlap(transfer, worker, self.follow, 1)

    def test_requires_progress_in_an_identified_decode_step(self):
        self.assertEqual(self.proof()["receive_during_decode_ms"], 60 / 1e6)
        self.rows[2]["span_end_ns"] = (
            245  # Receive spans a step but no observed NIXL progress inside it.
        )
        self.assertIsNone(self.proof())

    def test_reject_misordered_grant_and_incomplete_mapping(self):
        self.sched[-1]["monotonic_ns"] = (
            200  # Previous request has not released its only active credit.
        )
        self.assertIsNone(self.proof())
        self.sched[-1]["monotonic_ns"] = 350  # H2D already started.
        self.assertIsNone(self.proof())
        self.sched[-1]["monotonic_ns"] = 320
        self.rows[0]["remote_block_ids"] = [[]]
        self.assertIsNone(self.proof())

    def test_gpu_reference_requires_cuda_destination_and_full_epp_decision(self):
        target = dict(self.producer, url="http://127.0.0.1:3", backend="cuda")
        decision = dict(request_id="b", path="full", target="gpu", prefill=None)
        self.save_rows(
            self.work / "epp.log",
            [dict(decision, body="hetero_route", epp_request_id="e")],
        )
        access = dict(
            decision, event="gateway_access", http_status=200, destination="127.0.0.1:3"
        )
        self.save_rows(self.work / "gateway.log", [access])
        self.assertEqual(_gpu_full_route(self.work, "b", target)["epp_request_id"], "e")
        self.save_rows(
            self.work / "gateway.log", [dict(access, destination="127.0.0.1:2")]
        )
        with self.assertRaisesRegex(ValueError, "direct CUDA"):
            _gpu_full_route(self.work, "b", target)

    def test_selected_route_alone_does_not_confirm_transfer(self):
        self.proof()
        self.rows[0]["remote_engine_id"] = "another-engine"
        self.proof()
        (self.work / "run.json").write_text(
            json.dumps({"endpoints": [self.ep, self.producer]})
        )
        (self.work / "profile").mkdir()
        self.save_rows(
            self.work / "profile/requests.jsonl",
            [
                {
                    **self.row,
                    "error": None,
                    "http_status": 200,
                    "actual_output_tokens": 8,
                    "requested_output_tokens": 8,
                    "actual_send_monotonic": 0.5,
                    "first_output_monotonic": 0.6,
                    "completed_monotonic": 0.9,
                },
            ],
        )
        self.save_rows(
            self.work / "epp.log",
            [
                dict(
                    body="hetero_route",
                    epp_request_id="e",
                    request_id="b",
                    path="pdd",
                    target="npu",
                    prefill="gpu",
                ),
                dict(
                    body="hetero_route_prediction",
                    epp_request_id="e",
                    request_id="b",
                    estimate=dict(Preload=True),
                ),
            ],
        )
        self.save_rows(
            self.work / "gateway.log",
            [
                dict(
                    event="gateway_access",
                    request_id="b",
                    path="pdd",
                    target="npu",
                    prefill="gpu",
                    http_status=200,
                    destination="localhost:2",
                )
            ],
        )
        result = analyze_routes(self.work)
        self.assertEqual(result["requested_paths"], {"pdd": 1})
        self.assertEqual(result["executed_paths"], {"unverified": 1})
        self.assertEqual(
            (
                result["preload_selected"],
                result["preload_approved"],
                result["actual_overlap"],
            ),
            (1, 1, 0),
        )
