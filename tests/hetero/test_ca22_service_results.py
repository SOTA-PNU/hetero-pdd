# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.hetero.profiler.ca22_service_results import counts, worker_evidence


class WorkerJoinTests(TestCase):
    def test_fragmented_transfer_and_ambiguous_h2d(self):
        native = "cmpl-test-0-ffee3311"
        start = dict(
            kind="kv_xfer_start",
            req_id=native,
            rank="0",
            monotonic_ns=1,
            remote_request_id="producer-native",
            local_block_ids=[[29, 20, 31, 2]],
            remote_block_ids=[[5, 14, 7, 8]],
        )
        done = dict(kind="kv_xfer_done", req_id=native, rank="0", monotonic_ns=901)
        h2d = dict(
            kind="kv_h2d",
            req_id=native,
            rank="0",
            start_ns=1001,
            end_ns=2101,
            h2d_ns=1000,
            reorder_ns=0,
            bytes_est=56 << 20,
            copy_calls=224,
            num_blocks=4,
            import_threads_applied=8,
        )
        transfer = dict(
            native_id=native,
            remote_id=start["remote_request_id"],
            local_block_ids=start["local_block_ids"],
            remote_block_ids=start["remote_block_ids"],
            blocks=4,
            bytes=56 << 20,
            copy_calls=224,
            receive_start_ns=1,
            host_ready_ns=901,
            receive_ms=0.0009,
            import_start_ns=1001,
            import_end_ns=2101,
            import_ms=0.0011,
            host_copy_calls_ms=0.001,
        )
        requests = [dict(target="npu0", transfer=transfer)]
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            path = work / "events/npu0/events_0.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in [start, done, h2d])
            )
            result = worker_evidence(work, requests)[("npu0", native)]
            self.assertEqual(result["physical_runs"], 4)
            self.assertEqual(result["import_threads_applied"], 8)
            with path.open("a") as f:
                f.write(json.dumps(h2d) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate worker kv_h2d"):
                worker_evidence(work, requests)

    def test_exact_request_tokens_exclude_failed_gpu_comparison(self):
        rows = [
            dict(
                executed_path="pdd",
                preload_selected=False,
                preload_approved=False,
                transfer=None,
                overlap=None,
                gpu_reference=g,
            )
            for g in (
                None,
                dict(matching=False, reference_output_length=4),
                dict(matching=True, reference_output_length=4),
            )
        ]
        result = counts(rows)
        self.assertEqual(result["gpu_checked_requests"], 2)
        self.assertEqual(result["gpu_exact_requests"], 1)
        self.assertEqual(result["gpu_reference_tokens"], 8)
        self.assertEqual(result["gpu_tokens_in_exact_requests"], 4)
