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
"""Hardware-free checks for the extracted requests and diagnostic summaries."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pdd"))


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/pdd" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workload = load_module("workload")
summary = load_module("summary")


class MockRequests(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((body, dict(self.headers)))
        response = {"choices": [{"token_ids": [1, 2, 3], "text": "text",
                                 "finish_reason": "length"}]}
        if self.server.omit_tokens:
            del response["choices"][0]["token_ids"]
        self.send_response(self.server.response_status)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pdd workload ")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockRequests)
        self.server.requests = []
        self.server.omit_tokens = False
        self.server.response_status = 200
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        port = str(self.server.server_port)
        self.environment = {
            "RESULT_DIR": str(self.directory), "RUN_ID": "test-run",
            "PREFILL_PORT": port, "DECODE_PORT": port, "PROXY_PORT": port,
            "WARMUP_REQUESTS": "3", "MEASURE_REQUESTS": "5", "CONCURRENCY": "2",
            "MAX_TOKENS": "128", "PROMPT_MODE": "long",
        }

    def read_rows(self, name):
        return [
            json.loads(line)
            for line in (self.directory / name).read_text().splitlines()
        ]

    def run_phase(self, phase):
        with contextlib.redirect_stdout(io.StringIO()):
            return workload.Workload(self.environment).run(phase)

    def test_separate_shell_processes_preserve_warmup_and_only_measure(self):
        command = ["bash", "-c", 'exec "$1" "$2" --phase "$3"', "pdd-test",
                   sys.executable, str(ROOT / "tools/pdd/workload.py")]
        warmup_result = subprocess.run(
            [*command, "warmup"], env=self.environment, cwd="/", capture_output=True,
            text=True, timeout=15,
        )
        self.assertEqual(warmup_result.returncode, 0, warmup_result.stderr)
        warmup_bytes = (self.directory / "warmup_results.jsonl").read_bytes()
        self.assertEqual(len(self.server.requests), 3)
        self.assertNotIn("READY_FOR_PROFILING", warmup_result.stdout)
        self.assertFalse((self.directory / "measure_results.jsonl").exists())
        measure_result = subprocess.run(
            [*command, "measure"], env=self.environment, cwd="/", capture_output=True,
            text=True, timeout=15,
        )
        self.assertEqual(measure_result.returncode, 0, measure_result.stderr)
        self.assertEqual(len(self.server.requests), 8)
        self.assertEqual(
            (self.directory / "warmup_results.jsonl").read_bytes(), warmup_bytes
        )
        self.assertEqual(len(self.read_rows("measure_results.jsonl")), 5)
        self.assertEqual(len(self.read_rows("workload_results.jsonl")), 8)
        self.assertNotIn("READY_FOR_PROFILING", measure_result.stdout)
        phases = json.loads((self.directory / "workload_phases.json").read_text())
        self.assertEqual(phases["run_id"], "test-run")
        for name in ("warmup", "measure"):
            self.assertEqual(phases["phases"][name]["status"], "SUCCESS")
            self.assertGreater(phases["phases"][name]["elapsed_seconds"], 0)
        self.assertIn(
            "READY_DELAY_SECONDS=0",
            (self.directory / "workload_summary.txt").read_text(),
        )

    def test_legacy_all_keeps_default_counts_and_five_second_delay(self):
        del self.environment["MEASURE_REQUESTS"]
        with mock.patch.object(workload.time, "sleep") as sleep:
            self.assertEqual(self.run_phase("all"), 0)
        sleep.assert_called_once_with(5)
        rows = self.read_rows("workload_results.jsonl")
        self.assertEqual(sum(row["phase"] == "warmup" for row in rows), 3)
        self.assertEqual(sum(row["phase"] == "measure" for row in rows), 20)
        self.assertIn(
            "READY_DELAY_SECONDS=5",
            (self.directory / "workload_summary.txt").read_text(),
        )

    def test_original_body_and_prompt_hashes_are_preserved(self):
        for mode, prompt_id, checksum in (
            ("short", 2,
             "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116"),
            ("long", 4,
             "57242b91f80632b4eec84b172bfe170fa52b0d33940af0b59a9a43be51889921"),
        ):
            request = workload.Workload({**self.environment, "PROMPT_MODE": mode})
            body = json.loads(request.body_bytes)
            self.assertEqual(request.prompt_id, prompt_id)
            self.assertEqual(request.prompt_sha256, checksum)
            expected_prompt = (
                "Complete the sequence: 1, 1, 2, 3, 5,"
                if mode == "short" else
                "A clear glass of water rests on the wooden table. " * 29
                + "Continue with the next three words only:"
            )
            self.assertEqual(body, {
                "model": "Qwen/Qwen3-0.6B", "prompt": expected_prompt,
                "temperature": 0, "top_p": 1, "seed": 0, "max_tokens": 128,
                "logprobs": 5, "n": 1, "return_token_ids": True,
                "return_tokens_as_token_ids": True, "stream": False,
            })

    def test_repeat_phase_is_rejected_without_modifying_results(self):
        self.run_phase("warmup")
        original = (self.directory / "workload_results.jsonl").read_bytes()
        for phase in ("warmup", "all"):
            with self.assertRaisesRegex(ValueError, "already attempted"):
                self.run_phase(phase)
        self.assertEqual(
            (self.directory / "workload_results.jsonl").read_bytes(), original
        )

    def test_measure_requires_successful_warmup(self):
        with self.assertRaisesRegex(ValueError, "requires.*warmup"):
            self.run_phase("measure")
        self.assertFalse(self.server.requests)

    def test_warmup_failure_stops_integrated_run_and_missing_tokens_fail(self):
        self.server.omit_tokens = True
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
            workload.time, "sleep"
        ) as sleep:
            self.assertEqual(self.run_phase("all"), 1)
        sleep.assert_not_called()
        self.assertEqual(len(self.server.requests), 3)
        self.assertFalse((self.directory / "measure_results.jsonl").exists())
        rows = self.read_rows("warmup_results.jsonl")
        self.assertTrue(all(not row["success"] for row in rows))
        self.assertTrue(all("omitted" in row["error"] for row in rows))
        with self.assertRaisesRegex(ValueError, "requires.*warmup"):
            self.run_phase("measure")

    def test_measure_http_failure_and_zero_warmup(self):
        self.environment["WARMUP_REQUESTS"] = "0"
        self.assertEqual(self.run_phase("warmup"), 0)
        self.assertFalse(self.server.requests)
        self.server.response_status = 503
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_phase("measure"), 1)
        self.assertEqual(len(self.server.requests), 5)
        phases = json.loads((self.directory / "workload_phases.json").read_text())
        self.assertEqual(phases["phases"]["measure"]["failed"], 5)
        self.assertEqual(phases["phases"]["warmup"]["status"], "SUCCESS")

    def test_cross_run_phase_state_rejected(self):
        (self.directory / "workload_phases.json").write_text(
            json.dumps({"run_id": "other", "phases": {}})
        )
        with self.assertRaisesRegex(ValueError, "run ID mismatch"):
            workload.Workload(self.environment)

    def test_completed_requests_are_flushed_before_whole_batch_finishes(self):
        request = workload.Workload({**self.environment, "CONCURRENCY": "1"})
        original = request.issue_request
        second_started = threading.Event()
        release_second = threading.Event()

        def issue(phase, index):
            if index == 1:
                second_started.set()
                release_second.wait(5)
            return original(phase, index)

        with mock.patch.object(request, "issue_request", side_effect=issue):
            thread = threading.Thread(target=request.run_phase, args=("warmup", 2))
            thread.start()
            try:
                self.assertTrue(second_started.wait(5))
                path = self.directory / "warmup_results.jsonl"
                for _ in range(100):
                    if path.exists() and path.stat().st_size:
                        break
                    release_second.wait(0.01)
                self.assertEqual(len(self.read_rows(path.name)), 1)
                state = json.loads(
                    (self.directory / "workload_phases.json").read_text()
                )
                self.assertEqual(state["phases"]["warmup"]["status"], "RUNNING")
            finally:
                release_second.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(self.read_rows("warmup_results.jsonl")), 2)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.runtime = {"RUN_ID": "test-run", "COMMIT": "saved-commit",
                        "WARMUP_REQUESTS": "1", "MEASURE_REQUESTS": "2",
                        "MODEL_NAME": "fixed-model", "CUDA_DEVICE": "0"}
        self.stages = {"run_id": "test-run", "steps": {
            name: {"status": "SUCCESS", "elapsed_seconds": 0.5, "exit_code": 0}
            for name in summary.STEPS
        }}
        self.phases = {"run_id": "test-run", "phases": {}}

    def phase(self, name, rows, status="SUCCESS", elapsed=4):
        self.phases["phases"][name] = {
            "status": status, "requested": 1 if name == "warmup" else 2,
            "elapsed_seconds": elapsed, "config": {"CONCURRENCY": 1},
        }
        (self.directory / "workload_phases.json").write_text(json.dumps(self.phases))
        result_rows = [
            {"run_id": "test-run", "phase": name, "request_index": index,
             "success": success, "generated_token_count": tokens,
             "latency_seconds": latency, "error": None if success else "HTTP 503"}
            for index, (success, tokens, latency) in enumerate(rows)
        ]
        (self.directory / f"{name}_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in result_rows)
        )

    def generate(self):
        return summary.generate(self.directory, self.runtime, self.stages, [])

    def complete(self):
        self.phase("warmup", [(True, 3, 1)])
        self.phase("measure", [(True, 3, 2), (True, 5, 4)])

    def test_success_uses_actual_tokens_interval_and_observed_latencies(self):
        self.complete()
        result = self.generate()
        self.assertEqual(result["experiment_status"], "SUCCESS")
        phase = result["phases"]["measure"]
        self.assertEqual(phase["generated_token_total"], 8)
        self.assertEqual(phase["tokens_per_second"], 2)
        self.assertEqual(phase["latency_mean_seconds"], 3)
        self.assertEqual(phase["latency_p50_seconds"], 2)
        self.assertEqual(phase["latency_p95_seconds"], 4)
        self.assertEqual(result["commit"], "saved-commit")
        self.assertTrue(
            all(value is None for value in result["unavailable_metrics"].values())
        )
        for filename in ("summary.json", "summary.csv", "summary.md"):
            self.assertTrue((self.directory / filename).stat().st_size)
        self.assertEqual(
            json.loads((self.directory / "summary.json").read_text()), result
        )
        with (self.directory / "summary.csv").open() as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertTrue(any(
            row["name"] == "MODEL_NAME" and row["value"] == "fixed-model"
            for row in csv_rows
        ))
        self.assertTrue(any(
            row["name"] == "ttft_seconds" and row["value"] == "unavailable"
            for row in csv_rows
        ))

    def test_failed_measure_excludes_failed_tokens_from_throughput(self):
        self.phase("warmup", [(True, 3, 1)])
        self.phase("measure", [(True, 3, 2), (False, 2, 10)], status="FAILED")
        self.stages["steps"]["run_workload"]["status"] = "FAILED"
        result = self.generate()
        self.assertEqual(result["experiment_status"], "FAILED")
        self.assertEqual(result["summary_status"], "SUCCESS")
        phase = result["phases"]["measure"]
        self.assertEqual(phase["generated_token_total"], 5)
        self.assertEqual(phase["successful_generated_token_total"], 3)
        self.assertEqual(phase["tokens_per_second"], 0.75)
        self.assertEqual(phase["latency_mean_seconds"], 2)

    def test_absent_results_do_not_pass_even_when_steps_claim_success(self):
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertEqual(result["phases"]["measure"]["status"], "NOT_RUN")
        self.assertIsNone(result["phases"]["measure"]["tokens_per_second"])

    def test_interrupted_and_truncated_results_are_diagnostic(self):
        self.phase("warmup", [(True, 3, 1)])
        self.phase("measure", [(True, 3, 2)], status="RUNNING", elapsed=None)
        with (self.directory / "measure_results.jsonl").open("a") as handle:
            handle.write('{"run_id":')
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertEqual(result["phases"]["measure"]["completed"], 1)
        self.assertEqual(result["phases"]["measure"]["missing"], 1)
        self.assertIsNone(result["phases"]["measure"]["tokens_per_second"])
        self.assertGreaterEqual(len(result["diagnostics"]), 3)

    def test_corrupt_phase_state_and_wrong_run_never_pass(self):
        self.complete()
        self.phases["run_id"] = "other-run"
        (self.directory / "workload_phases.json").write_text(json.dumps(self.phases))
        self.assertEqual(self.generate()["experiment_status"], "INCOMPLETE")
        (self.directory / "workload_phases.json").write_text("[")
        self.assertEqual(self.generate()["experiment_status"], "INCOMPLETE")

    def test_cleanup_failure_is_recorded_separately_from_measurement(self):
        self.complete()
        self.stages["steps"]["stop_pdd"] = {
            "status": "FAILED", "exit_code": 1, "reason": "PID identity mismatch",
        }
        result = self.generate()
        self.assertEqual(result["experiment_status"], "SUCCESS")
        self.assertEqual(result["cleanup_status"], "FAILED")
        self.assertEqual(result["phases"]["measure"]["status"], "SUCCESS")

    def test_missing_correctness_is_incomplete_and_not_run(self):
        self.complete()
        del self.stages["steps"]["check_correctness"]
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertEqual(result["steps"]["check_correctness"]["status"], "NOT_RUN")

    def test_summary_writes_propagate_io_failure(self):
        (self.directory / "summary.json").mkdir()
        with self.assertRaises(OSError):
            self.generate()

    def test_failed_prepare_without_runtime_still_reports_failed(self):
        self.runtime = {}
        self.stages = {"run_id": "failed-run", "steps": {
            "prepare_run": {"status": "FAILED", "reason": "compiler unavailable"},
        }}
        result = self.generate()
        self.assertEqual(result["run_id"], "failed-run")
        self.assertEqual(result["experiment_status"], "FAILED")
        self.assertEqual(result["steps"]["prepare_run"]["status"], "FAILED")

    def test_successful_retry_does_not_hide_prior_failed_step(self):
        self.complete()
        self.stages["steps"]["start_decode"]["attempts"] = [{"status": "FAILED"}]
        self.assertEqual(self.generate()["experiment_status"], "FAILED")

    def test_unfinished_script_is_explicitly_incomplete(self):
        self.stages["steps"]["start_decode"]["status"] = "RUNNING"
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertIn("interrupted", result["steps"]["start_decode"]["reason"])

    def test_wrong_run_and_duplicate_rows_cannot_satisfy_expected_count(self):
        self.complete()
        path = self.directory / "measure_results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["run_id"] = "unrelated-run"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertEqual(result["phases"]["measure"]["completed"], 1)
        path.write_text((json.dumps(rows[0]) + "\n") * 2)
        result = self.generate()
        self.assertEqual(result["experiment_status"], "INCOMPLETE")
        self.assertEqual(result["phases"]["measure"]["completed"], 1)


if __name__ == "__main__":
    unittest.main()
