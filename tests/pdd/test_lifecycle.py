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
"""Exercise public shell entry points using isolated CPU-only fixtures."""

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
STAGES = (
    "prepare_run", "start_prefill", "start_decode", "start_proxy",
    "check_correctness", "warmup", "run_workload", "stop_pdd",
    "summarize_results", "start_pdd",
)


def read_env(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            words = shlex.split(value)
            values[key] = words[0] if words else ""
    return values


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pdd fixture ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.repo = self.directory / "repository with spaces"
        self.repo.mkdir()
        (self.repo / "tools").mkdir()
        shutil.copytree(REPO / "tools/pdd", self.repo / "tools/pdd")
        for stage in STAGES:
            shutil.copy2(REPO / f"{stage}.sh", self.repo / f"{stage}.sh")
        shutil.copy2(REPO / "run_all_once.sh", self.repo / "run_all_once.sh")
        (self.repo / "pdd_profile_proxy.py").write_text("# fixture proxy path\n")
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        fixture = (REPO / "tests/pdd/fake_tools.py").read_text()
        for name in ("git", "nvidia-smi", "rbln-smi", "cuda-python", "rbln-python"):
            executable = self.bin / name
            executable.write_text(f"#!{sys.executable}\n" + fixture)
            executable.chmod(0o755)
        self.result = self.directory / "results" / "run with spaces"
        self.env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": str(self.directory),
            "CUDA_PY": str(self.bin / "cuda-python"),
            "RBLN_PY": str(self.bin / "rbln-python"),
            "RESULT_ROOT": str(self.directory / "results"),
            "WARMUP_REQUESTS": "2",
            "MEASURE_REQUESTS": "3",
            "MAX_TOKENS": "4",
            "CONCURRENCY": "1",
        }
        self.addCleanup(self.kill_fixture_servers)

    def kill_fixture_servers(self):
        directories = {self.result, *(self.directory / "results").glob("*")}
        for directory in directories:
            pids_file = directory / "pids.env"
            if not pids_file.exists():
                continue
            for key, value in read_env(pids_file).items():
                if key.endswith("_PID") and value.isdigit():
                    pid = int(value)
                    try:
                        # Only clean processes from this fixture directory.
                        command = Path(f"/proc/{pid}/cmdline").read_bytes()
                        if str(self.bin).encode() in command:
                            os.killpg(pid, signal.SIGKILL)
                    except (FileNotFoundError, ProcessLookupError):
                        pass

    def control(self, **settings):
        (self.bin / "control.json").write_text(json.dumps(settings))

    def call(self, stage, *arguments, expected=0, explicit=True, environment=None):
        command = ["bash", str(self.repo / f"{stage}.sh")]
        if explicit:
            command += ["--result-dir", str(self.result)]
        command += list(arguments)
        result = subprocess.run(
            command, cwd=self.directory, env=environment or self.env,
            capture_output=True, text=True, timeout=60,
        )
        if expected == 0:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        elif expected is not None:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def prepare(self):
        return self.call("prepare_run")

    def start(self, compatibility=False):
        if compatibility:
            return self.call("start_pdd")
        else:
            self.prepare()
            for stage in ("start_prefill", "start_decode", "start_proxy"):
                self.call(stage)

    def test_help_and_argument_errors_do_not_need_devices(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                self.call(stage, "--help", explicit=False)
                self.call(stage, "--not-a-real-flag", explicit=False, expected=1)
                self.call(stage, "--result-dir", explicit=False, expected=1)
        self.assertFalse(self.result.exists())
        self.call("run_workload", "--phase", "invalid", expected=1)

    def test_missing_state_and_no_newest_fallback(self):
        for stage in STAGES:
            if stage in {"prepare_run", "start_pdd", "summarize_results"}:
                continue
            with self.subTest(stage=stage):
                self.call(stage, expected=1)
        for stage in ("start_prefill", "start_decode", "start_proxy", "warmup",
                      "stop_pdd", "summarize_results"):
            self.call(stage, explicit=False, expected=1)

    def test_prepare_preserves_existing_results_and_configuration(self):
        prepared = self.prepare()
        self.assertIn(f"RESULT_DIR={self.result}", prepared.stdout)
        runtime = (self.result / "runtime.env").read_bytes()
        state = read_env(self.result / "runtime.env")
        self.assertEqual(state["MODEL_DTYPE"], "float16")
        self.assertEqual(state["WARMUP_REQUESTS"], "2")
        self.assertEqual(state["CUDA_PY"], str(self.bin / "cuda-python"))
        self.assertFalse((self.result / "fixture_prefill_invocation.json").exists())
        self.call("prepare_run", expected=1)
        self.assertEqual((self.result / "runtime.env").read_bytes(), runtime)

    def test_independent_shells_split_requests_and_summary(self):
        self.prepare()
        state_before = (self.result / "runtime.env").read_bytes()
        env = {"PATH": self.env["PATH"], "HOME": self.env["HOME"],
               "RESULT_DIR": str(self.directory / "wrong-run")}
        for stage in ("start_prefill", "start_decode", "start_proxy"):
            self.call(stage, environment=env)
        self.call("check_correctness", environment=env)
        self.call("check_correctness", environment={**env, "PROMPT_ID": "4"})
        self.call("warmup", environment=env)
        combined_before = rows(self.result / "workload_results.jsonl")
        self.assertEqual([row["phase"] for row in combined_before], ["warmup"] * 2)
        self.call("run_workload", "--phase", "measure", environment=env)
        combined_after = rows(self.result / "workload_results.jsonl")
        self.assertEqual(combined_after[:2], combined_before)
        self.assertEqual([row["phase"] for row in combined_after[2:]], ["measure"] * 3)
        self.assertEqual((self.result / "runtime.env").read_bytes(), state_before)
        self.call("stop_pdd", environment=env)
        self.call("summarize_results", environment=env)
        self.assertTrue((self.result / "summary.csv").is_file())
        summary = json.loads((self.result / "summary.json").read_text())
        self.assertEqual(summary["experiment_status"], "SUCCESS")
        self.assertEqual(summary["cleanup_status"], "SUCCESS")
        self.assertEqual(len(summary["steps"]["check_correctness"]["attempts"]), 1)
        self.assertEqual(summary["phases"]["measure"]["completed"], 3)
        self.assertEqual(summary["phases"]["measure"]["generated_token_total"], 12)
        self.assertIn("unavailable", json.dumps(summary).lower())
        self.assertIn(
            read_env(self.result / "runtime.env")["RUN_ID"], json.dumps(summary)
        )
        self.call("warmup", expected=1, environment=env)
        self.call("run_workload", "--phase", "measure", expected=1, environment=env)
        self.assertEqual(rows(self.result / "workload_results.jsonl"), combined_after)
        self.call("summarize_results", environment=env)
        rejected = json.loads((self.result / "summary.json").read_text())
        self.assertEqual(rejected["experiment_status"], "FAILED")
        self.assertEqual(rejected["phases"]["measure"]["status"], "SUCCESS")
        for step in ("warmup", "run_workload"):
            record = rejected["steps"][step]
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["attempts"][0]["status"], "SUCCESS")

    def test_compatibility_start_and_combined_workload(self):
        started = self.start(compatibility=True)
        paths = [
            line for line in started.stdout.splitlines()
            if line.startswith("RESULT_DIR=")
        ]
        self.assertEqual(paths, [f"RESULT_DIR={self.result}"])
        result = self.call("run_workload")
        self.assertIn("READY_FOR_PROFILING", result.stdout)
        observed = rows(self.result / "workload_results.jsonl")
        self.assertEqual(sum(row["phase"] == "warmup" for row in observed), 2)
        self.assertEqual(sum(row["phase"] == "measure" for row in observed), 3)
        self.assertEqual(len(rows(self.result / "fixture_requests.jsonl")), 5)
        self.call("stop_pdd")

    def test_legacy_run_all_once_parses_directory_and_cleans_up(self):
        result = self.call(
            "run_all_once", explicit=False,
            environment={**self.env, "RESULT_DIR": str(self.result)},
        )
        paths = {
            line.removeprefix("RESULT_DIR=")
            for line in result.stdout.splitlines()
            if line.startswith("RESULT_DIR=")
        }
        self.assertEqual(len(paths), 1, result.stdout)
        self.result = Path(paths.pop())
        self.assertTrue(self.result.name.startswith("native-dtype-"))
        self.assertIn("-runall-", self.result.name)
        records = rows(self.result / "workload_results.jsonl")
        self.assertEqual(sum(row["phase"] == "warmup" for row in records), 2)
        self.assertEqual(sum(row["phase"] == "measure" for row in records), 3)
        self.assertIn("READY_FOR_PROFILING", result.stdout)
        self.assertIn("PID_SCOPED_CLEANUP=PASS", result.stdout)
        stages = json.loads((self.result / "stages.json").read_text())
        self.assertEqual(stages["steps"]["stop_pdd"]["status"], "SUCCESS")
        for name, pid in read_env(self.result / "pids.env").items():
            if name.endswith("_PID"):
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists():
                    state = stat.read_text().rsplit(") ", 1)[1].split()[0]
                    self.assertIn(state, ("Z", "X"))

    def test_warmup_failure_preserves_evidence_and_fails(self):
        self.start()
        self.control(fail_requests=True)
        self.call("warmup", expected=1)
        observed = rows(self.result / "workload_results.jsonl")
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(not row["success"] for row in observed))
        self.call("run_workload", "--phase", "measure", expected=1)
        self.call("stop_pdd")
        self.call("summarize_results")
        self.assertIn("FAILED", (self.result / "summary.json").read_text())

    def test_duplicate_start_partial_and_repeated_stop(self):
        self.prepare()
        self.call("start_prefill")
        identity = read_env(self.result / "pids.env")
        self.call("start_prefill", expected=None)
        self.assertEqual(read_env(self.result / "pids.env"), identity)
        self.call("stop_pdd")
        self.call("stop_pdd")
        self.call("summarize_results")
        self.assertIn("INCOMPLETE", (self.result / "summary.json").read_text())

    def test_early_exit_records_failure_and_diagnostic(self):
        self.prepare()
        self.control(exit_role="prefill")
        result = self.call("start_prefill", expected=1)
        output = (result.stdout + result.stderr).lower()
        self.assertIn("exit", output)
        self.assertIn("prefill.log", output)
        self.call("stop_pdd")
        self.call("summarize_results")
        self.assertIn("FAILED", (self.result / "summary.json").read_text())

    def test_launch_rechecks_device_and_saved_port_without_reallocation(self):
        self.prepare()
        before = (self.result / "runtime.env").read_bytes()
        self.control(gpu_busy=True)
        self.call("start_prefill", expected=1)
        self.assertFalse((self.result / "fixture_prefill_invocation.json").exists())
        self.control()
        port = int(read_env(self.result / "runtime.env")["PREFILL_PORT"])
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", port))
            occupied.listen()
            self.call("start_prefill", expected=1)
        self.assertEqual((self.result / "runtime.env").read_bytes(), before)

    def test_preflight_rejects_dirty_tree(self):
        self.control(dirty=True)
        result = self.call("prepare_run", expected=1)
        self.assertIn("clean", result.stdout + result.stderr)

    def test_launch_rejects_changed_commit(self):
        self.prepare()
        self.control(commit="a" * 40)
        result = self.call("start_prefill", expected=1)
        self.assertIn("commit", result.stdout + result.stderr)
        self.assertFalse((self.result / "fixture_prefill_invocation.json").exists())

    def test_summary_with_missing_state_is_diagnostic(self):
        self.result.mkdir(parents=True)
        self.call("summarize_results")
        summary = json.loads((self.result / "summary.json").read_text())
        self.assertNotEqual(summary["experiment_status"], "PASS")
        self.assertEqual(summary["steps"]["start_prefill"]["status"], "NOT_RUN")
        self.assertTrue(summary["diagnostics"])

    def test_summary_with_corrupt_stage_schema_is_diagnostic(self):
        self.result.mkdir(parents=True)
        cases = (
            [],
            {"run_id": "fixture", "steps": []},
            {"run_id": "fixture", "steps": {"start_prefill": []}},
            {"run_id": "fixture", "steps": {"prepare_run": {"attempts": {}}}},
        )
        for state in cases:
            with self.subTest(state=state):
                (self.result / "stages.json").write_text(json.dumps(state))
                self.call("summarize_results")
                summary = json.loads((self.result / "summary.json").read_text())
                self.assertEqual(summary["experiment_status"], "INCOMPLETE")
                self.assertTrue(summary["diagnostics"])

    def test_saved_configuration_cannot_be_injected_or_reassigned(self):
        self.prepare()
        runtime = self.result / "runtime.env"
        marker = self.directory / "injected"
        with runtime.open("a") as handle:
            handle.write(f"MODEL_NAME=$(touch {shlex.quote(str(marker))})\n")
        self.call("start_prefill", expected=1)
        self.assertFalse(marker.exists())


class CommandPreservationTests(unittest.TestCase):
    def legacy_section(self, name):
        text = (REPO / "tools/pdd/legacy_steps.sh").read_text()
        return text.split(f"# BEGIN {name}\n", 1)[1].split(
            f"# END {name}\n", 1
        )[0]

    def test_server_command_and_environment_block_is_unchanged(self):
        # Digest of the original block at commit 199605d9; no tokens/settings
        # are regenerated by a test or obtained from the modified implementation.
        text = self.legacy_section("commands")
        block = text[text.index("COMMON_SERVER_ARGS=("):text.index('\ncase "$1" in')]
        self.assertEqual(
            hashlib.sha256(block.strip().encode()).hexdigest(),
            "6c82a74b5c3640207382f5437db4470012087bc13bec48799267fa719a5b6179",
        )

    def test_correctness_expectations_and_transfer_checks_are_unchanged(self):
        text = self.legacy_section("correctness")
        block = text[text.index("successful_transfer_reports_before="):]
        self.assertEqual(
            hashlib.sha256(block.strip().encode()).hexdigest(),
            "f1356eadc793d3b04dbed016cde19c1bf6df4a33661fff160087f643835101cf",
        )


class DriverUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "pdd_runner", REPO / "tools/pdd/runner.py"
        )
        cls.runner = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(REPO / "tools/pdd"), *sys.path]):
            spec.loader.exec_module(cls.runner)

    def test_readiness_timeout_is_distinct_from_early_exit(self):
        runtime = {"PREFILL_PORT": "9999", "RESULT_DIR": "/fixture"}
        with (
            mock.patch.object(self.runner, "role_identity", return_value={}),
            mock.patch.object(
                self.runner.urllib.request,
                "urlopen",
                side_effect=urllib.error.URLError("fixture unavailable"),
            ),
        ):
            with self.assertRaisesRegex(
                self.runner.RunError, "readiness timed out.*prefill.log"
            ):
                self.runner.wait_ready(runtime, {}, "PREFILL", 0)
        with mock.patch.object(
            self.runner,
            "role_identity",
            side_effect=self.runner.RunError("PREFILL exited before readiness"),
        ):
            with self.assertRaisesRegex(
                self.runner.RunError, "exited before readiness"
            ):
                self.runner.wait_ready(runtime, {}, "PREFILL", 0)

    def test_http_success_from_unowned_listener_is_not_readiness(self):
        runtime = {"PREFILL_PORT": "9999", "RESULT_DIR": "/fixture"}
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        with (
            mock.patch.object(self.runner, "role_identity", return_value={}),
            mock.patch.object(
                self.runner.urllib.request, "urlopen", return_value=response
            ),
            mock.patch.object(self.runner, "owns_listener", return_value=False),
        ):
            with self.assertRaisesRegex(self.runner.RunError, "readiness timed out"):
                self.runner.wait_ready(runtime, {}, "PREFILL", 0)

    def test_runtime_values_roundtrip_without_executing_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            marker = Path(directory) / "unexpected"
            values = {
                "A": "spaces ' quotes \" and\nnewlines",
                "B": f"$(touch {marker})",
            }
            self.runner.write_env(path, values)
            self.assertEqual(self.runner.read_env(path), values)
            self.assertFalse(marker.exists())

    def test_run_lock_rejects_competing_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".pdd.lock"
            with self.runner.lock(path):
                with self.assertRaisesRegex(self.runner.RunError, "holds lock"):
                    with self.runner.lock(path):
                        self.fail("competing writer acquired lock")


if __name__ == "__main__":
    unittest.main()
