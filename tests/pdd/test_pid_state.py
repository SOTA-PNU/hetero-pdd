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
"""Per-role PID recovery remains conservative when another record is damaged."""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/pdd"))
SPEC = importlib.util.spec_from_file_location(
    "pdd_pid_runner", REPO / "tools/pdd/runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class PidStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pdd pid recovery ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.runtime = {"RUN_ID": "test-run"}
        self.stages = {"runtime_sha256": "0123456789abcdef" * 4}
        self.bindings = {
            "RESULT_DIR": str(self.directory),
            "RUN_ID": self.runtime["RUN_ID"],
            "RUNTIME_SHA256": self.stages["runtime_sha256"],
        }

    def process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def cleanup():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

        self.addCleanup(cleanup)
        return process

    def record(self, values, role, process):
        observed = runner.proc_identity(process.pid)
        values[f"{role}_PID"] = str(process.pid)
        values.update({
            f"{role}_{key}": observed[key] for key in ("PGID", "SID", "STARTTIME")
        })
        values[f"{role}_SHA256"] = runner.pid_record_digest(values, role)

    def load(self):
        return runner.load_pids(self.directory, self.runtime, self.stages)

    def test_corrupt_role_does_not_prevent_safe_other_role_cleanup(self):
        damaged = self.process()
        safe = self.process()
        values = dict(self.bindings)
        self.record(values, "PREFILL", damaged)
        self.record(values, "PROXY", safe)
        path = self.directory / "pids.env"
        runner.write_env(path, values)
        text = path.read_text().replace(
            f"PREFILL_STARTTIME={values['PREFILL_STARTTIME']}",
            "PREFILL_STARTTIME='damaged",
        )
        path.write_text(text)
        recovered, diagnostics = self.load()
        self.assertTrue(diagnostics)
        self.assertFalse(any(key.startswith("PREFILL_") for key in recovered))
        self.assertEqual(recovered["PROXY_PID"], str(safe.pid))
        result = subprocess.run(
            ["bash", str(REPO / "tools/pdd/legacy_steps.sh"), "stop"],
            env={
                "PATH": os.environ["PATH"],
                "PDD_CONTROL_PY": sys.executable,
                "PDD_STATE_ERRORS": "; ".join(diagnostics),
                **recovered,
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(safe.wait(timeout=5), -signal.SIGTERM)
        self.assertIsNone(damaged.poll())

    def test_numeric_identity_edit_requires_matching_digest(self):
        process = self.process()
        values = dict(self.bindings)
        self.record(values, "PREFILL", process)
        values["PREFILL_STARTTIME"] = "0"
        runner.write_env(self.directory / "pids.env", values)
        recovered, diagnostics = self.load()
        self.assertEqual(recovered, self.bindings)
        self.assertIn("checksum mismatch", "; ".join(diagnostics))

    def test_unknown_keys_never_override_the_execution_environment(self):
        process = self.process()
        values = dict(self.bindings)
        self.record(values, "PROXY", process)
        values["PATH"] = "/unsafe"
        values["REPO"] = "/other-repository"
        runner.write_env(self.directory / "pids.env", values)
        recovered, diagnostics = self.load()
        self.assertNotIn("PATH", recovered)
        self.assertNotIn("REPO", recovered)
        self.assertEqual(recovered["PROXY_PID"], str(process.pid))
        self.assertEqual(len(diagnostics), 2)

    def test_damaged_or_duplicate_run_binding_refuses_all_roles(self):
        process = self.process()
        values = dict(self.bindings)
        self.record(values, "PREFILL", process)
        path = self.directory / "pids.env"
        runner.write_env(path, values)
        original = path.read_text()
        for text in (original.replace("RUN_ID=test-run", "RUN_ID=other"),
                     original + "RUN_ID=test-run\n"):
            with self.subTest(text=text):
                path.write_text(text)
                recovered, diagnostics = self.load()
                self.assertEqual(recovered, {})
                self.assertTrue(diagnostics)
                self.assertIsNone(process.poll())

    def test_quoted_newline_in_result_directory_survives_recovery(self):
        self.directory /= "run\nwith newline"
        self.directory.mkdir()
        self.bindings["RESULT_DIR"] = str(self.directory)
        runner.write_env(self.directory / "pids.env", self.bindings)
        with (self.directory / "pids.env").open("a") as handle:
            handle.write("PREFILL_PID='truncated\n")
        recovered, diagnostics = self.load()
        self.assertEqual(recovered, self.bindings)
        self.assertTrue(diagnostics)

    def test_corrupt_stage_records_and_diagnostics_preserve_original_bytes(self):
        runtime = {"RESULT_DIR": str(self.directory), "RUN_ID": "test-run"}
        runner.write_env(self.directory / "runtime.env", runtime)
        digest = hashlib.sha256(
            (self.directory / "runtime.env").read_bytes()
        ).hexdigest()
        stages = {
            "run_id": "test-run", "runtime_sha256": digest,
            "steps": {"warmup": {"attempts": [None]}},
            "diagnostics": {"not": "a list"},
        }
        source = json.dumps(stages).encode()
        (self.directory / "stages.json").write_bytes(source)
        _, _, loaded, diagnostics = runner.load_state(self.directory, tolerate=True)
        self.assertNotIn("warmup", loaded["steps"])
        self.assertTrue(any("diagnostics record" in item for item in diagnostics))
        backup = next(self.directory.glob("stages.corrupt.*.json"))
        self.assertEqual(backup.read_bytes(), source)

    def test_non_utf8_stage_corruption_preserves_original_bytes(self):
        source = b"{\xffbroken"
        (self.directory / "stages.json").write_bytes(source)
        _, _, _, diagnostics = runner.load_state(self.directory, tolerate=True)
        self.assertTrue(diagnostics)
        backup = next(self.directory.glob("stages.corrupt.*.json"))
        self.assertEqual(backup.read_bytes(), source)


if __name__ == "__main__":
    unittest.main()
