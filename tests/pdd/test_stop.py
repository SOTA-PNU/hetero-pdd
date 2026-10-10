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
"""CPU-only cleanup tests using isolated sessions, never a device or SDK."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
LEGACY_STEPS = REPO / "tools/pdd/legacy_steps.sh"


def identity(process, role="PREFILL"):
    fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {
        f"{role}_PID": str(process.pid),
        f"{role}_PGID": fields[2],
        f"{role}_SID": fields[3],
        f"{role}_STARTTIME": fields[19],
    }


class StopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = {
            "PATH": os.environ["PATH"],
            "RESULT_DIR": self.directory.name,
            "PDD_CONTROL_PY": sys.executable,
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

    def stop(self):
        return subprocess.run(
            ["bash", str(LEGACY_STEPS), "stop"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_empty_run_and_repeated_stop_need_no_sdk(self):
        first = self.stop()
        second = self.stop()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("RBLN_CONTEXT_CLEAR=NOT_APPLICABLE", first.stdout)
        self.assertEqual(
            len(list(
                Path(self.directory.name).glob("unrelated_processes_before_stop*")
            )),
            2,
        )

    def test_partial_start_and_repeated_stop(self):
        process = self.process()
        self.env.update(identity(process))
        result = self.stop()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)
        result = self.stop()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pid_starttime_mismatch_does_not_signal(self):
        process = self.process()
        self.env.update(identity(process))
        self.env["PREFILL_STARTTIME"] = "0"
        result = self.stop()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("was reused", result.stderr)
        self.assertIsNone(process.poll())

    def test_session_mismatch_does_not_signal(self):
        process = self.process()
        self.env.update(identity(process))
        self.env["PREFILL_SID"] = str(process.pid + 1)
        result = self.stop()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not recorded as an isolated session", result.stderr)
        self.assertIsNone(process.poll())

    def test_unsafe_record_does_not_prevent_safe_cleanup(self):
        process = self.process()
        self.env.update(identity(process, "PROXY"))
        self.env["PREFILL_PID"] = "invalid"
        self.env["PDD_STATE_ERRORS"] = "PREFILL record is incomplete"
        result = self.stop()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe state records were excluded", result.stderr)
        self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)

    def test_decode_context_check_uses_control_python(self):
        process = self.process()
        self.env.update(identity(process, "DECODE"))
        self.env["RBLN_DEVICE"] = "0"
        self.env["CUDA_PY"] = "/unavailable/cuda/python"
        smi = Path(self.directory.name) / "rbln-smi"
        smi.write_text('#!/bin/sh\nprintf \'{"contexts": []}\\n\'\n')
        smi.chmod(0o700)
        self.env["PATH"] = self.directory.name + os.pathsep + self.env["PATH"]
        result = self.stop()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(process.wait(timeout=5), -signal.SIGTERM)
        self.assertIn("RBLN_CONTEXT_CLEAR=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
