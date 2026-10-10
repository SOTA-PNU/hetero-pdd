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
"""Interrupted helpers and failed PID persistence cannot leave untracked work."""

import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/pdd"))
SPEC = importlib.util.spec_from_file_location(
    "pdd_signal_runner", REPO / "tools/pdd/runner.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class InterruptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pdd interrupted ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.env = {"PATH": os.environ["PATH"], "REPO": str(self.directory)}

    def test_helper_interrupt_terminates_its_descendants(self):
        script = (
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(120)']); "
            "print(child.pid, flush=True); time.sleep(120)"
        )
        log = self.directory / "helper.log"
        with mock.patch.object(
            runner, "print", side_effect=KeyboardInterrupt, create=True
        ):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_helper([sys.executable, "-c", script], self.env, log)
        child = runner.proc_identity(int(log.read_text().strip()))
        self.assertTrue(child is None or child["state"] in ("Z", "X"))

    def start_with_mocked_environment(self, write):
        spawned = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            spawned.append(process)

            def cleanup():
                if process.poll() is None:
                    runner.terminate_spawned_session(
                        process, runner.proc_identity(process.pid)
                    )

            self.addCleanup(cleanup)
            return process

        pids = {
            "RESULT_DIR": str(self.directory), "RUN_ID": "test-run",
            "RUNTIME_SHA256": "test-settings",
        }
        stages = {"steps": {"prepare_run": {"status": "SUCCESS"}}}
        with contextlib.ExitStack() as stack:
            for name in ("check_code", "check_devices", "check_ports", "wait_ready"):
                stack.enter_context(mock.patch.object(runner, name))
            stack.enter_context(mock.patch.object(
                runner, "launch_lock", return_value=contextlib.nullcontext()
            ))
            stack.enter_context(mock.patch.object(
                runner, "execution_env", return_value=self.env
            ))
            stack.enter_context(mock.patch.object(
                runner.subprocess, "Popen", side_effect=spawn
            ))
            stack.enter_context(mock.patch.object(
                runner, "write_env", side_effect=write
            ))
            try:
                runner.start_role(self.directory, self.env, pids, stages, "PREFILL")
            except BaseException as exc:
                return spawned, pids, exc
        self.fail("expected a failure or deferred interruption")

    def test_failed_pid_write_rolls_back_only_the_new_session(self):
        spawned, pids, error = self.start_with_mocked_environment(OSError("disk full"))
        self.assertIsInstance(error, OSError)
        self.assertEqual(spawned[0].wait(timeout=5), -signal.SIGTERM)
        self.assertNotIn("PREFILL_PID", pids)

    def test_signal_during_pid_write_is_delayed_until_record_is_durable(self):
        write_env = runner.write_env
        previous = signal.getsignal(signal.SIGTERM)

        def write_then_signal(path, values):
            write_env(path, values)
            os.kill(os.getpid(), signal.SIGTERM)

        spawned, pids, error = self.start_with_mocked_environment(write_then_signal)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertIsNone(spawned[0].poll())
        self.assertEqual(runner.read_env(self.directory / "pids.env"), pids)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_changed_spawned_pid_identity_never_authorizes_signals(self):
        process = mock.Mock(pid=987654)
        original = {"PGID": "987654", "SID": "987654", "STARTTIME": "10", "state": "S"}
        changed = dict(original, STARTTIME="11")
        with mock.patch.object(runner, "proc_identity", return_value=changed):
            with mock.patch.object(runner.os, "killpg") as killpg:
                with self.assertRaisesRegex(runner.RunError, "identity changed"):
                    runner.terminate_spawned_session(process, original)
                killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
