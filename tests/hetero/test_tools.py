# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from socket import SO_REUSEADDR, SOL_SOCKET
from unittest.mock import patch

from tests.hetero.fakes import install_fake_tools
from tools.hetero.profiler import monitors
from tools.hetero.profiler.tools import check_tools


class ToolTests(unittest.TestCase):
    def test_required_tools_by_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            binaries = {
                "ALLSMI_BIN": "all-smi",
                "OTELCOL_BIN": "otelcol",
                "PERFETTO_TP_BIN": "trace_processor_shell",
            }
            versions = {
                "all-smi": "all-smi 0.28.1",
                "otelcol": "otelcol-contrib 0.163.2",
                "trace_processor_shell": "Perfetto v58.2-build",
            }
            install_fake_tools(temp, versions)
            env = {name: str(Path(temp) / binary) for name, binary in binaries.items()}
            env["PDD_ALLSMI_PORT"] = str(monitors.free_port())
            with patch.dict(os.environ, env):
                monitor = monitors.Monitors(temp)
                self.addCleanup(monitor.stop, "all-smi")
                for _ in range(2):
                    monitor.allsmi()
                    self.assertEqual(monitor.stop("all-smi"), 0)
                for mode, count in (
                    ("none", 0),
                    ("monitor", 1),
                    ("trace", 2),
                    ("deep", 3),
                ):
                    with self.subTest(mode=mode):
                        self.assertEqual(len(check_tools(mode)), count)
                        for i, variable in enumerate(binaries):
                            with patch.dict(
                                os.environ, {variable: str(Path(temp) / "missing")}
                            ):
                                if i < count:
                                    with self.assertRaises(FileNotFoundError):
                                        check_tools(mode)
                                else:
                                    self.assertEqual(len(check_tools(mode)), count)
                install_fake_tools(temp, {**versions, "all-smi": "all-smi 0.28.10"})
                with self.assertRaisesRegex(RuntimeError, "expected 0.28.1"):
                    check_tools("monitor")
            with patch.object(monitors.socket, "socket") as sock:
                monitors.free_port()
                options = sock.return_value.__enter__().setsockopt.call_args.args
                self.assertEqual(options, (SOL_SOCKET, SO_REUSEADDR, 1))

    def test_scripts_reject_repository_paths(self):
        repo = Path(__file__).resolve().parents[2]
        scripts = repo / "tools" / "hetero" / "third_party"
        with tempfile.TemporaryDirectory() as temp:
            for script, args in (
                ("make_offline_bundle.sh", ["--out", str(repo)]),
                (
                    "install_offline.sh",
                    ["--bundle", temp, "--venv", str(repo), "--tools-dir", temp],
                ),
            ):
                with self.subTest(script=script):
                    result = subprocess.run(
                        ["bash", str(scripts / script), *args],
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("outside repository", result.stderr)
