"""Exercise the local launcher at its failure and process-ownership boundaries."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import yaml

from llmd import config, runtime

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        ex = ROOT / "experiments"
        self.base = yaml.safe_load((ex / "baseline.yaml").read_text())
        for key in ("model", "profile"):
            self.base[key] = str((ex / self.base[key]).resolve())
        self.bins = self.work / "bin"
        self.bins.mkdir()
        for name in ("epp", "envoy", "pd-sidecar"):
            (self.bins / name).symlink_to(sys.executable)

    def test_invalid_experiments(self):
        path = self.work / "invalid.yaml"
        for change, expected in (
            ({"model": {"name": "Org/NewModel", "revision": "latest"}}, "immutable"),
            ({"policy": "lookahead"}, "unknown keys"),
            ({"endpoints": self.base["endpoints"] * 2}, "duplicate"),
            ({"ports": {"gateway": self.base["endpoints"][0]["port"]}}, "duplicate"),
        ):
            with self.subTest(change=change):
                path.write_text(yaml.safe_dump({**self.base, **change}))
                with self.assertRaisesRegex(ValueError, expected):
                    config.load(path)
        path.write_text("policy: baseline\npolicy: greedy\n")
        with self.assertRaisesRegex(ValueError, "unique string key"):
            config.load(path)

    def test_failed_cli_start_returns_nonzero_and_preserves_foreign_listener(self):
        # An unavailable ingress port must not cause adoption or teardown of
        # the process already listening there. Repeated stop is harmless.
        with socket.socket() as other:
            other.bind(("127.0.0.1", 0))
            other.listen()
            self.base["ports"] = {"gateway": other.getsockname()[1]}
            path = self.work / "config.yaml"
            path.write_text(yaml.safe_dump(self.base))
            work = self.work / "run"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llmd",
                    "start",
                    str(path),
                    "--work-dir",
                    str(work),
                ],
                env={**os.environ, "LLMD_BIN_DIR": str(self.bins)},
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(run.returncode, 0)
            state = json.loads((work / "run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["processes"], {})
            self.assertIn("unavailable", state["error"])
            for _ in range(2):
                subprocess.run(
                    [sys.executable, "-m", "llmd", "stop", "--work-dir", str(work)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    timeout=20,
                )
            self.assertEqual(
                json.loads((work / "run.json").read_text())["status"], "stopped"
            )
            self.assertEqual(
                other.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1
            )

    def test_all_examples_and_full_pair_graph(self):
        for path in (ROOT / "experiments").glob("*.yaml"):
            self.assertEqual(config.load(path)["apiVersion"], "hetero-pdd/v2")
        cfg = config.load(ROOT / "experiments/pool-2p2d.yaml")
        self.assertEqual(
            [e["peers"] for e in cfg["endpoints"] if "decode" in e["roles"]],
            [["gpu0", "gpu1"], ["gpu0", "gpu1"]],
        )
        p0, p1, d, _ = cfg["endpoints"]
        d["node"] = "remote"
        self.assertIn("same node", config.pair_error(p0, d, cfg["model"]))
        d["backend"] = "cuda"
        self.assertIsNone(config.pair_error(p0, d, cfg["model"]))
        d["engine"]["tp"] = 2
        self.assertIn("shard", config.pair_error(p1, d, cfg["model"]))

    def test_verify_invalidates_stale_success(self):
        from llmd.verify import verify

        report = self.work / "verification.json"
        report.write_text('{"status":"passed"}')
        state = {
            "status": "stopped",
            "model": {},
            "experiment": self.base,
            "run_id": uuid.uuid4().hex,
            "endpoints": [],
        }
        with self.assertRaisesRegex(ValueError, "requires ready"):
            verify(self.work, state, None)
        self.assertEqual(json.loads(report.read_text())["status"], "failed")

    def test_overlapping_launches_cannot_claim_the_same_device(self):
        endpoints = [{"kind": "gpu", "devices": [2 * os.getpid()]}]
        with (
            runtime.claim_devices(endpoints),
            self.assertRaisesRegex(RuntimeError, "another local launch"),
            runtime.claim_devices(endpoints),
        ):
            self.fail("conflicting claim was granted")
        with runtime.claim_devices(endpoints):
            pass  # The kernel released the first claim.

    def test_start_failure_cleans_only_its_process_tree(self):
        from llmd import __main__ as launcher

        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env={**os.environ, "HETERO_LLMD_RUN_ID": uuid.uuid4().hex},
        )
        self.addCleanup(other.wait)
        self.addCleanup(other.terminate)
        endpoints = []
        with (
            socket.socket() as port1,
            socket.socket() as port2,
            socket.socket() as gateway,
        ):
            for s in (port1, port2, gateway):
                s.bind(("127.0.0.1", 0))
            for i, s in enumerate((port1, port2)):
                endpoints.append(
                    {
                        **self.base["endpoints"][0],
                        "id": f"gpu-{i}",
                        "devices": [2 * os.getpid() + i],
                        "port": s.getsockname()[1],
                        "slots": 1,
                    }
                )
            ingress_port = gateway.getsockname()[1]

        # Launch genuine subprocesses. One engine starts an independently
        # sessioned worker, another exits with code 7 after the worker is up.
        worker_file = self.work / "worker-pid"
        launch_script = """
import os, pathlib, subprocess, sys, time
worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                          start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(worker.pid))
time.sleep(60)
"""
        fail_script = """
import pathlib, sys, time
until = time.monotonic() + 5
while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < until:
    time.sleep(.05)
sys.exit(7)
"""

        def spec(_model, ep, **_unused):
            script = launch_script if ep["id"] == "gpu-0" else fail_script
            return {
                "engine_id": ep["id"],
                "argv": [sys.executable, "-c", script, str(worker_file)],
                "env": os.environ,
            }

        with contextlib.ExitStack() as hold:
            ports = {"gateway": ingress_port}
            for key in ("admin", "epp", "epp_health", "epp_metrics"):
                s = hold.enter_context(socket.socket())
                s.bind(("127.0.0.1", 0))
                ports[key] = s.getsockname()[1]
        raw = {
            **self.base,
            "endpoints": endpoints,
            "ports": ports,
            "cache_root": str(self.work / "cache"),
        }
        cfg_file = self.work / "experiment.yaml"
        cfg_file.write_text(yaml.safe_dump(raw))
        cfg = config.load(cfg_file)
        run_dir = self.work / "run"
        run_dir.mkdir()
        with (
            patch.dict(os.environ, LLMD_BIN_DIR=str(self.bins)),
            patch.object(
                launcher.engines,
                "preflight",
                return_value={e["id"]: {"device": e["id"]} for e in endpoints},
            ),
            patch.object(launcher.engines, "engine_spec", side_effect=spec),
            self.assertRaisesRegex(RuntimeError, "exited with code 7"),
        ):
            launcher.start(cfg, run_dir, 8)

        state = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(set(state["processes"]), {"gpu-0", "gpu-1"})
        self.assertTrue(worker_file.exists())
        self.assertEqual(runtime.owned_processes(state["run_id"]), [])
        self.assertIsNone(other.poll())


if __name__ == "__main__":
    unittest.main()
