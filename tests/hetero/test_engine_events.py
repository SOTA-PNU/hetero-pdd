# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[2]


def load_metrics(enabled, directory):
    envs = types.SimpleNamespace(
        VLLM_RBLN_METRICS=enabled, VLLM_RBLN_METRICS_DIR=directory
    )
    logger = Mock()
    modules = {
        "numpy": Mock(),
        "rebel": types.ModuleType("rebel"),
        "vllm_rbln": types.SimpleNamespace(envs=envs),
        "vllm_rbln.envs": envs,
        "vllm_rbln.logger": types.SimpleNamespace(init_logger=lambda _: logger),
    }
    spec = importlib.util.spec_from_file_location(
        "wp4_metrics", REPO / "vllm_rbln/v1/worker/metrics_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    return module, logger


class EngineEventTests(unittest.TestCase):
    def test_flush_and_saved_step_times(self):
        with tempfile.TemporaryDirectory() as temp:
            m, _ = load_metrics(True, temp)
            with patch.object(m, "_rank_tag", return_value=""):
                context = m.PerformanceContext()
            log = m._event_log
            self.addCleanup(log.close)
            with patch.object(log.file, "flush", wraps=log.file.flush) as flush:
                for _ in range(255):
                    m.record_event("test")
                flush.assert_not_called()
                m.record_event("test")
                self.assertEqual(flush.call_count, 1)
                m.record_event("kv_layout")
                self.assertEqual(flush.call_count, 2)
                m.record_event("clock_anchor")
                self.assertEqual(flush.call_count, 3)
            sample = m._Sample(1.0, start_mono_ns=10, end_mono_ns=20)
            merged = sample.merged(m._Sample(2.0, start_mono_ns=30, end_mono_ns=40))
            self.assertEqual(
                (merged.latency, merged.start_mono_ns, merged.end_mono_ns), (3, 10, 40)
            )
            with (
                patch.object(log, "write") as write,
                patch.object(m.time, "monotonic_ns", side_effect=AssertionError),
            ):
                context._record(merged)
            self.assertEqual(write.call_args.kwargs["monotonic_ns"], 40)
            log.close()

    def test_time_based_flush(self):
        with tempfile.TemporaryDirectory() as temp:
            m, _ = load_metrics(True, temp)
            with patch.object(m.time, "monotonic_ns", return_value=0) as clock:
                log = m._EventLog(temp, "")
                self.addCleanup(log.close)
                with patch.object(log.file, "flush", wraps=log.file.flush) as flush:
                    for tick, expected in enumerate((0, 1, 1, 2), 1):
                        clock.return_value = tick * 500_000_000
                        log.write("test", monotonic_ns=0)
                        self.assertEqual(flush.call_count, expected)
                log.close()

    def test_errors_off_and_copy_totals(self):
        with tempfile.TemporaryDirectory() as temp:
            for enabled in (False, True):
                m, logger = load_metrics(enabled, str(Path(temp) / "absent"))
                with (
                    patch.object(m, "_rank_tag", return_value=""),
                    patch.object(m.time, "monotonic_ns", side_effect=AssertionError),
                    patch.object(
                        m.os, "makedirs", side_effect=PermissionError("denied")
                    ),
                ):
                    m.PerformanceContext()
                    m.record_event("test")
                    m.record_event("test")
                self.assertEqual(logger.warning.call_count, int(enabled))
                self.assertFalse(list(Path(temp).iterdir()))
            m.reset_copy_stats()
            m.add_copy_stats(10, 20, 2)
            m.add_copy_stats(30, 40, 3)
            self.assertEqual(m.take_copy_stats(), (40, 60, 5))
            m.reset_copy_stats()
            self.assertEqual(m.take_copy_stats(), (0, 0, 0))

    @unittest.skipUnless(os.environ.get("RBLN_PY"), "RBLN_PY is not set")
    def test_rbln_imports(self):
        modules = (
            "vllm_rbln.v1.worker.metrics_v2",
            "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker",
            "vllm_rbln.v1.worker.rbln_model_runner",
        )
        subprocess.run(
            [os.environ["RBLN_PY"], "-c", "import " + ",".join(modules)],
            cwd=REPO,
            check=True,
            timeout=120,
        )
