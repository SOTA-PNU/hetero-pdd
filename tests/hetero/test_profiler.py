# SPDX-License-Identifier: Apache-2.0
import gzip
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from tests.hetero.fakes import FakePopen
from tests.hetero.mock_server import MockServer
from tests.hetero.test_smoke import assert_log_line
from tools.hetero.profiler import expected, monitors, run, scrape

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
CASE = json.loads((FIXTURES / "profiler.json").read_text())


class ProfilerTests(unittest.TestCase):
    def test_options_routing_and_validation(self):
        for config, mode, env, args in CASE["options"]:
            actual = run.role_options(config, mode, Path("run"), 12345)
            self.assertEqual(actual, (env, args))
        for work, config, routing in CASE["routing"]:
            run.validate_workloads([work], config["configuration"])
            if config["configuration"] == "baseline":
                targets = ("gpu", "npu0", "npu1", "npu0", "npu1", "npu1")
                routing = [
                    v
                    for r in targets
                    for v in ("--worker", f"{r}={config['endpoints'][r]}=1")
                ]
            self.assertEqual(run.loadgen_args(work, config)[-len(routing) :], routing)
        for config, bad in CASE["invalid"]:
            with self.assertRaises(ValueError):
                run.validate_workloads(bad, config)
        self.assertRaises(ValueError, run.main, CASE["repo_args"])

    def test_expected_ids_and_span_wait(self):
        for name in ("b0_three_devices", "pdd_skew", "failed"):
            root = FIXTURES / name
            rows = list(expected.live_rows(root / "requests.jsonl"))
            rows += CASE["failed_rows"]
            config = json.loads((root / "service_config.json").read_text())
            wanted = json.loads((root / "expected.json").read_text())
            wanted = {r: set(ids) for r, ids in wanted["expected_ids_by_role"].items()}
            self.assertEqual(expected.expected_ids(rows, config), wanted)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scrape.append_rows(root / "otel/traces.jsonl", [CASE["trace"]])
            for rid, missing in CASE["span_cases"]:
                ids = {"decode0": {rid}, "decode1": set()}
                result = expected.wait_for_spans(root, ids, timeout=0.01)
                self.assertEqual(result["missing"], missing)
                saved = json.loads((root / "trace_wait.json").read_text())
                self.assertEqual(saved, result)

    def test_scrape(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            endpoints = {}
            for enabled in (True, False):
                server = stack.enter_context(MockServer(metrics=enabled))
                endpoints[str(enabled)] = server.url + "/metrics"
            path = Path(temp) / "boundaries.jsonl"
            scrape.boundary_snapshot(endpoints, path, "before")
            rows = list(expected.live_rows(path))
            self.assertEqual([r["status"] for r in rows], [200, 404])
            keys = {"mono_ns", "real_ns", "endpoint", "status", "text", "label"}
            self.assertEqual(set(rows[0]), keys)

    def test_execution_and_failure_cleanup(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root, fake = Path(temp), FakePopen()
                output = stack.enter_context(redirect_stdout(StringIO()))

                def popen(*args, fake=fake, **kwargs):
                    process = fake(*args, **kwargs)
                    process.terminate, process.wait = Mock(), Mock(return_value=0)
                    return process

                def command(directory, name, argv, fail=fail):
                    if name == "service_start":
                        self.assertFalse((directory / "service").exists())
                        (directory / "service").mkdir()
                        text = json.dumps(CASE["config"])
                        (directory / "service/config.json").write_text(text)
                    elif name == "w-measure":
                        if fail:
                            raise RuntimeError("measurement failed")
                        rows = expected.live_rows(FIXTURES / "pdd_skew/requests.jsonl")
                        scrape.append_rows(directory / "w/requests.jsonl", rows)
                    return 0

                stubs: dict = dict.fromkeys(
                    (
                        "check_tools",
                        "capture_links",
                        "wait_for_layout",
                        "profile_control",
                        "wait_for_traces",
                        "boundary_snapshot",
                    ),
                    lambda *_: None,
                )
                stubs.update(
                    command=command, wait_for_spans=lambda *_: {"complete": True}
                )
                stack.enter_context(patch.multiple(run, **stubs))
                stack.enter_context(patch.object(monitors.subprocess, "Popen", popen))
                stack.enter_context(
                    patch.multiple(
                        monitors, wait_until=lambda *_: None, free_port=lambda *_: 12345
                    )
                )
                args = Mock(run_dir=root, mode="deep", configuration="pdd")
                service = Mock(role_env_json={}, role_args_json={})
                forwarded = ["--run-dir", str(root / "service")]
                result = run.execute(args, forwarded, service, [CASE["workload"]], [])
                self.assertEqual(result, int(fail))
                data = json.loads((root / "run.json").read_text())
                steps = [s for s in CASE["steps"] if not (fail and s == "span_wait")]
                steps.insert(steps.index("w:scrape_stop") + 1, "w:trace_wait")
                steps += ["w:analysis"]
                self.assertEqual([s["name"] for s in data["steps"]], steps)
                for process in fake.processes:
                    process.terminate.assert_called_once()
                    process.wait.assert_called_once()
                for line in output.getvalue().splitlines():
                    assert_log_line(self, line)

    def test_trace_completion(self):
        trace = gzip.compress(b"{}")
        cases = [(trace, True), (trace[:-8], False), (b"", False), (None, False)]
        for content, ok in cases:
            with self.subTest(ok=ok), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "gpu").mkdir()
                if content is not None:
                    (root / "gpu/trace.json.gz").write_bytes(content)
                data: dict = {"steps": []}
                with (
                    patch.object(run.time, "monotonic", side_effect=range(200)),
                    patch.object(run.time, "sleep"),
                    run.recorded(root, data, "trace_wait", optional=True),
                ):
                    run.wait_for_traces(root, ["gpu"], set(), timeout=8)
                row = json.loads((root / "run.json").read_text())["steps"][0]
                self.assertEqual(row["ok"], ok)
                if not ok:
                    self.assertIn("gpu", row["error"])
