# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.hetero.mock_protocol import DECODE_KV
from tests.hetero.mock_server import MockServer
from tools.hetero.profiler import analysis, expected, pdd_ci, resources
from tools.pdd import runner

ROOT = Path(__file__).resolve().parents[2]


class PDDIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        settings = """RUN_ID=test COMMIT=abc MODEL_NAME=test MODEL_DTYPE=float16
        KV_CACHE_DTYPE=float16 CUDA_DEVICE=0 RBLN_DEVICE=1 PREFILL_PORT=1
        DECODE_PORT=2 PROXY_PORT=3 SEED=0 MAX_TOKENS=4 MAX_MODEL_LEN=1024
        PROMPT_MODE=long WARMUP_REQUESTS=1 MEASURE_REQUESTS=2 CONCURRENCY=1"""
        self.runtime = dict(item.split("=", 1) for item in settings.split())
        self.runtime.update(RESULT_DIR=str(self.root), MODEL_PATH=str(self.root))

    def test_options_and_service(self):
        path = self.root / "profile/server_options.json"
        original = runner.server_options(self.root, {}, "DECODE")
        for exists in (False, True):
            if exists:
                path.parent.mkdir()
                options = dict(
                    env={"OTEL_SERVICE_NAME": "decode0"},
                    args=["--otlp-traces-endpoint", "test"],
                )
                path.write_text(json.dumps(dict(decode0=options)))
            command, env = runner.server_options(self.root, {}, "DECODE")
            self.assertEqual(
                command,
                original[0] + (["--otlp-traces-endpoint", "test"] if exists else []),
            )
            self.assertEqual(
                env.get("OTEL_SERVICE_NAME"),
                "decode0" if exists else original[1].get("OTEL_SERVICE_NAME"),
            )
        config = pdd_ci.service(self.runtime)
        (self.root / "service.json").write_text(json.dumps(config))
        self.assertEqual(expected.service_config(self.root), config)
        self.assertEqual(config["active_devices"], ["gpu", "npu1"])
        row = json.loads(
            (ROOT / "tests/hetero/fixtures/synthetic/pdd_skew/requests.jsonl")
            .read_text()
            .splitlines()[0]
        )
        row["response_headers"]["x-decode-target"] = "127.0.0.1:2"
        ids = expected.expected_ids([row], config)
        self.assertEqual(
            ids, dict(prefill={row["request_id"]}, decode0={row["request_id"]})
        )
        rates = analysis.client_metrics([row], 2)["summary"]
        self.assertEqual(
            rates["tok_per_s_per_device"]["value"],
            rates["output_tokens_per_second"]["value"] / 2,
        )
        power = {
            d: [(0, p), (1, p), (2, p)] for d, p in zip(resources.DEVICES, [1, 99, 2])
        }
        energy = resources.energy_efficiency(power, (0, 2), 2, config["active_devices"])
        self.assertEqual(energy["total"]["energy_j"]["value"], 6)
        self.assertEqual(energy["npu0"]["energy_j"]["value"], 198)

    def test_prepare_and_workload_order(self):
        manager = Mock()
        manager.collector.return_value = 4318
        with (
            patch.object(pdd_ci, "check_tools"),
            patch.object(pdd_ci, "make_tokenizer") as tok,
        ):
            tok.return_value.encode.return_value = [1] * 327
            with patch.dict(
                pdd_ci.os.environ, {"PDD_PROFILE_MODE": "trace", "INPUT_TOKENS": ""}
            ):
                pdd_ci.prepare(self.root, self.runtime, manager)
        self.assertEqual(
            json.loads((self.root / "workload.json").read_text())["input_tokens"], 327
        )
        (self.root / "service.json").write_text(
            json.dumps(pdd_ci.service(self.runtime))
        )
        calls = []

        def invoke(argv, **_):
            calls.append("warmup" if "--warmup-only" in argv else "measure")

        with (
            patch.object(pdd_ci.subprocess, "run", side_effect=invoke),
            patch.object(pdd_ci.time, "sleep"),
            patch.object(
                pdd_ci, "boundary_snapshot", side_effect=lambda *a: calls.append(a[-1])
            ),
        ):
            pdd_ci.workload(self.root, "all", "trace")
        self.assertEqual(calls, ["warmup", "before", "measure", "after"])
        manager.root = self.root
        manager.stop.side_effect = [RuntimeError("broken"), None, None]
        with self.assertRaisesRegex(RuntimeError, "broken"):
            pdd_ci.stop(manager, ("scrape", "all-smi", "otelcol"))
        self.assertEqual(manager.stop.call_count, 3)

    def test_summary(self):
        with patch.object(sys, "path", [str(ROOT / "tools/pdd"), *sys.path]):
            from tools.pdd.summary import generate
        folder = self.root / "profile/measure/profile"
        folder.mkdir(parents=True)
        node = dict(value=1, status="measured", unit="s", reason=None)
        data = dict(
            run_status="PASS",
            analysis_status="PASS",
            active_devices=["gpu", "npu1"],
            client=dict(summary=dict(ttft=node)),
            transfer=dict(decode0=dict(h2d=node)),
            other=dict(value=None, status="not_collected", unit="s", reason="missing"),
        )
        (folder / "summary.json").write_text(json.dumps(data))
        stages = dict(
            run_id="test",
            steps=dict(run_workload=dict(status="FAILED", reason="test failure")),
        )
        generate(self.root, self.runtime, stages)
        text = (self.root / "summary.md").read_text()
        labels = (
            "Script execution steps|Saved configuration|Diagnostics|Original artifacts"
        )
        labels += "|Observed requests|KV Load|M14a|Device resources|Energy"
        labels += "|missing|test failure"
        for label in labels.split("|"):
            self.assertIn(label, text)

    def test_proxy(self):
        try:
            import httpx

            from pdd_profile_proxy import create_app
        except ImportError:
            self.skipTest("proxy dependencies unavailable")
        with (
            patch.dict(DECODE_KV, remote_request_id="test", tp_size=1),
            MockServer(role="prefill") as prefill,
            MockServer() as decode,
        ):
            args = argparse.Namespace(
                prefill_host="127.0.0.1",
                prefill_port=prefill.server_port,
                decode_host="127.0.0.1",
                decode_port=decode.server_port,
            )
            app = create_app(args)

            async def request():
                async with (
                    app.router.lifespan_context(app),
                    httpx.AsyncClient(
                        transport=httpx.ASGITransport(app), base_url="http://proxy"
                    ) as client,
                ):
                    for stream in (False, True):
                        response = await client.post(
                            "/v1/completions",
                            json=dict(prompt=[1], max_tokens=2, stream=stream),
                            headers={"X-Request-Id": "keep"},
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(
                            response.headers["x-decode-target"],
                            f"127.0.0.1:{decode.server_port}",
                        )

            asyncio.run(request())
            for server in (prefill, decode):
                self.assertEqual(
                    [r["request_id"] for r in server.requests], ["keep", "keep"]
                )
