# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from tests.hetero.mock_proxy import MockProxy
from tests.hetero.mock_server import MockServer
from tools.hetero import loadgen

TOKENIZER = SimpleNamespace(encode=lambda text, **_: list(map(ord, text)))
COMMON = [
    "--model=mock",
    "--tokenizer=unused",
    "--input-tokens=8",
    "--output-tokens=3",
    "--requests=3",
    "--workload=smoke",
]


def invoke(directory, routing, warmup=False):
    options = ["--warmup-per-endpoint", str(int(warmup)), "--out-dir", str(directory)]
    if warmup:
        options.append("--warmup-only")
    loadgen.main([*COMMON, *routing, *options], tokenizer=TOKENIZER)


class LoadgenTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.server = stack.enter_context(MockServer(role="prefill"))
        self.proxy = stack.enter_context(MockProxy(self.server.url, [self.server.url]))

    def test_raw_records_routes_and_anchors(self):
        fixtures = Path(__file__).parent / "fixtures/real"
        real = next(fixtures.glob("*/*/requests.jsonl"))
        legacy = json.loads(real.read_text().splitlines()[0])
        raw_keys = legacy.keys() - {k for k in legacy if k.endswith("_seconds")}
        raw_keys -= {"success", "token_itl_available"}
        roles = {"gpu", "npu0", "npu1"}
        workers = [
            v for r in sorted(roles) for v in ("--worker", f"{r}={self.server.url}=1")
        ]
        cases = [(workers, roles), (["--endpoint", self.proxy.url], {"pdd"})]
        for routing, targets in cases:
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                invoke(directory, routing)
                lines = (directory / "requests.jsonl").read_text().splitlines()
                rows = list(map(json.loads, lines))
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertEqual(set(row), raw_keys)
                self.assertEqual({r["assigned_target"] for r in rows}, targets)
                if targets == {"pdd"}:
                    headers = {r["response_headers"]["x-decode-target"] for r in rows}
                    self.assertEqual(headers, {urlsplit(self.server.url).netloc})
                run = json.loads((directory / "loadgen_run.json").read_text())
                for key in ("clock_anchor_start", "clock_anchor_end"):
                    self.assertEqual(set(run[key]), {"time_ns", "monotonic_ns"})
                    self.assertTrue(all(type(v) is int for v in run[key].values()))
                self.assertFalse((directory / "summary.json").exists())

    def test_warmup_only(self):
        with tempfile.TemporaryDirectory() as temp:
            invoke(temp, ["--endpoint", self.server.url], warmup=True)
            names = {p.name for p in Path(temp).iterdir()}
            self.assertEqual(names, {"warmup_requests.jsonl"})
            row = json.loads((Path(temp) / "warmup_requests.jsonl").read_text())
            self.assertEqual(row["http_status"], 200)
