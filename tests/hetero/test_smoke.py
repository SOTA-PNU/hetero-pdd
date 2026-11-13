# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
import unittest
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from tests.hetero.mock_proxy import MockProxy
from tests.hetero.mock_server import MockServer, post_completion


def assert_log_line(case, line):
    """Require one UTC ISO8601 timestamp, run ID, component and message."""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)) "
        r"\[([^\[\]\r\n]+)\] \[([^\[\]\r\n]+)\] ([^\r\n]+)",
        line,
    )
    if match is None:
        raise AssertionError(f"invalid log format: {line!r}")
    try:
        datetime.fromisoformat(match[1].replace("Z", "+00:00"))
    except ValueError as exc:
        case.fail(f"invalid log timestamp: {exc}")


def completion(url):
    body = dict(
        model="mock",
        prompt=[11, 12],
        max_tokens=3,
        ignore_eos=True,
        stream=True,
        return_token_ids=True,
    )
    return post_completion(url, body, "smoke-seed7-req0000")


class SmokeTests(unittest.TestCase):
    def test_streamed_token_count(self):
        with MockServer() as server, completion(server.url) as response:
            tokens = []
            for line in response:
                if line.strip() == b"data: [DONE]":
                    break
                if line.startswith(b"data: "):
                    for choice in json.loads(line[6:])["choices"]:
                        tokens.extend(choice["token_ids"])
            self.assertEqual(len(tokens), 3)

    def test_proxy_selects_decoder_and_returns_target(self):
        with ExitStack() as stack:
            prefill = stack.enter_context(MockServer(role="prefill"))
            decoders = [
                stack.enter_context(MockServer(role="decoder")) for _ in range(2)
            ]
            decoder_urls = [s.url for s in decoders]
            proxy = stack.enter_context(MockProxy(prefill.url, decoder_urls))
            with completion(proxy.url) as response:
                response.read()
                self.assertIn(
                    response.headers["x-decode-target"],
                    [urlsplit(url).netloc for url in decoder_urls],
                )

    def test_real_fixtures_load_without_absolute_paths(self):
        def check(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    check(key)
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, str):
                self.assertFalse(PurePosixPath(value).is_absolute(), value)
                self.assertFalse(PureWindowsPath(value).is_absolute(), value)
                self.assertIsNone(
                    re.search(r"/(?:home|root|Users|tmp|opt|mnt)/", value)
                )

        root = Path(__file__).resolve().parent / "fixtures" / "real"
        files = sorted(root.rglob("*.json*"))
        self.assertEqual(len(files), 16)
        for path in files:
            with self.subTest(path=path.relative_to(root)):
                text = path.read_text()
                data = (
                    [json.loads(line) for line in text.splitlines()]
                    if path.suffix == ".jsonl"
                    else json.loads(text)
                )
                self.assertTrue(data)
                check(data)
