# SPDX-License-Identifier: Apache-2.0
"""The WP1 toy proxy contract, using only the standard library."""

from __future__ import annotations

import itertools
import json
import uuid
from urllib.error import HTTPError
from urllib.parse import urlsplit

from tests.hetero.mock_protocol import PREFILL_KV
from tests.hetero.mock_server import Handler, MockHTTPServer, post_completion


class MockProxy(MockHTTPServer):
    def __init__(self, prefill_url, decoder_urls, *, port=0, pattern=None):
        indices = list(range(len(decoder_urls))) if pattern is None else pattern
        if not indices or any(i < 0 or i >= len(decoder_urls) for i in indices):
            raise ValueError("pattern must contain valid decoder indices")
        self.prefill_url, self.decoder_urls = prefill_url, list(decoder_urls)
        self.selection = itertools.cycle(indices)
        super().__init__(ProxyHandler, port)


class ProxyHandler(Handler):
    server: MockProxy

    def do_POST(self):
        if self.path != "/v1/completions":
            self.respond(status=404)
            return
        body = self.read_request()
        rid = self.headers.get("x-request-id") or str(uuid.uuid4())
        prefill = dict(body, max_tokens=1, stream=False, kv_transfer_params=PREFILL_KV)
        prefill.pop("stream_options", None)

        try:
            with post_completion(self.server.prefill_url, prefill, rid) as response:
                result = json.load(response)
            if result.get("kv_transfer_params"):
                body = {**body, "kv_transfer_params": result["kv_transfer_params"]}
            with self.server.lock:
                decoder = self.server.decoder_urls[next(self.server.selection)]
            with post_completion(decoder, body, rid) as response:
                # StreamingResponse in the repository toy proxy uses application/json.
                target = urlsplit(decoder).netloc
                self.respond(None, response.status, **{"x-decode-target": target})
                for line in response:
                    self.wfile.write(line)
                    self.wfile.flush()
        except HTTPError as exc:
            with exc:
                self.respond(exc.read(), status=exc.code)
        except (BrokenPipeError, ConnectionResetError):
            pass
