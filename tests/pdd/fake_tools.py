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
"""Temporary fixture executables for PDD tests; never installed on a runner."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROGRAM = Path(sys.argv[0]).name
DIRECTORY = Path(sys.argv[0]).absolute().parent
CONTROL_FILE = DIRECTORY / "control.json"


def control():
    return json.loads(CONTROL_FILE.read_text()) if CONTROL_FILE.exists() else {}


def main():
    args = sys.argv[1:]
    if PROGRAM == "git":
        if args[:1] == ["-C"]:
            args = args[2:]
        if args == ["rev-parse", "--is-inside-work-tree"]:
            print("true")
        elif args == ["rev-parse", "HEAD"]:
            print(control().get("commit", "199605d9d22fef5e0e31c416fea5e479a7b1ce12"))
        elif args == ["branch", "--show-current"]:
            print(control().get("branch", "0.11.1-2_native_dtype"))
        elif args[:1] == ["status"]:
            if control().get("dirty"):
                print(" M example.py")
        elif args[:2] == ["merge-base", "--is-ancestor"]:
            return 1 if control().get("wrong_ancestor") else 0
        else:
            raise SystemExit(f"unhandled fixture Git invocation: {args!r}")
        return 0
    if PROGRAM == "nvidia-smi":
        if any(arg.startswith("--query-compute-apps=") for arg in args):
            if control().get("gpu_busy"):
                print("GPU-PDD-FIXTURE")
        else:
            print("0, GPU-PDD-FIXTURE, Fixture CUDA")
        return 0
    if PROGRAM == "rbln-smi":
        contexts = [{"npu": 0}] if control().get("npu_busy") else []
        print(
            json.dumps(
                {
                    "devices": [{"npu": 0, "status": "normal"}],
                    "contexts": contexts,
                }
            )
        )
        return 0
    if args[:1] == ["-c"] and "importlib.metadata" in args[1]:
        if "rebel-compiler" in args[1]:
            print("0.11.1.post2.dev2+g2995098f.prod")
        else:
            print("0.22.0+cpu" if PROGRAM == "rbln-python" else "0.22.0")
        return 0
    if args[:3] == ["-m", "vllm.entrypoints.cli.main", "serve"]:
        role = "decode" if PROGRAM == "rbln-python" else "prefill"
    elif args and args[0].endswith("pdd_profile_proxy.py"):
        role = "proxy"
    else:
        os.execv(sys.executable, [sys.executable, *args])
    result_dir = Path(os.environ["RESULT_DIR"])
    (result_dir / f"fixture_{role}_invocation.json").write_text(
        json.dumps(
            {
                "arguments": args,
                "environment": {
                    key: value
                    for key, value in os.environ.items()
                    if key.startswith(("RBLN_", "VLLM_", "CUDA_", "PYTHON", "HF_"))
                },
            }
        )
    )
    if control().get("exit_role") == role:
        print("fixture server exited deliberately", flush=True)
        return 19
    port = int(args[args.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            ready = control().get("unready_role") != role
            self.send_response(200 if ready else 503)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_id = self.headers.get("X-Request-Id", "")
            with (result_dir / "fixture_requests.jsonl").open("a") as handle:
                handle.write(
                    json.dumps({"body": body, "request_id": request_id}) + "\n"
                )
            failed = control().get("fail_requests", False)
            if request_id.startswith("pdd-correctness-p4-"):
                tokens = [
                    279, 1790, 2326, 4244, 525, 1112, 30, 576,
                    4226, 374, 330, 1782, 1790, 2326, 4244, 525,
                    1112, 3263, 576, 4226, 374, 330, 1782, 1790,
                    2326, 4244, 525, 1112, 3263, 576, 4226, 374,
                ]
                text = "fixture long-prompt correctness completion"
            elif request_id.startswith("pdd-correctness"):
                tokens = [
                    220, 23, 11, 220, 16, 18, 11, 220,
                    17, 16, 11, 220, 18, 19, 11, 220,
                    20, 20, 11, 220, 23, 24, 11, 220,
                    16, 19, 19, 11, 220, 17, 18, 18,
                ]
                text = " 8, 13, 21, 34, 55, 89, 144, 233"
            else:
                tokens = [17] * int(body["max_tokens"])
                text = "fixture completion"
            with (result_dir / "decode.log").open("a") as handle:
                handle.write("KV Transfer metrics: Num successful transfers=1\n")
            payload = {
                "choices": [
                    {"token_ids": tokens, "text": text, "finish_reason": "length"}
                ]
            }
            self.send_response(500 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
