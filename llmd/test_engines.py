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
"""Backend, compile-cache and correctness contracts independent of hardware."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmd import config, engines
from llmd.verify import reference_output


class EngineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.cfg = config.load(engines.ROOT / "experiments/pool-2p2d.yaml")
        self.model = engines.model_info(self.cfg["model"], {"hub_cache": self.work})

    def test_connector_identity_and_peer_devices(self):
        with patch.dict(
            "os.environ", CUDA_PY="/cuda/bin/python", RBLN_PY="/rbln/bin/python"
        ):
            for ep in self.cfg["endpoints"]:
                if ep["kind"] == "npu":
                    ep["engine"]["coalesce_kv_import"] = True
                    ep["engine"]["kv_import_threads"] = 8
                    ep["engine"]["canonicalize_kv_import"] = True
                spec = engines.engine_spec(
                    self.model,
                    ep,
                    work=self.work,
                    engine_id="run-" + ep["id"],
                    cache=self.work / "cache",
                    peer_devices=[1, 3],
                    instrumentation={},
                )
                argv, env = spec["argv"], spec["env"]
                transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                self.assertEqual(transfer["engine_id"], "run-" + ep["id"])
                self.assertEqual(transfer["kv_load_failure_policy"], "fail")
                if ep["kind"] == "npu":
                    self.assertEqual(env["VLLM_RBLN_COALESCE_KV_IMPORT"], "1")
                    self.assertEqual(env["VLLM_RBLN_KV_IMPORT_THREADS"], "8")
                    self.assertEqual(env["VLLM_RBLN_CANONICALIZE_KV_IMPORT"], "1")
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,3")
                    self.assertEqual(transfer["kv_buffer_device"], "cpu")
                ep["roles"] = ["full"]
                standalone = engines.engine_spec(
                    self.model,
                    ep,
                    work=self.work,
                    engine_id="full",
                    cache=self.work / "cache",
                    peer_devices=[],
                    instrumentation={},
                )
                self.assertNotIn("--kv-transfer-config", standalone["argv"])

    def test_snapshot_shard_presence(self):
        with self.assertRaisesRegex(ValueError, "missing pinned snapshot"):
            engines._snapshot(self.model)
        path = Path(self.model["path"])
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"architectures":["NewArchitecture"]}')
        for f in ("tokenizer.json", "tokenizer_config.json"):
            (path / f).touch()
        (path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"missing.safetensors"}}'
        )
        with self.assertRaisesRegex(ValueError, "missing.safetensors"):
            engines._snapshot(self.model)

    def test_only_ready_compatible_generation_is_reused(self):
        ep = next(ep for ep in self.cfg["endpoints"] if ep["kind"] == "npu")

        def cache():
            return engines.compilation_cache(
                self.work, self.model, ep, {"driver": "one"}
            )

        with cache() as (bad, _, _):
            pass
        self.assertFalse(bad.exists())
        with cache() as (built, meta, publish):
            self.assertFalse(meta["reused"])
            publish()
        ep["engine"]["coalesce_kv_import"] = True
        ep["engine"]["kv_import_threads"] = 8
        ep["engine"]["canonicalize_kv_import"] = True
        with cache() as (again, meta, _):
            self.assertTrue(meta["reused"])
            self.assertEqual(again, built)
        ep["engine"]["tp"] = 2
        with cache() as (other, meta, _):
            self.assertFalse(meta["reused"])
            self.assertNotEqual(other, built)

    def test_tokenizer_or_generation_change_rejects_reference(self):
        path = self.work / "reference.json"
        source = dict(
            schema_version=2,
            status="smoke_passed",
            correctness="not_evaluated",
            smoke="passed",
            gpu_only=True,
            run_id="independent",
            model=self.model,
            conditions={"seed": 0},
            requests=[dict(path="full", prompt_ids=[1], token_ids=[2])],
        )
        path.write_text(json.dumps(source))
        self.assertEqual(
            reference_output(path, self.model, "new", [[1]], {"seed": 0}),
            source["requests"],
        )
        with self.assertRaises(ValueError):
            reference_output(
                path,
                {**self.model, "tokenizer_revision": "f" * 40},
                "new",
                [[1]],
                {"seed": 0},
            )
        with self.assertRaises(ValueError):
            reference_output(path, self.model, "new", [[1]], {"seed": 1})
        self.assertIsNone(reference_output(None, self.model, "new", [[1]], {}))
        path.write_text(json.dumps(dict(source, status="failed", correctness="failed")))
        with self.assertRaises(ValueError):
            reference_output(path, self.model, "new", [[1]], {"seed": 0})
