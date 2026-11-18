"""Engine/connector contracts that can be checked without accelerator access."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llmd import engines


class NativeEngineTests(unittest.TestCase):
    def test_baseline_and_transfer_have_distinct_launch_contracts(self):
        with TemporaryDirectory() as work:
            options = {
                "gpu_python": "/venv/cuda/bin/python",
                "npu_python": "/venv/npu/bin/python",
                "hub_cache": work,
                "max_model_len": 4096,
            }
            common = dict(
                run_dir=Path(work),
                run_id="abc",
                cuda_device=3,
                instrumentation={"rbln_events": True},
            )
            for role, kind in (
                ("full", "gpu"),
                ("full", "npu"),
                ("prefill", "gpu"),
                ("decode", "npu"),
            ):
                ep = dict(
                    id=kind + role,
                    kind=kind,
                    role=role,
                    device=3 if kind == "gpu" else 1,
                    port=8111,
                    kv_port=8222,
                )
                with self.subTest(role=role, kind=kind):
                    spec = engines.engine_spec("qwen3-0.6b", options, ep, **common)
                    argv, env = spec["argv"], spec["env"]
                    self.assertIn(
                        str(Path(work) / "cache" / ep["id"]), env["VLLM_CACHE_ROOT"]
                    )
                    self.assertEqual(
                        argv[argv.index("--num-gpu-blocks-override") + 1], "33"
                    )
                    if role == "full":
                        self.assertNotIn("--kv-transfer-config", argv)
                        self.assertNotIn("VLLM_NIXL_SIDE_CHANNEL_PORT", env)
                    else:
                        transfer = json.loads(
                            argv[argv.index("--kv-transfer-config") + 1]
                        )
                        self.assertEqual(transfer["engine_id"], "abc-" + ep["id"])
                        self.assertEqual(transfer["kv_load_failure_policy"], "fail")
                        self.assertEqual(
                            transfer["kv_buffer_device"],
                            "cuda" if kind == "gpu" else "cpu",
                        )
                        self.assertEqual(env["VLLM_KV_CACHE_LAYOUT"], "HND")
                        self.assertEqual(env["VLLM_NIXL_SIDE_CHANNEL_PORT"], "8222")
                    if kind == "npu":
                        self.assertEqual(env["RBLN_DEVICES"], "1")
                        self.assertEqual(
                            env["CUDA_VISIBLE_DEVICES"],
                            "3" if role == "decode" else "",
                        )
                        self.assertTrue(Path(env["VLLM_RBLN_METRICS_DIR"]).is_dir())
                        self.assertTrue(Path(env["RBLN_OFFLOAD_DIR"]).is_dir())

    def test_incomplete_or_wrong_snapshot_is_rejected(self):
        with TemporaryDirectory() as cache:
            model = engines.model_info("qwen3-1.7b", {"hub_cache": cache})
            path = Path(model["path"])
            with self.assertRaisesRegex(ValueError, "missing pinned snapshot"):
                engines._snapshot(model)
            path.mkdir(parents=True)
            (path / "config.json").write_text('{"architectures": ["Unknown"]}')
            with self.assertRaisesRegex(ValueError, "unquantized"):
                engines._snapshot(model)
            (path / "config.json").write_text('{"architectures": ["Qwen3ForCausalLM"]}')
            (path / "tokenizer.json").touch()
            (path / "tokenizer_config.json").touch()
            (path / "model.safetensors.index.json").write_text(
                '{"weight_map": {"a": "missing.safetensors"}}'
            )
            with self.assertRaisesRegex(ValueError, "missing.safetensors"):
                engines._snapshot(model)


if __name__ == "__main__":
    unittest.main()
