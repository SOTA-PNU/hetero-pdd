# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class WorkflowTests(unittest.TestCase):
    def test_workflow_yaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        for name in ("profiler_test", "experiment_test", "pdd_test"):
            data = yaml.load(
                (REPO / f".github/workflows/{name}.yml").read_text(),
                Loader=yaml.BaseLoader,
            )
            self.assertIn("workflow_dispatch", data["on"])
            self.assertTrue(data["jobs"])

    def test_pdd_and_overlap_require_evidence_on_each_decoder(self):
        from tools.hetero.profiler.experiment_ci import StepError, require_routes

        decoders = [
            {"id": f"npu{i}", "backend": "rbln", "roles": ["decode"]} for i in range(2)
        ]
        full = {"executed_paths": {"gpu_full": 9}, "per_endpoint": {}}
        require_routes(full, None, decoders)
        with self.assertRaisesRegex(StepError, "required PDD"):
            require_routes(full, "pdd", decoders)
        npu = dict(
            requested_paths={"pdd": 1},
            executed_paths={"pdd": 1},
            preload_selected=1,
            preload_approved=1,
            actual_overlap=1,
        )
        routes: dict = {
            "executed_paths": {"pdd": 2},
            "per_endpoint": {"npu0": npu, "npu1": npu | {"actual_overlap": 0}},
        }
        require_routes(routes, "pdd", decoders)
        with self.assertRaisesRegex(StepError, "npu1"):
            require_routes(routes, "overlap", decoders)
        routes["per_endpoint"]["npu1"] = npu
        require_routes(routes, "overlap", decoders)

    def test_only_matching_gpu_outputs_and_paired_repeats_contribute_to_gain(self):
        from tools.hetero.profiler.experiment_ci import METRICS, benchmark_results

        arms = ["gpu-only", "selective", "candidate", "fixed-pdd"]

        def run(name, repeat, value):
            return dict(
                arm=name,
                repeat=repeat,
                workload={"requests": 1, "prompts_sha256": "p"},
                output_ids_sha256="wrong" if name == "fixed-pdd" else "ids",
                route_evidence={"executed_paths": {"gpu_full": 1}},
                metrics={key: value for key, _, _ in METRICS},
            )

        runs = [
            run(name, n, 2 if name == "candidate" else 1)
            for n in (1, 2, 3)
            for name in arms
        ]
        gains = benchmark_results(runs, arms, [])
        self.assertEqual(gains["selective"]["Output throughput"], 100)
        self.assertTrue(
            all("correctness" in r["error"] for r in runs if r["arm"] == "fixed-pdd")
        )
        runs[0]["route_evidence"]["executed_paths"] = {}
        self.assertEqual(benchmark_results(runs, arms, []), {})
        self.assertIn("missing verified GPU", runs[0]["error"])
