"""Check answer references and profiler comparisons used by experiment CI."""

import json
import tempfile
import unittest
from pathlib import Path

from llmd import config
from tools.hetero.profiler.experiment_ci import (
    performance_baseline_yaml,
    reference_yaml,
    report,
)

ROOT = Path(__file__).resolve().parents[2]


def request(index, elapsed, ttft, target):
    start = index * (elapsed + 0.1)
    return {
        "request_id": f"{target}-{index}",
        "scheduled_arrival_monotonic": start,
        "assigned_monotonic": start,
        "actual_send_monotonic": start,
        "first_output_monotonic": start + ttft,
        "last_output_monotonic": start + elapsed - 0.01,
        "completed_monotonic": start + elapsed,
        "chunk_timestamps": [
            {"t_rel_arrival_seconds": ttft, "token_ids": [1], "text_chars": 1},
            {
                "t_rel_arrival_seconds": elapsed - 0.01,
                "token_ids": [2],
                "text_chars": 1,
            },
        ],
        "streamed_token_ids": [1, 2],
        "assigned_target": target,
        "requested_input_tokens": 128,
        "actual_input_tokens": 128,
        "requested_output_tokens": 2,
        "actual_output_tokens": 2,
        "http_status": 200,
        "error": None,
    }


class ExperimentCITest(unittest.TestCase):
    def test_reference_uses_existing_gpu_answer_sheet(self):
        source = ROOT / "experiments/fixed-pdd.yaml"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "answer-reference.yaml"
            selected = reference_yaml(config.load(source), target)
            original, baseline = config.load(source), config.load(target)
            self.assertEqual(selected.name, "qwen3-baseline.yaml")
            self.assertEqual(baseline["model"], original["model"])
            self.assertEqual([e["backend"] for e in baseline["endpoints"]], ["cuda"])
            self.assertEqual(baseline["endpoints"][0]["roles"], ["full"])
            self.assertEqual(baseline["verify"]["paths"], ["full"])
            qwen2 = ROOT / "experiments/qwen2-fixed.yaml"
            self.assertEqual(
                reference_yaml(config.load(qwen2), target).name,
                "qwen2-baseline.yaml",
            )

    def test_model_without_answer_sheet_has_no_reference(self):
        source = ROOT / "experiments/cuda-pair-tp2.yaml"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "answer-reference.yaml"
            self.assertIsNone(reference_yaml(config.load(source), target))
            self.assertFalse(target.exists())

    def test_performance_baseline_uses_heterogeneous_config(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "performance-baseline.yaml"
            cfg = config.load(ROOT / "experiments/fixed-pdd.yaml")
            self.assertTrue(performance_baseline_yaml(cfg, target))
            baseline = config.load(target)
            self.assertEqual([ep["backend"] for ep in baseline["endpoints"]], ["cuda", "rbln"])
            other = config.load(ROOT / "experiments/qwen2-fixed.yaml")
            self.assertFalse(performance_baseline_yaml(other, target))

    def test_report_shows_measured_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, elapsed, ttft in (
                ("performance-baseline", 0.4, 0.1),
                ("candidate", 0.2, 0.05),
            ):
                folder = root / name / "profile"
                folder.mkdir(parents=True)
                rows = [request(i, elapsed, ttft, name) for i in range(2)]
                (folder / "requests.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                (folder / "config.json").write_text(
                    json.dumps(
                        {
                            "model": "Qwen/Qwen3-0.6B",
                            "tokenizer": "/model",
                            "input_tokens": 128,
                            "output_tokens": 2,
                            "requests": 2,
                            "seed": 20260924,
                            "slots": [{"target": name}],
                        }
                    )
                )
            (root / "candidate/verification.json").write_text(
                json.dumps({"correctness": "passed"})
            )
            report(
                root,
                ROOT / "experiments/fixed-pdd.yaml",
                ROOT / "experiments/qwen3-baseline.yaml",
                True,
            )
            data = json.loads((root / "profile-summary.json").read_text())
            metrics = {item["metric"]: item for item in data["comparison"]}
            self.assertAlmostEqual(metrics["Mean TTFT"]["improvement_percent"], 50)
            self.assertAlmostEqual(metrics["Mean E2E"]["improvement_percent"], 50)
            self.assertGreater(metrics["Output throughput"]["improvement_percent"], 0)
            self.assertIn("baseline-hetero.yaml", (root / "summary.md").read_text())

    def test_other_model_reports_metrics_without_false_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "candidate/profile"
            folder.mkdir(parents=True)
            rows = [request(i, 0.3, 0.08, "candidate") for i in range(2)]
            (folder / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (folder / "config.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen2.5-0.5B",
                        "tokenizer": "/model",
                        "input_tokens": 128,
                        "output_tokens": 2,
                        "requests": 2,
                        "seed": 20260924,
                        "slots": [{"target": "candidate"}],
                    }
                )
            )
            (root / "candidate/verification.json").write_text(
                json.dumps({"correctness": "passed"})
            )
            report(
                root,
                ROOT / "experiments/qwen2-fixed.yaml",
                ROOT / "experiments/qwen2-baseline.yaml",
                False,
            )
            data = json.loads((root / "profile-summary.json").read_text())
            self.assertIsNone(data["performance_baseline"])
            self.assertIsNone(data["comparison"][0]["improvement_percent"])
            self.assertIn(
                "No performance baseline exists for this model",
                (root / "summary.md").read_text(),
            )

    def test_report_without_reference_marks_correctness_not_evaluated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "candidate/profile"
            folder.mkdir(parents=True)
            rows = [request(i, 0.3, 0.08, "candidate") for i in range(2)]
            (folder / "requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (folder / "config.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-1.7B",
                        "tokenizer": "/model",
                        "input_tokens": 128,
                        "output_tokens": 2,
                        "requests": 2,
                        "seed": 20260924,
                        "slots": [{"target": "candidate"}],
                    }
                )
            )
            verification = root / "candidate/verification.json"
            verification.write_text(json.dumps({"correctness": "not_evaluated"}))
            experiment = ROOT / "experiments/cuda-pair-tp2.yaml"
            report(root, experiment, None, False)
            data = json.loads((root / "profile-summary.json").read_text())
            self.assertIsNone(data["answer_reference"])
            self.assertEqual(data["correctness"], "not_evaluated")
            summary = (root / "summary.md").read_text()
            self.assertIn("**not evaluated**", summary)
            self.assertIn("| Mean TTFT |", summary)
            verification.write_text(json.dumps({"correctness": "failed"}))
            with self.assertRaises(RuntimeError):
                report(root, experiment, None, False)


if __name__ == "__main__":
    unittest.main()
