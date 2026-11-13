# SPDX-License-Identifier: Apache-2.0
import csv
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.hetero import finalize_results as final
from tools.hetero.profiler import analysis, sweep
from tools.hetero.profiler.common import live_rows

REPO = Path(__file__).resolve().parents[2]
INITIAL = REPO / "tools/hetero/scenarios/initial.json"
REAL = Path(__file__).parent / "fixtures/real"


class SweepTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "sweep"
        self.spec = json.loads(INITIAL.read_text())
        self.spec["scenarios"] = self.spec["scenarios"][:2]
        self.raw = next(REAL.glob("b0_qwen3*/workload*/requests.jsonl"))
        self.rows = list(live_rows(self.raw))
        for scenario in self.spec["scenarios"]:
            scenario["workloads"][0]["requests"] = len(self.rows)

    def test_pairs_cache_and_initial(self):
        sweep.validate_pairs(json.loads(INITIAL.read_text())["scenarios"])
        cases = [(k, 1) for k in sweep.PAIR_FIELDS] + [("concurrency", 1)]
        for key, delta in cases:
            spec = deepcopy(self.spec["scenarios"])
            for s in spec:
                s["comparison"] = "equal_concurrency"
            spec[1]["workloads"][0]["concurrency"] = 3
            spec[1]["workloads"][0][key] += delta
            with self.subTest(key=key), self.assertRaises(ValueError):
                sweep.validate_pairs(spec)
        a = self.spec["scenarios"][0]
        for change, same in [
            ({}, True),
            ({"mode": "trace"}, True),
            ({"mode": "deep"}, False),
            ({"model": "other"}, False),
            ({"service_args": {"layout_reorder": 1}}, False),
        ]:
            self.assertEqual(sweep.cache_key(a) == sweep.cache_key(a | change), same)
        shuffled = dict(reversed(list(a["service_args"].items())))
        self.assertEqual(
            sweep.cache_key(a), sweep.cache_key(a | {"service_args": shuffled})
        )

    def materialize(self, argv, **kwargs):
        opts = dict(zip(argv[2::2], argv[3::2]))
        root = Path(opts["--run-dir"])
        self.calls.append(root.name)
        if root.name == "s1_pdd" and self.inject_failure:
            return SimpleNamespace(returncode=1)
        work = json.loads(Path(opts["--workloads"]).read_text())[0]
        folder = root / work["name"]
        (folder / "profile").mkdir(parents=True, exist_ok=True)
        (root / "service").mkdir(exist_ok=True)
        config = dict(dtype="float16", kv_cache_dtype="float16", max_num_seqs=1)
        (root / "service/config.json").write_text(json.dumps(config))
        (folder / "requests.jsonl").write_bytes(self.raw.read_bytes())
        summary = dict(
            run_status="PASS",
            analysis_status="PASS",
            client=analysis.client_metrics(self.rows),
        )
        if opts["--configuration"] == "pdd":
            summary["client"]["summary"]["output_tokens_per_second"]["value"] *= 2
            summary["client"]["summary"]["makespan_seconds"]["value"] /= 2
        (folder / "profile/summary.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode=0)

    def test_resume_and_exports(self):
        self.calls: list[str] = []
        self.inject_failure = True
        models = {"qwen3-0.6b": "unused-model"}
        with patch.object(sweep.subprocess, "run", self.materialize):
            self.assertEqual(sweep.execute(deepcopy(self.spec), self.root, models), 1)
            self.inject_failure = False
            self.assertEqual(sweep.execute(deepcopy(self.spec), self.root, models), 0)
            self.assertEqual(sweep.execute(deepcopy(self.spec), self.root, models), 0)
        self.assertEqual(self.calls, ["s1_b0", "s1_pdd", "s1_pdd_1"])
        self.assertTrue((self.root / "runs/s1_pdd").exists())
        with (self.root / "results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["run_status"] for r in rows], ["PASS", "FAILED", "PASS"])
        old = json.loads(self.raw.with_name("summary.json").read_text())
        for key in (
            "output_tokens_per_second",
            "ttft_mean_seconds",
            "queue_mean_seconds",
        ):
            self.assertAlmostEqual(float(rows[0][key]), old[key], msg=key)
        self.assertEqual(rows[0]["correctness"], "not_evaluated")
        with (self.root / "comparison.csv").open() as handle:
            compared = list(csv.DictReader(handle))[0]
        for key, value in [
            ("throughput_improvement_percent_vs_b0", 100),
            ("makespan_reduction_percent_vs_b0", 50),
        ]:
            self.assertEqual(float(compared[key]), value)
        for name in final.TABLES + ["DATA_DICTIONARY"]:
            suffix = ".md" if name == "DATA_DICTIONARY" else ".csv"
            self.assertTrue((self.root / "dataset" / (name + suffix)).exists())
        self.assertFalse((self.root / "dataset/operators.csv").exists())
        self.assertEqual(
            final.output_tokens(
                self.raw.parent, dict(requests=len(self.rows), output_tokens=128)
            ),
            {r["request_index"]: r["streamed_token_ids"] for r in self.rows},
        )
        with patch.object(sweep.subprocess, "run", self.materialize):
            sweep.execute(
                deepcopy(self.spec), self.root.parent / "only", models, only=["s1_b0"]
            )
        self.assertEqual(self.calls[-1], "s1_b0")

    def test_kv_size_and_reorder(self):
        model = self.root.parent / "config.json"
        model.write_text(
            json.dumps(dict(num_hidden_layers=2, num_key_value_heads=2, head_dim=4))
        )
        config = dict(model_path=str(model.parent), kv_cache_dtype="float16")
        self.assertEqual(
            final.kv_size(config, dict(input_tokens=8))["value"], 512 / 2**20
        )
        work = dict(name="w", requests=len(self.rows), output_tokens=128)
        for complete in (True, False):
            work["requests"] = len(self.rows) + int(not complete)
            records: list[tuple] = []
            for reorder in (0, 1):
                row = dict(
                    run_status="PASS",
                    analysis_status="PASS",
                    returncode=0,
                    configuration="pdd",
                    pair_id="s7",
                    comparison="capacity",
                    workload="w",
                    layout_reorder=reorder,
                    outputs_identical="not_evaluated",
                )
                records.append(
                    (row, {}, self.raw.parent, dict(name=str(reorder)), work)
                )
            final.comparisons(records)
            self.assertEqual(
                records[0][0]["outputs_identical"],
                True if complete else "not_evaluated",
            )
