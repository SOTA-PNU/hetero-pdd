from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from llmd import config
from tests.hetero.mock_server import MockServer
from tools.hetero import loadgen
from tools.hetero.profiler import experiment_ci as ci
from tools.hetero.profiler.select_experiments import select


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.file = self.root / "experiments/case.yaml"
        self.file.parent.mkdir()
        source = ci.ROOT / "experiments/fixed-pdd.yaml"
        self.raw = config.read_yaml(source)
        self.raw["model"] = str(source.parent / self.raw["model"])
        self.raw["profile"] = str(source.parent / self.raw["profile"])
        self.raw["endpoints"][1]["engine"] = {
            "lookahead": {
                "active_decode_limit": 1,
                "lookahead_depth": 1,
                "prefetch_byte_budget": 131072,
            }
        }
        self.raw["benchmark"] = dict(
            input_tokens=256, output_tokens=128, requests=8, concurrency=4, warmup=1
        )

    def load(self):
        self.file.write_text(yaml.safe_dump(self.raw))
        return config.load(self.file)

    def test_independent_lengths_limits_and_engine_compatibility(self):
        cfg = self.load()
        self.assertEqual(ci.workload_config(cfg)["output_tokens"], 128)
        self.assertEqual(cfg["verify"]["output_tokens"], 8)
        self.assertEqual(cfg["endpoints"][1]["engine"]["max_num_seqs"], 1)
        labels = cfg["endpoints"][1]["labels"]
        self.assertEqual(labels["pdd.sota.dev/lookahead-depth"], "1")
        self.assertEqual(labels["pdd.sota.dev/lookahead-bytes"], "131072")
        good = self.raw["benchmark"]
        for invalid in (
            {"requests": 2},
            {"warmup": -1},
            {"repeats": False},
            {"ci": 1},
            {"output_tokens": 1},
            {"input_tokens": 999},
            {"prefill": 512},
            {"comparisons": {"../bad": "missing"}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.raw["benchmark"] = good | invalid
                self.load()

    def test_comparison_profiles_keep_engine_settings_and_repeat_order(self):
        native = ci.ROOT / "llmd/epp/profiles/baseline.yaml"
        self.raw["benchmark"].update(comparisons={"control": str(native)}, repeats=3)
        cfg = self.load()
        control = ci.comparison_yamls(self.file, cfg, self.root)["control"]
        self.assertEqual(config.load(control)["endpoints"], cfg["endpoints"])
        self.assertNotIn("paths", config.load(control)["verify"])
        with patch.object(ci, "run_service") as service:
            ci.benchmark_runs(
                self.file, self.root, cfg, ci.workload_config(cfg), None, ci.Steps()
            )
        self.assertEqual(
            [call.args[2] for call in service.call_args_list],
            ["control", "candidate", "candidate", "control", "control", "candidate"],
        )

    def test_profile_changes_deduplicated_and_manual_runs_remain_available(self):
        custom = self.root / "alternate.yaml"
        self.raw["benchmark"]["comparisons"] = {"control": str(custom)}
        self.file.write_text(yaml.safe_dump(self.raw))
        self.raw["benchmark"]["ci"] = False
        manual = self.file.with_name("large.yml")
        manual.write_text(yaml.safe_dump(self.raw))
        self.assertEqual(select(self.root, [str(custom)]), ["experiments/case.yaml"])
        self.assertEqual(
            select(self.root, [str(custom), str(self.file)]), ["experiments/case.yaml"]
        )
        self.assertEqual(select(self.root, ["unrelated"]), [])
        self.assertEqual(select(self.root, [], manual.name), ["experiments/large.yml"])
        with self.assertRaises(ValueError):
            select(self.root, [], "../alternate.yaml")

    def test_yaml_changes_reach_full_length_warmup_and_measurement(self):
        tokenizer = SimpleNamespace(encode=lambda text, **_: list(map(ord, text)))
        with MockServer() as server:
            self.raw["ports"] = {"gateway": server.server_port}
            for index, (length, output, requests, concurrency, waves) in enumerate(
                [(8, 3, 2, 1, 1), (260, 18, 4, 2, 2)]
            ):
                self.raw["benchmark"].update(
                    input_tokens=length,
                    output_tokens=output,
                    requests=requests,
                    concurrency=concurrency,
                    warmup=waves,
                    seed=13,
                )
                cfg = self.load()
                work = self.root / str(index)
                args = ci.benchmark_args(
                    cfg, work, "candidate", ci.workload_config(cfg)
                )
                loadgen.main(args[2:], tokenizer=tokenizer)
                folder = work / "profile"
                warm = (folder / "warmup_requests.jsonl").read_text().splitlines()
                rows = (folder / "requests.jsonl").read_text().splitlines()
                self.assertEqual(
                    (len(warm), len(rows)), (concurrency * waves, requests)
                )
                for row in map(json.loads, warm + rows):
                    self.assertEqual(row["actual_output_tokens"], output)
                    self.assertEqual(row["actual_input_tokens"], length)
                    self.assertEqual(row["seed"], 13)

    def test_retry_archives_cold_burst_and_preserves_input_identity(self):
        cfg = self.load()
        work = self.root / "arm"
        prefix = []

        def execute(*argv):
            directory = Path(argv[argv.index("--run-dir") + 1])
            directory.mkdir(parents=True)
            uid = argv[argv.index("--workload") + 1]
            prefix.append(uid)
            (directory / "warmup_requests.jsonl").write_text(uid + "-warm\n")
            (directory / "requests.jsonl").write_text(uid + "-live\n")

        def evidence(_work):
            cold = ["npu_full"] if len(prefix) == 1 else []
            return {"measured_unwarmed_paths": cold, "executed_paths": {"npu_full": 3}}

        with patch.object(ci, "analyze_routes", side_effect=evidence):
            ci.run_benchmark(
                cfg, work, "candidate", ci.workload_config(cfg), execute=execute
            )
        self.assertEqual(len(set(prefix)), 2)
        self.assertTrue((work / "excluded-cold-burst-1/requests.jsonl").exists())
        self.assertEqual(
            (work / "profile/requests.jsonl").read_text().splitlines(),
            [prefix[1] + "-live"],
        )
        self.assertEqual(
            (work / "profile/warmup_requests.jsonl").read_text().splitlines(),
            [prefix[0] + "-warm", prefix[0] + "-live", prefix[1] + "-warm"],
        )

    def test_counts_separate_http_errors_from_short_streams_and_missing_rows(self):
        work = self.root / "arm"
        profile = work / "profile"
        profile.mkdir(parents=True)
        fixture = next(
            (Path(__file__).parent / "fixtures/real").glob("*/*/requests.jsonl")
        )
        row = json.loads(fixture.read_text().splitlines()[0])
        rows = [
            row,
            row | {"http_status": 503, "error": "unavailable"},
            row | {"actual_output_tokens": 1},
        ]
        (profile / "requests.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (profile / "config.json").write_text(json.dumps({"requests": 4}))
        (work / "run.json").write_text(json.dumps({"endpoints": [{"devices": [0]}]}))
        with patch.object(ci, "analyze_routes", return_value={}):
            data = ci.profile(work)
        self.assertEqual(data["counts"], {"completed": 1, "failed": 1, "incomplete": 2})

    def test_report_keeps_variation_and_flags_disagreement_against_gpu(self):
        work = self.root / "case"
        config = {
            "model": "mock",
            "tokenizer": "unused",
            "seed": 42,
            "warmup_waves": 1,
            "slots": [{}],
            "prompts_sha256": "same",
            "generation": {"temperature": 0},
            "requests": 1,
            "input_tokens": 4,
            "output_tokens": 2,
        }
        runs: list[dict] = []
        for i, (arm, output, path) in enumerate(
            [
                ("candidate", [0, 1], "pdd"),
                ("gpu", [0, 2], "gpu_full"),
                ("candidate", [0, 2], "pdd"),
                ("candidate", [0, 2], "pdd"),
            ]
        ):
            directory = work / str(i)
            folder = directory / "profile"
            folder.mkdir(parents=True)
            row = {
                "request_index": 0,
                "streamed_token_ids": output,
                "actual_input_tokens": 4,
            }
            (folder / "requests.jsonl").write_text(json.dumps(row) + "\n")
            runs.append(
                dict(
                    arm=arm,
                    repeat=max(1, i),
                    work=str(directory),
                    profile={
                        "workload": config,
                        "metrics": {k: i + 1 for k, _, _ in ci.METRICS},
                        "counts": {"completed": 1, "failed": 0, "incomplete": 0},
                        "route_evidence": {"executed_paths": {path: 1}},
                    },
                )
            )
        result = ci.summarize_runs(self.root, runs)
        self.assertEqual(
            result["output_agreement_reference"], "repeat 1 / gpu (GPU full)"
        )
        self.assertEqual(runs[0]["mismatched_request_indices"], [0])
        self.assertTrue(result["errors"])
        self.assertTrue(runs[0]["timing_valid"])
        self.assertFalse(runs[0]["comparison_valid"])
        self.assertEqual(runs[0]["profile"]["metrics"]["output_tokens_per_second"], 1)
        self.assertEqual(
            result["arms"]["candidate"]["output_tokens_per_second"],
            {
                "values": [3, 4],
                "median": 3.5,
                "min": 3,
                "max": 4,
            },
        )
        self.assertTrue((self.root / "benchmark-summary.json").is_file())
        runs[2]["profile"]["route_evidence"]["measured_unwarmed_paths"] = ["pdd"]
        retry = ci.summarize_runs(self.root, runs)
        self.assertEqual(
            retry["arms"]["candidate"]["output_tokens_per_second"]["values"], [4]
        )
        self.assertFalse(runs[2]["timing_valid"])
        self.assertTrue(any("cold paths" in err for err in retry["errors"]))

    def test_legacy_workload_defaults(self):
        self.raw.pop("benchmark")
        workload = ci.workload_config(self.load())
        self.assertEqual(workload["output_tokens"], 32)
        self.assertEqual(workload["requests"], 20)
