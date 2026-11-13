# SPDX-License-Identifier: Apache-2.0
import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.hetero.profiler.ci_summary import render
from tools.hetero.profiler.sweep import validate_pairs

ROOT = Path(__file__).resolve().parents[2]


class CISummaryTests(unittest.TestCase):
    def test_summary(self):
        fixture = next((ROOT / "tests/hetero/fixtures/real").rglob("summary.json"))
        old = json.loads(fixture.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(fixture.read_text())
            good = dict(scenario="real", source_summary="summary.json")
            good["run_status"] = "PASS"
            for key in ("output_tokens_per_second", "makespan_seconds"):
                good[key] = old[key]
            bad = dict(scenario="failed", run_status="FAILED", error="service failed")
            rows = [good, bad]
            with (root / "results.csv").open("w") as f:
                writer = csv.DictWriter(f, sorted(set().union(*rows)))
                writer.writeheader()
                writer.writerows(rows)
            stats = dict(d_tp=dict(mean=0.01, low=0, high=0.02))
            data = dict(verdict="PASS", decisions=[dict(stats=stats)])
            (root / "overhead.json").write_text(json.dumps(data))
            for overhead in (None, root):
                with self.subTest(overhead=bool(overhead)):
                    render(root, overhead)
                    text = (root / "summary.md").read_text()
                    for value in ("real", "failed", "FAILED", "service failed"):
                        self.assertIn(value, text)
                    self.assertEqual("## PER-001" in text, overhead is not None)
            self.assertIn("90% CI [0, 0.02]", text)

    def test_ci_pairs(self):
        spec = json.loads((ROOT / "tools/hetero/scenarios/ci.json").read_text())
        validate_pairs(spec["scenarios"])
        initial = json.loads((ROOT / "tools/hetero/scenarios/initial.json").read_text())
        selected = [
            s
            for s in initial["scenarios"]
            if s["name"] in ("s1_b0", "s1_pdd", "s8_pdd")
        ]
        self.assertEqual(spec["scenarios"], selected)

    def test_yaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML unavailable")
        data = yaml.safe_load((ROOT / ".github/workflows/pdd_profile.yml").read_text())
        self.assertEqual(data["concurrency"]["group"], "pdd-test")
        jobs = ["prepare", "profile", "per001", "summarize", "upload"]
        self.assertEqual(set(data["jobs"]), set(jobs))
