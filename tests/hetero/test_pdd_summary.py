# SPDX-License-Identifier: Apache-2.0
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/pdd/fixtures/ci_summary_43900805216.json"


class SummaryLayoutTests(unittest.TestCase):
    def test_recorded_summary_layout(self):
        with patch.object(sys, "path", [str(ROOT / "tools/pdd"), *sys.path]):
            from tools.hetero.profiler.pdd_summary import LABELS
            from tools.pdd.summary import render_markdown
        data = json.loads(FIXTURE.read_text())
        before = json.dumps(data)
        text = render_markdown(data)
        self.assertEqual(json.dumps(data), before)
        self.assertNotRegex(text, r"[가-힣]")
        self.assertNotIn("M14a", text)
        self.assertNotRegex(text, r"(?m)^\|.*not_collected")
        self.assertIn("GiB", text)
        self.assertIn("req 0000", text)
        self.assertIn(
            "| device | active | utilization (%) | power (W) | memory (GiB) |", text
        )
        for label in (
            "## Key results",
            "Service TTFT",
            "Service E2E",
            "TPOT",
            "Mean",
            "p50",
            "p90",
            "<details>",
            "npu1.energy_j",
            "concurrency 1, 20 requests arriving at once",
            "| device | phase | metric | mean | max | unit |",
            "success_count",
            "failure_count",
            "incomplete_count",
        ):
            self.assertIn(label, text)
        self.assertLess(text.index("## Key results"), text.index("<details>"))
        phases = data["profiler"]["request_phase_share"]["summary"]["value"]
        section = "### Request time breakdown"
        m14 = text.split(section)[1].split("<details>")[0]
        self.assertEqual(
            sum(line.startswith("| ") for line in m14.splitlines()), len(phases) + 2
        )
        for phase, ratio in phases.items():
            self.assertIn(f"| {LABELS.get(phase, phase)} | {ratio * 100:.4g} |", m14)
        transfer = text.split("### KV Transfer")[1].split(section)[0]
        self.assertIn("#### Decode server (NPU0)", transfer)
        self.assertNotIn("#### Prefill server (GPU)", transfer)
        for label in (
            "npu1.energy_per_output_token",
            "npu1.tok_per_s_per_w",
            "samples[",
            "WP9",
            "H4",
            "H5",
            "미구현",
            "transfer.prefill.",
        ):
            self.assertNotIn(label, text)
