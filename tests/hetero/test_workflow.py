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
        for name in ("hetero_unit_tests",):
            data = yaml.safe_load((REPO / f".github/workflows/{name}.yml").read_text())
            self.assertIn("workflow_dispatch", data["on"])
            self.assertTrue(data["jobs"])
