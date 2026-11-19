# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import yaml

from tools.hetero.profiler.kv_import_values import order, settings


class ValueSettingsTests(TestCase):
    def test_shipped_fixtures_and_order(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "experiments/diagnostics/ca22-kv-values.yaml"
        )
        cfg = settings(path)
        self.assertEqual(cfg["layers"] * cfg["layer_shape"][0], 56)
        self.assertEqual(cfg["block_counts"], [1, 4])
        for rep in range(12):
            self.assertCountEqual(
                order(["normal", "canonical", "integer"], rep),
                ["normal", "canonical", "integer"],
            )

    def test_reject_missing_guards_and_incomplete_encoding_probe(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "experiments/diagnostics/ca22-kv-values.yaml"
        )
        cfg = settings(path)
        with TemporaryDirectory() as directory:
            choice = Path(directory) / "case.yaml"
            for change, error in (
                (dict(first_block=28), "untouched guard"),
                (dict(block_counts=[4, 1]), "sorted unique"),
                (dict(layer_shape=[2, 32, 1, 1, 1, 1]), "every FP16"),
                (dict(canonical_implementation="skip"), "diagnostic or worker"),
            ):
                with self.subTest(change=change):
                    choice.write_text(yaml.safe_dump({**cfg, **change}))
                    with self.assertRaisesRegex(ValueError, error):
                        settings(choice)
