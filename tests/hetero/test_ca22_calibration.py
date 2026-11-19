"""Verify that a calibration changes cost values inside the native profile."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import yaml

from llmd import config
from tools.hetero.profiler.ca22_costs import line, write_profiles


class CostCalibrationTests(TestCase):
    def test_nonnegative_fit(self):
        self.assertEqual(line([1, 2, 4], [5, 5, 5]), (5, 0))
        intercept, slope = line([1, 2, 4], [1, 4, 10])
        self.assertEqual(intercept, 0)
        self.assertGreater(slope, 0)

    def test_fit_reprices_both_native_profiles_from_gpu_matched_observations(self):
        profiles = Path(__file__).resolve().parents[2] / "llmd/epp/profiles"
        sources = [profiles / "tiktok.yaml", profiles / "tiktok-off.yaml"]
        full = {
            str(n): dict(prefill_ms=n / 10, ttft_ms=n / 10 + 3, tpot_ms=1)
            for n in (128, 256, 512)
        }
        split = {
            k: dict(
                row,
                blocks=int(k) // 128,
                receive_ms=int(k) / 128 * 2,
                import_ms=1 + int(k) / 128 * 3,
            )
            for k, row in full.items()
        }
        data = dict(cuda=full, rbln=full, pdd=split)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "matching-medians.json").write_text(json.dumps(data))
            (root / "settings.yaml").write_text(yaml.safe_dump({"output_tokens": 32}))
            write_profiles(root, sources, root / "frozen")
            params = []
            for path in sources:
                native = config.read_yaml(root / "frozen" / path.name)
                (handler,) = [
                    p for p in native["plugins"] if p["type"] == "cost-route-handler"
                ]
                params.append(handler["parameters"])
            self.assertEqual([p["preload"] for p in params], [True, False])
            self.assertEqual(params[0]["links"], params[1]["links"])
            self.assertEqual(params[0]["costs"], params[1]["costs"])
            link = params[0]["links"]["gpu0"]["npu0"]
            self.assertEqual(link["importFixedMs"], 1)
            self.assertEqual(link["importMsPerBlock"], 3)
