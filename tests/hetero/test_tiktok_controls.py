"""Keep shipped preload controls matched at both routing and engine layers."""

from pathlib import Path
from unittest import TestCase

from llmd.config import load, read_yaml


class TikTokControlsTests(TestCase):
    def test_controls_share_engine_configuration_and_cost_table(self):
        root = Path(__file__).resolve().parents[2]
        profiles = root / "llmd/epp/profiles"
        base = read_yaml(profiles / "tiktok.yaml")
        for fixed in (False, True):
            for off in (False, True):
                name = "tiktok" + ("-fixed" if fixed else "") + ("-off" if off else "")
                with self.subTest(name=name):
                    native = read_yaml(profiles / f"{name}.yaml")
                    native_contract = next(
                        p for p in native["plugins"] if p["type"] == "route-contract"
                    )
                    native_contract["name"] = "tiktok"
                    costs = next(
                        p["parameters"]
                        for p in native["plugins"]
                        if p["type"] == "cost-route-handler"
                    )
                    self.assertIs(costs["preload"], not off)
                    costs["preload"] = True
                    expected = {
                        **base,
                        "schedulingProfiles": [
                            p
                            for p in base["schedulingProfiles"]
                            if not fixed or p["name"] != "full"
                        ],
                    }
                    self.assertEqual(native, expected)
        ref = load(root / "experiments/tiktok.yaml")
        for name in ("tiktok-off", "tiktok-parallel", "tiktok-long", "tiktok-overlap"):
            with self.subTest(experiment=name):
                cfg = load(root / "experiments" / f"{name}.yaml")
                for key in ("model", "endpoints", "instrumentation"):
                    self.assertEqual(cfg[key], ref[key])
