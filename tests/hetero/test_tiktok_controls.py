"""Keep shipped preload controls matched at both routing and engine layers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import yaml

from llmd.config import OVERLAP_PROBE_DEFAULTS, load, read_yaml
from llmd.overlap import _pair_conditions


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

    def test_fixed_pair_yaml_controls_replay_and_checks_context(self):
        root = Path(__file__).resolve().parents[2]
        experiment = root / "experiments/tiktok-overlap.yaml"
        raw = read_yaml(experiment)
        raw.pop("benchmark")
        for key in ("model", "profile"):
            raw[key] = str((experiment.parent / raw[key]).resolve())
        override = dict(anchor_output_tokens=128, arrival_offset_ms=350)
        raw["verify"]["overlap_probe"] = override
        with TemporaryDirectory() as temp:
            path = Path(temp) / "replay.yaml"
            path.write_text(yaml.safe_dump(raw))
            cfg = load(path)
            state = dict(experiment=cfg, endpoints=cfg["endpoints"])
            self.assertEqual(
                _pair_conditions(state, None, None),
                ({**OVERLAP_PROBE_DEFAULTS, **override}, False),
            )
            cfg["native"] = read_yaml(root / "llmd/epp/profiles/tiktok-fixed-off.yaml")
            probe, ordinary = _pair_conditions(state, None, 0.75)
            self.assertTrue(ordinary)
            self.assertEqual(probe["arrival_offset_ms"], 750)
            cfg["verify"].pop("overlap_probe")
            self.assertEqual(
                _pair_conditions(state, False, None), (OVERLAP_PROBE_DEFAULTS, False)
            )
            for bad in (
                {"unknown": 123},
                {"arrival_offset_ms": 180000},
                {"anchor_input_tokens": True},
                {"following_output_tokens": 9000},
            ):
                with self.subTest(bad=bad):
                    raw["verify"]["overlap_probe"] = bad
                    path.write_text(yaml.safe_dump(raw))
                    with self.assertRaisesRegex(ValueError, "overlap_probe"):
                        load(path)
