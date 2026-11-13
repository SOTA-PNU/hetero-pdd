# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.hetero.profiler import analysis, resources, server, transfer


class MeasurementTests(unittest.TestCase):
    def test_exclusive_phase_shares(self):
        cases = [
            ("p_queue", (-1, 3), 0.1),
            ("p_prefill", (1, 6), 0.1),
            ("kv_transfer", (2, 7), 0.1),
            ("layout", (3, 8), 0.3),
            ("d_queue", (2, 9), 0.0),
            ("decode", (6, 11), 0.4),
        ]
        parts = {name: span for name, span, _ in cases}
        result = resources.exclusive_phase_shares(parts, (0, 10))
        for name, _, value in cases:
            self.assertAlmostEqual(result[name], value, msg=name)
        self.assertAlmostEqual(sum(result.values()), 1)
        self.assertGreaterEqual(result["other"], 0)
        self.assertAlmostEqual(
            resources.exclusive_phase_shares(parts, (0, 12))["other"], 1 / 12
        )

    def test_boundary_deltas(self):
        def histogram(count, total):
            return f"# TYPE delay histogram\ndelay_count {count}\ndelay_sum {total}\n"

        for after, value, reason in [
            ((4, 10), 3, None),
            ((2, 4), None, "zero_count"),
            ((1, 2), None, "counter_reset"),
        ]:
            with self.subTest(after=after):
                result = server.boundary_deltas(histogram(2, 4), histogram(*after))[
                    "delay"
                ]
                self.assertEqual((result["value"], result["reason"]), (value, reason))
        before = '# TYPE ok_total counter\nok_total{role="a"} 5\nok_total{role="b"} 1\n'
        reset = before.replace(" 5", " 2").replace(" 1", " 9")
        self.assertEqual(
            server.boundary_deltas(before, reset)["ok"]["reason"], "counter_reset"
        )

    def test_attribution_and_union(self):
        for intervals, wanted in [
            ([(0.5, 1.5)], (15, 20, 1, 15)),
            ([(0, 4)], (15, 30, 0.5, 30)),
        ]:
            self.assertEqual(
                resources.attribute_samples([(0, 10), (1, 20), (2, 30)], intervals),
                wanted,
            )
        self.assertEqual(
            resources.attribute_samples([(0, 10), (2, 20)], [(0, 2)])[2], 0
        )
        self.assertEqual(resources.active_ratio([(0, 2), (1, 3)], (0, 4)), 0.75)
        states = resources.device_states(
            {"prefill": [(0, 2)], "decode": [(1, 3)]}, (0, 4)
        )
        self.assertEqual(
            states,
            {
                "prefill": [(0, 1)],
                "mixed": [(1, 2)],
                "decode": [(2, 3)],
                "idle": [(3, 4)],
            },
        )

    def test_energy_and_gates(self):
        power = {r: [(0.0, 10), (1.0, 20), (2.0, 30)] for r in resources.DEVICES}
        result = resources.energy_efficiency(power, (0, 2), 120)
        a, b = (
            result["total"][k]["value"]
            for k in ("energy_per_output_token_j_est", "tok_per_s_per_w_est")
        )
        self.assertAlmostEqual(a * b, 1)
        self.assertEqual(result["total"]["energy_j"]["value"], 120)
        for missing in ([], [(0.0, 10)], [(0.0, 10), (0.5, 10)]):
            power["npu1"] = missing
            self.assertIsNone(
                resources.energy_efficiency(power, (0, 2), 120)["total"]["energy_j"][
                    "value"
                ]
            )
        for raw, value in [(0.8, 0.8), (1.2, None)]:
            ratio = transfer.device_time_ratio([{"device": raw * 1e6}], (0, 1))
            self.assertEqual((ratio["raw_value"], ratio["value"]), (raw, value))
        self.assertFalse(transfer.COPY_SYNC_CONFIRMED)
        self.assertEqual(
            transfer.h2d_bandwidth(100, 1)["reason"], "copy_ 동기 여부 미확인(H4)"
        )

    def test_classify_real_devices(self):
        text = (
            Path(__file__).parent / "fixtures/real/allsmi_metrics_sample.txt"
        ).read_text()
        text = text.replace('gpu_index="1"', 'gpu_index="99"')
        devices = [
            {"npu": i, "pci": {"bus_id": bus}}
            for i, bus in enumerate(("0000:34:00.0", "0000:ac:00.0"))
        ]
        links = dict(
            nvidia_smi="<REDACTED_DEVICE_UUID_0>, bus, 5, 16",
            rbln_smi=json.dumps({"devices": devices}),
        )
        self.assertEqual(
            resources.classify_devices(text, links),
            {
                f"<REDACTED_DEVICE_UUID_{i}>": role
                for i, role in enumerate(resources.DEVICES)
            },
        )

    def test_validation(self):
        boundaries = [
            dict(endpoint="gpu", label=label, status=200)
            for label in ("before", "after")
        ]
        for count, status in [(1, "INCOMPLETE"), (8, "PASS")]:
            samples = [
                dict(endpoint=r, mono_ns=i * 10**9, status=200)
                for r in ("gpu", "all-smi")
                for i in range(count)
            ]
            result = analysis.validate_measurement(
                samples, boundaries, ["gpu"], (0, 10)
            )
            self.assertEqual(result["analysis_status"], status)

    def test_memory_bandwidth(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            model = dict(
                num_hidden_layers=2,
                num_key_value_heads=1,
                num_attention_heads=2,
                hidden_size=8,
            )
            (path / "config.json").write_text(json.dumps(model))
            (path / "model.safetensors").write_bytes(bytes(100))
            config = dict(model_path=temp, kv_cache_dtype="float16")
            steps = [dict(device=2e6, context_len_sum=3, phase="decode")]
            for device, seconds, wanted in ((None, 4, 49), (2e6, 4, 98)):
                steps[0].update(
                    device=device, span_start_ns=0, span_end_ns=seconds * 10**9
                )
                with patch.dict("os.environ", NPU_MEM_BW_GBPS="2"):
                    result = transfer.memory_bandwidth(config, steps, None, [])
                for key, divisor in zip(transfer.MEMORY_FIELDS, (1, 2)):
                    self.assertAlmostEqual(result[key]["value"] * 1e9, wanted / divisor)
                    self.assertEqual(result[key]["status"], "estimated")
                    reason: str | None = (
                        "device 시간 없음: 호스트 관측 구간 기준 하한 추정"
                    )
                    reason = reason if device is None else None
                    self.assertEqual(result[key]["reason"], reason)
