# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.hetero.profiler import expected
from tools.hetero.profiler import perfetto_timeline as timeline

FIXTURE = Path(__file__).parent / "fixtures/synthetic"
SCRIPT = Path(timeline.__file__)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


class TimelineTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("PDD_TOOLS_PY"), "tools Python unset")
    def test_gpu_clock_filter_and_lanes(self):
        anchors = [dict(time_ns=10**9, monotonic_ns=2 * 10**9)]
        events = []
        for cat, ts, dur in (
            ("kernel", 100, 20),
            ("gpu_memcpy", 101, 5),
            ("cuda_runtime", 100, 10),
            ("cuda_runtime", 102, 10),
            ("cpu_op", 100, 20),
            ("cpu_op", 103, 2),
            ("kernel", 900, 10),
            ("python_function", 100, 1),
        ):
            events.append(dict(cat=cat, ts=ts, dur=dur, name=cat, ph="X", tid=1, pid=1))
        data = dict(baseTimeNanoseconds=10**9, traceEvents=events)
        rows, report = timeline.gpu_slices(data, anchors, (2, 2.0005), "gpu")
        self.assertEqual((report["included"], report["excluded"]), (5, 3))
        self.assertEqual(min(x["start"] for x in rows), 2000100000)
        runtime = [x for x in rows if x["name"] == "cuda_runtime"]
        self.assertEqual(len({x["track"] for x in runtime}), 2)
        self.assertEqual(sum(x["name"] == "cpu_op" for x in rows), 1)
        for key in ("baseTimeNanoseconds",):
            invalid = {k: v for k, v in data.items() if k != key}
            rows, report = timeline.gpu_slices(invalid, anchors, (2, 3), "gpu")
            self.assertEqual((rows, report["status"]), ([], "not_collected"))

    def test_lanes(self):
        for intervals, wanted in (
            ([(0, 3, "a"), (1, 2, "b")], [0, 1]),
            ([(0, 1, "a"), (1, 2, "a")], [0, 0]),
            ([(0, 1, "a"), (1, 2, "b")], [0, 0]),
            ([(0, 3, "a"), (1, 2, "b"), (3, 4, "a")], [0, 1, 0]),
        ):
            rows: list[dict] = [
                dict(track=("GPU",), start=a, end=b, rid=r) for a, b, r in intervals
            ]
            timeline.assign_lanes(rows)
            self.assertEqual(
                [r["track"][-1] for r in rows], [f"Lane {i}" for i in wanted]
            )

    def test_slices_without_perfetto(self):
        row = next(expected.live_rows(FIXTURE / "b0_three_devices/requests.jsonl"))
        args: tuple = ([row], {"gpu": {row["request_id"]}}, {}, {}, [], False)
        result = timeline.slices(*args)
        self.assertEqual(len(result), 5)
        self.assertEqual(sum(x["args"].get("status") == "추정" for x in result), 2)
        fields = dict(h2d_ns=1, reorder_ns=2, num_blocks=3, bytes_est=4)
        event = dict(kind="kv_h2d", start_ns=10, end_ns=20, **fields)
        result = timeline.slices([], {}, {}, {"decode0": [event]}, [], True)[0]
        self.assertEqual(result["name"], "KV Load (H2D + transform)")
        self.assertEqual(result["track"][1], result["name"])
        self.assertEqual(result["args"], fields)

    @unittest.skipUnless(
        os.getenv("PDD_TOOLS_PY") and os.getenv("PERFETTO_TP_BIN"),
        "tools Python/trace_processor required",
    )
    def test_monitor_and_trace(self):
        for case, mode, count, flows in (
            ("b0_three_devices", "monitor", 4, 2),
            ("pdd_skew", "trace", 9, 4),
            ("pdd_skew", "ci", 9, 4),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, fixture = Path(temp), FIXTURE / case
                row = next(expected.live_rows(fixture / "requests.jsonl"))
                config = json.loads((fixture / "service_config.json").read_text())
                if mode == "ci":
                    root = root / "profile"
                    del config["endpoints"]["decode1"]
                    config["role_devices"] = dict(prefill="gpu", decode0="npu1")
                write(
                    root / ("service.json" if mode == "ci" else "service/config.json"),
                    config,
                )
                write(root / "run.json", dict(mode="trace" if mode == "ci" else mode))
                write(root / "w/requests.jsonl", row)
                anchor = dict(time_ns=0, monotonic_ns=0)
                write(root / "w/loadgen_run.json", dict(clock_anchor_start=anchor))
                sid = expected.server_request_id(row["request_id"])
                batches = []
                for role in ("prefill", "decode0"):
                    attrs = {"gen_ai.request.id": sid}
                    for key, value in (
                        ("time_in_queue", 0.1),
                        ("time_to_first_token", 2),
                        ("time_in_model_prefill", 1),
                        ("time_in_model_decode", 6),
                    ):
                        attrs["gen_ai.latency." + key] = value
                    fields = [
                        dict(key=k, value={"stringValue": v}) for k, v in attrs.items()
                    ]
                    span = dict(
                        startTimeUnixNano="102000000000",
                        endTimeUnixNano="110000000000",
                        attributes=fields,
                    )
                    label = dict(key="service.name", value=dict(stringValue=role))
                    resource = dict(attributes=[label])
                    batches.append(
                        dict(resource=resource, scopeSpans=[dict(spans=[span])])
                    )
                write(root / "otel/traces.jsonl", dict(resourceSpans=batches))
                for i, kind in enumerate(
                    ("kv_xfer_start", "kv_xfer_done", "kv_h2d", "step")
                ):
                    event = dict(
                        kind=kind,
                        req_id=sid + "-abcdef01",
                        phase="decode",
                        monotonic_ns=(103 + i) * 10**9,
                        start_ns=104 * 10**9,
                        end_ns=105 * 10**9,
                        span_start_ns=105 * 10**9,
                        span_end_ns=106 * 10**9,
                    )
                    write(root / f"rbln_metrics/decode0/events_{i}.jsonl", event)
                cmd = [os.environ["PDD_TOOLS_PY"], str(SCRIPT), "--run-dir", str(root)]
                subprocess.run(cmd, check=True)
                result = json.loads((root / "w/timeline_validation.json").read_text())
                self.assertEqual(
                    (result["status"], result["actual_slices"], result["actual_flows"]),
                    ("PASS", count, flows),
                )
                self.assertTrue(all(r["n"] == r["expected"] for r in result["tracks"]))
                if mode == "ci":
                    self.assertEqual(result["role_tracks"]["decode0"], "Decode(NPU1)")
                original = (root / "w/timeline.pftrace").read_bytes()
                subprocess.run(cmd, check=True)
                self.assertEqual(original, (root / "w/timeline.pftrace").read_bytes())
