# SPDX-License-Identifier: Apache-2.0
import csv
import gzip
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.hetero import finalize_results
from tools.hetero.profiler import operator_stats as ops


class OperatorTests(unittest.TestCase):
    def test_npu_last_module_and_invalid_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta.json").write_text(
                json.dumps(dict(device="npu", phase="decode", model="toy"))
            )
            names = [f"rbln_20261110_121548351{s}.pb" for s in ("", "_9", "_10")]
            for name in names:
                (root / name).touch()
            for dropped, unfinished in [(0, 0), (17474, 0), (0, 511)]:

                def read(path, health, dropped=dropped, unfinished=unfinished):
                    self.assertEqual(path.name, names[-1])
                    health.update(
                        errors={"slice_max_depth_exceeded": dropped},
                        unfinished={"Neural Engine Clusters": unfinished},
                    )
                    track = "Neural Engine Clusters"
                    row = dict(
                        name="op",
                        track=track,
                        category=track,
                        start_ns=0,
                        duration_ns=2000,
                    )
                    return [row, dict(row, category="Host")]

                with patch.object(ops, "pb_rows", side_effect=read):
                    args = ["--input", str(root), "--out", str(root / "stats")]
                    if dropped or unfinished:
                        with self.assertRaises(ValueError):
                            ops.main(args)
                    else:
                        ops.main(args)
                with (root / "stats/operators.csv").open() as stream:
                    result = list(csv.DictReader(stream))
                self.assertEqual(len(result), 1)
                self.assertEqual(
                    result[0]["status"],
                    "not_collected" if dropped or unfinished else "measured",
                )
                self.assertEqual(
                    result[0]["ratio"], "" if dropped or unfinished else "1.0"
                )
                if dropped or unfinished:
                    self.assertEqual(result[0]["reason"], ops.TRACE_ERROR)

    def test_types(self):
        cases = [
            ("gpu", "gemm", "kernel", "matmul"),
            ("gpu", "flash_fwd", "kernel", "attention"),
            ("npu", "256_linear_19_14", "Neural Engine Clusters", "linear"),
            ("npu", "256_linear_19", "Neural DMA", "DMA"),
            ("npu", "unknown_12", "Neural Engine Clusters", "기타"),
        ]
        for device, name, category, kind in cases:
            rows = [dict(name=name, category=category, duration_ns=1000)]
            result = ops.aggregate_types(rows, device)
            self.assertEqual(
                (result[0]["operator_type"], result[0]["ratio"]), (kind, 1)
            )

    def test_aggregate(self):
        cat = "kernel"
        cases = [("a", cat, 2), ("a", cat, 4), ("b", cat, 3), ("a", "cpu_op", 8)]
        rows = [
            dict(name=n, category=c, start_ns=0, duration_ns=d * 1000)
            for n, c, d in cases
        ]
        for top, count in [(1, 2), (2, 3)]:
            result = ops.aggregate(rows, top)
            self.assertEqual(len(result), count)
            a = next(
                r for r in result if r["category"] == "kernel" and r["name"] == "a"
            )
            self.assertEqual((a["calls"], a["total_ms"], a["mean_us"]), (2, 0.006, 3))
            self.assertAlmostEqual(a["ratio"], 2 / 3)

    def test_chrome_exports_and_finalize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            step = "execute_context_0(0)_generation_4(4)"
            cases = [
                (step, "user_annotation", 10, 10, 1, 1),
                ("launch", "cuda_runtime", 11, 1, 1, 1),
                ("gemm", "kernel", 30, 2, 0, 7),
            ]
            columns = ["name", "cat", "ts", "dur", "pid", "tid"]
            events = [
                dict(zip(columns, row), ph="X", args={"correlation": 7})
                for row in cases
            ]
            path = root / "rank0.pt.trace.json.gz"
            path.write_bytes(
                gzip.compress(json.dumps(dict(traceEvents=events)).encode())
            )
            rows = ops.chrome_rows(path)
            self.assertEqual((rows[0]["duration_ns"], rows[0]["step"]), (2000, 0))
            self.assertTrue(rows[0]["decode"])
            (root / "meta.json").write_text(
                json.dumps(dict(device="gpu", phase="decode", model="toy"))
            )
            out = root / "stats"
            ops.main(["--input", str(root), "--out", str(out)])
            self.assertTrue((out / "operators.md").is_file())
            self.assertTrue((out / "operators_by_type.csv").is_file())
            (root / "scenarios.json").write_text('{"scenarios": []}')
            self.assertEqual(finalize_results.finalize(root, out), 0)
            self.assertTrue((root / "dataset/operators_by_type.csv").is_file())
            with (root / "dataset/operators.csv").open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["name"], "gemm")

    @unittest.skipUnless(
        os.getenv("PDD_TOOLS_PY") and os.getenv("PERFETTO_TP_BIN"), "tools unset"
    )
    def test_pb(self):
        script = """
import sys
from pathlib import Path
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tools.hetero.profiler.operator_stats import pb_rows
b = TraceProtoBuilder()
p = b.add_packet()
p.track_descriptor.uuid = 1
p.track_descriptor.thread.pid = 0
p.track_descriptor.thread.tid = 229376
p.track_descriptor.thread.thread_name = "Neural Engine Clusters"
for t, kind in [(1000, 1), (3000, 2)]:
    p = b.add_packet()
    p.timestamp, p.trusted_packet_sequence_id = t, 1
    p.track_event.type, p.track_event.track_uuid = kind, 1
    if kind == 1:
        p.track_event.name = "op"
        p.track_event.categories.append("Neural Engine Clusters")
f = Path(sys.argv[1])
f.write_bytes(b.serialize())
r = pb_rows(f)
assert r[0]["track"] == "Neural Engine Clusters" and r[0]["duration_ns"] == 2000
"""
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(
                [os.environ["PDD_TOOLS_PY"], "-c", script, str(Path(temp) / "test.pb")],
                check=True,
            )
