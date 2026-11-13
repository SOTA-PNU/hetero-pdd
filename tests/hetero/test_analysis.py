# SPDX-License-Identifier: Apache-2.0
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.hetero.profiler import analysis as a
from tools.hetero.profiler import expected as e
from tools.hetero.profiler.scrape import append_rows

FIXTURES = Path(__file__).parent / "fixtures"


class AnalysisTests(unittest.TestCase):
    def equal_number(self, actual, wanted, label):
        if isinstance(wanted, list):
            self.assertEqual(len(actual), len(wanted), label)
            for x, y in zip(actual, wanted):
                self.equal_number(x, y, label)
        elif isinstance(wanted, (int, float)):
            self.assertTrue(math.isclose(actual, wanted, rel_tol=1e-9), label)
        else:
            self.assertEqual(actual, wanted, label)

    def test_reproduction(self):
        for path in sorted((FIXTURES / "real").glob("*/*/requests.jsonl")):
            rows = list(e.live_rows(path))
            actual = a.client_metrics(rows)
            for old, new in zip(rows, actual["requests"]):
                for key, metric in new["metrics"].items():
                    field = "tp_ot_seconds" if key == "tpot_seconds" else key
                    if field in old:
                        self.equal_number(metric["value"], old[field], (path, field))
            old = json.loads(path.with_name("summary.json").read_text())
            for key, metric in actual["summary"].items():
                if key in old:
                    self.equal_number(metric["value"], old[key], f"{path}: {key}")

    def test_synthetic_and_validation(self):
        for folder in sorted((FIXTURES / "synthetic").glob("*/requests.jsonl")):
            rows = list(e.live_rows(folder))
            wanted = json.loads(folder.with_name("expected.json").read_text())
            config = json.loads(folder.with_name("config.json").read_text())
            actual = a.client_metrics(rows)
            for key, value in wanted["summary"].items():
                self.equal_number(actual["summary"][key]["value"], value, key)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                append_rows(root / "requests.jsonl", rows)
                (root / "loadgen_run.json").write_text("{}")
                result = a.validate_raw(root, rows, config)
                self.assertEqual(result["run_status"], wanted["run_status"])
                (root / "loadgen_run.json").unlink()
                result = a.validate_raw(root, rows, config)
                self.assertEqual(result["run_status"], "FAILED")
            if folder.parent.name == "one_token":
                for key in ("tpot_seconds", "second_token_delay_seconds"):
                    node = actual["requests"][0]["metrics"][key]
                    self.assertEqual(node["status"], "not_collected")
                    self.assertEqual(node["reason"], "not_applicable")
            self.assertEqual(
                actual["summary"]["success_count"]["value"], wanted["success_count"]
            )

    def test_join(self):
        for rid in ("x-ABCDEF12", "x-0123abcd\n", "x-123456789"):
            self.assertIsNone(e.engine_request_base(rid))
        attr = lambda key, value: {"key": key, "value": {"stringValue": value}}
        for case in ("b0_three_devices", "pdd_skew"):
            folder = FIXTURES / "synthetic" / case
            rows = list(e.live_rows(folder / "requests.jsonl"))
            config = json.loads((folder / "service_config.json").read_text())
            wanted = json.loads((folder / "expected.json").read_text())
            wanted = wanted["expected_ids_by_role"]
            with tempfile.TemporaryDirectory() as temp:
                root, batches = Path(temp), []
                for role, ids in wanted.items():
                    for rid in ids:
                        sid = e.server_request_id(rid)
                        span: dict = dict(traceId="a" * 32, spanId="b" * 16)
                        span.update(startTimeUnixNano="100", endTimeUnixNano="200")
                        span["attributes"] = [attr("gen_ai.request.id", sid)]
                        resource = {"attributes": [attr("service.name", role)]}
                        res = dict(resource=resource, scopeSpans=[{"spans": [span]}])
                        batches.append({"resourceSpans": [res]})
                        if role.startswith("decode"):
                            eid = sid + "-0123abcd"
                            events = [dict(kind=k, req_id=eid) for k in a.KV_KINDS]
                            path = root / "rbln_metrics" / role / "events_0.jsonl"
                            append_rows(path, events)
                trace = root / "otel" / "traces.jsonl"
                for missing, duplicate in ((0, 0), (1, 0), (0, 1)):
                    trace.unlink(missing_ok=True)
                    append_rows(trace, batches[missing:] + batches[:duplicate])
                    result = a.join_requests(root, rows, config)
                    self.assertEqual(result["complete"], not (missing or duplicate))
                    self.assertEqual(len(result["missing"]), int(missing))
                    self.assertEqual(len(result["duplicates"]), int(duplicate))
                    if not (missing or duplicate):
                        counts = {r: v["span"] for r, v in result["matched"].items()}
                        counts_wanted = {r: len(ids) for r, ids in wanted.items()}
                        self.assertEqual(counts, counts_wanted)
                if case == "pdd_skew":
                    for rid, category, count in (
                        (sid + "-deadbeef", "duplicates", 2),
                        (sid, "unmatched", 1),
                    ):
                        append_rows(path, [dict(kind="kv_h2d", req_id=rid)])
                        result = a.join_requests(root, rows, config)
                        records = [r for r in result[category] if r["kind"] == "kv_h2d"]
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]["request_id"], sid)
                        self.assertEqual(records[0]["count"], count)
                        self.assertFalse(result["complete"])
