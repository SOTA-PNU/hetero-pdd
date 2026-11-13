# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from sys import executable
from unittest.mock import Mock, patch

from tests.hetero.test_profiler import CASE
from tools.hetero.profiler import overhead as oh
from tools.hetero.profiler import run


def record(path, on, **values):
    basic = dict(path=path, mode="on" if on else "off", valid=True)
    return dict(tp=100, lat=1, p95=1, cpu={}, **basic) | values


class OverheadTests(unittest.TestCase):
    def test_statistics_and_decisions(self):
        inputs = [("off", 100, 1), ("on", 90, 2), ("on", 108, 4), ("off", 120, 3)]
        rows = [dict(mode=m, tp=t, lat=d) for m, t, d in inputs]
        self.assertEqual(oh.block_metrics(rows), dict(d_tp=0.1, d_lat=0.5))
        cases = [
            ([0.05] * 4, [0.05] * 4, "PASS", False),
            ([0.02] * 4, [0.06] * 4, "FAIL", False),
            ([0, 0.08] * 2, [0.02] * 4, "ADD", False),
            ([0.06] * 4, [0, 0.08] * 2, "FAIL", False),
            ([0, 0.08] * 3, [0.02] * 6, "PASS", True),
            ([0.02, 0.10] * 3, [0.02] * 6, "FAIL", True),
        ]
        for tp, lat, verdict, uncertain in cases:
            result = oh.decide([dict(d_tp=t, d_lat=d) for t, d in zip(tp, lat)])
            self.assertEqual(result["verdict"], verdict)
            self.assertEqual(bool(result["uncertainty"]), uncertain)
        blocks = [dict(d_tp=v, d_lat=v) for v in (0, 0.02, 0.04, 0.06)]
        stats = oh.decide(blocks)["stats"]["d_tp"]
        for k, value in dict(mean=0.03, low=-0.0003771, high=0.0603771).items():
            self.assertAlmostEqual(stats[k], value, places=8, msg=k)

    def test_cpu_and_retry(self):
        ticks = oh.cpu_ticks("1 (name ) with space) S " + "0 " * 10 + "10 20 30 40")
        before = dict(mono_ns=0, ticks={"all-smi": ticks})
        after = dict(mono_ns=2 * 10**9, ticks={"all-smi": 300})
        cpu = oh.cpu_usage(before, after, 100, 4)["all-smi"]
        self.assertEqual(cpu, dict(single_core_pct=100, all_cores_pct=25))
        reason = dict(failures=["bad count"])
        for failures in (1, 2):
            data: dict = dict(runs=[], attempts=[])

            def measure(name, on, data=data, failures=failures):
                valid = len(data["attempts"]) >= failures
                return record(name, on, valid=valid, validity=reason)

            if failures == 1:
                self.assertEqual(oh.measure_block(1, measure, data)["d_tp"], 0)
            else:
                with self.assertRaisesRegex(RuntimeError, "bad count"):
                    oh.measure_block(1, measure, data)
            accepted = [a["valid"] for a in data["attempts"]]
            self.assertEqual(accepted, [False, failures == 1])
            self.assertEqual(len({r["path"] for r in data["runs"]}), 8)

    def test_execution_and_failure_cleanup(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temp, redirect_stdout(StringIO()):
                root, names = Path(temp) / "run", list[str]()
                work = Path(temp) / "work.json"
                payload = CASE["workload"]

                def measure(folder, work, config, on, names=names, fail=fail):
                    names.append(folder.name)
                    if fail and on:
                        raise RuntimeError("fake failure")
                    low = on and folder.name.startswith(("02_", "04_", "06_"))
                    return record(folder.name, on, tp=92 if low else 100)

                options = dict(configuration="pdd", run_dir=root, workload=work)
                options.update(model_path=temp, model_name="mock")
                options.update(cuda_python=executable, rbln_python=executable)
                args = run.flags(**options)
                work.write_text(json.dumps([payload] * 2))
                self.assertRaises(ValueError, run.parse_args, args, single=True)
                work.write_text(json.dumps([payload] if fail else payload))
                stubs = dict(command=Mock(return_value=0), check_tools=Mock())
                stubs["service_config"] = Mock(return_value=CASE["config"])
                with patch.multiple(run, **stubs), patch.object(oh, "trial", measure):
                    self.assertEqual(oh.main(args), int(fail))
                calls = [c.args[1] for c in stubs["command"].call_args_list]
                self.assertEqual(calls, ["service_start", "service_stop"])
                data = json.loads((root / "overhead/overhead.json").read_text())
                self.assertEqual(data["verdict"], "ABORTED" if fail else "PASS")
                if not fail:
                    modes = [n.rsplit("_", 1)[1] for n in names]
                    self.assertEqual(
                        modes, ["off", "on"] + ["off", "on", "on", "off"] * 6
                    )
                    self.assertEqual(
                        [d["verdict"] for d in data["decisions"]], ["ADD", "PASS"]
                    )
                self.assertIn(
                    "fake failure" if fail else "PASS",
                    (root / "overhead/overhead.md").read_text(),
                )
