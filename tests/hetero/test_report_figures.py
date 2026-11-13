# SPDX-License-Identifier: Apache-2.0
import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class FigureSmoke(unittest.TestCase):
    @unittest.skipUnless(os.getenv("PDD_TOOLS_PY"), "tools Python unset")
    def test_synthetic_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset").mkdir()
            rows = []
            for config in ("baseline", "pdd"):
                row: dict[str, object] = dict(
                    run_id=config,
                    configuration=config,
                    model="toy",
                    mode="monitor",
                    scenario="s2_" + config,
                    pair_id="s2",
                    workload="i4096_o256",
                )
                for metric in (
                    "output_tokens_per_second",
                    "makespan_seconds",
                    "ttft_mean_seconds",
                    "tpot_mean_seconds",
                    "e2e_mean_seconds",
                ):
                    row[metric] = 1
                row.update(
                    phase_resources={"samples": []},
                    transfer={},
                    energy={"total": {"energy_per_output_token_j_est": {"value": 1}}},
                )
                rows.append(row)
            for row in rows:
                row.update(run_status="PASS", analysis_status="PASS", returncode=0)
            rows.insert(0, dict(rows[0], run_id="old", phase_resources={}))
            for status, code in (("INCOMPLETE", 0), ("PASS", 1)):
                rows.append(
                    dict(
                        rows[1],
                        run_id="failed",
                        phase_resources={},
                        analysis_status=status,
                        returncode=code,
                    )
                )
            pairs = [
                dict(
                    pair_id="s2",
                    workload="i4096_o256",
                    comparison="capacity",
                    baseline="baseline",
                    candidate="pdd",
                )
            ]
            for name, records in (("results.csv", rows), ("comparison.csv", pairs)):
                with (root / name).open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=records[0])
                    writer.writeheader()
                    writer.writerows(
                        {
                            k: json.dumps(v) if isinstance(v, dict) else v
                            for k, v in r.items()
                        }
                        for r in records
                    )
            (root / "dataset/resource_timeseries.csv").write_text("run_id,workload\n")
            (root / "overhead.json").write_text(
                json.dumps(
                    dict(verdict="PASS", blocks=[dict(block=1, d_tp=0.01, d_lat=-0.01)])
                )
            )
            script = (
                Path(__file__).resolve().parents[2] / "tools/hetero/report_figures.py"
            )
            subprocess.run(
                [
                    os.environ["PDD_TOOLS_PY"],
                    str(script),
                    "--sweep-dir",
                    str(root),
                    "--opprofile-dir",
                    str(root),
                    "--overhead",
                    str(root / "overhead.json"),
                    "--out",
                    str(root / "figures"),
                ],
                check=True,
                env=dict(os.environ, MPLCONFIGDIR=str(root / "mplconfig")),
            )
            files = list((root / "figures").glob("*.png"))
            self.assertEqual(len(files), 9)
            self.assertTrue(all(p.read_bytes().startswith(b"\x89PNG") for p in files))
