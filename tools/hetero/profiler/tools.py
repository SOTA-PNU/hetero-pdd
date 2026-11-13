# SPDX-License-Identifier: Apache-2.0
"""Check pinned executables and preserve device link information verbatim."""

import json
import os
import re
import subprocess
from pathlib import Path

VERSIONS = Path(__file__).resolve().parents[1] / "third_party" / "versions.env"
TOOLS = (
    ("ALLSMI", "all-smi"),
    ("OTELCOL", "otelcol-contrib"),
    ("PERFETTO", "trace_processor_shell"),
)


def check_tools(mode):
    count = {"none": 0, "monitor": 1, "trace": 2, "deep": 3}[mode]
    pins = dict(re.findall(r"^(\w+_VERSION)=([\d.]+)$", VERSIONS.read_text(), re.M))
    found = {}
    for name, default in TOOLS[:count]:
        variable = "PERFETTO_TP_BIN" if name == "PERFETTO" else name + "_BIN"
        binary = os.environ.get(variable, default)
        output = subprocess.check_output(
            [binary, "--version"], text=True, stderr=subprocess.STDOUT, timeout=10
        ).strip()
        versions = re.findall(r"(?<![\w.])v?(\d+\.\d+(?:\.\d+)?)(?![\w.])", output)
        if pins[name + "_VERSION"] not in versions:
            raise RuntimeError(
                f"{binary}: expected {pins[name + '_VERSION']}, got {output}"
            )
        found[name] = output
    return found


def capture_links(gpu_index, out_path):
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=uuid,pci.bus_id,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader",
        ],
        "rbln_smi": ["rbln-smi", "--json"],
    }
    raw = {
        name: subprocess.check_output(argv, text=True, timeout=30)
        for name, argv in commands.items()
    }
    Path(out_path).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
