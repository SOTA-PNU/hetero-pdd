# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fit native TikTok route costs from an independent calibration directory."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from llmd import config


def line(x, y):
    """Least-squares line with a nonnegative intercept and slope."""
    xm, ym = statistics.mean(x), statistics.mean(y)
    denom = sum((v - xm) ** 2 for v in x)
    if denom == 0:
        raise ValueError("need at least two distinct calibration sizes")
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denom
    candidates = [
        (ym - slope * xm, slope),
        (ym, 0),
        (0, sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)),
    ]
    return min(
        (p for p in candidates if min(p) >= 0),
        key=lambda p: sum((p[0] + p[1] * a - b) ** 2 for a, b in zip(x, y)),
    )


def fit(measured, output):
    lengths = sorted(map(int, measured["cuda"]))
    changes = {}
    for backend in ("cuda", "rbln"):
        points = [measured[backend][str(n)] for n in lengths]
        intercept, slope = line(lengths, [p["prefill_ms"] for p in points])
        full = max(
            0,
            statistics.median(
                p["ttft_ms"] - intercept - slope * n for n, p in zip(lengths, points)
            ),
        )
        decode = [p["tpot_ms"] for p in points]
        if backend == "rbln":
            base, context = line([n + (output - 1) / 2 for n in lengths], decode)
        else:
            base, context = statistics.median(decode), 0
        changes[backend] = dict(
            prefillFixedMs=intercept,
            prefillMsPerToken=slope,
            fullFixedMs=full,
            decodeMsPerToken=base,
            decodeMsPerContextToken=context,
        )
    pdd = [measured["pdd"][str(n)] for n in lengths]
    blocks = [p["blocks"] for p in pdd]
    transfer_fixed, transfer_slope = line(blocks, [p["receive_ms"] for p in pdd])
    import_fixed, import_slope = line(blocks, [p["import_ms"] for p in pdd])
    gpu, npu = changes["cuda"], changes["rbln"]
    dispatch = [
        p["ttft_ms"]
        - (gpu["prefillFixedMs"] + n * gpu["prefillMsPerToken"])
        - transfer_fixed
        - transfer_slope * b
        - import_fixed
        - import_slope * b
        - (
            npu["decodeMsPerToken"]
            + (n + (output - 1) / 2) * npu["decodeMsPerContextToken"]
        )
        for p, n, b in zip(pdd, lengths, blocks)
    ]
    link = dict(
        dispatchFixedMs=max(0, statistics.median(dispatch)),
        transferFixedMs=transfer_fixed,
        transferMsPerBlock=transfer_slope,
        importFixedMs=import_fixed,
        importMsPerBlock=import_slope,
    )
    return changes, link, dispatch


def write_profiles(source, templates, output):
    measured_path = source / "matching-medians.json"
    spec = config.read_yaml(source / "settings.yaml")
    measured = json.loads(measured_path.read_text())
    changes, link, residuals = fit(measured, spec["output_tokens"])
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(measured_path.read_bytes()).hexdigest()
    hashes = {}
    for path in templates:
        native = config.read_yaml(path)
        (handler,) = [p for p in native["plugins"] if p["type"] == "cost-route-handler"]
        table = handler["parameters"]["costs"]
        for backend, data in changes.items():
            table[backend].update({k: round(v, 6) for k, v in data.items()})
        # Restrict calibration to the physical pair described by these samples.
        links = handler["parameters"]["links"]
        (producer,) = links
        (consumer,) = links[producer]
        links[producer][consumer].update(
            {k: round(v, 6) for k, v in link.items()}
        )
        profile = output / path.name
        profile.write_text(
            f"# Independent CA22 calibration medians SHA256 {digest}\n"
            + yaml.safe_dump(native, sort_keys=False)
        )
        hashes[path.name] = hashlib.sha256(profile.read_bytes()).hexdigest()
    (output / "calibration.json").write_text(
        json.dumps(
            dict(
                medians_sha256=digest,
                medians=measured,
                fitted_devices=changes,
                link=link,
                dispatch_residuals_ms=residuals,
                profile_sha256=hashes,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument(
        "templates", type=Path, nargs="+", help="native profiles to reprice"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_profiles(args.calibration, args.templates, args.output)
