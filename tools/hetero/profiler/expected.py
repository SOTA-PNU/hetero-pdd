# SPDX-License-Identifier: Apache-2.0
import json
import re
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from tools.hetero.loadgen import request_succeeded
from tools.hetero.profiler.common import live_rows
from tools.hetero.profiler.scrape import wait_until


def service_config(run_dir):
    return json.loads((Path(run_dir) / "service" / "config.json").read_text())


# H0 확인(2026-11-06). vLLM 0.22.0:
# entrypoints/openai/engine/serving.py:399; completion/serving.py:144,180;
# v1/engine/output_processor.py:749 uses external_req_id (single prompt, n=1).
def server_request_id(client_id):
    return f"cmpl-{client_id}-0"


def engine_request_base(req_id):
    # vLLM 0.22.0 v1/engine/input_processor.py:224,232 preserves external_req_id
    # and appends f"-{random_uuid():.8}" to the internal request_id.
    match = re.fullmatch(r"^(?P<base>.+)-[0-9a-f]{8}$", req_id or "")
    return match["base"] if match else None


def expected_ids(rows, config):
    expected = {role: set() for role in config["endpoints"] if role != "proxy"}
    decoders = {
        urlsplit(url).netloc: role
        for role, url in config["endpoints"].items()
        if role.startswith("decode")
    }
    decoders = config.get("decoder_roles", decoders)
    for row in filter(request_succeeded, rows):
        if config["configuration"] == "baseline":
            roles = [row["assigned_target"]]
        else:
            target = row["response_headers"].get("x-decode-target")
            if target not in decoders:
                raise ValueError(f"unknown decoder target: {target!r}")
            roles = ["prefill", decoders[target]]
        for role in roles:
            if role not in expected:
                raise ValueError(f"unknown request role: {role}")
            expected[role].add(row["request_id"])
    return expected


def attribute(item, key):
    values = {a["key"]: a["value"] for a in item.get("attributes", [])}
    return values.get(key, {}).get("stringValue")


def wait_for_spans(run_dir, expected, timeout=30):
    wanted = {
        (role, server_request_id(rid)) for role, ids in expected.items() for rid in ids
    }
    root, missing = Path(run_dir), wanted

    def complete():
        nonlocal missing
        seen = set()
        for batch in live_rows(root / "otel" / "traces.jsonl"):
            for resource in batch.get("resourceSpans", []):
                role = attribute(resource.get("resource", {}), "service.name")
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        seen.add((role, attribute(span, "gen_ai.request.id")))
        missing = wanted - seen
        return not missing

    with suppress(TimeoutError):
        wait_until(complete, timeout, "spans")
    gaps = [dict(role=role, request_id=rid) for role, rid in sorted(missing)]
    result = dict(complete=not missing, missing=gaps)
    (root / "trace_wait.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
