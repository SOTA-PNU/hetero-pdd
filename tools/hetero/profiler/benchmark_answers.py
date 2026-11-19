"""Check measured completions against a separate, exact-length GPU full burst."""

import hashlib
import json
from pathlib import Path

from llmd.engines import MODEL_IDENTITY
from tools.hetero.profiler.common import live_rows


def output_set(work):
    work = Path(work)
    state = json.loads((work / "run.json").read_text())
    folder = work / "profile"
    load = json.loads((folder / "config.json").read_text())
    prompts = {
        r["request_index"]: r["prompt_ids"] for r in live_rows(folder / "prompts.jsonl")
    }
    planned = set(range(load["requests"]))
    if (
        set(prompts) != planned
        or hashlib.sha256(
            json.dumps([prompts[i] for i in sorted(planned)]).encode()
        ).hexdigest()
        != load["prompts_sha256"]
    ):
        raise ValueError(f"{work}: saved prompts differ from benchmark input hash")
    route_log = json.loads((folder / "route_evidence.json").read_text())
    routes = {r["request_id"]: r for r in route_log["requests"]}
    eps = {ep["id"]: ep for ep in state["endpoints"]}
    conditions = {k: state["model"][k] for k in MODEL_IDENTITY}
    conditions.update(
        {k: load[k] for k in ("seed", "generation", "input_tokens", "output_tokens")}
    )
    results = {}
    request_ids = set()
    for row in live_rows(folder / "requests.jsonl"):
        index = row["request_index"]
        if index in results or index not in prompts or row["request_id"] in request_ids:
            raise ValueError(
                f"{work}: duplicate request identity or invalid index {index}"
            )
        request_ids.add(row["request_id"])
        route = routes.get(row["request_id"], {})
        engine = eps.get(route.get("target"), {}).get("engine", {})
        results[index] = {
            "request_id": row["request_id"],
            "prompt_ids": prompts[index],
            "output_ids": row["streamed_token_ids"] or [],
            "executed_path": route.get("executed_path", "unverified"),
            "target": route.get("target"),
            "dtype": engine.get("dtype"),
            "kv_dtype": engine.get("kv_dtype"),
            "complete": (
                row["http_status"] == 200
                and not row["error"]
                and row["actual_input_tokens"]
                == len(prompts[index])
                == load["input_tokens"]
                and row["actual_output_tokens"] == load["output_tokens"]
                and len(row["streamed_token_ids"] or []) == load["output_tokens"]
            ),
        }
    return dict(
        work=str(work), conditions=conditions, planned=load["requests"], rows=results
    )


def gpu_answer_key(work):
    answers = output_set(work)
    rows = answers["rows"]
    if len(rows) != answers["planned"] or any(
        r["executed_path"] != "gpu_full" or not r["complete"] or not r["dtype"]
        for r in rows.values()
    ):
        raise ValueError(
            f"{work}: every reference needs a complete verified GPU full output"
        )
    return answers


def compare(work, answers):
    measured = output_set(work)
    same_conditions = measured["conditions"] == answers["conditions"]
    rows = []
    for i in sorted(answers["rows"]):
        ref = answers["rows"][i]
        actual = measured["rows"].get(i, {})
        observed = actual.get("output_ids", [])
        expected = ref["output_ids"]
        first = next(
            (j for j, (a, b) in enumerate(zip(expected, observed)) if a != b), None
        )
        if first is None and len(expected) != len(observed):
            first = min(len(expected), len(observed))
        same_input = ref["prompt_ids"] == actual.get("prompt_ids")
        valid = (
            same_conditions
            and same_input
            and actual.get("complete")
            and actual.get("dtype") == ref["dtype"]
            and actual.get("kv_dtype") == ref["kv_dtype"]
            and actual.get("executed_path") in {"gpu_full", "npu_full", "pdd"}
            and actual.get("request_id") != ref["request_id"]
            and first is None
        )
        rows.append(
            {
                "request_index": i,
                "request_id": actual.get("request_id"),
                "reference_request_id": ref["request_id"],
                "reference_target": ref["target"],
                "reference_dtype": ref["dtype"],
                "reference_kv_dtype": ref["kv_dtype"],
                "target": actual.get("target"),
                "executed_path": actual.get("executed_path"),
                "dtype": actual.get("dtype"),
                "kv_dtype": actual.get("kv_dtype"),
                "same_prompt_ids": same_input,
                "matching": bool(valid),
                "first_mismatch_index": first,
                "first_expected_token": expected[first]
                if first is not None and first < len(expected)
                else None,
                "first_observed_token": observed[first]
                if first is not None and first < len(observed)
                else None,
                "reference_output_length": len(expected),
                "observed_output_length": len(observed),
            }
        )
    mismatched = [r["request_index"] for r in rows if not r["matching"]]
    report = dict(
        reference=answers["work"],
        reference_conditions=answers["conditions"],
        measured_conditions=measured["conditions"],
        measured_requests=measured["planned"],
        reference_requests=answers["planned"],
        same_conditions=same_conditions,
        correct=(
            not mismatched
            and measured["planned"] == answers["planned"] == len(measured["rows"])
        ),
        mismatched_request_indices=mismatched,
        requests=rows,
    )
    (Path(work) / "profile/gpu_correctness.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
