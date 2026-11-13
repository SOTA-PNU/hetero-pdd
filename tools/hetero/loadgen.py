# SPDX-License-Identifier: Apache-2.0
"""Finite-burst load generator with exact token inputs and streaming timestamps."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Slot:
    index: int
    target: str
    base_url: str


def make_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )


def clock_anchor():
    return {"time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def request_succeeded(row):
    return (
        row["error"] is None
        and 200 <= row["http_status"] < 300
        and row["actual_output_tokens"] == row["requested_output_tokens"]
    )


def make_prompt(tokenizer, length: int, request_index: int, seed: int) -> list[int]:
    prefix = tokenizer.encode(
        f"Independent request {request_index}, deterministic seed {seed}. ",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        "The quick brown fox studies a different systems performance observation. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced an empty filler")
    tokens = list(prefix[:length])
    offset = (request_index * 7 + seed) % len(filler)
    while len(tokens) < length:
        take = min(length - len(tokens), len(filler))
        tokens.extend((filler + filler)[offset : offset + take])
        offset = (offset + take + request_index + 1) % len(filler)
    return tokens[:length]


def parse_sse_line(raw_line: bytes) -> dict | None:
    line = raw_line.decode("utf-8", errors="replace").strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data:"):
        line = line[5:].strip()
    if line == "[DONE]":
        return {"_done": True}
    try:
        value = json.loads(line)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def issue_request(
    *,
    slot: Slot,
    request_index: int,
    prompt_ids: list[int],
    scheduled_arrival: float,
    model: str,
    output_tokens: int,
    seed: int,
    timeout: float,
    workload: str,
) -> dict:
    assigned = time.monotonic()
    request_id = f"{workload}-seed{seed}-req{request_index:04d}"
    body = {
        "model": model,
        "prompt": prompt_ids,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        slot.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    sent = time.monotonic()
    status = 0
    first_output = None
    last_output = None
    chunks: list[dict] = []
    output_fragments: list[str] = []
    streamed_token_ids: list[int] = []
    usage: dict = {}
    error = None
    response_headers: dict[str, str] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "x-request-id", "x-decode-target"}
            }
            for raw_line in response:
                now = time.monotonic()
                event = parse_sse_line(raw_line)
                if not event or event.get("_done"):
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                text = choice.get("text")
                token_ids = choice.get("token_ids")
                has_output = isinstance(text, str) and bool(text)
                if isinstance(token_ids, list) and token_ids:
                    valid = [
                        value
                        for value in token_ids
                        if isinstance(value, int) and not isinstance(value, bool)
                    ]
                    if valid:
                        streamed_token_ids.extend(valid)
                        has_output = True
                if isinstance(text, str):
                    output_fragments.append(text)
                if has_output:
                    first_output = now if first_output is None else first_output
                    last_output = now
                chunks.append(
                    {
                        "t_rel_arrival_seconds": now - scheduled_arrival,
                        "text_chars": len(text) if isinstance(text, str) else 0,
                        "token_ids": token_ids if isinstance(token_ids, list) else None,
                        "finish_reason": choice.get("finish_reason"),
                    }
                )
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    actual_prompt_tokens = usage.get("prompt_tokens")
    actual_output_tokens = usage.get("completion_tokens")
    if not isinstance(actual_prompt_tokens, int):
        actual_prompt_tokens = len(prompt_ids)
    if not isinstance(actual_output_tokens, int):
        actual_output_tokens = len(streamed_token_ids) if streamed_token_ids else None
    return {
        "request_id": request_id,
        "request_index": request_index,
        "workload": workload,
        "seed": seed,
        "scheduled_arrival_monotonic": scheduled_arrival,
        "actual_send_monotonic": sent,
        "assigned_monotonic": assigned,
        "first_output_monotonic": first_output,
        "last_output_monotonic": last_output,
        "completed_monotonic": ended,
        "chunk_timestamps": chunks,
        "assigned_slot": slot.index,
        "assigned_target": slot.target,
        "execution_path": "PDD" if slot.target == "pdd" else "direct",
        "requested_input_tokens": len(prompt_ids),
        "actual_input_tokens": actual_prompt_tokens,
        "requested_output_tokens": output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "streamed_token_ids": streamed_token_ids or None,
        "generated_text": "".join(output_fragments),
        "http_status": status,
        "response_headers": response_headers,
        "error": error,
        "fallback": None,
    }


def parse_slots(args: argparse.Namespace) -> list[Slot]:
    slots: list[Slot] = []
    if args.endpoint:
        for index in range(args.concurrency):
            slots.append(Slot(index=index, target=args.target, base_url=args.endpoint))
        return slots
    index = 0
    for spec in args.worker:
        try:
            target, base_url, capacity_text = spec.split("=", 2)
            capacity = int(capacity_text)
        except Exception as exc:
            raise ValueError(
                f"invalid --worker {spec!r}; expected NAME=URL=CAPACITY"
            ) from exc
        for _ in range(capacity):
            slots.append(Slot(index=index, target=target, base_url=base_url))
            index += 1
    if not slots:
        raise ValueError("at least one endpoint or worker is required")
    return slots


def main(argv=None, *, tokenizer=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    started_anchor = clock_anchor()
    parser = argparse.ArgumentParser()
    endpoint_group = parser.add_mutually_exclusive_group(required=True)
    endpoint_group.add_argument("--endpoint")
    endpoint_group.add_argument("--worker", action="append")
    parser.add_argument("--target", default="pdd")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--warmup-per-endpoint", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--run-dir", "--out-dir", dest="run_dir", type=Path, required=True
    )
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if min(args.input_tokens, args.output_tokens, args.requests, args.concurrency) <= 0:
        raise ValueError("token lengths, requests, and concurrency must be positive")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        tokenizer = make_tokenizer(args.tokenizer)
    prompts = [
        make_prompt(tokenizer, args.input_tokens, index, args.seed)
        for index in range(args.requests)
    ]
    slots = parse_slots(args)

    unique_endpoints: list[Slot] = []
    seen = set()
    for slot in slots:
        key = (slot.target, slot.base_url)
        if key not in seen:
            seen.add(key)
            unique_endpoints.append(slot)
    if args.endpoint and len(slots) >= 2:
        unique_endpoints = slots[:2]
    warmup_rows = []
    for repeat in range(args.warmup_per_endpoint):
        for slot in unique_endpoints:
            prompt = make_prompt(
                tokenizer, min(args.input_tokens, 256), -repeat - 1, args.seed
            )
            now = time.monotonic()
            warmup_rows.append(
                issue_request(
                    slot=slot,
                    request_index=-(len(warmup_rows) + 1),
                    prompt_ids=prompt,
                    scheduled_arrival=now,
                    model=args.model,
                    output_tokens=min(args.output_tokens, 16),
                    seed=args.seed,
                    timeout=args.timeout,
                    workload=args.workload + "-warmup",
                )
            )
    with (args.run_dir / "warmup_requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in warmup_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not all(request_succeeded(row) for row in warmup_rows):
        raise RuntimeError("warmup failed; see warmup_requests.jsonl")
    if args.warmup_only:
        return

    arrival = time.monotonic() + 0.25
    time.sleep(max(0.0, arrival - time.monotonic()))
    rows: list[dict] = []
    next_request = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as executor:
        active: dict[concurrent.futures.Future, Slot] = {}

        def submit(slot: Slot, request_index: int) -> None:
            future = executor.submit(
                issue_request,
                slot=slot,
                request_index=request_index,
                prompt_ids=prompts[request_index],
                scheduled_arrival=arrival,
                model=args.model,
                output_tokens=args.output_tokens,
                seed=args.seed,
                timeout=args.timeout,
                workload=args.workload,
            )
            active[future] = slot

        for slot in slots:
            if next_request >= args.requests:
                break
            submit(slot, next_request)
            next_request += 1
        while active:
            done, _ = concurrent.futures.wait(
                active, return_when=concurrent.futures.FIRST_COMPLETED
            )
            completed = []
            for future in done:
                slot = active.pop(future)
                row = future.result()
                completed.append((row["completed_monotonic"], slot.index, slot, row))
            for _, _, slot, row in sorted(completed):
                rows.append(row)
                if next_request < args.requests:
                    submit(slot, next_request)
                    next_request += 1

    rows.sort(key=lambda row: row["request_index"])
    with (args.run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (args.run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "tokenizer": args.tokenizer,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "requests": args.requests,
                "seed": args.seed,
                "warmup_per_endpoint": args.warmup_per_endpoint,
                "slots": [slot.__dict__ for slot in slots],
                "prefix_reuse": "low_unique_prompts",
                "generation": {"temperature": 0, "top_p": 1, "ignore_eos": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "argv": argv,
        "slots": [slot.__dict__ for slot in slots],
        "arrival": arrival,
        "clock_anchor_start": started_anchor,
        "clock_anchor_end": clock_anchor(),
        "workload_args": vars(args),
    }
    (args.run_dir / "loadgen_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if not all(request_succeeded(row) for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
