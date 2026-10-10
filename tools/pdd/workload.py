# SPDX-License-Identifier: Apache-2.0
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
"""The original fixed PDD requests, shared by warmup and measurement steps.

The shell entry points validate and lock the saved run before invoking this file
with its recorded CUDA Python and environment. No third-party imports are needed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from runner import atomic_text


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def percentile_nearest_rank(values: list[float], percentile: float) -> str:
    if not values:
        return "NA"
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return f"{sorted(values)[index]:.9f}"


class Workload:
    def __init__(self, environment: dict[str, str]):
        self.result_dir = Path(environment["RESULT_DIR"])
        self.run_id = environment["RUN_ID"]
        self.workload_id = f"profile-workload-{uuid.uuid4()}"
        self.prefill_port = int(environment["PREFILL_PORT"])
        self.decode_port = int(environment["DECODE_PORT"])
        self.proxy_port = int(environment["PROXY_PORT"])
        self.model = environment.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
        self.warmup_requests = int(environment.get("WARMUP_REQUESTS", "3"))
        self.measure_requests = int(environment.get("MEASURE_REQUESTS", "20"))
        self.concurrency = int(environment.get("CONCURRENCY", "1"))
        self.max_tokens = int(environment.get("MAX_TOKENS", "128"))
        self.prompt_mode = environment.get("PROMPT_MODE", "long")
        if self.warmup_requests < 0:
            raise ValueError("WARMUP_REQUESTS must be a nonnegative integer")
        for name in ("measure_requests", "concurrency", "max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name.upper()} must be a positive integer")
        if self.prompt_mode not in ("short", "long"):
            raise ValueError("PROMPT_MODE must be short or long")

        self.prompt_id = 2 if self.prompt_mode == "short" else 4
        prompt = (
            "Complete the sequence: 1, 1, 2, 3, 5,"
            if self.prompt_id == 2
            else (
                "A clear glass of water rests on the wooden table. " * 29
                + "Continue with the next three words only:"
            )
        )
        expected_prompt_sha256 = {
            2: "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116",
            4: "57242b91f80632b4eec84b172bfe170fa52b0d33940af0b59a9a43be51889921",
        }[self.prompt_id]
        self.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != expected_prompt_sha256:
            raise ValueError(f"embedded prompt {self.prompt_id} checksum mismatch")
        self.body_bytes = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": self.max_tokens,
            "logprobs": 5,
            "n": 1,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "stream": False,
        }).encode("utf-8")
        self.results_path = self.result_dir / "workload_results.jsonl"
        self.summary_path = self.result_dir / "workload_summary.txt"
        self.phases_path = self.result_dir / "workload_phases.json"
        self.phases = {"run_id": self.run_id, "phases": {}}
        if self.phases_path.exists():
            self.phases = json.loads(self.phases_path.read_text(encoding="utf-8"))
            if self.phases.get("run_id") != self.run_id:
                raise ValueError("workload_phases.json run ID mismatch")
            if not isinstance(self.phases.get("phases"), dict):
                raise ValueError("invalid workload_phases.json phases")

    def check_ready(self) -> None:
        for port, endpoint, label in (
            (self.prefill_port, "/health", "CUDA prefiller"),
            (self.decode_port, "/health", "RBLN decoder"),
            (self.proxy_port, "/healthcheck", "PDD proxy"),
        ):
            url = f"http://127.0.0.1:{port}{endpoint}"
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if not 200 <= response.status < 300:
                        raise RuntimeError(f"HTTP {response.status}")
            except Exception as exc:
                raise RuntimeError(
                    f"{label} readiness failed at {url}: {exc}"
                ) from exc

    def issue_request(self, phase: str, index: int) -> dict:
        request_id = f"{self.workload_id}-{phase}-{index}"
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}/v1/completions",
            data=self.body_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
                "X-Request-Id": request_id,
                "X-Prompt-Sha256": self.prompt_sha256,
            },
            method="POST",
        )
        started = time.monotonic()
        status = 0
        payload: object = None
        error = None
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                status = response.status
                raw = response.read()
            payload = json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            raw = exc.read()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"non_json_body": raw.decode("utf-8", errors="replace")}
            error = f"HTTP {status}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        ended = time.monotonic()

        token_ids = None
        generated_text = None
        finish_reason = None
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                generated_text = choices[0].get("text")
                finish_reason = choices[0].get("finish_reason")
                candidate = choices[0].get("token_ids")
                if isinstance(candidate, list) and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in candidate
                ):
                    token_ids = candidate
        if 200 <= status < 300 and (token_ids is None or not token_ids):
            error = "HTTP response omitted choices[0].token_ids"

        return {
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "phase": phase,
            "request_index": index,
            "request_id": request_id,
            "prompt_mode": self.prompt_mode,
            "prompt_id": self.prompt_id,
            "started_monotonic": started,
            "ended_monotonic": ended,
            "latency_seconds": ended - started,
            "http_status": status,
            "generated_token_count": len(token_ids) if token_ids is not None else None,
            "generated_token_ids": token_ids,
            "generated_text": generated_text,
            "finish_reason": finish_reason,
            "success": error is None and 200 <= status < 300,
            "error": error,
        }

    def save_phases(self) -> None:
        atomic_text(self.phases_path, json.dumps(self.phases, indent=2) + "\n")

    def run_phase(self, phase: str, count: int) -> list[dict]:
        rows = []
        # Exclusive creation also protects against accidental direct module reruns.
        with (self.result_dir / f"{phase}_results.jsonl").open(
            "x", encoding="utf-8"
        ) as phase_handle, self.results_path.open("a", encoding="utf-8") as combined:
            record = {
                "status": "RUNNING",
                "started_at": utc_now(),
                "ended_at": None,
                "elapsed_seconds": None,
                "requested": count,
                "completed": 0,
                "succeeded": 0,
                "failed": 0,
                "reason": None,
                "results_path": str(phase_handle.name),
                "config": {
                    "MODEL_NAME": self.model,
                    "WARMUP_REQUESTS": self.warmup_requests,
                    "MEASURE_REQUESTS": self.measure_requests,
                    "CONCURRENCY": self.concurrency,
                    "MAX_TOKENS": self.max_tokens,
                    "PROMPT_MODE": self.prompt_mode,
                    "PROMPT_ID": self.prompt_id,
                    "PROMPT_SHA256": self.prompt_sha256,
                    "TEMPERATURE": 0,
                    "TOP_P": 1,
                    "SEED": 0,
                    "LOGPROBS": 5,
                    "N": 1,
                    "STREAM": False,
                },
            }
            self.phases["phases"][phase] = record
            self.save_phases()
            started = time.monotonic()
            try:
                if count:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self.concurrency, count)
                    ) as executor:
                        futures = [
                            executor.submit(self.issue_request, phase, index)
                            for index in range(count)
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            row = future.result()
                            rows.append(row)
                            line = json.dumps(row, sort_keys=True) + "\n"
                            phase_handle.write(line)
                            phase_handle.flush()
                            combined.write(line)
                            combined.flush()
                record["status"] = (
                    "SUCCESS" if all(row["success"] for row in rows) else "FAILED"
                )
                if record["status"] == "FAILED":
                    record["reason"] = "one or more requests failed"
            except BaseException as exc:
                record["status"] = "FAILED"
                record["reason"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.update(
                    ended_at=utc_now(),
                    elapsed_seconds=time.monotonic() - started,
                    completed=len(rows),
                    succeeded=sum(row["success"] for row in rows),
                    failed=sum(not row["success"] for row in rows),
                )
                self.save_phases()
        return sorted(rows, key=lambda row: row["request_index"])

    def write_legacy_summary(self, started_at: str, delay: int) -> None:
        phases = self.phases["phases"]
        warmup = phases.get("warmup", {})
        measure = phases.get("measure", {})
        measured = []
        if measure:
            with (self.result_dir / "measure_results.jsonl").open(
                encoding="utf-8"
            ) as handle:
                measured = [json.loads(line) for line in handle if line.strip()]
        passed = [row for row in measured if row["success"]]
        latencies = [float(row["latency_seconds"]) for row in passed]
        if warmup.get("status") == "FAILED":
            status = "WARMUP_FAILED"
        elif measure:
            status = "PASS" if measure["status"] == "SUCCESS" else "FAILED"
        else:
            status = "WARMUP_COMPLETE"
        summary = {
            "RESULT_DIR": self.result_dir,
            "RUN_ID": self.run_id,
            "STARTED_UTC": started_at,
            "STATUS": status,
            "PROMPT_MODE": self.prompt_mode,
            "PROMPT_ID": self.prompt_id,
            "WARMUP_REQUESTS": self.warmup_requests,
            "MEASURE_REQUESTS": self.measure_requests,
            "CONCURRENCY": self.concurrency,
            "MAX_TOKENS": self.max_tokens,
            "READY_DELAY_SECONDS": delay,
            "WARMUP_SUCCEEDED": warmup.get("succeeded", 0),
            "WARMUP_FAILED": warmup.get("failed", 0),
            "MEASURE_SUCCEEDED": len(passed),
            "MEASURE_FAILED": sum(not row["success"] for row in measured),
            "MEASURE_WALL_SECONDS": (
                f"{measure['elapsed_seconds']:.9f}" if measure else "NA"
            ),
            "GENERATED_TOKEN_TOTAL": sum(
                row["generated_token_count"] for row in passed
            ),
            "LATENCY_MEAN_SECONDS": (
                f"{statistics.mean(latencies):.9f}" if latencies else "NA"
            ),
            "LATENCY_P50_SECONDS": percentile_nearest_rank(latencies, 0.50),
            "LATENCY_P95_SECONDS": percentile_nearest_rank(latencies, 0.95),
        }
        atomic_text(
            self.summary_path,
            "".join(f"{key}={value}\n" for key, value in summary.items()),
        )

    def run(self, phase: str) -> int:
        selected = ("warmup", "measure") if phase == "all" else (phase,)
        for name in selected:
            if (
                name in self.phases["phases"]
                or (self.result_dir / f"{name}_results.jsonl").exists()
            ):
                raise ValueError(
                    f"{name} was already attempted; use a new RESULT_DIR to rerun"
                )
        if phase == "measure" and self.phases["phases"].get("warmup", {}).get(
            "status"
        ) != "SUCCESS":
            raise ValueError("measure requires a successfully completed warmup step")
        self.check_ready()
        started_at = utc_now()
        for name in selected:
            if name == "measure" and phase == "all":
                print("READY_FOR_PROFILING", flush=True)
                time.sleep(5)
            count = self.warmup_requests if name == "warmup" else self.measure_requests
            rows = self.run_phase(name, count)
            self.write_legacy_summary(started_at, 5 if phase == "all" else 0)
            failures = sum(not row["success"] for row in rows)
            if failures:
                print(
                    f"WORKLOAD_FAIL: {failures} {name} request(s) failed",
                    file=sys.stderr,
                )
                return 1
        print("WARMUP_COMPLETE" if phase == "warmup" else "WORKLOAD_COMPLETE")
        print(f"RESULTS_FILE={self.results_path}")
        print(f"SUMMARY_FILE={self.summary_path}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("warmup", "measure", "all"), required=True)
    args = parser.parse_args()
    try:
        return Workload(dict(os.environ)).run(args.phase)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"WORKLOAD_FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
