"""Exercise independent GPU answers and YAML residency/arm order without hardware."""

import hashlib
import json
import socket
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llmd.engines import MODEL_IDENTITY
from tools.hetero.profiler import benchmark_answers as answers
from tools.hetero.profiler import experiment_ci as ci
from tools.hetero.profiler import resident
from tools.hetero.profiler import service_suite as suite


def recorded(work, route, output, *, revision="first"):
    folder = work / "profile"
    folder.mkdir(parents=True)
    rid = work.name + "-0"
    model = {k: revision if k == "revision" else k for k in MODEL_IDENTITY}
    state = dict(
        model=model,
        endpoints=[dict(id="x", engine=dict(dtype="float16", kv_dtype="float16"))],
    )
    (work / "run.json").write_text(json.dumps(state))
    tokens = [1, 2]
    cfg = dict(
        input_tokens=2,
        output_tokens=3,
        requests=1,
        seed=7,
        generation={"temperature": 0},
        prompts_sha256=hashlib.sha256(json.dumps([tokens]).encode()).hexdigest(),
    )
    (folder / "config.json").write_text(json.dumps(cfg))
    (folder / "prompts.jsonl").write_text(
        json.dumps(dict(request_index=0, prompt_ids=tokens)) + "\n"
    )
    (folder / "requests.jsonl").write_text(
        json.dumps(
            dict(
                request_id=rid,
                request_index=0,
                streamed_token_ids=output,
                http_status=200,
                error=None,
                actual_input_tokens=2,
                actual_output_tokens=len(output),
            )
        )
        + "\n"
    )
    (folder / "route_evidence.json").write_text(
        json.dumps(
            {"requests": [dict(request_id=rid, target="x", executed_path=route)]}
        )
    )


class ServiceTests(unittest.TestCase):
    def test_gateway_listeners_skip_active_and_ephemeral_ports(self):
        with socket.socket() as active:
            active.bind(("127.0.0.1", 0))
            busy = active.getsockname()[1]
            with (
                patch.object(resident.secrets, "randbelow", return_value=busy - 20000),
                patch.object(Path, "read_text", return_value=f"{busy + 1} {busy + 5}"),
            ):
                ports = resident.isolated_ports({"ports": {"gateway": 0, "epp": 0}})
        self.assertEqual(len(set(ports.values())), 2)
        self.assertTrue(all(p not in range(busy, busy + 6) for p in ports.values()))

    def test_external_exact_length_gpu_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recorded(root / "gpu", "gpu_full", [3, 4, 5])
            recorded(root / "pdd", "pdd", [3, 8, 9])
            key = answers.gpu_answer_key(root / "gpu")
            check = answers.compare(root / "pdd", key)
            self.assertFalse(check["correct"])
            self.assertEqual(check["mismatched_request_indices"], [0])
            row = check["requests"][0]
            self.assertEqual(
                (
                    row["first_mismatch_index"],
                    row["first_expected_token"],
                    row["first_observed_token"],
                ),
                (1, 4, 8),
            )
            self.assertEqual(row["executed_path"], "pdd")
            with self.assertRaisesRegex(ValueError, "verified GPU"):
                answers.gpu_answer_key(root / "pdd")
            self.assertFalse(answers.compare(root / "gpu", key)["correct"])
            recorded(root / "same", "pdd", [3, 4, 5])
            self.assertTrue(answers.compare(root / "same", key)["correct"])
            recorded(root / "short", "gpu_full", [3, 4])
            with self.assertRaisesRegex(ValueError, "complete"):
                answers.gpu_answer_key(root / "short")
            recorded(root / "model", "pdd", [3, 4, 5], revision="different")
            self.assertFalse(answers.compare(root / "model", key)["correct"])

    def test_alternation_reuses_matching_engines_and_separates_answers(self):
        measurements = []
        starts = []

        @contextmanager
        def boot(experiment, work):
            work.mkdir(parents=True)
            (work / "run.json").write_text('{"model":{"tokenizer_path":"unused"}}')
            starts.append(experiment)
            yield work

        def measure(live, cfg, work, label, tokenizer):
            work.mkdir(parents=True)
            measurements.append(
                (work.parent.parent.name, label, cfg["benchmark"]["input_tokens"])
            )
            return {}

        spec = ci.ROOT / "experiments/suites/ca22-import.yaml"
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(suite, "managed_engine", side_effect=boot),
            patch.object(suite, "measure", side_effect=measure),
            patch.object(suite, "PromptTokenizer"),
            patch.object(answers, "gpu_answer_key"),
            patch.object(
                ci, "summarize_runs", return_value={"errors": [], "runs": []}
            ) as aggregate,
            patch.object(ci, "benchmark_lines", return_value=[]),
        ):
            self.assertEqual(suite.execute_suite(spec, Path(tmp)), [])
            self.assertEqual(len(starts), 7)
            self.assertEqual([r[1] for r in measurements[:2]], ["gpu-reference"] * 2)
            self.assertEqual(
                [r[1] for r in measurements[2::2]],
                [
                    "ordinary",
                    "coalesced",
                    "canonical",
                    "canonical",
                    "ordinary",
                    "coalesced",
                    "coalesced",
                    "canonical",
                    "ordinary",
                ],
            )
            self.assertEqual({r[2] for r in measurements}, {128, 512})
            for call in aggregate.call_args_list:
                self.assertEqual(
                    call.kwargs["gpu_reference"], call.args[0] / "gpu-reference"
                )
