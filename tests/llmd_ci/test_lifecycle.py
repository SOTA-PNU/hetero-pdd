"""Exercise the CI supervisor with subprocess exits, signals and persisted state."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llmd import config
from llmd.verify import reference_output
from tools.llmd_ci import runner

NATIVE = """
import json, os, signal, sys, time, uuid
from pathlib import Path

mode, action, *args = sys.argv[1:]
log = Path(os.environ["TEST_JOURNAL"])


def publish(path, value):
    path.write_text(json.dumps(value))


if mode == "helper":
    if action == "local":
        publish(
            Path(args[-1]), {"binaries": {"epp": "1", "envoy": "2", "pd-sidecar": "3"}}
        )
    elif action == "collect-local":
        work = Path(args[0])
        phase = work.name
        with log.open("a") as f:
            print(phase, "collect", file=f)
        if os.getenv("TEST_COLLECT_FAILURE"):
            sys.exit(19)
        publish(work / "collection.json", {"status": "collected"})
    sys.exit(0)
assert mode == "llmd"
work = Path(args[args.index("--work-dir") + 1])
phase = work.name
with log.open("a") as f:
    print(phase, action, file=f)
state = work / "run.json"
owned = work / "live"
if action == "start":
    assert not any(
        p.exists()
        for p in (work.parent / "reference/live", work.parent / "target/live")
    )
    native = dict(
        run_id=uuid.uuid4().hex,
        status="ready",
        work_dir=str(work),
        processes={},
        control_plane=[{"argv": ["/bin/epp"]}, {"argv": ["/bin/envoy"]}],
        binary_sha256={"epp": "1", "envoy": "2"},
    )
    if os.getenv("TEST_BINARY_CHANGE"):
        native["binary_sha256"]["envoy"] = "changed"
    publish(state, native)
    owned.write_text(native["run_id"])
if action == "verify":
    native = json.loads(state.read_text())
    if phase == "target":
        f = args[args.index("--reference") + 1]
        ref = json.loads(Path(f).read_text())
        assert ref["run_id"] != native["run_id"]
    publish(
        work / "verification.json",
        {
            "run_id": native["run_id"],
            "smoke": "passed",
            "gpu_only": phase == "reference",
            "requests": [{"path": "full", "prompt_ids": [42]}],
            "conditions": {},
            "model": {},
            "schema_version": 2,
            "status": "smoke_passed" if phase == "reference" else "verified",
            "correctness": "not_evaluated" if phase == "reference" else "passed",
        },
    )
if os.getenv("TEST_FAIL_ACTION") == phase + ":" + action:
    print("original failure in " + action, file=sys.stderr)
    sys.exit(13)
if os.getenv("TEST_DELAY_ACTION") == phase + ":" + action:
    time.sleep(10)
if os.getenv("TEST_SIGNAL_ACTION") == phase + ":" + action:
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(10)
if action == "stop":
    assert owned.read_text() == json.loads(state.read_text())["run_id"]
    owned.unlink()
"""


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        fake = self.base / "fake.py"
        fake.write_text(NATIVE)
        self.journal = self.base / "calls"
        self.out = self.base / "run"
        self.plan = self.base / "plan.json"
        self.target = dict(
            id="chosen",
            path="experiments/qwen2-fixed.yaml",
            environment="unit",
            target="local",
            reference=dict(profile="llmd/epp/profiles/baseline.yaml", endpoint="gpu0"),
            settings=dict(timeout=60, collection_timeout=4, cleanup_timeout=4),
        )
        self.options = {"TEST_JOURNAL": str(self.journal)}
        for p in (
            patch.object(runner, "PYTHON", [sys.executable, str(fake)]),
            patch.object(runner, "HELPER", [sys.executable, str(fake), "helper"]),
            patch("tools.llmd_ci.plan.verify_checkout", return_value=runner.ROOT),
        ):
            p.start()
            self.addCleanup(p.stop)

    def execute(self, **env):
        self.plan.write_text(
            json.dumps(dict(targets=[self.target], source={"dirty": True}))
        )
        with patch.dict(os.environ, self.options | env):
            return runner.run(self.plan, "chosen", self.out)

    def test_distinct_reference_ends_before_target_and_replays_do_not_redeploy(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed", result)
        a, b = result["phases"].values()
        self.assertNotEqual(a["run_id"], b["run_id"])
        actions = self.journal.read_text().splitlines()
        self.assertEqual(
            actions,
            [
                f"{phase} {step}"
                for phase in ("reference", "target")
                for step in ("start", "ready", "verify", "collect", "stop")
            ],
        )
        before = (runner.ROOT / self.target["path"]).read_text()
        target = config.load(self.out / "configs/target.yaml")
        reference = config.load(self.out / "configs/reference.yaml")
        self.assertEqual(target["model"], reference["model"])
        self.assertEqual(reference["verify"], dict(target["verify"], paths=["full"]))
        self.assertEqual(reference["endpoints"][0]["roles"], ["full"])
        self.assertEqual(
            reference["endpoints"][0]["engine"], target["endpoints"][0]["engine"]
        )
        self.assertEqual((runner.ROOT / self.target["path"]).read_text(), before)
        with patch.dict(os.environ, self.options):
            self.assertEqual(runner.collect(self.out)["status"], "completed")
            self.assertEqual(runner.cleanup(self.out)["status"], "completed")
        self.assertEqual(self.journal.read_text().splitlines(), actions)
        with self.assertRaisesRegex(ValueError, "fresh"):
            runner.run(self.plan, "chosen", self.out)

    def test_start_failure_preserved_when_diagnostics_fail(self):
        result = self.execute(
            TEST_FAIL_ACTION="reference:start", TEST_COLLECT_FAILURE="yes"
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("original failure in start", result["error"])
        self.assertEqual(
            self.journal.read_text().splitlines(),
            ["reference start", "reference collect", "reference stop"],
        )
        ref = result["phases"]["reference"]
        self.assertEqual(ref["collection"]["status"], "failed")
        self.assertEqual(ref["cleanup"]["status"], "cleaned")
        self.assertFalse(result["phases"]["target"]["start_attempted"])

    def test_deadline_covers_whole_native_call_and_allows_teardown(self):
        self.target["settings"]["timeout"] = 1
        result = self.execute(TEST_DELAY_ACTION="reference:ready")
        self.assertEqual(result["status"], "failed")
        self.assertIn("deadline", result["error"])
        self.assertEqual(
            self.journal.read_text().splitlines()[-2:],
            ["reference collect", "reference stop"],
        )
        self.assertFalse((self.out / "reference/live").exists())

    def test_sigterm_keeps_non_success_and_collects_before_cleanup(self):
        previous = signal.getsignal(signal.SIGTERM)
        result = self.execute(TEST_SIGNAL_ACTION="reference:ready")
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["signal"], "SIGTERM")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertEqual(
            self.journal.read_text().splitlines()[-2:],
            ["reference collect", "reference stop"],
        )

    def test_stop_failure_retries_only_same_native_owner(self):
        result = self.execute(TEST_FAIL_ACTION="reference:stop")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["phases"]["reference"]["cleanup"]["status"], "failed")
        native = self.out / "reference/run.json"
        original = native.read_text()
        native.write_text(json.dumps(dict(json.loads(original), run_id="a" * 32)))
        count = self.journal.read_text().count("reference stop")
        with patch.dict(os.environ, self.options):
            rejected = runner.cleanup(self.out)
            self.assertIn(
                "ID changed", rejected["phases"]["reference"]["cleanup"]["reason"]
            )
            self.assertEqual(self.journal.read_text().count("reference stop"), count)
            self.assertTrue((self.out / "reference/live").exists())
            native.write_text(original)
            final = runner.cleanup(self.out)
            self.assertEqual(final["status"], "failed")
            self.assertEqual(
                final["phases"]["reference"]["cleanup"]["status"], "cleaned"
            )
            runner.cleanup(self.out)
        self.assertEqual(self.journal.read_text().count("reference stop"), count + 1)

    def test_sigkill_recovery_collects_and_stops_with_stored_owner(self):
        supervisor = """
import sys
from pathlib import Path
from tools.llmd_ci import plan, runner
runner.PYTHON = [sys.executable, sys.argv[1]]
runner.HELPER = [*runner.PYTHON, "helper"]
plan.verify_checkout = lambda plan, root: root
runner.run(sys.argv[2], "chosen", sys.argv[3])
"""
        self.plan.write_text(json.dumps(dict(targets=[self.target], source={})))
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                supervisor,
                self.base / "fake.py",
                self.plan,
                self.out,
            ],
            cwd=runner.ROOT,
            env=os.environ | self.options | {"TEST_DELAY_ACTION": "reference:ready"},
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not self.journal.exists()
                or "reference ready" not in self.journal.read_text()
            ):
                self.assertIsNone(proc.poll())
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            proc.kill()
            proc.wait(timeout=5)
            with patch.dict(os.environ, self.options):
                while True:
                    try:
                        result = runner.cleanup(self.out)
                        break
                    except RuntimeError as exc:
                        self.assertIn("another CI lifecycle command", str(exc))
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
            self.assertEqual(result["status"], "failed")
            self.assertFalse((self.out / "reference/live").exists())
            self.assertEqual(
                self.journal.read_text().splitlines()[-2:],
                ["reference collect", "reference stop"],
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_unschedulable_or_missing_images_are_blocked_by_recorded_pod_evidence(self):
        state = {
            "target": "kubernetes",
            "phases": {
                "target": {"status": "failed", "commands": [{"action": "start"}]}
            },
        }
        path = self.out / "target/diagnostics/start-abc/pods.json"
        path.parent.mkdir(parents=True)
        pod = dict(name="pdd-test", states={}, conditions=[])
        path.write_text(json.dumps([pod]))
        self.assertIsNone(runner.cluster_blocker(self.out, state))
        pod["states"] = {
            "engine": {
                "waiting": {"reason": "ErrImagePull", "message": "digest unavailable"}
            }
        }
        path.write_text(json.dumps([pod]))
        self.assertIn("digest unavailable", runner.cluster_blocker(self.out, state))
        pod.update(
            states={},
            conditions=[
                {
                    "reason": "Unschedulable",
                    "status": "False",
                    "message": "insufficient accelerator",
                }
            ],
        )
        path.write_text(json.dumps([pod]))
        self.assertIn(
            "insufficient accelerator", runner.cluster_blocker(self.out, state)
        )
        state["phases"]["target"]["commands"].append({"action": "verify"})
        self.assertIsNone(runner.cluster_blocker(self.out, state))

    def test_blocked_run_keeps_planner_reason_and_environment(self):
        self.plan.write_text(
            json.dumps(
                {
                    "targets": [],
                    "blocked": [
                        {
                            "id": "chosen",
                            "environment": "lab",
                            "reason": "missing approved site",
                        }
                    ],
                }
            )
        )
        result = runner.run(self.plan, "chosen", self.out)
        self.assertEqual(result["environment"], "lab")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("missing approved site", result["error"])

    def test_running_binary_digest_checked_before_ready(self):
        result = self.execute(TEST_BINARY_CHANGE="yes")
        self.assertIn("running binary hashes", result["error"])
        self.assertEqual(
            self.journal.read_text().splitlines(),
            ["reference start", "reference collect", "reference stop"],
        )

    def test_native_verifier_rejects_reference_conditions(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed")
        path = self.out / "reference/verification.json"
        with self.assertRaisesRegex(ValueError, "conditions differ"):
            reference_output(path, {}, "different-run", [[42]], {"temperature": 1})

    def test_kubernetes_context_is_required_before_allocation(self):
        self.target.update(target="kubernetes", site="ci/llmd/site.yaml")
        self.target["settings"].update(
            context_env="TEST_LLMD_CI_CONTEXT", namespace_env="TEST_LLMD_CI_NAMESPACE"
        )
        result = self.execute(TEST_LLMD_CI_CONTEXT="")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("TEST_LLMD_CI_CONTEXT", result["error"])
        self.assertFalse(result["phases"]["reference"]["start_attempted"])
        self.assertFalse(self.journal.exists())
