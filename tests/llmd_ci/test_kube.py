"""Exercise the native Kubernetes lifecycle with real manifests and a simulated API."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from llmd import cluster, config, kube
from llmd.__main__ import resolved
from tools.llmd_ci.kube_diagnostics import MAX_BYTES, preflight

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/pdd@" + DIGEST


class KubernetesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        cfg = config.load(ROOT / "experiments/fixed-pdd.yaml")
        cfg["images"] = dict.fromkeys(cluster.IMAGES, IMAGE)
        cfg["cluster"] = dict(
            peerGpuVisibility="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            nodeSelector={"approved": "yes"},
            runtimeClassName="nvidia",
        )
        self.run_id = "02a4ef6f8731a85b3327d9a28321c452"
        self.deploy = cluster.render(
            cfg, resolved(cfg, self.run_id), self.work, "approved"
        )
        self.owned = kube.read_manifest(self.work, self.deploy)
        self.pods = []
        for doc in self.owned:
            if doc["kind"] != "Deployment":
                continue
            tmpl = doc["spec"]["template"]
            name = doc["metadata"]["name"] + "-abc"
            statuses = [
                dict(name=c["name"], ready=True, imageID="docker-pullable://" + IMAGE)
                for c in tmpl["spec"]["containers"]
            ]
            self.pods.append(
                dict(
                    metadata=dict(
                        name=name, uid="uid-" + name, labels=tmpl["metadata"]["labels"]
                    ),
                    spec=dict(tmpl["spec"], nodeName="node"),
                    status=dict(
                        podIP="127.0.0.1", phase="Running", containerStatuses=statuses
                    ),
                )
            )
        self.model = {}
        self.trace = []
        self.break_wait = False
        self.break_logs = False
        self.break_delete = False
        self.uid = "uid-approved"

    def command(self, context, ns, args, budget, *, limit=None):
        self.assertGreater(budget, 0)
        self.assertEqual((context, ns), ("test-context", "approved"))
        self.trace.append(args)
        if args[:2] in (("get", "ns"), ("get", "namespace")):
            return json.dumps(
                dict(metadata=dict(uid=self.uid), status=dict(phase="Active"))
            )
        if args[:2] == ("get", "pvc"):
            return json.dumps(dict(metadata={}, status=dict(phase="Bound")))
        if args[:2] == ("get", "runtimeclass"):
            return "{}"
        if args[:2] == ("get", "nodes"):
            return json.dumps(
                dict(
                    items=[
                        dict(
                            metadata=dict(name="node", labels={"approved": "yes"}),
                            spec={},
                            status=dict(
                                allocatable={
                                    e["resource"]: "1" for e in self.deploy["instances"]
                                },
                                conditions=[dict(type="Ready", status="True")],
                            ),
                        )
                    ]
                )
            )
        if args[:2] == ("get", "pods"):
            return json.dumps(dict(items=self.pods))
        if args[0] == "exec" and "cat" in args:
            return json.dumps(
                dict(
                    model=self.model,
                    runtime={},
                    cache={},
                    endpoint=dict(engine_id=args[1]),
                )
            )
        if args[0] == "logs":
            if self.break_logs:
                raise RuntimeError("logs unavailable")
            return "a" * (MAX_BYTES + 1)
        if args[0] == "wait" and self.break_wait:
            raise RuntimeError("original readiness error")
        if args[0] == "delete":
            if self.break_delete:
                raise RuntimeError("API delete temporarily denied")
            self.assertIn("pdd.sota.dev/run=" + self.run_id, args)
            self.assertIn("--field-selector", args)
            name = args[args.index("--field-selector") + 1].split("=", 1)[1]
            self.assertIn(
                (args[1], name),
                [(d["kind"].lower(), d["metadata"]["name"]) for d in self.owned],
            )
        return ""

    def action(self, name, **kwargs):
        with patch.object(kube, "command", self.command):
            return kube.run(self.work, name, 10, context="test-context", **kwargs)

    def stored(self, filename):
        return json.loads((self.work / filename).read_text())

    def test_metrics_command_bounds_large_and_stalled_responses(self):
        producer = (
            "import sys; sys.stdout.buffer.write(b'x' * 2000000); sys.stdout.flush()"
        )
        self.assertEqual(
            len(kube.limited_command([sys.executable, "-c", producer], 2, 100)), 100
        )
        with self.assertRaises(TimeoutError):
            kube.limited_command(
                [sys.executable, "-c", "import time; time.sleep(5)"], 0.05, 100
            )

    def test_preflight_and_pinned_image_match(self):
        with patch.object(kube, "command", self.command):
            metadata = preflight(self.work, "test-context", 5)
        self.assertEqual(metadata["namespace_uid"], self.uid)
        self.assertEqual(set(metadata["claims"]), {"pdd-models", "pdd-compiled"})
        self.action("start", require_image_digests=True, namespace_uid=self.uid)
        state = self.stored("kube-run.json")
        self.assertTrue(
            all(
                r["matched"]
                for p in state["pods"]
                for r in p["image_comparison"].values()
            )
        )
        self.pods[0]["status"]["containerStatuses"][0]["imageID"] = (
            "registry.example/pdd:local"
        )
        with self.assertRaisesRegex(ValueError, "imageID"):
            self.action("ready")
        pod = self.stored("kube-run.json")["pods"][0]
        self.assertFalse(
            all(info["matched"] for info in pod["image_comparison"].values())
        )
        self.assertTrue(self.stored("diagnostics/index.json")["truncated"])

    def test_preflight_rejects_unbounded_placement_visibility_and_capacity(self):
        path = self.work / "kubernetes.yaml"
        original = path.read_text()
        for before, after, error in (
            ('"approved": "yes"', '"unavailable": "node"', "gateway placement"),
            ("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "all", "producer GPU UUIDs"),
            (
                "registry.example/pdd@" + DIGEST,
                "registry.example/pdd:local",
                "immutable",
            ),
        ):
            with self.subTest(error=error):
                self.assertIn(before, original)
                path.write_text(original.replace(before, after))
                with (
                    patch.object(kube, "command", self.command),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    preflight(self.work, "test-context", 2)
        path.write_text(original)
        render_path = self.work / "deployment.json"
        deploy = self.stored("deployment.json")
        deploy["instances"][0]["devices"].append(1)
        render_path.write_text(json.dumps(deploy))
        with (
            patch.object(kube, "command", self.command),
            self.assertRaisesRegex(ValueError, "capacity"),
        ):
            preflight(self.work, "test-context", 2)
        self.assertNotIn("create", [c[0] for c in self.trace])

    def test_broken_start_preserves_original_and_retries_cleanup(self):
        self.break_wait = self.break_delete = self.break_logs = True
        other = dict(metadata=dict(name="foreign", labels={}), spec={}, status={})
        self.pods.append(other)
        with self.assertRaisesRegex(RuntimeError, "original readiness error"):
            self.action("start", require_image_digests=True)
        verbs = [c[0] for c in self.trace]
        self.assertLess(verbs.index("logs"), verbs.index("delete"))
        state = self.stored("kube-run.json")
        self.assertEqual(state["error"], "original readiness error")
        self.assertTrue(state["cleanup"]["remaining"])
        self.assertTrue(self.stored("diagnostics/index.json")["missing"])
        self.assertFalse(any("foreign" in c for c in self.trace))
        self.break_delete = False
        self.assertEqual(self.action("stop"), "stopped")
        after_stop = len(self.trace)
        self.assertEqual(self.action("stop"), "stopped")
        self.assertEqual(self.action("collect"), "partial")
        self.assertEqual(len(self.trace), after_stop)
        self.assertEqual(
            self.stored("kube-run.json")["error"], "original readiness error"
        )

    def test_no_mutation_for_wrong_context_or_namespace_uid(self):
        with self.assertRaisesRegex(ValueError, "UID"):
            self.action("start", namespace_uid="wrong")
        self.assertNotIn("create", [c[0] for c in self.trace])
        self.action("start")
        before = len(self.trace)
        with self.assertRaisesRegex(ValueError, "context"):
            kube.run(self.work, "stop", 1, context="different")
        self.assertEqual(len(self.trace), before)
        self.uid = "recreated"
        with self.assertRaisesRegex(ValueError, "UID"):
            self.action("stop")
        self.assertNotIn("delete", [c[0] for c in self.trace])

    def test_verify_failing_before_http_request_preserves_snapshot(self):
        self.action("start")
        with (
            patch.object(
                kube, "port_forward", side_effect=RuntimeError("port forward failed")
            ),
            self.assertRaisesRegex(RuntimeError, "port forward failed"),
        ):
            self.action("verify")
        self.assertEqual(self.stored("diagnostics/index.json")["reason"], "verify")
        self.assertFalse((self.work / "verification.json").exists())
        self.action("collect")
        snapshots = list((self.work / "diagnostics").glob("*/index.json"))
        self.assertEqual(len(snapshots), 2)
        self.action("stop")

    def test_tokenizer_config_and_truncated_proof_fail_verification(self):
        (self.work / "tokenizer.json").write_bytes(b"vocabulary")
        (self.work / "tokenizer_config.json").write_bytes(b"configuration")
        self.model = {
            **self.deploy["model"],
            "tokenizer_sha256": hashlib.sha256(b"vocabulary").hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(b"configuration").hexdigest(),
        }
        self.action("start")
        runner_model = dict(self.model, tokenizer_path=str(self.work))
        with (
            patch.object(
                kube, "port_forward", return_value=nullcontext("http://127.0.0.1:1")
            ),
            patch.object(kube.engines, "model_info", return_value=runner_model),
            patch.object(kube, "verify", side_effect=lambda *a, collect: collect()),
            patch.object(kube, "PROOF_BYTES", 100),
        ):
            (self.work / "tokenizer_config.json").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "tokenizer_config_sha256"):
                self.action("verify")
            (self.work / "tokenizer_config.json").write_bytes(b"configuration")
            with self.assertRaisesRegex(ValueError, "verification evidence truncated"):
                self.action("verify")
        self.assertEqual((self.work / "epp.log").stat().st_size, 100)
        self.assertIn("epp.log", self.stored("kube-run.json")["error"])

    def test_sigterm_real_cli_uses_recorded_context_and_rolls_back(self):
        executable = self.work / "kubectl"
        executable.write_text("""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
d = Path(os.environ['TEST_WORK'])
argv = sys.argv[1:]
assert argv[:4] == ['--context', 'test-context', '-n', 'approved']
verb = argv[4]
with (d / 'trace').open('a') as log:
    print(verb, *argv[5:7], file=log)
if argv[4:6] == ['get', 'ns']:
    print(json.dumps(dict(metadata=dict(uid='uid-approved'))))
elif verb == 'wait':
    (d / 'waiting').touch()
    time.sleep(10)
elif argv[4:6] == ['get', 'pods']:
    print(json.dumps(dict(items=[])))
elif verb == 'port-forward':
    print('Forwarding from 127.0.0.1:1234 -> 8081', flush=True)
    time.sleep(10)
""")
        executable.chmod(0o755)
        env = dict(
            os.environ,
            PATH=str(self.work) + ":" + os.environ["PATH"],
            TEST_WORK=str(self.work),
        )
        with (
            patch.dict(os.environ, env),
            kube.port_forward("test-context", "approved", "pod1", 8081) as address,
        ):
            self.assertEqual(address, "http://127.0.0.1:1234")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "llmd.kube",
                "start",
                "--work-dir",
                str(self.work),
                "--context",
                "test-context",
                "--timeout",
                "5",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
        )
        try:
            deadline = time.monotonic() + 4
            while not (self.work / "waiting").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.work / "waiting").exists())
            proc.send_signal(signal.SIGTERM)
            _, stderr = proc.communicate(timeout=8)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(json.loads(stderr)["status"], "cancelled")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        state = self.stored("kube-run.json")
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["cleanup"]["remaining"], [])
        calls = (self.work / "trace").read_text()
        self.assertLess(calls.index("get pods"), calls.index("delete "))
