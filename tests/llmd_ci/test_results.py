import json
import platform
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from llmd.engines import MODEL_IDENTITY
from tools.llmd_ci import checks, results


class ResultsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.phases = {}
        self.ref = str(self.out / "reference/verification.json")
        self.reports = {}
        for phase in ("reference", "target"):
            ref = phase == "reference"
            (self.out / phase).mkdir()
            self.phases[phase] = dict(
                run_id=phase,
                status="completed",
                start_attempted=True,
                collection={"status": "collected"},
                cleanup={"status": "cleaned"},
                commands=[
                    dict(
                        action="verify",
                        returncode=0,
                        argv=[] if ref else ["--reference", self.ref],
                    )
                ],
            )
            self.reports[phase] = dict(
                schema_version=2,
                run_id=phase,
                status="smoke_passed" if ref else "verified",
                smoke="passed",
                transfer="not_applicable",
                gpu_only=True,
                requests=[dict(path="full", prompt_ids=[1], token_ids=[7])],
                model={k: k for k in MODEL_IDENTITY},
                conditions={"max_tokens": 1},
                correctness="not_evaluated" if ref else "passed",
                reference=None if ref else self.ref,
            )
        self.journal = dict(experiment="one", status="completed", phases=self.phases)

    def verdict(self):
        for name, report in self.reports.items():
            checks.save(self.out / name / "verification.json", report)
        return results.summarize(self.out, self.journal)

    def test_full_requires_real_verifier_and_disjoint_gpu_reference(self):
        self.assertEqual("passed", self.verdict()["status"])
        self.assertEqual("not_run", self.verdict()["overlap"])
        self.reports["target"]["conditions"]["max_tokens"] += 1
        self.assertNotEqual("passed", self.verdict()["correctness"])
        self.reports["target"]["conditions"]["max_tokens"] -= 1
        self.reports["target"]["run_id"] = "reference"
        self.assertEqual("failed", self.verdict()["status"])

    def test_smoke_ready_and_exit_codes_do_not_promote_correctness(self):
        for status in ("smoke_passed", "ready", "not_run"):
            with self.subTest(status=status):
                self.reports["target"]["status"] = status
                self.assertNotEqual("passed", self.verdict()["correctness"])
        self.reports["target"]["status"] = "verified"
        self.phases["target"]["commands"][0]["returncode"] = 1
        self.assertEqual("failed", self.verdict()["status"])
        self.phases["target"]["commands"].clear()
        self.assertEqual("failed", self.verdict()["status"])

    def test_aggregate_distinct_from_per_request_and_overlap(self):
        target = self.reports["target"]
        target["requests"][0]["path"] = "pdd"
        target["transfer"] = "aggregate_observed"
        self.assertEqual("failed", self.verdict()["status"])
        checks.save(
            self.out / "plan.json",
            {
                "targets": [
                    {"id": "one", "settings": {"transfer_evidence": "aggregate"}}
                ]
            },
        )
        self.assertEqual("aggregate_observed", self.verdict()["transfer"])
        self.assertEqual("passed", self.verdict()["status"])
        target["transfer"] = "passed"
        self.assertEqual("failed", self.verdict()["status"])
        target["requests"][0]["worker_import"] = {"imported_blocks": 1}
        self.assertEqual("passed", self.verdict()["status"])
        self.assertEqual("not_run", self.verdict()["overlap"])

    def test_cleanup_cannot_erase_failure_and_is_required(self):
        self.journal["status"] = "cancelled"
        self.assertEqual("cancelled", self.verdict()["status"])
        self.journal["status"] = "completed"
        self.phases["target"]["cleanup"]["status"] = "failed"
        self.assertEqual("failed", self.verdict()["status"])

    def test_inventory_excludes_caches_symlinks_and_secret_values(self):
        phase = self.out / "target"
        (phase / "offload").mkdir()
        (phase / "offload/model.log").write_text("private")
        (phase / "kubeconfig").write_text("private")
        (phase / "epp.log").symlink_to(phase / "kubeconfig")
        checks.save(
            phase / "run.json",
            {"engine_env": {"OTEL_EXPORTER_OTLP_HEADERS": "private"}},
        )
        self.reports["target"]["requests"][0]["headers"] = {
            "x-hetero-path": "full",
            "Authorization": "private",
        }
        results.publish(self.out, self.verdict())
        with tarfile.open(self.out / "artifacts.tar.gz") as archive:
            self.assertFalse(
                any(
                    "offload" in n or "kubeconfig" in n or n.endswith("epp.log")
                    for n in archive.getnames()
                )
            )
            self.assertNotIn(b"private", archive.extractfile("target/run.json").read())
            uploaded = json.load(archive.extractfile("target/verification.json"))
            headers = uploaded["requests"][0]["headers"]
            self.assertEqual(headers["x-hetero-path"], "full")
            self.assertNotIn("private", str(headers))
            manifest = json.load(archive.extractfile("artifacts.json"))
            self.assertTrue(
                all(f["path"] in archive.getnames() for f in manifest["files"])
            )


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for path in (
            "llmd/serving/build.sh",
            "llmd/serving/images.sh",
            "llmd/serving/patches/up.patch",
            "llmd/epp/go.mod",
            "llmd/epp/go.sum",
            "llmd/epp/cmd/main.go",
        ):
            p = self.root / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(path)

    def test_engine_source_fingerprint_is_independent_of_deployment_digest(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        before = checks.image_inputs(self.root, "engine")
        site = self.root / "ci/llmd/site.yaml"
        site.parent.mkdir(parents=True)
        site.write_text("images: {engine: registry/engine@sha256:digest}")
        self.assertEqual(before, checks.image_inputs(self.root, "engine"))
        (self.root / "llmd/epp/cmd/main.go").write_text("different build input")
        self.assertNotEqual(before, checks.image_inputs(self.root, "engine"))

    def test_cpu_job_rejects_other_checkout_before_commands(self):
        root = Path(__file__).resolve().parents[2]
        proof = self.root / "different-plan.json"
        checks.save(
            proof,
            dict(schema_version=1, head_sha="0" * 40, source={"sha256": "0" * 64}),
        )
        result = checks.run_checks(root, "unit", self.root / "unit", plan_path=proof)
        self.assertEqual(result["status"], "failed")
        self.assertIn("differs from plan", result["error"])
        self.assertEqual(result["commands"], [])

    def test_modified_binary_and_source_rejected(self):
        dest = self.root / "bin"
        dest.mkdir()
        hashes = {}
        for name in checks.BINARIES:
            p = dest / name
            p.write_bytes(name.encode())
            p.chmod(0o755)
            hashes[name] = checks.file_sha(p)
        checks.save(
            dest / checks.BUILD_RECORD,
            dict(
                schema_version=1,
                platform=platform.machine(),
                inputs_sha256=checks.build_inputs(self.root),
                binaries=hashes,
            ),
        )
        self.assertEqual(checks.verify_binaries(self.root, dest), hashes)
        (dest / "envoy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "envoy"):
            checks.verify_binaries(self.root, dest)
        (self.root / "llmd/epp/cmd/main.go").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checkout"):
            checks.verify_binaries(self.root, dest)

    def test_build_record_requires_current_inputs_and_pinned_repository(self):
        wanted = {"epp": "registry.site:5000/ours/epp:v1"}
        image = "registry.site:5000/ours/epp@sha256:" + "1" * 64
        proof = self.root / "proof.json"
        checks.save(
            proof,
            dict(
                schema_version=1,
                images={
                    "epp": {
                        "image": image,
                        "inputs_sha256": checks.image_inputs(self.root, "epp"),
                    }
                },
            ),
        )
        self.assertEqual(
            {"epp": image},
            checks.verify_image_record(self.root, proof, wanted, ["epp"]),
        )
        with self.assertRaisesRegex(ValueError, "repository"):
            checks.verify_image_record(
                self.root, proof, {"epp": "foreign/epp:v1"}, ["epp"]
            )
        with self.assertRaisesRegex(ValueError, "digest"):
            checks.verify_image_record(
                self.root, proof, {"epp": image.replace("1", "2")}, ["epp"]
            )
        (self.root / "llmd/serving/patches/up.patch").write_text("new")
        with self.assertRaisesRegex(ValueError, "inputs"):
            checks.verify_image_record(self.root, proof, wanted, ["epp"])
