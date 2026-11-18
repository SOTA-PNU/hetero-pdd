"""Test planner contracts with real Git histories and relocated checkouts."""

import json
import shutil
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from tools.llmd_ci.plan import plan, verify_checkout, write_outputs

REPO = Path(__file__).resolve().parents[2]
MODEL = "experiments/models/qwen3-0.6b.yaml"
PROFILE = "llmd/epp/profiles/fixed-pdd.yaml"
SITE = "ci/site.yaml.example"
SUITE = "ci/llmd/suite.yaml"


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "planner checkout"
        self.root.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "CI Fixture")
        self.git("config", "user.email", "ci@example.invalid")
        for rel in (
            "experiments/baseline.yaml",
            "experiments/fixed-pdd.yaml",
            "experiments/qwen2-fixed.yaml",
            "experiments/models/qwen2.5-0.5b.yaml",
            MODEL,
            PROFILE,
            "llmd/epp/profiles/baseline.yaml",
            "llmd/serving/envoy.yaml",
            "llmd/serving/upstream/standalone.yaml",
            "experiments/cluster-site.yaml.example",
        ):
            self.write(rel, (REPO / rel).read_text())
        self.dump(
            SITE,
            {
                "cluster": {
                    "modelClaim": "approved",
                    "cacheClaim": "dedicated",
                    "peerGpuVisibility": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                    "nodeSelector": {"kubernetes.io/hostname": "lab-fixture"},
                }
            },
        )
        local = {
            "target": "local",
            "allowed": {"gpu": 1, "npu": 1, "nodes": ["local"]},
            "serving": ["llmd/serving/envoy.yaml"],
            "timeout": 300,
            "collection_timeout": 60,
            "cleanup_timeout": 30,
        }
        self.spec = {
            "version": 1,
            "reference": {"profile": "llmd/epp/profiles/baseline.yaml"},
            "environments": {
                "desk": local,
                "lab": dict(
                    deepcopy(local),
                    target="kubernetes",
                    site=SITE,
                    serving=["llmd/serving/upstream/standalone.yaml"],
                    context_env="APPROVED_CONTEXT",
                    namespace_env="APPROVED_NAMESPACE",
                    image_record_env="APPROVED_IMAGES",
                ),
            },
            "experiments": {
                "base": {"path": "experiments/baseline.yaml", "environment": "desk"},
                "pair": {"path": "experiments/fixed-pdd.yaml", "environment": "lab"},
                "second-model": {
                    "path": "experiments/qwen2-fixed.yaml",
                    "environment": "lab",
                },
            },
        }
        self.dump(SUITE, self.spec)
        self.base = self.commit("base")

    def git(self, *args):
        return (
            subprocess.check_output(
                ["git", *args], cwd=self.root, stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )

    def write(self, path, data):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(data)

    def dump(self, path, data):
        self.write(path, yaml.safe_dump(data))

    def append(self, path):
        with (self.root / path).open("a") as f:
            f.write("\n# change\n")

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-qm", msg)
        return self.git("rev-parse", "HEAD")

    def auto(self, base=None, **kw):
        return plan(
            self.base if base is None else base, "HEAD", SUITE, root=self.root, **kw
        )

    def manual(self, exp):
        return plan(None, "HEAD", SUITE, manual=True, experiment=exp, root=self.root)

    def selected(self, report):
        self.assertFalse(report["blocked"], report["blocked"])
        return [row["id"] for row in report["targets"]]

    def test_single_experiment_then_shared_model_and_profile(self):
        self.append("experiments/baseline.yaml")
        previous = self.commit("one experiment")
        self.assertEqual(self.selected(self.auto()), ["base"])
        self.append(MODEL)
        self.append(PROFILE)
        self.commit("model and policy")
        report = self.auto(previous)
        self.assertCountEqual(self.selected(report), ["base", "pair", "second-model"])
        reasons = {t["id"]: t["reasons"] for t in report["targets"]}
        self.assertEqual(len(reasons["base"]), 1)
        self.assertEqual(len(reasons["pair"]), 2)
        self.assertEqual(len(reasons["second-model"]), 1)
        self.assertEqual(
            report["matrix"]["include"][1], {"experiment": "pair", "environment": "lab"}
        )

    def test_site_and_renderer_inputs_are_environment_specific(self):
        self.append(SITE)
        self.append("experiments/cluster-site.yaml.example")
        changed = self.commit("approved site")
        self.assertCountEqual(self.selected(self.auto()), ["pair", "second-model"])
        self.append("experiments/cluster-site.yaml.example")
        self.commit("unused site")
        self.assertFalse(self.auto(changed)["hardware_requested"])
        base = self.git("rev-parse", "HEAD")
        self.append("llmd/serving/envoy.yaml")
        self.commit("local renderer")
        self.assertEqual(self.selected(self.auto(base)), ["base"])

    def test_registered_add_rename_delete_and_dangling_reference(self):
        model = "experiments/models/new model.yaml"
        profile = "llmd/epp/profiles/new profile.yaml"
        path = "experiments/new exp.yaml"
        self.write(model, (self.root / MODEL).read_text())
        self.write(profile, (self.root / PROFILE).read_text())
        raw = yaml.safe_load((self.root / "experiments/fixed-pdd.yaml").read_text())
        raw.update(model="models/new model.yaml", profile="../" + profile)
        self.dump(path, raw)
        self.spec["experiments"]["space"] = {"path": path, "environment": "desk"}
        self.dump(SUITE, self.spec)
        added = self.commit("new experiment, model and profile")
        (target,) = self.auto()["targets"]
        self.assertEqual(target["id"], "space")
        self.assertTrue({model, profile, path} <= target["inputs"].keys())
        renamed = "experiments/moved exp.yaml"
        self.git("mv", path, renamed)
        self.spec["experiments"]["space"]["path"] = renamed
        self.dump(SUITE, self.spec)
        moved = self.commit("renamed experiment")
        report = self.auto(added)
        self.assertEqual(self.selected(report), ["space"])
        self.assertIn(
            {"status": "R100", "old_path": path, "path": renamed}, report["changes"]
        )
        self.git("rm", profile)
        broken = self.commit("broken shared input")
        report = self.auto(moved)
        self.assertTrue(report["hardware_requested"])
        self.assertEqual(report["matrix"]["include"], [])
        self.assertIn(profile, report["blocked"][0]["reason"])
        self.git("rm", renamed)
        self.commit("deleted experiment")
        report = self.auto(broken)
        self.assertFalse(report["hardware_requested"])
        self.assertIn("deleted", report["excluded"][0]["reason"])
        self.assertFalse(report["blocked"])

    def test_inline_model_unused_yaml_and_code_only(self):
        raw = yaml.safe_load((self.root / "experiments/baseline.yaml").read_text())
        raw["model"] = yaml.safe_load((self.root / MODEL).read_text())
        self.dump("experiments/baseline.yaml", raw)
        base = self.commit("inline model")
        self.append(MODEL)
        after = self.commit("shared model")
        self.assertEqual(self.selected(self.auto(base)), ["pair"])
        self.write("llmd/serving/patches/change.patch", "changed\n")
        self.write("experiments/unregistered.yaml", "apiVersion: example/v1\n")
        self.write(".github/workflows/nothing.yaml", "jobs: {}\n")
        self.commit("code and unrelated examples")
        report = self.auto(after)
        self.assertTrue(report["checks_required"])
        self.assertFalse(report["hardware_requested"])
        self.assertEqual(report["excluded"][0]["reason"], "experiment not registered")

    def test_manual_noop_invalid_base_and_limits(self):
        report = self.auto()
        self.assertFalse(report["hardware_requested"])
        self.assertFalse(report["checks_required"])
        (target,) = self.manual("pair")["targets"]
        self.assertEqual(target["reference"]["endpoint"], "gpu0")
        self.assertEqual(target["site"], SITE)
        self.assertEqual(
            target["requirements"], {"gpu": 1, "npu": 1, "nodes": ["local"]}
        )
        self.assertEqual(target["settings"]["namespace_env"], "APPROVED_NAMESPACE")
        for ref in ("missing", "--help"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.auto(ref)
        with self.assertRaises(ValueError):
            plan(None, "HEAD", SUITE, root=self.root)
        with self.assertRaises(ValueError):
            self.manual("typo")
        self.spec["environments"]["lab"]["allowed"]["npu"] = 0
        self.dump(SUITE, self.spec)
        self.commit("reduce lab capacity")
        report = self.auto()
        self.assertTrue(report["hardware_requested"])
        self.assertEqual(report["targets"], [])
        self.assertEqual(len(report["blocked"]), 2)
        self.assertIn("requires 1 npu", report["blocked"][0]["reason"])

    def test_merge_base_and_checkout_head(self):
        self.git("checkout", "-qb", "feature")
        self.append("experiments/baseline.yaml")
        head = self.commit("feature")
        self.git("checkout", "-q", "main")
        self.append("experiments/qwen2-fixed.yaml")
        side = self.commit("main moved")
        self.git("checkout", "-q", "feature")
        report = self.auto(side, comparison="merge-base")
        self.assertEqual(report["base_sha"], self.base)
        self.assertEqual(report["head_sha"], head)
        self.assertEqual(self.selected(report), ["base"])
        with self.assertRaisesRegex(ValueError, "checkout HEAD"):
            plan(self.base, side, SUITE, root=self.root)

    def test_portable_hashes_and_dirty_source_contents(self):
        other = Path(self.tmp.name) / "execution checkout"
        self.git("clone", "-q", str(self.root), str(other))
        clean = self.manual("pair")
        self.assertFalse(clean["source"]["dirty"])
        self.assertEqual(verify_checkout(clean, other), other)
        filename = "llmd/new_adapter.py"
        self.write(filename, "revision = 1\n")
        dirty = self.manual("pair")
        self.assertTrue(dirty["source"]["dirty"])
        self.assertIn(filename, dirty["source"]["dirty_paths"])
        with self.assertRaisesRegex(ValueError, "source hash"):
            verify_checkout(dirty, other)
        shutil.copyfile(self.root / filename, other / filename)
        self.assertEqual(verify_checkout(dirty, other), other)
        (other / filename).write_text("revision = 2\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            verify_checkout(dirty, other)
        (other / filename).unlink()
        with (other / MODEL).open("a") as f:
            f.write("\n# drift\n")
        with self.assertRaisesRegex(ValueError, "input hash"):
            verify_checkout(clean, other)
        self.commit("different head")
        with self.assertRaisesRegex(ValueError, "checkout HEAD"):
            verify_checkout(clean, self.root)

    def test_portable_files_and_github_outputs(self):
        report = self.manual("pair")
        out, gha = [
            Path(self.tmp.name) / name for name in ("plan.json", "actions output")
        ]
        write_outputs(report, out, gha)
        self.assertEqual(json.loads(out.read_text()), report)
        outputs = dict(row.split("=", 1) for row in gha.read_text().splitlines())
        self.assertEqual(Path(outputs["plan"]), out)
        self.assertEqual(json.loads(outputs["matrix"]), report["matrix"])
        self.assertEqual(outputs["hardware_requested"], "true")
        outside = Path(self.tmp.name) / "outside.yaml"
        outside.write_text("apiVersion: example/v1\n")
        (self.root / SITE).unlink()
        (self.root / SITE).symlink_to(outside)
        self.assertIn("outside checkout", self.manual("pair")["blocked"][0]["reason"])

    def test_explicit_reference_and_placement(self):
        raw = yaml.safe_load((self.root / "experiments/baseline.yaml").read_text())
        (ep,) = raw["endpoints"]
        raw["endpoints"].append(dict(ep, id="second", devices=[1], port=19020))
        self.dump("experiments/baseline.yaml", raw)
        self.spec["environments"]["desk"]["allowed"]["gpu"] = 2
        self.dump(SUITE, self.spec)
        self.assertIn(
            "specify a reference", self.manual("base")["blocked"][0]["reason"]
        )
        self.spec["experiments"]["base"]["reference"] = {"endpoint": "second"}
        self.dump(SUITE, self.spec)
        self.assertEqual(
            self.manual("base")["targets"][0]["reference"]["endpoint"], "second"
        )
        self.spec["environments"]["desk"]["allowed"]["gpu_devices"] = [0]
        self.dump(SUITE, self.spec)
        self.assertIn(
            "outside allowed placement", self.manual("base")["blocked"][0]["reason"]
        )

    def test_disabled_then_enabled_and_invalid_site(self):
        self.spec["experiments"]["pair"]["enabled"] = False
        self.dump(SUITE, self.spec)
        disabled = self.commit("disable pair")
        report = self.auto()
        self.assertFalse(report["hardware_requested"])
        self.assertEqual(report["excluded"][0]["id"], "pair")
        self.assertTrue(self.manual("pair")["blocked"])
        self.spec["experiments"]["pair"]["enabled"] = True
        self.dump(SUITE, self.spec)
        self.commit("enable pair")
        self.assertEqual(self.selected(self.auto(disabled)), ["pair"])
        self.dump(SITE, {"cluster": {"peerGpuVisibility": "all"}})
        self.assertIn("nodeSelector", self.manual("pair")["blocked"][0]["reason"])
        self.dump(
            SITE,
            {
                "cluster": {
                    "nodeSelector": {"node": "approved"},
                    "peerGpuVisibility": "all",
                }
            },
        )
        self.assertIn("producer GPU", self.manual("pair")["blocked"][0]["reason"])

    def test_dirty_runtime_dependency_and_reference_tampering(self):
        report = self.manual("pair")
        self.write("tools/hetero/loadgen.py", "fixture = True\n")
        with self.assertRaisesRegex(ValueError, "source hash"):
            verify_checkout(report, self.root)
        (self.root / "tools/hetero/loadgen.py").unlink()
        report["targets"][0]["settings"]["timeout"] += 10
        with self.assertRaisesRegex(ValueError, "configuration"):
            verify_checkout(report, self.root)

    def test_github_outputs_distinguish_runnable_and_blocked_in_same_plan(self):
        # Both local and cluster registrations reference this shared model.
        self.append(MODEL)
        (self.root / SITE).unlink()
        self.commit("new model and removed site")
        report = self.auto()
        output = Path(self.tmp.name) / "outputs"
        write_outputs(report, Path(self.tmp.name) / "plan.json", output)
        values = dict(row.split("=", 1) for row in output.read_text().splitlines())
        self.assertEqual(values["has_runnable"], "true")
        self.assertEqual(values["blocked"], "true")
        self.assertEqual(
            json.loads(values["matrix"])["include"],
            [{"experiment": "base", "environment": "desk"}],
        )
        output.unlink()
        write_outputs(self.manual("pair"), Path(self.tmp.name) / "manual.json", output)
        self.assertIn("has_runnable=false", output.read_text().splitlines())
