# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.hetero import orchestrate as harness


class OrchestrateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_python_path_precedence(self):
        cli, env = (self.directory / name for name in ("cli", "env"))
        cases = [(cli, str(env), cli), (None, str(env), env), (None, "", None)]
        for explicit, variable, wanted in cases:
            with self.subTest(wanted), patch.dict(os.environ, CUDA_PY=variable):
                resolve = partial(harness.python_path, explicit, "CUDA_PY")
                if wanted is None:
                    self.assertRaises(ValueError, resolve)
                else:
                    self.assertEqual(resolve(), wanted)

    def test_role_options(self):
        cases = [
            ({"decode0": {"bad-key": "1"}}, {}),
            ({"decode0": {"GOOD": "line\nbreak"}}, {}),
            ({}, {"decode0": ["--port", "1"]}),
            ({"missing": {"GOOD": "1"}}, {}),
        ]
        args = SimpleNamespace(configuration="pdd")
        for role_env, role_args in cases:
            args.role_env_json, args.role_args_json = role_env, role_args
            with self.subTest(env=role_env, argv=role_args):
                self.assertRaises(ValueError, harness.validate_role_options, args)
        args.role_env_json = {"decode0": {"TEST_TAG": "ok"}}
        args.role_args_json = {"decode0": ["--profiler-config", "{}"]}
        harness.validate_role_options(args)
        for role in ("decode0", "decode1"):
            env, delta, argv = {"BASE": "keep"}, dict[str, str](), ["serve"]
            harness.apply_role_options(args, role, env, delta, argv)
            self.assertEqual(delta, args.role_env_json.get(role, {}))
            self.assertEqual(env, dict(BASE="keep", **delta))
            self.assertEqual(argv, ["serve", *args.role_args_json.get(role, [])])

    def test_clean_repository(self):
        repo = self.directory
        git = partial(subprocess.check_output, cwd=repo)
        git(["git", "init", "--quiet"])
        git(["git", "config", "user.name", "Fixture"])
        git(["git", "config", "user.email", "fixture@example.invalid"])
        tracked = repo / "tracked"
        tracked.write_text("initial")
        git(["git", "add", "tracked"])
        git(["git", "commit", "-m", "fixture"])
        (repo / ".git/info/exclude").write_text("local-only\n")
        (repo / "local-only").write_text("ignored")
        original = harness.run_checked
        versions = {"cuda": "0.22.0", "rbln": "0.22.0+cpu"}

        def checked(argv):
            return versions.get(argv[0]) or original(argv)

        with patch.object(harness, "run_checked", side_effect=checked):
            check = partial(
                harness.verify_environment, repo, Path("cuda"), Path("rbln")
            )
            self.assertTrue(check()["commit"])
            self.assertRaisesRegex(RuntimeError, "compiler mismatch", check, "other")
            tracked.write_text("modified")
            self.assertRaisesRegex(RuntimeError, "dirty", check)
