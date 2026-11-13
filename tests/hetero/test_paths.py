# SPDX-License-Identifier: Apache-2.0
import re
import unittest
from pathlib import Path


class PathTests(unittest.TestCase):
    def test_no_absolute_filesystem_literals(self):
        repo = Path(__file__).resolve().parents[2]
        pattern = re.compile(
            r"/(?:home|root|Users|mnt|tmp|opt|var|proc)/|(?<![A-Za-z])[A-Za-z]:[\\/]"
        )
        for directory in (repo / "tools" / "hetero", repo / "tests" / "hetero"):
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".json", ".jsonl", ".md"):
                    with self.subTest(path=path.relative_to(repo)):
                        self.assertIsNone(pattern.search(path.read_text()), str(path))
