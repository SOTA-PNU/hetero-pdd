"""Select CI experiments affected by experiment/profile edits or a manual dispatch."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from llmd.config import read_yaml


def select(root: Path, changed: list[str], manual: str | None = None) -> list[str]:
    directory = root / "experiments"
    if manual:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\.ya?ml", manual):
            raise ValueError("Enter a YAML filename from experiments/")
        file = directory / manual
        if not file.is_file():
            raise ValueError(f"{file} does not exist")
        return [str(file.relative_to(root))]
    paths = {(root / path).resolve() for path in changed}
    selected = []
    for file in sorted(directory.iterdir()):
        if file.suffix not in {".yml", ".yaml"}:
            continue
        cfg = read_yaml(file)
        bench = cfg.get("benchmark", {})
        if bench.get("ci", True) is False:
            continue
        profiles = [cfg["profile"], *bench.get("comparisons", {}).values()]
        sources = {file.resolve(), *((directory / p).resolve() for p in profiles)}
        if paths & sources:
            selected.append(str(file.relative_to(root)))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual")
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    changed = []
    if not args.manual:
        files = subprocess.check_output(
            ["git", "diff", "--name-only", "--no-renames", "-z", args.base, args.head]
        )
        changed = [name.decode() for name in files.split(b"\0") if name]
    print(json.dumps(select(Path.cwd(), changed, args.manual)))


if __name__ == "__main__":
    main()
