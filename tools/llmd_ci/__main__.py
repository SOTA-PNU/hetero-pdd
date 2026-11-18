"""Public command line used by workflows and local checkouts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from . import checks, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan", help="select affected experiments")
    p.add_argument("--base")
    p.add_argument("--head", required=True)
    p.add_argument("--comparison", choices=["direct", "merge-base"], default="direct")
    p.add_argument("--suite", default="ci/llmd/suite.yaml")
    p.add_argument("--manual", action="store_true")
    p.add_argument("--experiment")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--github-output", type=Path)
    p = commands.add_parser(
        "checks", help="run CPU or official-process checks, or rebuild binaries"
    )
    p.add_argument("--kind", choices=["unit", "build", "process"], required=True)
    p.add_argument("--bin-dir", type=Path)
    p.add_argument(
        "--plan", type=Path, help="require the planned HEAD and runtime source"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p = commands.add_parser(
        "image-inputs", help="fingerprint image inputs in the trusted build checkout"
    )
    p.add_argument("--role", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser(
        "image-record",
        help="record platform manifest references produced by the trusted build job",
    )
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument(
        "--image",
        metavar="ROLE=REPOSITORY@sha256:DIGEST",
        required=True,
        action="append",
    )
    p.add_argument("--index", metavar="ROLE=sha256:DIGEST", action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run", help="execute a planned reference and target")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    for name in ("collect", "cleanup"):
        commands.add_parser(name).add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "checks" and args.kind != "unit" and not args.bin_dir:
        parser.error("--bin-dir is required for process and build")
    if args.command == "plan" and bool(args.experiment) != args.manual:
        parser.error("--manual and --experiment must be supplied together")
    root = Path(__file__).resolve().parents[2]
    rc = 0
    try:
        if args.command == "plan":
            output = plan.plan(
                args.base,
                args.head,
                args.suite,
                args.manual,
                args.experiment,
                args.comparison,
                root=root,
            )
            plan.write_outputs(output, args.output, args.github_output)
            rc = int(bool(output["blocked"]))
        elif args.command == "checks":
            output = checks.run_checks(
                root, args.kind, args.output_dir, args.bin_dir, args.plan
            )
            rc = int(output["status"] != "passed")
        elif args.command == "image-inputs":
            output = dict(
                schema_version=1,
                inputs={r: checks.image_inputs(root, r) for r in args.role},
            )
            checks.save(args.output, output)
        elif args.command == "image-record":
            captured = json.loads(args.inputs.read_text())
            images = dict(i.split("=", 1) for i in args.image)
            indexes = dict(i.split("=", 1) for i in args.index)
            if (
                captured.get("schema_version") != 1
                or images.keys() != captured["inputs"].keys()
                or len(images) != len(args.image)
                or not indexes.keys() <= images.keys()
            ):
                raise ValueError("captured inputs and built image roles must agree")
            output = dict(schema_version=1, images={})
            for role, image in images.items():
                name, _, digest = image.partition("@")
                if not name or not checks.DIGEST.fullmatch(digest):
                    raise ValueError(f"{role}: immutable platform manifest required")
                fingerprint = checks.image_inputs(root, role)
                if fingerprint != captured["inputs"][role]:
                    raise ValueError(f"{role}: build inputs changed since capture")
                entry = output["images"][role] = dict(
                    image=image, inputs_sha256=fingerprint
                )
                if role in indexes:
                    if not checks.DIGEST.fullmatch(indexes[role]):
                        raise ValueError(f"{role}: invalid index digest")
                    entry["index_digest"] = indexes[role]
            checks.save(args.output, output)
        else:
            from . import results, runner

            if args.command == "run":
                runner.run(args.plan, args.experiment, args.output_dir)
                output = results.write_results(args.output_dir)
                rc = int(output["status"] != "passed")
            else:
                fn = runner.collect if args.command == "collect" else runner.cleanup
                journal = fn(args.run_dir)
                output = results.write_results(args.run_dir)
                field = "collection" if args.command == "collect" else "cleanup"
                rc = int(
                    any(
                        p[field]["status"] == "failed"
                        for p in journal["phases"].values()
                    )
                )
        print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        yaml.YAMLError,
    ) as exc:
        print(json.dumps(dict(status="failed", error=str(exc)), ensure_ascii=True))
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
