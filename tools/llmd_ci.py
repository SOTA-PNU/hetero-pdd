r"""Run existing llm-d checks or one Kubernetes reference/target pair.

From the checkout root (RUN_DIR must be new or empty):
  python tools/llmd_ci.py check
  python tools/llmd_ci.py run --experiment experiments/qwen2-fixed.yaml \
      --reference-experiment experiments/qwen2-baseline.yaml --site SITE.yaml \
      --namespace NAMESPACE --context CONTEXT --output-dir RUN_DIR
  python tools/llmd_ci.py cleanup --output-dir RUN_DIR --context CONTEXT

CI wiring: keep the existing PDD regression track. Configure workflow paths
to run check on code/config changes. For related experiment/model/profile/site
YAML pushes or manual dispatch, pass explicit target/reference paths from
inputs or a small matrix. Register new combinations there. The workflow owns
the push trigger, runner, pdd-test concurrency shared with legacy CI, job
timeout, always() cleanup and artifact upload. Adjust the existing workflow's
path filters to avoid duplicate YAML-only runs.

CPU: Python with PyYAML, Git, the Go version in llmd/epp/go.mod on PATH and
a C compiler for -race. Kubernetes: supply kubectl, one job-private KUBECONFIG
with CONTEXT already selected, an approved site and namespace, device plugins,
devices, PVCs, prepared images/models and the matching runner tokenizer and
Python dependencies (tokenizers or transformers). Use the existing build step
when image source changes; experiment YAML can reuse the prepared images.

Keep the job's kubeconfig in place through cleanup. Upload commands.log and
reference/ and target/ native artifacts, particularly verification.json,
deployment.json and kube-run.json (configured images and observed Pod imageIDs).
The output directory also holds small ownership records for cleanup on this run.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLMD = [sys.executable, "-m", "llmd"]
KUBE = [sys.executable, "-m", "llmd.kube"]
ERRORS = (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError)


def command(argv, *, cwd=ROOT, env=None, log=None):
    line = "+ " + shlex.join([str(arg) for arg in argv])
    print(line, file=sys.stderr, flush=True)
    if log is not None:
        print(line, file=log, flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT if log else None,
        start_new_session=True,
    )
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        # kubectl port-forward is a grandchild of the native verifier.
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def check():
    experiments = sorted((ROOT / "experiments").glob("*.yaml"))
    if not experiments:
        raise ValueError("no experiment YAML files found")
    for experiment in experiments:
        command([*LLMD, "validate", experiment])
    modules = ["llmd.test_engines", "llmd.test_runtime"]
    command([sys.executable, "-m", "unittest", *modules])
    for options in (["./..."], ["-race", "./plugins"]):
        command(["go", "test", "-mod=readonly", *options], cwd=ROOT / "llmd/epp")


def kube_env():
    path = os.environ.get("KUBECONFIG", "")
    if not path or os.pathsep in path or not Path(path).is_file():
        raise ValueError("set KUBECONFIG to one existing job-private file")
    return {**os.environ, "KUBECONFIG": str(Path(path).resolve())}


def check_context(expected, env, log):
    argv = ["kubectl", "config", "current-context"]
    current = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"+ kubectl config current-context\n{current}", file=log, flush=True)
    if current != expected:
        raise ValueError(f"expected Kubernetes context {expected!r}, found {current!r}")


def kube(action, work, context, env, log, *options):
    check_context(context, env, log)
    command([*KUBE, action, "--work-dir", work, *options], env=env, log=log)


def read_json(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def stop(work, context, log, env=None):
    state_file = work / "kube-run.json"
    if work.is_symlink() or state_file.is_symlink():
        raise ValueError(f"unexpected symlink in {work}")
    if not state_file.exists():  # Native start saves ownership before creating objects.
        return
    deploy_file = work / "deployment.json"
    owner_file = work.parent / (work.name + ".owner")
    if deploy_file.is_symlink() or owner_file.is_symlink():
        raise ValueError(f"unexpected ownership symlink for {work}")
    state = read_json(state_file)
    deploy = read_json(deploy_file)
    expected = [context, deploy["namespace"], deploy["run_id"]]
    if (
        owner_file.read_text().splitlines() != expected
        or state.get("namespace") != deploy["namespace"]
        or state.get("run_id") != deploy["run_id"]
        or not re.fullmatch(r"[0-9a-f]{32}", deploy["run_id"])
        or not state.get("namespace_uid")
    ):
        raise ValueError(f"context, namespace or ownership mismatch for {work}")
    if state.get("status") != "stopped":
        kube("stop", work, context, env or kube_env(), log)


def cleanup(out, context, log, env=None):
    errors = []
    for phase in ("target", "reference"):
        try:
            stop(out / phase, context, log, env)
        except ERRORS as exc:
            errors.append(f"{phase}: {exc}")
    if errors:
        raise RuntimeError("cleanup failed: " + " | ".join(errors))


def verdict(work, run_id, reference):
    report = read_json(work / "verification.json")
    fields = {
        key: report.get(key) for key in ("status", "smoke", "transfer", "correctness")
    }
    print(f"{work.name}: {json.dumps(fields)}", flush=True)
    if (
        report.get("schema_version") != 2
        or report.get("run_id") != run_id
        or fields["smoke"] != "passed"
        or not report.get("requests")
    ):
        raise ValueError(f"{work.name}: missing successful native verification")
    if reference is None:
        if (
            fields["status"] != "smoke_passed"
            or report.get("gpu_only") is not True
            or fields["correctness"] != "not_evaluated"
            or fields["transfer"] != "not_applicable"
            or any(row.get("path") != "full" for row in report["requests"])
        ):
            raise ValueError("reference did not verify a GPU-only full route")
    elif (
        fields["status"] != "verified"
        or fields["correctness"] != "passed"
        or fields["transfer"] not in {"passed", "aggregate_observed", "not_applicable"}
        or report.get("reference") != str(reference)
    ):
        raise ValueError(
            "target did not pass native verification against this reference"
        )


def phase(work, experiment, args, env, log, reference=None):
    render = [*LLMD, "render", experiment, "--target", "kubernetes"]
    options = ["--namespace", args.namespace, "--site", args.site, "--work-dir", work]
    command([*render, *options], env=env, log=log)
    deploy = read_json(work / "deployment.json")
    run_id = deploy["run_id"]
    if deploy["namespace"] != args.namespace or not re.fullmatch(
        r"[0-9a-f]{32}", run_id
    ):
        raise ValueError("rendered namespace or run ID is invalid")
    if reference is None:
        engines = deploy["instances"]
        if not all(e["kind"] == "gpu" for e in engines) or not any(
            "full" in e["roles"] for e in engines
        ):
            raise ValueError("reference experiment must provide a GPU-only full route")
    with (work.parent / (work.name + ".owner")).open("x") as owner:
        print(args.context, args.namespace, run_id, sep="\n", file=owner)
    kube("start", work, args.context, env, log)
    options = ["--reference", reference] if reference else []
    try:
        kube("verify", work, args.context, env, log, *options)
    except subprocess.CalledProcessError:
        # A failed verifier can still leave its smoke/transfer/correctness fields.
        with suppress(OSError, ValueError, KeyError):
            verdict(work, run_id, reference)
        raise
    verdict(work, run_id, reference)
    stop(work, args.context, log, env)


def run(args, out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"run requires a new or empty directory: {out}")
    env = None
    with (out / "commands.log").open("x", buffering=1) as log:
        try:
            command(["git", "rev-parse", "HEAD"], log=log)
            command(["git", "status", "--short"], log=log)
            for experiment in (args.reference_experiment, args.experiment):
                command([*LLMD, "validate", experiment], log=log)
            env = kube_env()
            check_context(args.context, env, log)
            phase(out / "reference", args.reference_experiment, args, env, log)
            ref = out / "reference" / "verification.json"
            phase(out / "target", args.experiment, args, env, log, reference=ref)
        except (Exception, KeyboardInterrupt) as original:
            print(f"run failed: {original!r}", file=log)
            try:
                cleanup(out, args.context, log, env)
            except ERRORS as exc:
                print(str(exc), file=log)
                print(str(exc), file=sys.stderr)
            raise


def interrupt(signum, _frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check", help="validate example experiments and run CPU tests")
    execute = sub.add_parser("run", help="run a GPU full reference then the target")
    for option, help_text in (
        ("--experiment", "target experiment YAML"),
        ("--reference-experiment", "compatible independent GPU full experiment YAML"),
        ("--site", "approved Kubernetes site YAML"),
    ):
        execute.add_argument(option, type=Path, required=True, help=help_text)
    execute.add_argument("--namespace", required=True, help="approved namespace")
    clean = sub.add_parser("cleanup", help="stop this run's remaining deployments")
    for subcommand in (execute, clean):
        subcommand.add_argument("--context", required=True, help="expected context")
        subcommand.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        if args.action == "check":
            check()
        elif args.action == "run":
            args.site = args.site.resolve(strict=True)
            args.experiment = args.experiment.resolve(strict=True)
            args.reference_experiment = args.reference_experiment.resolve(strict=True)
            run(args, args.output_dir.resolve())
        else:
            out = args.output_dir.resolve()
            if any((out / name).exists() for name in ("reference", "target")):
                with (out / "commands.log").open("a") as log:
                    cleanup(out, args.context, log)
    except KeyboardInterrupt as exc:
        print(f"interrupted: {exc}", file=sys.stderr)
        return 130
    except ERRORS as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
