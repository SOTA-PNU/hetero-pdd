# SPDX-License-Identifier: Apache-2.0
"""Run one service session and its workloads; see docs/hetero/profiler.md."""

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
import zlib
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from urllib.request import Request, urlopen

if __package__ is None:
    sys.path[0] = str(Path(__file__).resolve().parents[3])

from tools.hetero import orchestrate
from tools.hetero.profiler import log
from tools.hetero.profiler.expected import expected_ids, service_config, wait_for_spans
from tools.hetero.profiler.monitors import Monitors, wait_for_layout
from tools.hetero.profiler.scrape import append_rows, boundary_snapshot, wait_until
from tools.hetero.profiler.tools import capture_links, check_tools

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "tools" / "hetero"
RESERVED = ("service", "monitor", "otel", "rbln_metrics", "torch_traces", "overhead")
RESERVED += ("run.json", "links.json", "trace_wait.json")
FIELDS = ("name", "input_tokens", "output_tokens", "requests", "seed", "warmup")
ROUTING = ("slots", "concurrency")
DEVICES = ("gpu", "npu0", "npu1")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def validate_workloads(workloads, configuration):
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be a nonempty JSON array")
    names = set(RESERVED)
    routing = ROUTING[0] if configuration == "baseline" else ROUTING[1]
    for work in workloads:
        if not isinstance(work, dict) or set(work) != set(FIELDS) | {routing}:
            raise ValueError(f"workload requires {FIELDS} and {routing}")
        name = work["name"]
        if not isinstance(name, str) or name in names or not NAME.match(name):
            raise ValueError(f"invalid, reserved or repeated workload name: {name!r}")
        names.add(name)
        limits = {key: int(key not in ("seed", "warmup")) for key in FIELDS[1:]}
        if routing == "slots":
            slots = work["slots"]
            if not isinstance(slots, dict) or set(slots) != set(DEVICES):
                raise ValueError(f"{name}: slots require gpu/npu0/npu1")
            work = {**work, **slots}
            limits.update(dict.fromkeys(slots, 0))
        else:
            limits["concurrency"] = 1
        if any(type(work[k]) is not int or work[k] < v for k, v in limits.items()):
            raise ValueError(f"{name}: invalid integer; minimum values: {limits}")
        if routing == "slots" and sum(slots.values()) == 0:
            raise ValueError(f"{name}: total slots must be positive")


def flags(**values):
    return [
        part for k, v in values.items() for part in ("--" + k.replace("_", "-"), str(v))
    ]


def role_options(configuration, mode, root, port):
    env, args = {}, {}
    endpoint = f"http://127.0.0.1:{port}/v1/traces"
    if mode in ("trace", "deep"):
        for role in sorted(orchestrate.ROLES[configuration] - {"proxy"}):
            env[role] = dict(
                OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf",
                OTEL_SERVICE_NAME=role,
                OTEL_BSP_SCHEDULE_DELAY="500",
            )
            args[role] = ["--otlp-traces-endpoint", endpoint]
            if role.startswith(("npu", "decode")):
                env[role]["VLLM_RBLN_METRICS"] = "1"
                env[role]["VLLM_RBLN_METRICS_DIR"] = str(root / "rbln_metrics" / role)
                if mode == "deep":
                    env[role]["RBLN_PROFILER"] = "1"
            if mode == "deep":
                path = str(root / "torch_traces" / role)
                profile = dict(profiler="torch", torch_profiler_dir=path)
                args[role] += flags(profiler_config=json.dumps(profile))
    return env, args


def loadgen_args(work, config):
    values = dict(model=config["model_name"], tokenizer=config["model_path"])
    values.update(workload=work["name"], **{k: work[k] for k in FIELDS[1:5]})
    argv = [sys.executable, str(HARNESS / "loadgen.py"), *flags(**values)]
    if config["configuration"] == "baseline":
        for slot in range(max(work["slots"].values())):
            for role in DEVICES:
                if slot < work["slots"][role]:
                    argv += flags(worker=f"{role}={config['endpoints'][role]}=1")
    else:
        target = config["endpoints"]["proxy"]
        argv += flags(endpoint=target, concurrency=work["concurrency"])
    return argv


def command(root, name, argv):
    with (root / "monitor" / f"{name}.log").open("ab") as output:
        return subprocess.run(
            argv, cwd=REPO, check=True, stdout=output, stderr=subprocess.STDOUT
        ).returncode


def profile_control(out, role, url, action):
    row = dict(
        role=role, action=action, mono_ns=time.monotonic_ns(), real_ns=time.time_ns()
    )
    try:
        request = Request(url + "/" + action, data=b"", method="POST")
        # The runner supplies loopback HTTP service URLs from its own config.
        with urlopen(request, timeout=10) as response:  # nosec B310
            row["status"] = response.status
    except Exception as exc:
        row.update(status=getattr(exc, "code", 0), error=str(exc))
        raise
    finally:
        append_rows(out, [row])


def wait_for_traces(root, roles, previous, timeout=120):
    sizes, changed, pending = {}, time.monotonic(), []

    def complete():
        nonlocal sizes, changed, pending
        files = sorted(set(root.rglob("*.gz")) - previous)
        current = {p: p.stat().st_size for p in files}
        if current != sizes:
            sizes, changed = current, time.monotonic()
        pending = [
            f"{role}: no new gzip trace"
            for role in roles
            if not any(p.parent.name == role for p in files)
        ]
        if time.monotonic() - changed < 3:
            pending += [f"{p.relative_to(root)}: size not stable for 3s" for p in files]
        else:
            for path in files:
                try:
                    if not current[path]:
                        raise EOFError("empty gzip trace")
                    with gzip.open(path, "rb") as stream:
                        while stream.read(1024 * 1024):
                            pass
                except (OSError, EOFError, zlib.error) as exc:
                    pending.append(f"{path.relative_to(root)}: {exc}")
        return not pending

    try:
        wait_until(complete, timeout, "torch traces")
    except TimeoutError as exc:
        raise TimeoutError(
            f"torch traces: {timeout}s timeout; {'; '.join(pending)}"
        ) from exc


@contextmanager
def recorded(root, data, name, optional=False):
    row = dict(name=name, start=time.monotonic_ns(), end=None, ok=False, error=None)
    data["steps"].append(row)
    try:
        yield
        row["ok"] = True
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if not optional or not isinstance(exc, Exception):
            raise
    finally:
        row["end"] = time.monotonic_ns()
        try:
            (root / "run.json").write_text(json.dumps(data, indent=2) + "\n")
        except OSError as exc:
            data["record_error"] = str(exc)
            log.log_line("run.json", f"write failed: {exc}")
        log.log_line(name, row["error"] or "done")


def stage_functions(root, data):
    def call(name, function, *values, optional=False):
        with recorded(root, data, name, optional):
            return function(*values)

    def cli(name, argv, optional=False):
        task = partial(command, root, name.replace(":", "-"))
        return call(name, task, argv, optional=optional)

    return call, cli


@contextmanager
def service_session(args, forwarded, service, data):
    root, mode = args.run_dir, args.mode
    (root / "monitor").mkdir(parents=True, exist_ok=True)
    log.RUN_ID = root.name
    monitors, started = Monitors(root), False
    traced = mode in ("trace", "deep")
    service_cmd = [sys.executable, str(HARNESS / "orchestrate.py")]
    service_dir = root / "service"

    call, cli = stage_functions(root, data)
    cleanup = partial(call, optional=True)
    cleanup_cli = partial(cli, optional=True)
    try:
        call("check_tools", check_tools, mode)
        port = call("collector_start", monitors.collector) if traced else None
        env, extra = role_options(args.configuration, mode, root, port)
        for role, values in env.items():
            service.role_env_json.setdefault(role, {}).update(values)
        for role, values in extra.items():
            existing = service.role_args_json.setdefault(role, [])
            if set(existing[::2]) & set(values[::2]):
                raise ValueError(f"{role}: conflicting profiler arguments")
            existing.extend(values)
        options = flags(
            role_env_json=json.dumps(service.role_env_json),
            role_args_json=json.dumps(service.role_args_json),
        )
        cli("service_start", service_cmd + ["start", *forwarded, *options])
        started = True
        config = service_config(root)
        if args.configuration == "pdd" and traced:
            call("host_buffer", wait_for_layout, root)
        endpoints = dict(config["metrics_endpoints"])
        if mode != "none":
            endpoints["all-smi"] = call("allsmi_start", monitors.allsmi)
            cleanup("links", capture_links, 0, root / "links.json")
        yield config, monitors, call, cli, endpoints
    except (Exception, KeyboardInterrupt) as exc:
        data["error"] = f"{type(exc).__name__}: {exc}"
        if not data["steps"] or data["steps"][-1]["ok"]:
            log.log_line("run", data["error"])
        raise
    finally:
        for name in ("scrape", "all-smi"):
            if name in monitors.processes:
                cleanup(name + "_stop", monitors.stop, name)
        if started or (service_dir / "processes.json").exists():
            stop = service_cmd + ["stop", *flags(run_dir=service_dir)]
            cleanup_cli("service_stop", stop)
        if "otelcol" in monitors.processes:
            cleanup("collector_stop", monitors.stop, "otelcol")


@contextmanager
def workload_session(work, config, folder, mode, monitors, call, cli, endpoints):
    cleanup = partial(call, optional=True)
    servers = {r: u for r, u in config["endpoints"].items() if r != "proxy"}
    name = work["name"]
    loadgen, metrics = loadgen_args(work, config), folder / "monitor"
    boundary = partial(
        boundary_snapshot, config["metrics_endpoints"], metrics / "boundaries.jsonl"
    )
    control = partial(profile_control, folder / "profile_control.jsonl")
    warmup = flags(warmup_per_endpoint=work["warmup"])
    warmup += flags(out_dir=folder / "warmup")
    cli(name + ":warmup", loadgen + ["--warmup-only", *warmup])
    profiles = {}
    traces = folder.parent / "torch_traces"
    previous = set(traces.rglob("*.gz")) if mode == "deep" else set()
    try:
        if mode != "none":
            call(
                name + ":scrape_start",
                monitors.scrape,
                endpoints,
                metrics / "metrics.jsonl",
            )
            call(name + ":before", boundary, "before")
        if mode == "deep":
            for role, url in servers.items():
                profiles[role] = url
                action = "start_profile"
                call(f"{name}:{action}:{role}", control, role, url, action)
        measurement = loadgen + flags(warmup_per_endpoint=0, out_dir=folder)
        yield measurement
    finally:
        for role, url in profiles.items():
            action = "stop_profile"
            cleanup(f"{name}:{action}:{role}", control, role, url, action)
        if "scrape" in monitors.processes:
            cleanup(name + ":after", boundary, "after")
            cleanup(name + ":scrape_stop", monitors.stop, "scrape")
        if profiles:
            cleanup(name + ":trace_wait", wait_for_traces, traces, profiles, previous)


def execute(args, forwarded, service, workloads, argv):
    root, mode = args.run_dir, args.mode
    data = dict(argv=argv, workloads=workloads, mode=mode, steps=[])
    data.update(configuration=args.configuration, analysis_status="PENDING")
    traced = mode in ("trace", "deep")
    try:
        with service_session(args, forwarded, service, data) as session:
            config, monitors, call, cli, endpoints = session
            expected = {role: set() for role in config["endpoints"] if role != "proxy"}
            for work in workloads:
                name, folder = work["name"], root / work["name"]
                with workload_session(
                    work, config, folder, mode, monitors, call, cli, endpoints
                ) as measurement:
                    cli(name + ":measure", measurement)
                if traced:
                    text = (folder / "requests.jsonl").read_text()
                    wanted = expected_ids(map(json.loads, text.splitlines()), config)
                    for role, ids in wanted.items():
                        expected[role].update(ids)
            if traced:
                result = call("span_wait", wait_for_spans, root, expected)
                if not result["complete"]:
                    data["analysis_status"] = "INCOMPLETE"
                    log.log_line("span_wait", f"missing: {result['missing']}")
    except (Exception, KeyboardInterrupt):
        if "error" not in data:
            raise
    cleanup_cli = partial(stage_functions(root, data)[1], optional=True)
    for work in workloads:
        analysis = Path(__file__).with_name("analysis.py")
        if analysis.exists():
            argv = [sys.executable, str(analysis)] + flags(run_dir=root / work["name"])
            cleanup_cli(work["name"] + ":analysis", argv)
        else:
            log.log_line(work["name"], "analysis pending (WP7a)")
    failed = bool(data.get("error") or data.get("record_error"))
    failed |= any(not s["ok"] for s in data["steps"] if s["name"] != "links")
    return int(failed or data["analysis_status"] == "INCOMPLETE")


def parse_args(argv=None, *, single=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Measure PER-001 monitor overhead." if single else __doc__,
        epilog="Append service arguments after the measurement options."
        if single
        else None,
        allow_abbrev=False,
    )
    parser.add_argument("--configuration", choices=("baseline", "pdd"), required=True)
    if single:
        parser.set_defaults(mode="none")
    else:
        parser.add_argument(
            "--mode", choices=("none", "monitor", "trace", "deep"), required=True
        )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--workload" if single else "--workloads",
        dest="workloads",
        type=Path,
        required=True,
    )
    args, forwarded = parser.parse_known_args(argv)
    root = args.run_dir = args.run_dir.expanduser().resolve()
    if root.is_relative_to(REPO) or (root.exists() and any(root.iterdir())):
        raise ValueError("run-dir must be empty and outside repository")
    workloads = json.loads(args.workloads.read_text())
    workloads = [workloads] if single and isinstance(workloads, dict) else workloads
    if single and isinstance(workloads, list) and len(workloads) != 1:
        raise ValueError("PER-001 requires exactly one workload")
    validate_workloads(workloads, args.configuration)
    forwarded += flags(configuration=args.configuration, run_dir=root / "service")
    service = orchestrate.parse_args(["start", *forwarded])
    if service.cache_dir and service.cache_dir.resolve().is_relative_to(REPO):
        raise ValueError("cache-dir must be outside repository")
    return args, forwarded, service, workloads, argv


def main(argv=None):
    return execute(*parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.log_line("run", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
