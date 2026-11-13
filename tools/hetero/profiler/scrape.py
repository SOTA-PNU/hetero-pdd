# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def wait_until(condition, timeout, name):
    deadline = time.monotonic() + timeout
    while not condition():
        if (remaining := deadline - time.monotonic()) <= 0:
            raise TimeoutError(f"{name}: readiness timeout")
        time.sleep(min(0.2, remaining))


def fetch(endpoint):
    name, url = endpoint
    row = dict(mono_ns=time.monotonic_ns(), real_ns=time.time_ns(), endpoint=name)
    try:
        try:
            # Operator-controlled URLs; the normal runner uses loopback HTTP.
            response = urlopen(url, timeout=0.8)  # nosec B310
        except HTTPError as exc:
            response = exc
        with response:
            row.update(status=response.status, text=response.read().decode("utf-8"))
    except (OSError, URLError, UnicodeError, HTTPException) as exc:
        row.update(status=0, text=f"{type(exc).__name__}: {exc}")
    return row


def append_rows(out_path, rows):
    path = Path(out_path)
    if path.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("output must be outside repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def boundary_snapshot(endpoints, out_path, label):
    append_rows(out_path, (dict(fetch(e), label=label) for e in endpoints.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    endpoints = dict(spec.split("=", 1) for spec in args.endpoint)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        while not stopped.is_set():
            started = time.monotonic()
            append_rows(args.out, pool.map(fetch, endpoints.items()))
            stopped.wait(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
