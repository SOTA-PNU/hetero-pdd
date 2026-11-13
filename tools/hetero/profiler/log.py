# SPDX-License-Identifier: Apache-2.0
"""Shared UTC console format; run.py sets RUN_ID before starting work."""

from datetime import datetime, timezone

RUN_ID = "-"


def log_line(component, message):
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    message = str(message).replace("\r", "\\r").replace("\n", "\\n")
    print(f"{timestamp} [{RUN_ID}] [{component}] {message}", flush=True)
