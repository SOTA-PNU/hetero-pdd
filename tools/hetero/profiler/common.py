# SPDX-License-Identifier: Apache-2.0
import json
from itertools import pairwise


def live_rows(path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle if line.endswith("\n"))


def metric(value, unit, source, reason="missing_raw", status="measured", **extra):
    status = status if value is not None else "not_collected"
    node = dict(value=value, unit=unit, source=source, status=status)
    node.update(reason=reason if value is None else None, **extra)
    return node


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b > a:
            if result and a <= result[-1][1]:
                result[-1] = (result[-1][0], max(b, result[-1][1]))
            else:
                result.append((a, b))
    return result


def active_ratio(intervals, window):
    clipped = [(max(a, window[0]), min(b, window[1])) for a, b in intervals]
    return sum(b - a for a, b in union(clipped)) / (window[1] - window[0])


def attribute_samples(points, intervals, linear=False):
    """Integrate adjacent samples without extrapolation or bridging missing values."""
    intervals = union(intervals)
    total, covered, maxima = 0, 0, []
    for (a, va), (b, vb) in pairwise(sorted(points)):
        # At 1 Hz, a gap beyond 1.5 periods is uncovered, not interpolated.
        if None in (va, vb) or not 0 < b - a <= 1.5:
            continue
        for start, end in intervals:
            lo, hi = max(start, a), min(end, b)
            if hi > lo:
                left = va + (vb - va) * (lo - a) / (b - a) if linear else va
                right = va + (vb - va) * (hi - a) / (b - a) if linear else va
                total += (left + right) / 2 * (hi - lo)
                covered += hi - lo
                maxima.extend((left, right))
    for t, v in points:
        if v is not None and any(a <= t <= b for a, b in intervals):
            maxima.append(v)
    duration = sum(b - a for a, b in intervals)
    avg = total / covered if covered else None
    coverage = covered / duration if duration else 0
    return avg, max(maxima, default=None), coverage, total


def interval_segments(intervals, window):
    """Partition a window at clipped interval endpoints."""
    points = {max(window[0], min(t, window[1])) for ab in intervals for t in ab}
    return pairwise(sorted(points | set(window)))
