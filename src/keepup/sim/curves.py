"""cumulative counts on an hourly grid, and the lookups the simulator needs.

every rate in the simulator is piecewise constant per hour, so cumulative
counts are piecewise linear and both directions (count at time t, time when
the count reaches v) are exact with linear interpolation.
"""

from __future__ import annotations

import math
from bisect import bisect_left

from ..workload import FloatArray

HOUR = 3600.0


class Curve:
    __slots__ = ("_list", "c", "hours")

    def __init__(self, cum: FloatArray):
        if len(cum) < 2:
            raise ValueError("curve needs at least one hour")
        self.c = cum
        self.hours = len(cum) - 1
        # python floats are a lot faster than numpy scalars for single lookups
        self._list: list[float] = cum.tolist()

    def at(self, t: float) -> float:
        c = self._list
        x = t / HOUR
        if x <= 0:
            # before the window: extrapolate with the first hour's rate. used
            # for "how many xids ago" questions that reach into the past
            return c[0] + (c[1] - c[0]) * x
        k = int(x)
        h = self.hours
        if k >= h:
            return c[h] + (c[h] - c[h - 1]) * (x - h)
        lo = c[k]
        return lo + (c[k + 1] - lo) * (x - k)

    def time_of(self, v: float) -> float:
        """first time the cumulative count reaches v (inf if never in range)."""
        c = self._list
        if v <= c[0]:
            r0 = c[1] - c[0]
            if r0 <= 0:
                return -math.inf if v < c[0] else 0.0
            return (v - c[0]) / r0 * HOUR
        k = bisect_left(c, v)
        if k > self.hours:
            r = c[-1] - c[-2]
            if r <= 0:
                return math.inf
            return (self.hours + (v - c[-1]) / r) * HOUR
        lo, hi = c[k - 1], c[k]
        if hi == lo:
            return k * HOUR
        return (k - 1 + (v - lo) / (hi - lo)) * HOUR

    def time_to_add(self, t0: float, delta: float) -> float:
        if delta <= 0:
            return t0
        t = self.time_of(self.at(t0) + delta)
        return max(t, t0)

    def rate_at(self, t: float) -> float:
        """per-second rate in the hour bin containing t."""
        k = min(max(int(t / HOUR), 0), self.hours - 1)
        return (self._list[k + 1] - self._list[k]) / HOUR

    def __add__(self, other: Curve) -> Curve:
        return Curve(self.c + other.c)

    def scaled(self, f: float) -> Curve:
        return Curve(self.c * f)
