"""change-rate models and the code that fits them to sampled counters.

a rate model is base * trend(day) * hod[hour] * dow[weekday], evaluated on an
hourly grid. the simulator never looks at instantaneous rates, only at the
cumulative count at hour boundaries, so everything in here produces or
consumes hourly bins.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

MAX_GROWTH_PER_DAY = 0.02  # +/- 2% of base per day; anything bigger from a week of data is noise


@dataclass(frozen=True)
class RateModel:
    """events per second, with optional daily/weekly shape and linear growth."""

    base: float = 0.0
    growth_per_day: float = 0.0  # fraction of base added per day after the forecast start
    hod: tuple[float, ...] | None = None  # 24 multipliers, local time, mean ~1
    dow: tuple[float, ...] | None = None  # 7 multipliers, monday first
    noise_sigma: float = 0.0  # stddev of hourly log residuals, for monte carlo
    noise_ar1: float = 0.0  # lag-1 autocorrelation of those residuals
    kind: str = "flat"

    def __post_init__(self) -> None:
        if self.base < 0:
            raise ValueError("rate can't be negative")
        if self.hod is not None and len(self.hod) != 24:
            raise ValueError("hod needs 24 values")
        if self.dow is not None and len(self.dow) != 7:
            raise ValueError("dow needs 7 values")

    @classmethod
    def constant(cls, per_second: float) -> RateModel:
        return cls(base=per_second)

    def scaled(self, factor: float) -> RateModel:
        return RateModel(
            self.base * factor,
            self.growth_per_day,
            self.hod,
            self.dow,
            self.noise_sigma,
            self.noise_ar1,
            self.kind,
        )

    def hourly_rates(
        self, start: datetime, hours: int, tz: str = "UTC", rng: np.random.Generator | None = None
    ) -> FloatArray:
        """per-second rate for each hour bin starting at `start`."""
        if hours <= 0:
            return np.zeros(0)
        idx = np.arange(hours, dtype=np.float64)
        trend = 1.0 + self.growth_per_day * (idx + 0.5) / 24.0
        np.maximum(trend, 0.0, out=trend)
        rates = self.base * trend
        if self.hod is not None or self.dow is not None:
            hod_i, dow_i = _calendar_index(start, hours, tz)
            if self.hod is not None:
                rates = rates * np.asarray(self.hod)[hod_i]
            if self.dow is not None:
                rates = rates * np.asarray(self.dow)[dow_i]
        if rng is not None and self.noise_sigma > 0:
            rates = rates * _ar1_lognormal(rng, hours, self.noise_sigma, self.noise_ar1)
        return rates

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"base": round(self.base, 6)}
        if self.growth_per_day:
            d["growth_per_day"] = round(self.growth_per_day, 6)
        if self.hod is not None:
            d["hod"] = [round(x, 4) for x in self.hod]
        if self.dow is not None:
            d["dow"] = [round(x, 4) for x in self.dow]
        if self.noise_sigma:
            d["noise_sigma"] = round(self.noise_sigma, 4)
            d["noise_ar1"] = round(self.noise_ar1, 4)
        if self.kind != "flat":
            d["kind"] = self.kind
        return d

    @classmethod
    def from_value(cls, v: Any) -> RateModel:
        """accept a bare number (per second) or a dict like to_dict() makes.

        dicts may also use per_hour / per_day / per_minute instead of base,
        which is nicer for hand-written scenarios.
        """
        if v is None:
            return cls()
        if isinstance(v, (int, float)):
            return cls(base=float(v))
        if not isinstance(v, dict):
            raise ValueError(f"can't read a rate from {v!r}")
        v = dict(v)
        base = v.pop("base", None)
        for key, div in (
            ("per_second", 1.0),
            ("per_minute", 60.0),
            ("per_hour", 3600.0),
            ("per_day", 86400.0),
        ):
            if key in v:
                base = float(v.pop(key)) / div
        hod = v.pop("hod", None)
        dow = v.pop("dow", None)
        m = cls(
            base=float(base or 0.0),
            growth_per_day=float(v.pop("growth_per_day", 0.0)),
            hod=tuple(float(x) for x in hod) if hod is not None else None,
            dow=tuple(float(x) for x in dow) if dow is not None else None,
            noise_sigma=float(v.pop("noise_sigma", 0.0)),
            noise_ar1=float(v.pop("noise_ar1", 0.0)),
            kind=str(v.pop("kind", "flat")),
        )
        if v:
            raise ValueError(f"unknown rate keys: {sorted(v)}")
        return m


def _calendar_index(
    start: datetime, hours: int, tz: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    # every rate model of every table asks for the same grid, and the search
    # and monte carlo ask again for each simulation
    return _calendar_index_cached(start, hours, tz)


@functools.lru_cache(maxsize=64)
def _calendar_index_cached(
    start: datetime, hours: int, tz: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    zone = ZoneInfo(tz)
    # dst makes this not quite arithmetic, so just walk it. 2-3k entries, cheap.
    hod = np.empty(hours, dtype=np.int64)
    dow = np.empty(hours, dtype=np.int64)
    for i in range(hours):
        t = (start + timedelta(hours=i, minutes=30)).astimezone(zone)
        hod[i] = t.hour
        dow[i] = t.weekday()
    hod.flags.writeable = False
    dow.flags.writeable = False
    return hod, dow


def _ar1_lognormal(rng: np.random.Generator, n: int, sigma: float, phi: float) -> FloatArray:
    phi = min(max(phi, 0.0), 0.99)
    eps = rng.standard_normal(n) * sigma * math.sqrt(1 - phi * phi)
    x = np.empty(n)
    prev = rng.standard_normal() * sigma
    for i in range(n):
        prev = phi * prev + eps[i]
        x[i] = prev
    # mean-preserving: E[exp(x)] = exp(sigma^2/2)
    return np.asarray(np.exp(x - sigma * sigma / 2.0), dtype=np.float64)


def cumulative(rates: FloatArray) -> FloatArray:
    """hourly per-second rates -> cumulative count at each hour boundary (len n+1)."""
    out = np.zeros(len(rates) + 1)
    np.cumsum(rates * 3600.0, out=out[1:])
    return out


# fitting ---------------------------------------------------------------------


@dataclass
class HourlyCounts:
    """counts per local-time hour bin, with how many seconds of each bin we observed."""

    start: datetime  # utc, start of first bin
    counts: FloatArray
    covered: FloatArray  # seconds observed per bin (0..3600)
    tz: str = "UTC"
    hod: npt.NDArray[np.int64] = field(init=False)
    dow: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        self.hod, self.dow = _calendar_index(self.start, len(self.counts), self.tz)

    @property
    def rates(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.covered > 0, self.counts / np.maximum(self.covered, 1e-9), 0.0)


def bin_intervals(
    intervals: Iterable[tuple[datetime, datetime, float]], tz: str = "UTC"
) -> HourlyCounts:
    """spread (start, end, count) deltas over hour bins, proportionally to overlap."""
    items = [(a.astimezone(UTC), b.astimezone(UTC), c) for a, b, c in intervals if b > a]
    if not items:
        raise ValueError("no intervals to fit")
    t0 = min(a for a, _, _ in items).replace(minute=0, second=0, microsecond=0)
    t1 = max(b for _, b, _ in items)
    n = math.ceil((t1 - t0).total_seconds() / 3600.0)
    counts = np.zeros(n)
    covered = np.zeros(n)
    for a, b, c in items:
        sa = (a - t0).total_seconds()
        sb = (b - t0).total_seconds()
        span = sb - sa
        k = int(sa // 3600)
        while k < n and k * 3600.0 < sb:
            lo = max(sa, k * 3600.0)
            hi = min(sb, (k + 1) * 3600.0)
            if hi > lo:
                w = (hi - lo) / span
                counts[k] += c * w
                covered[k] += hi - lo
            k += 1
    np.minimum(covered, 3600.0, out=covered)
    return HourlyCounts(t0, counts, covered, tz)


@dataclass
class FitReport:
    model: RateModel
    kind: str
    bic: dict[str, float]
    hours_used: int
    span_days: float
    growth_tstat: float


def fit_rate(h: HourlyCounts, min_coverage: float = 0.5, allow_growth: bool = True) -> FitReport:
    """fit flat / daily / weekly multiplicative models and keep the best by bic."""
    # short windows only ever partly cover their hour bins, so the cutoff is
    # relative to the best-covered bin rather than a full hour
    cutoff = min(min_coverage * 3600.0, 0.5 * float(np.max(h.covered)))
    keep = (h.covered >= cutoff) & (h.covered > 0)
    rates = h.rates[keep]
    w = h.covered[keep] / 3600.0
    hod = h.hod[keep]
    dow = h.dow[keep]
    days = np.asarray((np.nonzero(keep)[0] + 0.5) / 24.0, dtype=np.float64)
    n = int(keep.sum())
    span_days = float(len(h.counts)) / 24.0
    if n == 0:
        return FitReport(RateModel(), "flat", {}, 0, span_days, 0.0)
    total = float(np.sum(rates * w))
    if total <= 0:
        return FitReport(RateModel(), "flat", {}, n, span_days, 0.0)

    # center time so the base is the rate at the end of the sample window,
    # which is where the forecast starts
    t_end = span_days
    tcen = days - t_end

    candidates: list[str] = ["flat"]
    n_days_seen = len(np.unique(np.floor(days)))
    if n_days_seen >= 2 and len(np.unique(hod)) >= 12:
        candidates.append("daily")
    if n_days_seen >= 7 and len(np.unique(dow)) == 7:
        candidates.append("weekly")

    fit_growth = allow_growth and span_days >= 14.0
    eps = 0.05 * total / float(np.sum(w)) + 1e-12
    results: dict[str, tuple[RateModel, float, float]] = {}
    for kind in candidates:
        m, tstat = _fit_one(rates, w, hod, dow, tcen, kind, fit_growth)
        pred = _predict(m, hod, dow, tcen)
        resid = np.log((rates + eps) / (pred + eps))
        rss = float(np.sum(w * resid * resid))
        k = (
            1
            + (1 if fit_growth else 0)
            + (23 if kind != "flat" else 0)
            + (6 if kind == "weekly" else 0)
        )
        bic = n * math.log(max(rss / n, 1e-12)) + k * math.log(n)
        sigma = float(np.sqrt(rss / max(np.sum(w), 1e-9)))
        ar1 = _lag1(resid, np.nonzero(keep)[0])
        m = RateModel(m.base, m.growth_per_day, m.hod, m.dow, sigma, ar1, kind)
        results[kind] = (m, bic, tstat)

    best = min(results, key=lambda k: results[k][1])
    model, _, tstat = results[best]
    return FitReport(model, best, {k: v[1] for k, v in results.items()}, n, span_days, tstat)


def _predict(
    m: RateModel, hod: npt.NDArray[np.int64], dow: npt.NDArray[np.int64], tcen: FloatArray
) -> FloatArray:
    p = m.base * np.maximum(1.0 + m.growth_per_day * tcen, 0.0)
    if m.hod is not None:
        p = p * np.asarray(m.hod)[hod]
    if m.dow is not None:
        p = p * np.asarray(m.dow)[dow]
    return np.asarray(p, dtype=np.float64)


def _fit_one(
    rates: FloatArray,
    w: FloatArray,
    hod: npt.NDArray[np.int64],
    dow: npt.NDArray[np.int64],
    tcen: FloatArray,
    kind: str,
    fit_growth: bool,
) -> tuple[RateModel, float]:
    # shape first with a flat trend, then growth from daily totals, then the
    # shape again given the trend. fitting growth on hourly points directly
    # gave silly t-stats because hourly residuals are strongly autocorrelated.
    s_hod = np.ones(24)
    s_dow = np.ones(7)
    trend = np.ones_like(rates)
    base, g, tstat = float(np.average(rates, weights=w)), 0.0, 0.0
    for _ in range(3):
        for _ in range(6):
            if kind in ("daily", "weekly"):
                other = s_dow[dow] if kind == "weekly" else 1.0
                s_hod = _group_ratio(rates, base * trend * other, w, hod, 24, shrink=1.0)
            if kind == "weekly":
                s_dow = _group_ratio(rates, base * trend * s_hod[hod], w, dow, 7, shrink=4.0)
        shape = np.ones_like(rates)
        if kind in ("daily", "weekly"):
            shape = shape * s_hod[hod]
        if kind == "weekly":
            shape = shape * s_dow[dow]
        base, g, tstat = _daily_trend(rates, w, shape, tcen, fit_growth)
        trend = np.maximum(1.0 + g * tcen, 1e-9)
    return RateModel(
        base=max(base, 0.0),
        growth_per_day=g,
        hod=tuple(float(x) for x in s_hod) if kind != "flat" else None,
        dow=tuple(float(x) for x in s_dow) if kind == "weekly" else None,
        kind=kind,
    ), tstat


def _daily_trend(
    rates: FloatArray, w: FloatArray, shape: FloatArray, tcen: FloatArray, fit_growth: bool
) -> tuple[float, float, float]:
    """base rate at the end of the window plus relative growth per day.

    works on per-day ratios of observed to shape-predicted counts so each
    point is roughly independent. needs two weeks of data to say anything.
    """
    sw = float(np.sum(w * shape))
    mean = float(np.sum(w * rates)) / sw if sw > 0 else 0.0
    if not fit_growth:
        return mean, 0.0, 0.0
    day = np.floor(tcen).astype(np.int64)
    day -= np.min(day)
    num = np.bincount(day, weights=w * rates)
    den = np.bincount(day, weights=w * shape)
    ok = den > 0.5 * np.max(den)  # skip days we barely saw
    if ok.sum() < 14:
        return mean, 0.0, 0.0
    y = num[ok] / den[ok]
    x = np.nonzero(ok)[0].astype(np.float64)
    x = x - np.max(x) - 0.5  # 0 = end of window
    A = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    resid = y - A @ beta
    s2 = float(resid @ resid) / max(len(y) - 2, 1)
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_b = math.sqrt(s2 / sxx) if sxx > 0 and s2 > 0 else 0.0
    tstat = b / se_b if se_b > 0 else (math.inf if b else 0.0)
    if a <= 0 or abs(tstat) < 3.0:
        return mean, 0.0, tstat
    g = max(-MAX_GROWTH_PER_DAY, min(MAX_GROWTH_PER_DAY, b / a))
    return a, g, tstat


def _group_ratio(
    obs: FloatArray,
    pred: FloatArray,
    w: FloatArray,
    groups: npt.NDArray[np.int64],
    ngroups: int,
    shrink: float,
) -> FloatArray:
    num = np.bincount(groups, weights=w * obs, minlength=ngroups)
    den = np.bincount(groups, weights=w * pred, minlength=ngroups)
    cnt = np.bincount(groups, weights=w, minlength=ngroups)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(den > 0, num / den, 1.0)
    # pull sparsely observed groups towards 1
    s = (cnt * raw + shrink * 1.0) / (cnt + shrink)
    mean = float(np.average(s, weights=np.maximum(cnt, 1e-9)))
    return np.asarray(s / mean if mean > 0 else s, dtype=np.float64)


def _lag1(resid: FloatArray, positions: npt.NDArray[np.int64]) -> float:
    if len(resid) < 3:
        return 0.0
    consecutive = np.diff(positions) == 1
    if consecutive.sum() < 2:
        return 0.0
    a = resid[:-1][consecutive]
    b = resid[1:][consecutive]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -0.99, 0.99))


def counter_intervals(
    times: Sequence[datetime], values: Sequence[float | None]
) -> list[tuple[datetime, datetime, float]]:
    """turn monotonic counter samples into (start, end, delta) intervals.

    a drop means pg_stat_reset() or a restart; that interval is thrown away.
    """
    out = []
    for i in range(1, len(times)):
        a, b = values[i - 1], values[i]
        if a is None or b is None:
            continue
        d = b - a
        if d < 0:
            continue
        out.append((times[i - 1], times[i], float(d)))
    return out
