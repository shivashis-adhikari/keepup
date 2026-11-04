"""what a simulation run hands back, plus the per-table numbers we report."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..gucs import XID_STOP_LIMIT, Settings
from ..profile import BLCKSZ
from ..workload import FloatArray

MB = 1024.0 * 1024.0


@dataclass
class Run:
    table: str
    db: str
    worker: int
    start: float
    end: float | None
    vacuum: bool
    analyze: bool
    wraparound: bool
    aggressive: bool
    failsafe: bool
    scanned: float
    index_passes: int
    removed: float
    credits: float
    hits: float
    misses: float
    dirtied: float
    lag: float
    # the vacuum part alone; the server logs vacuum and analyze separately
    vacuum_seconds: float | None = None

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start

    @property
    def kind(self) -> str:
        if not self.vacuum:
            return "analyze"
        if self.wraparound:
            return "wraparound"
        if self.aggressive:
            return "aggressive"
        return "vacuum"


@dataclass
class TableSeries:
    name: str
    db: str
    tpp: float
    live: FloatArray
    dead_stats: FloatArray
    dead_phys: FloatArray
    pages: FloatArray
    age: FloatArray
    vac_threshold: FloatArray
    freeze_max_age: float

    @property
    def bloat(self) -> FloatArray:
        ideal = self.live / self.tpp
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(self.pages > 0, 1.0 - ideal / self.pages, 0.0)
        return np.asarray(np.clip(b, 0.0, 1.0), dtype=np.float64)

    @property
    def wasted_bytes(self) -> FloatArray:
        return np.asarray(
            np.maximum(self.pages - self.live / self.tpp, 0.0) * BLCKSZ, dtype=np.float64
        )


@dataclass
class TableSummary:
    name: str
    db: str
    size_mb_start: float
    size_mb_end: float
    bloat_start: float
    bloat_end: float
    bloat_max: float
    wasted_mb_end: float
    wasted_mb_mean: float
    dead_max: float
    vacuums: int
    analyzes: int
    aggressive: int
    wraparound: int
    failsafe: int
    age_max: float
    age_end: float
    lag_p95_s: float
    lag_max_s: float
    longest_run_s: float
    first_wraparound_h: float | None
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = None if math.isinf(v) or math.isnan(v) else round(v, 4)
        return d


@dataclass
class SimResult:
    start: datetime
    horizon: float
    sample_times: FloatArray
    tables: dict[str, TableSeries]
    runs: list[Run]
    settings: Settings
    busy_worker_seconds: float
    saturated_seconds: float
    naptime: float

    @property
    def utilization(self) -> float:
        cap = self.settings.autovacuum_max_workers * self.horizon
        return self.busy_worker_seconds / cap if cap > 0 else 0.0

    @property
    def saturated_fraction(self) -> float:
        return self.saturated_seconds / self.horizon if self.horizon > 0 else 0.0

    def runs_for(self, table: str) -> list[Run]:
        return [r for r in self.runs if r.table == table]

    def io_series(self, bucket: float = 300.0) -> tuple[FloatArray, FloatArray, FloatArray]:
        """(bucket start times, read MB/s, write MB/s) from vacuum activity."""
        n = max(math.ceil(self.horizon / bucket), 1)
        read = np.zeros(n)
        write = np.zeros(n)
        for r in self.runs:
            end = r.end if r.end is not None else self.horizon
            dur = end - r.start
            if dur <= 0:
                continue
            rb = r.misses * BLCKSZ / MB
            wb = r.dirtied * BLCKSZ / MB
            a = int(r.start // bucket)
            b = min(int(end // bucket), n - 1)
            for k in range(a, b + 1):
                lo = max(r.start, k * bucket)
                hi = min(end, (k + 1) * bucket)
                if hi > lo:
                    f = (hi - lo) / dur
                    read[k] += rb * f
                    write[k] += wb * f
        t = np.arange(n, dtype=np.float64) * bucket
        return t, read / bucket, write / bucket

    def busy_series(self, bucket: float = 300.0) -> tuple[FloatArray, FloatArray]:
        n = max(math.ceil(self.horizon / bucket), 1)
        busy = np.zeros(n)
        for r in self.runs:
            end = r.end if r.end is not None else self.horizon
            a = int(r.start // bucket)
            b = min(int(end // bucket), n - 1)
            for k in range(a, b + 1):
                lo = max(r.start, k * bucket)
                hi = min(end, (k + 1) * bucket)
                if hi > lo:
                    busy[k] += (hi - lo) / bucket
        return np.arange(n, dtype=np.float64) * bucket, busy

    def peak_io_mbps(self, bucket: float = 300.0, q: float = 0.99) -> float:
        _, r, w = self.io_series(bucket)
        tot = r + w
        return float(np.quantile(tot, q)) if len(tot) else 0.0

    def summarize(self) -> list[TableSummary]:
        out = []
        for name, ts in self.tables.items():
            runs = self.runs_for(name)
            vac = [r for r in runs if r.vacuum]
            lags = [r.lag for r in vac if r.lag >= 0]
            bloat = ts.bloat
            wasted = ts.wasted_bytes / MB
            durs = [r.duration for r in runs if r.duration is not None]
            if any(r.end is None for r in runs):
                durs.append(self.horizon - max(r.start for r in runs if r.end is None))
            wrap = [r for r in vac if r.wraparound]
            s = TableSummary(
                name=name,
                db=ts.db,
                size_mb_start=float(ts.pages[0] * BLCKSZ / MB),
                size_mb_end=float(ts.pages[-1] * BLCKSZ / MB),
                bloat_start=float(bloat[0]),
                bloat_end=float(bloat[-1]),
                bloat_max=float(np.max(bloat)),
                wasted_mb_end=float(wasted[-1]),
                wasted_mb_mean=float(wasted.mean()),
                dead_max=float(np.max(ts.dead_phys)),
                vacuums=len(vac),
                analyzes=sum(1 for r in runs if r.analyze),
                aggressive=sum(1 for r in vac if r.aggressive),
                wraparound=len(wrap),
                failsafe=sum(1 for r in vac if r.failsafe),
                age_max=float(np.max(ts.age)),
                age_end=float(ts.age[-1]),
                lag_p95_s=float(np.quantile(lags, 0.95)) if lags else 0.0,
                lag_max_s=float(max(lags)) if lags else 0.0,
                longest_run_s=float(max(durs)) if durs else 0.0,
                first_wraparound_h=wrap[0].start / 3600.0 if wrap else None,
                verdict="ok",
            )
            _judge(s, ts, self)
            out.append(s)
        return out


def _judge(s: TableSummary, ts: TableSeries, res: SimResult) -> None:
    """decide ok / behind / danger. thresholds are opinions, written down."""
    reasons = []
    danger = False
    behind = False
    if s.age_max > 0.75 * XID_STOP_LIMIT:
        danger = True
        reasons.append(
            f"xid age reaches {s.age_max / 1e9:.2f}B, "
            f"{s.age_max / XID_STOP_LIMIT:.0%} of the stop limit"
        )
    if s.failsafe:
        danger = True
        reasons.append(f"{s.failsafe} failsafe vacuum(s)")
    if s.age_max > 1.25 * ts.freeze_max_age and s.age_max <= 0.75 * XID_STOP_LIMIT:
        behind = True
        reasons.append(
            f"xid age overshoots autovacuum_freeze_max_age by "
            f"{s.age_max / ts.freeze_max_age - 1:.0%}"
        )
    if s.lag_p95_s > max(3600.0, 10 * res.naptime):
        behind = True
        reasons.append(
            f"p95 wait between crossing the threshold and getting a worker "
            f"is {_fmt_dur(s.lag_p95_s)}"
        )
    bloat = ts.bloat
    n = len(bloat)
    grew = s.bloat_end - s.bloat_start
    if grew > 0.10:
        # tables don't give space back without vacuum full / repack, so
        # bloat that piles up during a bad stretch stays
        behind = True
        reasons.append(
            f"bloat grows from {s.bloat_start:.0%} to {s.bloat_end:.0%} "
            f"({s.size_mb_end - s.size_mb_start:+,.0f} MB on disk)"
        )
    if n > 48:
        tail = bloat[int(n * 0.75) :]
        x = np.arange(len(tail), dtype=np.float64)
        slope_per_sample = float(np.polyfit(x, tail, 1)[0]) if len(tail) > 2 else 0.0
        samples_per_day = 86400.0 / max(float(res.sample_times[1] - res.sample_times[0]), 1.0)
        slope_day = slope_per_sample * samples_per_day
        if slope_day > 0.002:
            behind = True
            reasons.append(f"bloat still climbing at the end, {slope_day * 100:.1f} pts/day")
    if s.lag_max_s > max(4 * 3600.0, 20 * res.naptime) and s.lag_p95_s <= max(
        3600.0, 10 * res.naptime
    ):
        reasons.append(f"worst wait for a worker {_fmt_dur(s.lag_max_s)}")
    if s.wraparound and s.vacuums:
        every = res.horizon / s.wraparound
        label = f"{s.wraparound} anti-wraparound vacuum" + ("s" if s.wraparound > 1 else "")
        reasons.append(f"{label}, about one every {_fmt_dur(every)}" if s.wraparound > 1 else label)
    if s.longest_run_s > 86400 and s.vacuums:
        reasons.append(f"longest run {_fmt_dur(s.longest_run_s)}")
    s.verdict = "danger" if danger else "behind" if behind else "ok"
    s.reasons = reasons


def _fmt_dur(sec: float) -> str:
    if sec < 120:
        return f"{sec:.0f}s"
    if sec < 7200:
        return f"{sec / 60:.0f}min"
    if sec < 172800:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"
