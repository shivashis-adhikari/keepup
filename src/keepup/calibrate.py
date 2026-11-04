"""line the simulator up with what the server did.

two steps:

1. fit the throughput model from logged runs. every log entry has the
   buffer usage (so the exact credits it spent) and the elapsed time. what
   it doesn't say is how long it slept, so that comes from the cost settings
   and how many workers were running at the same time, worked out from the
   overlaps between logged runs. pg18 with track_cost_delay_timing logs the
   sleep directly, and then we use that.

       elapsed ~= overhead * sleep + credits / raw_rate

   is linear in (overhead, 1/raw_rate), so it's plain weighted least squares.

2. backtest: start the simulator from the state at the beginning of the
   sample window, run it to the end, and compare per-table vacuum counts,
   intervals and durations with the log.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np

from .gucs import TableOptions, effective
from .logs import LogRun
from .profile import Profile, TableProfile
from .sim import SimResult, simulate


@dataclass
class IoFit:
    raw_credit_rate: float
    sleep_overhead: float
    runs_used: int
    median_abs_err: float  # relative, on elapsed
    measured_sleep: bool
    analyze_row_seconds: float | None = None

    def describe(self) -> str:
        src = "logged delay time" if self.measured_sleep else "settings + overlap"
        return (
            f"raw rate {self.raw_credit_rate:,.0f} credits/s, sleep overhead "
            f"{self.sleep_overhead:.2f}x ({src}), {self.runs_used} runs, "
            f"median error {self.median_abs_err:.0%}"
            + (
                f", analyze {self.analyze_row_seconds * 1e6:.1f}us/row"
                if self.analyze_row_seconds is not None
                else ""
            )
        )


def _inv_limit_profile(runs: list[LogRun], limits: list[int], balanced: list[bool]) -> list[float]:
    """time-averaged 1/limit per run, splitting the limit across overlapping
    balanced runs the way autovac_recalculate_workers_for_balance does."""
    events: list[tuple[float, int, int]] = []
    for i, r in enumerate(runs):
        if balanced[i]:
            events.append((r.start.timestamp(), 1, i))
            events.append((r.end.timestamp(), -1, i))
    events.sort(key=lambda e: (e[0], e[1]))
    # piecewise-constant count of balanced runs
    times: list[float] = []
    counts: list[int] = []
    n = 0
    for t, d, _ in events:
        n += d
        times.append(t)
        counts.append(n)
    out = []
    for i, r in enumerate(runs):
        a, b = r.start.timestamp(), r.end.timestamp()
        if b <= a:
            out.append(1.0 / max(limits[i], 1))
            continue
        if not balanced[i]:
            out.append(1.0 / max(limits[i], 1))
            continue
        acc = 0.0
        k = int(np.searchsorted(times, a, side="right"))
        cur_t = a
        cur_n = counts[k - 1] if k > 0 else 1
        while cur_t < b:
            nxt = times[k] if k < len(times) else b
            nxt = min(nxt, b)
            lim = max(limits[i] // max(cur_n, 1), 1)
            acc += (nxt - cur_t) / lim
            cur_t = nxt
            if k < len(times):
                cur_n = counts[k]
            k += 1
        out.append(acc / (b - a))
    return out


def fit_io(runs: list[LogRun], profile: Profile, min_elapsed: float = 0.1) -> IoFit | None:
    s = profile.settings
    by_name = {(t.database, t.relname): t for t in profile.tables}
    allr = [r for r in runs if r.hits + r.misses > 0]
    if not allr:
        return None
    limits, delays, balanced = [], [], []
    for r in allr:
        tp = by_name.get((r.db, r.relname))
        e = effective(s, tp.options if tp else TableOptions())
        limits.append(e.cost_limit)
        delays.append(e.cost_delay_ms / 1000.0)
        balanced.append(e.dobalance)
    # overlaps are worked out over every logged run, short ones included,
    # since they all took a share of the limit while they ran
    inv_all = _inv_limit_profile(allr, limits, balanced)
    pick = [i for i, r in enumerate(allr) if r.kind == "vacuum" and r.elapsed >= min_elapsed]
    if len(pick) < 3:
        return None
    use = [allr[i] for i in pick]
    inv = np.array([inv_all[i] for i in pick])
    dl = np.array([delays[i] for i in pick])
    hit, miss, dirty = s.vacuum_cost_page_hit, s.vacuum_cost_page_miss, s.vacuum_cost_page_dirty
    credits = np.array([r.credits(hit, miss, dirty) for r in use])
    elapsed = np.array([r.elapsed for r in use])
    measured = all(r.delay_ms is not None for r in use)
    if measured:
        sleep = np.array([(r.delay_ms or 0.0) / 1000.0 for r in use])
    else:
        sleep = credits * dl * inv
    # elapsed is logged with 10ms resolution, don't let tiny runs dominate
    w = 1.0 / (np.maximum(elapsed, 1e-3) ** 2 + 0.01**2)

    def rate_only(alpha: float) -> float:
        y = elapsed - alpha * sleep
        return float(np.sum(w * credits * y) / max(np.sum(w * credits * credits), 1e-30))

    if measured or not np.any(sleep > 0):
        alpha, beta = 1.0, rate_only(1.0)
    elif _collinear(sleep, credits):
        # every run slept in proportion to its credits (same limit, same
        # concurrency), so overhead and raw rate can't be told apart. pin the
        # overhead at what pg_usleep usually overshoots by on linux.
        alpha = 1.05
        beta = rate_only(alpha)
    else:
        X = np.column_stack([sleep, credits])
        WX = X * w[:, None]
        try:
            alpha, beta = (float(v) for v in np.linalg.solve(WX.T @ X, WX.T @ elapsed))
        except np.linalg.LinAlgError:
            alpha, beta = 1.05, rate_only(1.05)
        if not 0.8 <= alpha <= 3.0 or beta <= 0:
            alpha = min(max(alpha, 0.8), 3.0) if beta > 0 else 1.05
            beta = rate_only(alpha)
    # a run can't go faster than its throttle allows; if everything is
    # explained by sleeping, raw throughput is "a lot" rather than infinite
    beta = max(beta, 1.0 / 5e6)
    pred = alpha * sleep + credits * beta
    err = float(np.median(np.abs(pred - elapsed) / np.maximum(elapsed, 1e-3)))
    fit = IoFit(1.0 / beta, alpha, len(use), err, measured)
    fit.analyze_row_seconds = _fit_analyze_cpu(allr, inv_all, delays, profile, fit)
    return fit


def _collinear(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return True
    ratio = a / np.maximum(b, 1e-12)
    return bool(np.std(ratio) / max(float(np.mean(ratio)), 1e-12) < 0.05)


def _fit_analyze_cpu(
    allr: list[LogRun], inv_all: list[float], delays: list[float], profile: Profile, io: IoFit
) -> float | None:
    s = profile.settings
    by_name = {(t.database, t.relname): t for t in profile.tables}
    per_row = []
    for i, r in enumerate(allr):
        if r.kind != "analyze" or r.elapsed < 0.05:
            continue
        tp = by_name.get((r.db, r.relname))
        if tp is None or tp.live_tuples <= 0:
            continue
        cr = r.credits(s.vacuum_cost_page_hit, s.vacuum_cost_page_miss, s.vacuum_cost_page_dirty)
        io_time = io.sleep_overhead * cr * delays[i] * inv_all[i] + cr / io.raw_credit_rate
        rows = min(tp.live_tuples, 300 * s.default_statistics_target)
        per_row.append(max(r.elapsed - io_time, 0.0) / rows)
    if len(per_row) < 3:
        return None
    return float(min(statistics.median(per_row), 1e-3))


@dataclass
class TableParams:
    cache_hit: float | None = None
    freeze_fraction: float | None = None
    active_fraction: float | None = None
    dirty_residency: float | None = None
    runs: int = 0


def fit_table_params(runs: list[LogRun], profile: Profile) -> dict[str, TableParams]:
    by_table: dict[str, list[LogRun]] = defaultdict(list)
    for r in runs:
        if r.kind == "vacuum":
            by_table[f"{r.db}:{r.relname}"].append(r)
    out: dict[str, TableParams] = {}
    for tp in profile.tables:
        rs = sorted(by_table.get(f"{tp.database}:{tp.relname}", []), key=lambda r: r.end)
        if not rs:
            continue
        p = TableParams(runs=len(rs))
        reads = hits = 0.0
        for r in rs:
            fsm = r.pages_scanned or 0
            h = max(r.hits - fsm, 0)
            hits += h
            reads += h + r.misses
        if reads > 0:
            p.cache_hit = hits / reads
        # dirtied pages are left to the checkpoint model in the simulator: the
        # share depends on how long since the last vacuum, so a single number
        # fitted from one stretch of logs doesn't carry over.
        #
        # opportunistic freezing only shows cleanly in runs that came sooner
        # than freeze_min_age xids after the previous one; later than that,
        # ordinary freezing of old pages is mixed in.
        fmin = effective(profile.settings, tp.options).freeze_min_age
        xps = profile.xid_rate.base
        scanned = frozen = 0.0
        for prev, cur in itertools.pairwise(rs):
            if cur.aggressive or not cur.pages_scanned:
                continue
            if xps * (cur.start - prev.end).total_seconds() >= fmin:
                continue
            scanned += cur.pages_scanned
            frozen += cur.frozen_pages
        p.freeze_fraction = frozen / scanned if scanned else None
        p.dirty_residency = _fit_residency(rs, profile.settings.checkpoint_timeout)
        p.active_fraction = _fit_active_fraction(rs, tp)
        out[tp.name] = p
    return out


def _fit_residency(rs: list[LogRun], ckpt: float) -> float | None:
    """how long pages stay dirty in memory: T = D * (1 - clean share).

    runs where vacuum dirtied next to nothing only say T >= D, so they're
    left out; if that's all there is, the default (checkpoint_timeout) stands.
    """
    ts = []
    for prev, cur in itertools.pairwise(rs):
        if not cur.pages_scanned:
            continue
        d = (cur.start - prev.end).total_seconds()
        clean = min(cur.dirtied / cur.pages_scanned, 1.0)
        if d <= 0 or clean < 0.02:
            continue
        ts.append(min(d * (1.0 - clean), ckpt))
    if len(ts) < 2:
        return None
    return float(statistics.median(ts))


def _fit_active_fraction(rs: list[LogRun], tp: TableProfile) -> float | None:
    mod_rate = tp.updates.base + tp.deletes.base
    ins_rate = tp.inserts.base
    qs = []
    for prev, cur in itertools.pairwise(rs):
        if cur.aggressive or cur.pages_scanned is None:
            continue
        P = float(cur.pages_remain)
        if P < 100:
            continue
        dt = (cur.start - prev.end).total_seconds()
        if dt <= 0:
            continue
        m = mod_rate * dt
        # inserts either extended the file (a contiguous run vacuum has to
        # read) or went into space the last vacuum freed, landing on about
        # as many pages as that vacuum found dead rows on
        grown = max(float(cur.pages_remain - prev.pages_remain), 0.0)
        reused_rows = max(ins_rate * dt - grown * tp.tpp, 0.0)
        # rows freed per page. on-access pruning already turned many dead rows
        # into dead item identifiers, which "tuples removed" doesn't count
        freed = max(prev.tuples_removed, prev.lpdead_items)
        density = freed / prev.lpdead_pages if prev.lpdead_pages else tp.tpp / 2.0
        target = cur.pages_scanned - grown - reused_rows / max(density, 1.0)
        if target <= 0:
            continue

        def f(a: float, m: float = m) -> float:
            return a * -math.expm1(-m / a)

        if f(P) <= target:
            qs.append(1.0)
            continue
        lo, hi = 1.0, P
        for _ in range(50):
            mid = math.sqrt(lo * hi)
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        qs.append(hi / P)
    if len(qs) < 2:
        return None
    return float(np.clip(statistics.median(qs), 1e-3, 1.0))


def apply_params(profile: Profile, io: IoFit | None, params: dict[str, TableParams]) -> Profile:
    tables = []
    for t in profile.tables:
        p = params.get(t.name)
        if p is not None:
            kw: dict[str, Any] = {}
            for k in ("cache_hit", "dirty_residency", "freeze_fraction", "active_fraction"):
                v = getattr(p, k)
                if v is not None:
                    kw[k] = round(float(v), 4)
            t = replace(t, **kw)
        tables.append(t)
    out = replace(profile, tables=tables)
    if io is not None:
        out = replace(out, raw_credit_rate=io.raw_credit_rate, sleep_overhead=io.sleep_overhead)
        if io.analyze_row_seconds is not None:
            out = replace(out, analyze_row_seconds=io.analyze_row_seconds)
    return out


# backtest --------------------------------------------------------------------


@dataclass
class TableComparison:
    table: str
    n_obs: int
    n_sim: int
    interval_obs: float | None
    interval_sim: float | None
    dur_obs: float | None
    dur_sim: float | None
    scanned_obs: float | None
    scanned_sim: float | None
    analyze_obs: int = 0
    analyze_sim: int = 0

    @property
    def count_err(self) -> float | None:
        if self.n_obs == 0:
            return None
        return (self.n_sim - self.n_obs) / self.n_obs

    @property
    def dur_err(self) -> float | None:
        if not self.dur_obs or self.dur_sim is None:
            return None
        return (self.dur_sim - self.dur_obs) / self.dur_obs

    @property
    def interval_err(self) -> float | None:
        if not self.interval_obs or self.interval_sim is None:
            return None
        return (self.interval_sim - self.interval_obs) / self.interval_obs


@dataclass
class Backtest:
    start: datetime
    end: datetime
    tables: list[TableComparison]
    io: IoFit | None
    min_duration: float
    notes: list[str] = field(default_factory=list)

    def _mape(self, attr: str, min_obs: int = 3) -> float | None:
        errs = [
            abs(v)
            for t in self.tables
            if t.n_obs >= min_obs and (v := getattr(t, attr)) is not None
        ]
        return float(np.mean(errs)) if errs else None

    @property
    def count_mape(self) -> float | None:
        return self._mape("count_err")

    @property
    def duration_mape(self) -> float | None:
        return self._mape("dur_err")

    @property
    def interval_mape(self) -> float | None:
        return self._mape("interval_err")

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "count_mape": self.count_mape,
            "interval_mape": self.interval_mape,
            "duration_mape": self.duration_mape,
            "io": None if self.io is None else self.io.__dict__,
            "tables": [
                {
                    **t.__dict__,
                    "count_err": t.count_err,
                    "dur_err": t.dur_err,
                    "interval_err": t.interval_err,
                }
                for t in self.tables
            ],
            "notes": self.notes,
        }


def _median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def _intervals(starts: list[float]) -> list[float]:
    s = sorted(starts)
    return [b - a for a, b in itertools.pairwise(s)]


def backtest(
    profile: Profile,
    runs: list[LogRun],
    end: datetime,
    min_duration: float = 0.0,
    io: IoFit | None = None,
) -> tuple[Backtest, SimResult]:
    """simulate from profile.start to `end` and compare with logged runs."""
    days = (end - profile.start).total_seconds() / 86400.0
    if days <= 0:
        raise ValueError("backtest window is empty")
    res = simulate(profile, days=days, sample_every=min(3600.0, days * 86400 / 50))
    t0 = profile.start
    obs = [r for r in runs if t0 <= r.start and r.end <= end]
    notes = []
    if min_duration > 0:
        notes.append(
            f"log_autovacuum_min_duration = {min_duration:g}s, shorter simulated "
            "runs are dropped before comparing"
        )
    comps = []
    for tp in profile.tables:
        o = [r for r in obs if r.db == tp.database and r.relname == tp.relname]
        ov = [r for r in o if r.kind == "vacuum"]
        s_all = [
            r
            for r in res.runs_for(tp.name)
            if r.end is not None and (r.duration or 0) >= min_duration
        ]
        sv = [r for r in s_all if r.vacuum]
        comps.append(
            TableComparison(
                table=f"{tp.database}:{tp.relname}",
                n_obs=len(ov),
                n_sim=len(sv),
                interval_obs=_median(_intervals([(r.start - t0).total_seconds() for r in ov])),
                interval_sim=_median(_intervals([r.start for r in sv])),
                dur_obs=_median([r.elapsed for r in ov]),
                dur_sim=_median([r.vacuum_seconds or r.duration or 0.0 for r in sv]),
                scanned_obs=_median([float(r.pages_scanned) for r in ov if r.pages_scanned]),
                scanned_sim=_median([r.scanned for r in sv]),
                analyze_obs=sum(1 for r in o if r.kind == "analyze"),
                analyze_sim=sum(1 for r in s_all if r.analyze),
            )
        )
    return Backtest(t0, end, comps, io, min_duration, notes), res


def log_min_duration(settings_raw: dict[str, Any]) -> float:
    v = settings_raw.get("log_autovacuum_min_duration")
    if not v:
        return 0.0
    try:
        ms = float(v[0])
    except (TypeError, ValueError):
        return 0.0
    return max(ms, 0.0) / 1000.0
