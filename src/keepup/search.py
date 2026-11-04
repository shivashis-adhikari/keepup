"""what-if search: find settings that keep bloat and xid age down without
spending more vacuum i/o than you're willing to.

the objective, lower is better:

    wasted space   bloat bytes added since the start, time-averaged and
                   summed over tables, in GB
  + wraparound     a steep penalty once a table's xid age passes
                   autovacuum_freeze_max_age by more than 25%, and a
                   bigger one near the failsafe / stop limit
  + lag            p95 wait for a worker, in hours, per table
  + churn          a small cost per changed setting, so we don't suggest
                   twelve changes when two do the job

and the i/o budget is a hard constraint on p99 vacuum read+write MB/s over
5 minute windows. candidates over budget are rejected outright.

the search is greedy coordinate descent over a fixed grid per knob. each
step tries every value of every knob, keeps the best improvement, repeats
until nothing helps. it's not clever, but each evaluation is a full
simulation and the grids are small, so it finishes in minutes and the
answer is easy to explain.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import IO, Any

import numpy as np

from .gucs import XID_STOP_LIMIT
from .profile import Profile
from .sim import SimResult, simulate

GB = 1024.0**3


@dataclass(frozen=True)
class Knob:
    """one thing the search may change. table=None means a server setting."""

    name: str
    values: tuple[Any, ...]
    table: str | None = None

    @property
    def label(self) -> str:
        return f"{self.table}.{self.name}" if self.table else self.name


GLOBAL_KNOBS = (
    Knob("autovacuum_max_workers", (3, 4, 5, 6, 8)),
    Knob("autovacuum_vacuum_cost_limit", (-1, 400, 800, 1500, 3000)),
    Knob("autovacuum_vacuum_cost_delay", (2.0, 1.0, 0.5)),
    Knob("autovacuum_naptime", (60.0, 30.0, 15.0)),
    Knob("autovacuum_vacuum_scale_factor", (0.2, 0.1, 0.05)),
    Knob("autovacuum_vacuum_insert_scale_factor", (0.2, 0.1, 0.05)),
)

TABLE_KNOBS = (
    ("vacuum_scale_factor", (None, 0.1, 0.05, 0.02, 0.01, 0.005)),
    ("vacuum_insert_scale_factor", (None, 0.1, 0.05, 0.01)),
    ("vacuum_cost_limit", (None, 1000, 2000, 5000)),
    ("freeze_min_age", (None, 10_000_000, 1_000_000)),
)


@dataclass
class Config:
    settings: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)

    def with_knob(self, k: Knob, v: Any) -> Config:
        s = dict(self.settings)
        t = {n: dict(o) for n, o in self.tables.items()}
        if k.table is None:
            s[k.name] = v
        else:
            t.setdefault(k.table, {})[k.name] = v
        return Config(s, t)

    def get(self, k: Knob, base: Profile) -> Any:
        if k.table is None:
            return self.settings.get(k.name, getattr(base.settings, k.name))
        cur = self.tables.get(k.table, {})
        if k.name in cur:
            return cur[k.name]
        return getattr(base.table(k.table).options, k.name)

    def apply(self, base: Profile) -> Profile:
        p = base
        if self.settings:
            p = p.with_settings(base.settings.with_values(self.settings))
        if self.tables:
            changes = {}
            for name, opts in self.tables.items():
                changes[name] = replace(base.table(name).options, **opts)
            p = p.with_table_options(changes)
        return p

    def changes_vs(self, base: Profile) -> int:
        n = 0
        for k, v in self.settings.items():
            if getattr(base.settings, k) != v:
                n += 1
        for name, opts in self.tables.items():
            cur = base.table(name).options
            n += sum(1 for k, v in opts.items() if getattr(cur, k) != v)
        return n


@dataclass
class Score:
    total: float
    wasted_gb: float
    wrap_penalty: float
    lag_penalty: float
    churn: float
    peak_io: float
    feasible: bool

    def describe(self) -> str:
        flag = "" if self.feasible else " (over i/o budget)"
        return (
            f"score {self.total:.3f}: bloat {self.wasted_gb:+.2f} GB avg, wraparound "
            f"{self.wrap_penalty:.2f}, lag {self.lag_penalty:.2f}, peak i/o "
            f"{self.peak_io:.1f} MB/s{flag}"
        )


@dataclass
class Weights:
    wasted_gb: float = 1.0
    wraparound: float = 50.0
    lag_hours: float = 0.5
    per_change: float = 0.02


def score(res: SimResult, base: Profile, cfg: Config, io_budget: float | None, w: Weights) -> Score:
    wasted = 0.0
    wrap = 0.0
    lag = 0.0
    for s in res.summarize():
        ts = res.tables[s.name]
        # bloat that's already there at the start can't be tuned away (files
        # don't shrink), so only count what gets added
        wb = ts.wasted_bytes
        wasted += float(np.mean(wb) - wb[0]) / GB
        over = s.age_max / ts.freeze_max_age - 1.25
        if over > 0:
            wrap += over
        near_stop = s.age_max / XID_STOP_LIMIT
        if near_stop > 0.5:
            wrap += 10.0 * (near_stop - 0.5)
        wrap += 0.5 * s.failsafe
        lag += s.lag_p95_s / 3600.0
    peak = res.peak_io_mbps()
    churn = cfg.changes_vs(base) * w.per_change
    feasible = io_budget is None or peak <= io_budget
    total = w.wasted_gb * wasted + w.wraparound * wrap + w.lag_hours * lag + churn
    if not feasible:
        total += 1e6
    return Score(total, wasted, wrap, lag, churn, peak, feasible)


def _evaluate(args: tuple[Profile, Config, float, float | None, Weights]) -> Score:
    base, cfg, days, budget, w = args
    res = simulate(cfg.apply(base), days=days)
    return score(res, base, cfg, budget, w)


def default_knobs(base: Profile, top_tables: int = 4, days: float = 30.0) -> list[Knob]:
    knobs = list(GLOBAL_KNOBS)
    # per-table knobs only for the tables that look like they need help, the
    # grid gets big fast otherwise
    res = simulate(base, days=days)
    ranked = sorted(res.summarize(), key=lambda s: -(s.wasted_mb_mean + 1e3 * (s.verdict != "ok")))
    for s in ranked[:top_tables]:
        for name, values in TABLE_KNOBS:
            knobs.append(Knob(name, values, table=s.name))
    return knobs


@dataclass
class SearchResult:
    base: Config
    best: Config
    base_score: Score
    best_score: Score
    history: list[tuple[str, Any, Score]]
    profile: Profile

    @property
    def improved(self) -> bool:
        return self.best_score.total < self.base_score.total - 1e-9


def search(
    base: Profile,
    io_budget: float | None = None,
    days: float = 30.0,
    knobs: Sequence[Knob] | None = None,
    weights: Weights | None = None,
    max_rounds: int = 6,
    jobs: int | None = None,
    log: IO[str] = sys.stderr,
    progress: Callable[[str], None] | None = None,
) -> SearchResult:
    w = weights or Weights()
    knobs = list(knobs) if knobs is not None else default_knobs(base, days=days)
    jobs = jobs or min(os.cpu_count() or 1, 8)
    say = progress or (lambda m: print(m, file=log))

    cur = Config()
    cur_score = _evaluate((base, cur, days, io_budget, w))
    base_score = cur_score
    say(f"baseline: {cur_score.describe()}")
    history: list[tuple[str, Any, Score]] = []

    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for rnd in range(max_rounds):
            cands: list[tuple[Knob, Any, Config]] = []
            for k in knobs:
                now = cur.get(k, base)
                for v in k.values:
                    if v == now:
                        continue
                    cands.append((k, v, cur.with_knob(k, v)))
            if not cands:
                break
            args = [(base, c, days, io_budget, w) for _, _, c in cands]
            scores = list(pool.map(_evaluate, args)) if pool else [_evaluate(a) for a in args]
            best_i = min(range(len(scores)), key=lambda i: scores[i].total)
            if scores[best_i].total >= cur_score.total - 1e-6:
                say(f"round {rnd + 1}: nothing beats the current config, stopping")
                break
            k, v, cfg = cands[best_i]
            cur, cur_score = cfg, scores[best_i]
            history.append((k.label, v, cur_score))
            say(f"round {rnd + 1}: {k.label} = {_fmt(v)} -> {cur_score.describe()}")
    finally:
        if pool:
            pool.shutdown()
    cur = _prune(base, cur, days, io_budget, w, cur_score, say)
    final = _evaluate((base, cur, days, io_budget, w))
    return SearchResult(Config(), cur, base_score, final, history, cur.apply(base))


def _prune(
    base: Profile,
    cfg: Config,
    days: float,
    budget: float | None,
    w: Weights,
    sc: Score,
    say: Callable[[str], None],
) -> Config:
    """drop changes that stopped mattering once later ones went in."""
    for name in list(cfg.settings):
        trial = Config({k: v for k, v in cfg.settings.items() if k != name}, cfg.tables)
        s = _evaluate((base, trial, days, budget, w))
        if s.total <= sc.total + 1e-6:
            say(f"dropping {name}, it no longer helps")
            cfg, sc = trial, s
    for table in list(cfg.tables):
        for opt in list(cfg.tables[table]):
            tables = {t: dict(o) for t, o in cfg.tables.items()}
            del tables[table][opt]
            if not tables[table]:
                del tables[table]
            trial = Config(cfg.settings, tables)
            s = _evaluate((base, trial, days, budget, w))
            if s.total <= sc.total + 1e-6:
                say(f"dropping {table}.{opt}, it no longer helps")
                cfg, sc = trial, s
    return cfg


def _fmt(v: Any) -> str:
    if v is None:
        return "default"
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:g}"
    return str(v)
