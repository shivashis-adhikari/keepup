"""the event loop: one launcher, up to autovacuum_max_workers workers.

the scheduling rules come from src/backend/postmaster/autovacuum.c (pg 16):

- the launcher keeps a list of databases, each with a next_worker time,
  spread evenly across autovacuum_naptime. it sleeps until the soonest one.
- when it wakes and a worker slot is free, do_start_worker picks a database:
  the one closest to xid wraparound if any is past autovacuum_freeze_max_age,
  otherwise the least recently autovacuumed one that wasn't just scheduled.
  that database's next_worker moves to now + naptime.
- if no slot is free it just waits; a worker exiting wakes it up again, but
  it still only launches if a next_worker time has come due.
- a worker collects every table that needs work at the moment it starts,
  walks them in pg_class order, rechecks each one right before vacuuming it,
  and skips anything another worker is busy with. then it exits.
- cost limit is split evenly across running workers whose table has no
  per-table cost settings (autovac_recalculate_workers_for_balance). per
  table settings take a worker out of the pool and it gets the full value.

durations come from the cost model: a run needs N credits, and each credit
costs (delay / limit) seconds of sleep plus 1/raw_rate seconds of real work.
whenever the number of balanced workers changes, every running job's rate
changes and its finish time gets rescheduled.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..gucs import Settings, effective
from ..profile import HorizonPin, Profile
from ..workload import cumulative
from .curves import Curve
from .results import Run, SimResult, TableSeries
from .table import TableSim
from .vacuum import Plan, finish, plan_vacuum

MIN_SLEEP = 0.1  # MIN_AUTOVAC_SLEEPTIME, seconds
MAX_SLEEP = 300.0  # MAX_AUTOVAC_SLEEPTIME

_WAKE, _DONE, _STOP = 0, 1, 2


@dataclass
class Job:
    worker: Worker
    table: TableSim
    plan: Plan
    start: float
    lag: float
    done: float = 0.0
    rate: float = 0.0
    last: float = 0.0
    version: int = 0
    limit: int = 0
    in_cpu_tail: bool = False


@dataclass
class Worker:
    id: int
    db: DbState
    queue: list[TableSim]
    job: Job | None = None


@dataclass
class DbState:
    name: str
    tables: list[TableSim]
    next_worker: float = 0.0
    last_autovac: float = -math.inf
    # cached mins over the tables; only change when a run finishes
    _next_need: float = math.inf
    _min_frozenxid: float = math.inf

    def refresh(self) -> None:
        self._next_need = min((t.next_need for t in self.tables), default=math.inf)
        self._min_frozenxid = min((t.relfrozenxid for t in self.tables), default=math.inf)

    def next_need(self) -> float:
        return self._next_need

    def max_age(self, t: float) -> float:
        # datfrozenxid is the min relfrozenxid; catalogs aren't modelled
        if not self.tables:
            return 0.0
        return self.tables[0].xid.at(t) - self._min_frozenxid


@dataclass
class Curves:
    ins: list[Curve]
    upd: list[Curve]
    dele: list[Curve]
    xid: Curve


def build_curves(profile: Profile, hours: int, rng: np.random.Generator | None = None) -> Curves:
    start = profile.start
    tz = profile.timezone

    def mk(rm_rates: np.ndarray) -> Curve:
        return Curve(cumulative(rm_rates))

    ins = [mk(t.inserts.hourly_rates(start, hours, tz, rng)) for t in profile.tables]
    upd = [mk(t.updates.hourly_rates(start, hours, tz, rng)) for t in profile.tables]
    dele = [mk(t.deletes.hourly_rates(start, hours, tz, rng)) for t in profile.tables]
    xid = mk(profile.xid_rate.hourly_rates(start, hours, tz, rng))
    return Curves(ins, upd, dele, xid)


@dataclass
class _Ev:
    t: float
    seq: int
    kind: int
    ref: Job | None = field(compare=False, default=None)
    version: int = field(compare=False, default=0)

    def __lt__(self, other: _Ev) -> bool:
        return (self.t, self.seq) < (other.t, other.seq)


class Simulation:
    def __init__(
        self,
        profile: Profile,
        days: float = 90.0,
        sample_every: float = 3600.0,
        curves: Curves | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.profile = profile
        self.s: Settings = profile.settings
        self.horizon = days * 86400.0
        hours = math.ceil(self.horizon / 3600.0) + 1
        self.curves = curves or build_curves(profile, hours, rng)
        self.sample_every = sample_every
        self.tables: list[TableSim] = []
        order = sorted(enumerate(profile.tables), key=lambda x: (x[1].database, x[1].oid))
        for i, (idx, tp) in enumerate(order):
            tb = TableSim(
                tp,
                effective(self.s, tp.options),
                self.s,
                self.curves.ins[idx],
                self.curves.upd[idx],
                self.curves.dele[idx],
                self.curves.xid,
                i,
            )
            tb.compute_next_need()
            self.tables.append(tb)
        self.dbs = [
            DbState(name, [t for t in self.tables if t.db == name]) for name in profile.databases
        ]
        for db in self.dbs:
            db.refresh()
        self._db_of = {tb.name: db for db in self.dbs for tb in db.tables}
        self.pins: Sequence[HorizonPin] = profile.pins
        self.workers: list[Worker] = []
        self.next_worker_id = 0
        self.runs: list[Run] = []
        self.heap: list[_Ev] = []
        self.seq = 0
        self.wake_version = 0
        self.busy_seconds = 0.0
        self.saturated_seconds = 0.0
        self._last_acct = 0.0
        self.now = 0.0

    # -- event plumbing ------------------------------------------------------

    def _push(self, t: float, kind: int, ref: Job | None = None, version: int = 0) -> None:
        self.seq += 1
        heapq.heappush(self.heap, _Ev(t, self.seq, kind, ref, version))

    def _account(self, t: float) -> None:
        dt = t - self._last_acct
        if dt > 0:
            n = len(self.workers)
            self.busy_seconds += dt * sum(1 for w in self.workers if w.job is not None)
            if n >= self.s.autovacuum_max_workers:
                self.saturated_seconds += dt
            self._last_acct = t

    # -- launcher ------------------------------------------------------------

    def _rebuild(self, t: float) -> None:
        # rebuild_database_list: spread next_worker over one naptime, keeping
        # the existing order
        n = len(self.dbs)
        inc = max(self.s.autovacuum_naptime / n, MIN_SLEEP * 1.1)
        order = sorted(self.dbs, key=lambda d: d.last_autovac)
        for i, db in enumerate(order):
            db.next_worker = t + (i + 1) * inc

    def _schedule_wake(self, t: float) -> None:
        self.wake_version += 1
        self._push(t, _WAKE, version=self.wake_version)

    def _wake(self, t: float) -> None:
        nap = self.s.autovacuum_naptime
        can_launch = len(self.workers) < self.s.autovacuum_max_workers
        if not can_launch:
            # sleep until a worker exits, _worker_exit wakes us
            return
        tail = min(self.dbs, key=lambda d: d.next_worker)
        if tail.next_worker <= t + 1e-9:
            self._launch(t)
        tail = min(self.dbs, key=lambda d: d.next_worker)
        if tail.next_worker <= t:
            self._rebuild(t)
            tail = min(self.dbs, key=lambda d: d.next_worker)
        sleep = min(max(tail.next_worker - t, MIN_SLEEP), MAX_SLEEP)
        nxt = t + sleep
        if len(self.dbs) == 1 and not self.workers:
            # nothing running and nothing to do: hop straight to the first
            # naptime tick at or after the next threshold crossing. a worker
            # started on an idle database exits immediately, so skipping
            # those wakes changes nothing.
            db = self.dbs[0]
            need = db.next_need()
            if need > nxt:
                if need >= self.horizon:
                    return
                k = math.ceil((need - db.next_worker) / nap - 1e-9)
                db.next_worker += max(k, 0) * nap
                db.last_autovac = db.next_worker - nap
                nxt = db.next_worker
        self._schedule_wake(nxt)

    def _pick_db(self, t: float) -> DbState | None:
        nap = self.s.autovacuum_naptime
        fmax = self.s.autovacuum_freeze_max_age
        if len(self.dbs) == 1:
            d = self.dbs[0]
            if d.max_age(t) > fmax:
                return d
        else:
            at_risk = [d for d in self.dbs if d.max_age(t) > fmax]
            if at_risk:
                return max(at_risk, key=lambda d: d.max_age(t))
        if not self.s.autovacuum:
            return None
        best = None
        for d in self.dbs:
            if t < d.next_worker < t + nap:
                continue  # just had one
            if best is None or d.last_autovac < best.last_autovac:
                best = d
        return best

    def _launch(self, t: float) -> None:
        db = self._pick_db(t)
        if db is None:
            self._rebuild(t)
            return
        db.last_autovac = t
        db.next_worker = t + self.s.autovacuum_naptime
        if db.next_need() > t + 1e-6:
            return  # the worker would find nothing and exit right away
        # pg puts busy tables on the list too and skips them later; same thing
        queue = [
            tb for tb in db.tables if not tb.busy and tb.next_need <= t + 1e-6 and tb.needs(t).any
        ]
        if not self.s.autovacuum:
            queue = [tb for tb in queue if tb.needs(t).wraparound]
        if not queue:
            return
        w = Worker(self.next_worker_id, db, queue)
        self.next_worker_id += 1
        self.workers.append(w)
        self._next_table(w, t)

    # -- workers -------------------------------------------------------------

    def _pin_start(self, t: float) -> float | None:
        active = [
            p.start_hours * 3600.0
            for p in self.pins
            if p.start_hours * 3600.0 <= t < p.end_hours * 3600.0
        ]
        return min(active) if active else None

    def _next_table(self, w: Worker, t: float) -> None:
        s = self.s
        while w.queue:
            tb = w.queue.pop(0)
            if tb.busy:
                continue
            need = tb.needs(t)
            if not s.autovacuum and not need.wraparound:
                continue
            if not need.any:
                continue
            plan = plan_vacuum(
                tb,
                t,
                need.vacuum,
                need.analyze,
                need.wraparound,
                self._pin_start(t),
                s.vacuum_cost_page_hit,
                s.vacuum_cost_page_miss,
                s.vacuum_cost_page_dirty,
                s.work_mem_kb,
                s.default_statistics_target,
                self.profile.analyze_row_seconds,
            )
            # next_need is the exact crossing time as long as nothing touched
            # the table since, which is the case: only a finished run does
            job = Job(w, tb, plan, start=t, lag=max(t - tb.next_need, 0.0), last=t)
            tb.busy = True
            w.job = job
            self._rebalance(t)
            return
        self._worker_exit(w, t)

    def _worker_exit(self, w: Worker, t: float) -> None:
        self.workers.remove(w)
        self._rebalance(t)
        self._schedule_wake(t)

    def _job_rate(self, job: Job, n_balanced: int) -> tuple[float, int]:
        e = job.table.eff
        raw = self.profile.raw_credit_rate
        if job.plan.failsafe or e.cost_delay_ms <= 0:
            return raw, 0
        if e.dobalance:
            limit = max(e.cost_limit // max(n_balanced, 1), 1)
        else:
            limit = e.cost_limit
        # vacuum_delay_point: once the balance reaches the limit, sleep
        # delay * balance / limit, capped at 4x delay. the balance overshoots
        # the limit by about half a page's cost on average.
        b = limit + job.plan.avg_page_cost / 2.0
        per_credit = (
            self.profile.sleep_overhead * e.cost_delay_ms / 1000.0 * min(1.0 / limit, 4.0 / b)
        )
        return 1.0 / (per_credit + 1.0 / raw), limit

    def _rebalance(self, t: float) -> None:
        jobs = [w.job for w in self.workers if w.job is not None]
        # a worker in failsafe mode stops sleeping but keeps its wi_dobalance
        # flag, so it still eats a share of everyone else's limit
        n_bal = sum(1 for j in jobs if j.table.eff.dobalance)
        for j in jobs:
            if j.in_cpu_tail:
                continue  # done with i/o, the throttle doesn't matter any more
            j.done += j.rate * (t - j.last)
            j.last = t
            rate, limit = self._job_rate(j, n_bal)
            if rate != j.rate or j.version == 0:
                j.rate = rate
                j.limit = limit
                j.version += 1
                remaining = max(j.plan.credits - j.done, 0.0)
                self._push(t + remaining / rate, _DONE, j, j.version)

    def _job_done(self, job: Job, t: float) -> None:
        tb = job.table
        finish(tb, job.plan, t)
        tb.busy = False
        tb.compute_next_need()
        self._db_of[tb.name].refresh()
        self._record(job, t)
        w = job.worker
        w.job = None
        self._next_table(w, t)

    def _record(self, job: Job, end: float | None) -> None:
        p = job.plan
        vac_s = None
        if end is not None and p.vacuum and p.credits > 0:
            io_s = max(end - job.start - p.cpu_seconds, 0.0)
            vac_s = io_s * p.vacuum_credits / p.credits
        self.runs.append(
            Run(
                table=job.table.name,
                db=job.table.db,
                worker=job.worker.id,
                start=job.start,
                end=end,
                vacuum=p.vacuum,
                analyze=p.analyze,
                wraparound=p.wraparound,
                aggressive=p.aggressive,
                failsafe=p.failsafe,
                scanned=p.scanned,
                index_passes=p.index_passes,
                removed=p.removable,
                credits=p.credits,
                hits=p.hits,
                misses=p.misses,
                dirtied=p.dirtied,
                lag=job.lag,
                vacuum_seconds=vac_s,
            )
        )

    # -- main loop -----------------------------------------------------------

    def run(self) -> SimResult:
        n_samples = int(self.horizon // self.sample_every) + 1
        times = np.arange(n_samples, dtype=np.float64) * self.sample_every
        n_tab = len(self.tables)
        live = np.zeros((n_tab, n_samples))
        dstat = np.zeros((n_tab, n_samples))
        dphys = np.zeros((n_tab, n_samples))
        pages = np.zeros((n_tab, n_samples))
        age = np.zeros((n_tab, n_samples))
        vth = np.zeros((n_tab, n_samples))
        si = 0

        self._rebuild(0.0)
        first = min(d.next_worker for d in self.dbs)
        self._schedule_wake(first)
        self._push(self.horizon, _STOP)

        while self.heap:
            ev = heapq.heappop(self.heap)
            t = ev.t
            while si < n_samples and times[si] <= t:
                ts = float(times[si])
                for i, tb in enumerate(self.tables):
                    tb.advance(ts)
                    live[i, si] = tb.live
                    dstat[i, si] = tb.dead_stats
                    dphys[i, si] = tb.dead_phys
                    pages[i, si] = tb.pages
                    age[i, si] = tb.age(ts)
                    vth[i, si] = tb.eff.vacuum_threshold(tb.reltuples)
                si += 1
            if ev.kind == _STOP:
                break
            self._account(t)
            self.now = t
            if ev.kind == _WAKE:
                if ev.version == self.wake_version:
                    self._wake(t)
            elif ev.kind == _DONE:
                job = ev.ref
                assert job is not None
                if job.version == ev.version and job.worker.job is job:
                    if not job.in_cpu_tail and job.plan.cpu_seconds > 0:
                        # i/o is done, analyze still has to crunch its sample
                        job.in_cpu_tail = True
                        job.version += 1
                        self._push(t + job.plan.cpu_seconds, _DONE, job, job.version)
                    else:
                        self._job_done(job, t)

        self._account(self.horizon)
        for w in self.workers:
            if w.job is not None:
                self._record(w.job, None)

        series = {}
        for i, tb in enumerate(self.tables):
            series[tb.name] = TableSeries(
                name=tb.name,
                db=tb.db,
                tpp=tb.tpp,
                live=live[i],
                dead_stats=dstat[i],
                dead_phys=dphys[i],
                pages=pages[i],
                age=age[i],
                vac_threshold=vth[i],
                freeze_max_age=float(tb.eff.freeze_max_age),
            )
        return SimResult(
            start=self.profile.start,
            horizon=self.horizon,
            sample_times=times,
            tables=series,
            runs=self.runs,
            settings=self.s,
            busy_worker_seconds=self.busy_seconds,
            saturated_seconds=self.saturated_seconds,
            naptime=self.s.autovacuum_naptime,
        )


def simulate(
    profile: Profile, days: float = 90.0, seed: int | None = None, sample_every: float = 3600.0
) -> SimResult:
    rng = np.random.default_rng(seed) if seed is not None else None
    return Simulation(profile, days=days, sample_every=sample_every, rng=rng).run()
