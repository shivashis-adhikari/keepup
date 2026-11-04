"""what one autovacuum run on one table costs, and what it leaves behind.

the shape follows lazy_scan_heap in vacuumlazy.c: one pass over the heap
pages the visibility map doesn't let us skip, then one or more passes over
every index if there were enough dead items, then a second heap pass over
the pages that had dead items. costs are in vacuum cost credits
(vacuum_cost_page_hit/miss/dirty) because that's what the throttle meters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .table import TableSim

SKIP_PAGES_THRESHOLD = 32
BYPASS_THRESHOLD_PAGES = 0.02
MAX_DEAD_ITEMS_BYPASS = 32 * 1024 * 1024 // 6  # MAXDEADITEMS(32MB)
ITEM_POINTER_BYTES = 6
# pg17's tidstore is a radix tree keyed by block. this per-page figure is a
# rough average from poking at it, not something the source states.
TIDSTORE_BYTES_PER_PAGE = 48
MAX_ALLOC = 1024 * 1024 * 1024
ANALYZE_PAGES_PER_TARGET = 300
EAGER_SCAN_REGION_SIZE = 4096
MAX_EAGER_FREEZE_SUCCESS_RATE = 0.2


@dataclass
class Plan:
    """everything decided when the vacuum starts."""

    start: float
    vacuum: bool
    analyze: bool
    wraparound: bool
    aggressive: bool
    failsafe: bool
    scanned: float = 0.0
    lpdead_pages: float = 0.0
    removable: float = 0.0
    pinned: float = 0.0
    index_passes: int = 0
    index_pages: float = 0.0
    analyze_pages: float = 0.0
    hits: float = 0.0
    misses: float = 0.0
    dirtied: float = 0.0
    credits: float = 0.0
    vacuum_credits: float = 0.0
    cpu_seconds: float = 0.0
    newly_frozen: float = 0.0
    eager_scanned: float = 0.0
    frozen_after: float = 0.0
    tail_at_start: float = 0.0
    scatter_at_start: float = 0.0
    advances_relfrozenxid: bool = False
    new_relfrozenxid: float = 0.0
    dead_curve_at_start: float = 0.0
    pages_at_start: float = 0.0

    @property
    def page_reads(self) -> float:
        return self.hits + self.misses

    @property
    def avg_page_cost(self) -> float:
        n = self.page_reads
        return self.credits / n if n > 0 else 1.0


def _skip_fraction(q: float) -> float:
    """share of skippable pages that sit in runs of >= 32 (so really get skipped),
    when each page in a region is skippable independently with probability q."""
    if q <= 0:
        return 0.0
    if q >= 1:
        return 1.0
    n = SKIP_PAGES_THRESHOLD
    return float(q ** (n - 1) * (n - (n - 1) * q))


def _scanned_pages(tb: TableSim, aggressive: bool) -> float:
    """pages lazy_scan_heap reads.

    the heap is three regions: the appended tail (all of it needs reading),
    the active region where updates and deletes scatter not-all-visible
    pages, and cold pages that are one long all-visible run. inside the
    active region the skippable pages sit in short runs between scattered
    dirty ones, and runs under SKIP_PAGES_THRESHOLD get read anyway.
    """
    P = tb.pages
    tail = min(tb.nav_tail, P)
    body = max(P - tail, 0.0)
    hot = min(max(tb.active_fraction * P, 1.0), max(body, 1.0))
    scatter = min(tb.nav_scatter, hot)
    if aggressive:
        must = max(P - tb.frozen, 0.0)
        av = max(P - tb.nav, 1.0)
        skippable_hot = tb.frozen * max(hot - scatter, 0.0) / av
    else:
        must = tb.nav
        skippable_hot = max(hot - scatter, 0.0)
    q = min(skippable_hot / hot, 1.0) if hot > 0 else 1.0
    read_anyway = skippable_hot * (1.0 - _skip_fraction(q))
    return min(must + read_anyway, P)


def plan_vacuum(
    tb: TableSim,
    t: float,
    do_vacuum: bool,
    do_analyze: bool,
    wraparound: bool,
    pin_start: float | None,
    page_hit: int,
    page_miss: int,
    page_dirty: int,
    work_mem_kb: int,
    stats_target: int,
    analyze_row_seconds: float = 0.0,
) -> Plan:
    tb.advance(t)
    e = tb.eff
    s = tb.settings
    age = tb.age(t)
    # vacuum_get_cutoffs decides this from freeze_table_age alone. a forced
    # anti-wraparound run is normally aggressive too, but not when a table
    # has its own low autovacuum_freeze_max_age: then it's "automatic vacuum
    # to prevent wraparound" without the aggressive part, which the source
    # calls a corner case and the lab hit straight away
    aggressive = do_vacuum and age > e.freeze_table_age
    failsafe = do_vacuum and age > e.failsafe_age
    p = Plan(t, do_vacuum, do_analyze, wraparound, aggressive, failsafe)
    P = tb.pages
    p.pages_at_start = P
    p.tail_at_start = tb.nav_tail
    p.scatter_at_start = tb.nav_scatter
    p.dead_curve_at_start = tb.dead_curve.at(t)
    reads = 0.0
    dirtied = 0.0

    if do_vacuum:
        active = max(tb.active_fraction * P, 1.0)
        p.scanned = _scanned_pages(tb, aggressive)

        pinned = 0.0
        if pin_start is not None and pin_start <= t:
            pinned = tb.dead_curve.at(t) - tb.dead_curve.at(pin_start)
            pinned = min(max(pinned, 0.0), tb.dead_phys)
        p.pinned = pinned
        p.removable = max(tb.dead_phys - pinned, 0.0)

        dead_region = max(active, 1.0)
        p.lpdead_pages = min(dead_region * -math.expm1(-p.removable / dead_region), p.scanned)

        n_idx = tb.prof.n_indexes
        idx_pages = tb.prof.index_ratio * P
        bypass = p.lpdead_pages < BYPASS_THRESHOLD_PAGES * P and p.removable < MAX_DEAD_ITEMS_BYPASS
        if n_idx > 0 and p.removable > 0 and not p.failsafe and not bypass:
            mem = work_mem_kb * 1024.0
            if s.version <= 16:
                need = p.removable * ITEM_POINTER_BYTES
                mem = min(mem, MAX_ALLOC)
            else:
                need = p.lpdead_pages * TIDSTORE_BYTES_PER_PAGE
            p.index_passes = max(1, math.ceil(need / max(mem, 1.0)))
            p.index_pages = idx_pages * p.index_passes
        reads += p.scanned + p.index_pages
        if p.index_passes:
            reads += p.lpdead_pages  # second heap pass

        # freezing. pages touched in the last freeze_min_age xids hold rows too
        # young to freeze, everything older gets frozen by an aggressive run
        if aggressive:
            young = _young_pages(tb, t, e.freeze_min_age, active)
            p.frozen_after = max(P - young, tb.frozen)
        else:
            # any vacuum freezes the pages it reads whose rows are all older
            # than the cutoff; normally few of the not-all-visible pages
            # qualify, with a small freeze_min_age most of them do. on top
            # of that, opportunistic freezing of some young pages (16+).
            scanned_unfrozen = min(tb.nav, p.scanned)
            young = _young_pages(tb, t, e.freeze_min_age, active)
            old = max(scanned_unfrozen - young, 0.0)
            opportunistic = tb.prof.freeze_fraction * (scanned_unfrozen - old)
            p.frozen_after = min(tb.frozen + old + opportunistic, P)
            fail_rate = s.vacuum_max_eager_freeze_failure_rate
            if fail_rate > 0 and P >= 2 * EAGER_SCAN_REGION_SIZE and age > e.freeze_min_age:
                eager, ok = _eager_scan(tb, t, e.freeze_min_age, active, fail_rate)
                p.eager_scanned = eager
                p.scanned = min(p.scanned + eager, P)
                p.frozen_after = min(p.frozen_after + ok, P)
        p.newly_frozen = max(p.frozen_after - tb.frozen, 0.0)

        # dirtied heap pages: setting PD_ALL_VISIBLE on the not-all-visible
        # ones (unless they're still dirty from whatever cleared the bit),
        # plus freezing pages that were all-visible but not frozen, which
        # are clean by definition
        dshare = _clean_share(tb, t)
        frozen_from_clean = max(p.newly_frozen - tb.nav, 0.0)
        dirtied += min(tb.nav * dshare + frozen_from_clean, p.scanned)
        if p.index_passes:
            # non-hot dead tuples have an entry in every index; hot ones don't.
            # an index page that keeps getting new entries stays in cache and
            # stays dirty, so vacuum only pays for the ones that went quiet
            # for a checkpoint interval
            nonhot = 1.0 - tb.prof.hot_fraction
            entries = p.removable * n_idx * nonhot
            touched = idx_pages * -math.expm1(-entries / max(idx_pages, 1.0))
            new_entries = (tb.ins.rate_at(t) + tb.upd.rate_at(t) * nonhot) * n_idx
            quiet = math.exp(-new_entries / max(idx_pages, 1.0) * s.checkpoint_timeout)
            dirtied += touched * quiet
            # the second heap pass re-dirties pages that got evicted and
            # written out after the first pass; only happens on cold tables
            dirtied += p.lpdead_pages * dshare * (1.0 - tb.prof.cache_hit) ** 2

        # relfrozenxid moves if the run was aggressive, or happened to look at
        # every page that isn't all-frozen
        p.advances_relfrozenxid = aggressive or p.scanned + tb.frozen >= P - 0.5
        # since 15 the new value is the oldest xid still unfrozen after the
        # run (NewRelfrozenXid). everything older than the freeze cutoff got
        # frozen; anything unfrozen that's younger stays where it is.
        x_now = tb.xid.at(t)
        new = max(tb.oldest_unfrozen, x_now - float(e.freeze_min_age))
        if pin_start is not None and pin_start <= t:
            new = min(new, tb.xid.at(pin_start))
        p.new_relfrozenxid = min(new, x_now)

    if do_analyze:
        p.analyze_pages = min(P, ANALYZE_PAGES_PER_TARGET * stats_target)
        reads += p.analyze_pages
        rows = min(tb.live, ANALYZE_PAGES_PER_TARGET * stats_target)
        p.cpu_seconds = rows * analyze_row_seconds
    vacuum_reads = reads - p.analyze_pages

    h = tb.prof.cache_hit
    # plus one free space map lookup per scanned heap page, always a hit.
    # shows up clearly in the logs: hits ~= 2x scanned on a cached table
    p.hits = reads * h + p.scanned
    p.misses = reads * (1.0 - h)
    p.dirtied = dirtied
    p.credits = p.hits * page_hit + p.misses * page_miss + p.dirtied * page_dirty
    # the server logs vacuum and analyze as separate entries, so keep track of
    # how much of this job was the vacuum part
    vac_credits = (vacuum_reads * h + p.scanned) * page_hit + vacuum_reads * (1 - h) * page_miss
    vac_credits += p.dirtied * page_dirty
    p.vacuum_credits = vac_credits if p.vacuum else 0.0
    return p


def _clean_share(tb: TableSim, t: float) -> float:
    """share of not-all-visible pages that are clean by the time vacuum gets there.

    a page stays dirty in shared_buffers for about T seconds after it was
    changed, then a checkpoint or the background writer writes it out. with
    changes spread over the D seconds since the last vacuum the clean share
    is 1 - T/D. T is checkpoint_timeout on a box with room in its cache (the
    lab logged 0.07%, 29%, 54% at D = 122, 428, 650s with T = 300; this gives
    0, 30, 54) and close to zero on one without.
    """
    if tb.prof.dirty_fraction is not None:
        return tb.prof.dirty_fraction
    # residency: checkpoint_timeout at most, much less when the cache is under
    # pressure and the background writer gets to pages first
    c = tb.settings.checkpoint_timeout
    if tb.prof.dirty_residency is not None:
        c = min(tb.prof.dirty_residency, c)
    d = t - tb.last_vacuum_end if tb.last_vacuum_end is not None else t + tb.since_vacuum_at_start
    return max(0.0, 1.0 - c / d) if d > 0 else 0.0


def _eager_scan(
    tb: TableSim, t: float, freeze_min_age: float, active: float, fail_rate: float
) -> tuple[float, float]:
    """pg18 eager scanning in a normal vacuum: returns (pages read, pages frozen).

    it reads all-visible-but-not-frozen pages until it has frozen 20% of them
    or failed on fail_rate of each 4096-page region. a page fails when it
    holds rows younger than the freeze cutoff.
    """
    cand = max(tb.pages - tb.nav - tb.frozen, 0.0)
    if cand <= 0:
        return 0.0, 0.0
    young = _young_pages(tb, t, freeze_min_age, active)
    old = 1.0 - min(young / max(tb.pages, 1.0), 1.0)
    want = MAX_EAGER_FREEZE_SUCCESS_RATE * cand
    by_success = want / old if old > 0 else math.inf
    by_fail = fail_rate * tb.pages / (1.0 - old) if old < 1 else math.inf
    scanned = min(cand, by_success, by_fail)
    return scanned, scanned * old


def _young_pages(tb: TableSim, t: float, freeze_min_age: float, active: float) -> float:
    x_now = tb.xid.at(t)
    t_cut = tb.xid.time_of(x_now - freeze_min_age)
    if t_cut == -math.inf:
        return 0.0
    mods = tb.mod_curve.at(t) - tb.mod_curve.at(t_cut)
    ins = tb.ins.at(t) - tb.ins.at(t_cut)
    young = active * -math.expm1(-mods / active) + ins / tb.tpp
    return min(young, tb.pages)


def finish(tb: TableSim, p: Plan, t: float) -> None:
    """apply a finished run to the table."""
    tb.advance(t)
    if p.vacuum:
        died_during = tb.dead_curve.at(t) - p.dead_curve_at_start
        tb.dead_phys = max(tb.dead_phys - p.removable, 0.0)
        # pgstat_report_vacuum overwrites n_dead_tup with what vacuum counted
        # as recently dead. rows that died behind the scan never get counted,
        # so the stat drifts low after long runs. assume half were seen.
        tb.dead_stats = min(p.pinned + 0.5 * died_during, tb.dead_phys)
        tb.ins_since_vac = 0.0
        tb.reltuples = tb.live
        pinned_pages = 0.0
        if p.pinned > 0:
            region = max(tb.active_fraction * tb.pages, 1.0)
            pinned_pages = region * -math.expm1(-p.pinned / region)
        # whatever got dirtied while we were scanning stays not-all-visible
        tb.nav_tail = max(tb.nav_tail - p.tail_at_start, 0.0)
        tb.nav_scatter = min(
            max(tb.nav_scatter - p.scatter_at_start, 0.0) + pinned_pages,
            max(tb.pages - tb.nav_tail, 0.0),
        )
        tb.frozen = min(p.frozen_after, tb.pages - tb.nav)
        if p.lpdead_pages > 1 and p.removable > 0:
            tb.free_density = max(p.removable / p.lpdead_pages, 1.0)
        if p.advances_relfrozenxid:
            # decided by the cutoffs taken at start
            tb.relfrozenxid = max(tb.relfrozenxid, p.new_relfrozenxid)
            tb.oldest_unfrozen = max(tb.oldest_unfrozen, tb.relfrozenxid)
        tb.last_vacuum_end = t
    if p.analyze:
        tb.mod_since_anl = 0.0
        tb.reltuples = tb.live
        # analyze also overwrites n_dead_tup, with its sample-based estimate
        tb.dead_stats = tb.dead_phys
