"""simulated state of one table.

state is advanced lazily: nothing happens to a table between events except
counters growing along its curves, so we only do arithmetic when someone
looks at it. vacuums change state at start (snapshot of what's removable)
and at end (apply the result).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..gucs import Effective, Settings
from ..profile import TableProfile
from .curves import Curve

INF = math.inf


@dataclass
class Needs:
    vacuum: bool
    analyze: bool
    wraparound: bool

    @property
    def any(self) -> bool:
        return self.vacuum or self.analyze


class TableSim:
    """one table's counters and physical shape over time."""

    def __init__(
        self,
        prof: TableProfile,
        eff: Effective,
        settings: Settings,
        ins: Curve,
        upd: Curve,
        dele: Curve,
        xid: Curve,
        index: int,
    ):
        self.prof = prof
        self.name = prof.name
        self.db = prof.database
        self.eff = eff
        self.settings = settings
        self.index = index
        self.ins, self.upd, self.dele, self.xid = ins, upd, dele, xid
        self.dead_yield = prof.dead_yield
        self.dead_curve = dele + upd.scaled(self.dead_yield)
        self.mod_curve = ins + upd + dele
        self._ci, self._cu, self._cd = ins.at(0.0), upd.at(0.0), dele.at(0.0)
        self.tpp = prof.tpp
        self.active_fraction = min(max(prof.active_fraction, 1e-4), 1.0)

        self.t = 0.0
        self.live = float(prof.live_tuples)
        self.dead_stats = float(prof.dead_tuples)
        self.dead_phys = float(prof.dead_tuples)
        self.ins_since_vac = float(prof.ins_since_vacuum)
        self.mod_since_anl = float(prof.mod_since_analyze)
        self.reltuples = float(prof.reltuples if prof.reltuples is not None else prof.live_tuples)
        need = (self.live + self.dead_phys) / self.tpp
        self.pages = max(float(prof.pages), need, 1.0)
        # not-all-visible pages come in two shapes: a contiguous run at the end
        # of the heap from appends, and pages scattered around the active
        # region by updates and deletes. the difference matters because vacuum
        # only skips all-visible runs of 32+ pages.
        nav = self.pages * (1.0 - min(max(prof.all_visible_fraction, 0.0), 1.0))
        r_ins = prof.inserts.base
        r_mod = prof.updates.base + prof.deletes.base
        tail_share = r_ins / (r_ins + r_mod) if r_ins + r_mod > 0 else 0.0
        self.nav_tail = nav * tail_share
        self.nav_scatter = nav - self.nav_tail
        self.frozen = self.pages * min(max(prof.all_frozen_fraction, 0.0), 1.0)
        self.frozen = min(self.frozen, self.pages - nav)
        # rows of free space per page, on pages that have any. the fsm sends
        # inserts to those pages, so this decides how many pages an insert
        # into reused space dirties. vacuum resets it from what it freed.
        self.free_density = max(self.tpp / 2.0, 1.0)
        # relfrozenxid on the same axis as the xid curve, which starts at 0
        self.relfrozenxid = xid.at(0.0) - float(prof.xid_age)
        # oldest xid still sitting unfrozen in the heap. we can't see it, so
        # start from the worst case: as old as relfrozenxid itself
        self.oldest_unfrozen = self.relfrozenxid
        self.busy = False
        self.next_need = 0.0
        # rough time since the last real vacuum, from how far the counters
        # have run: only used for the dirty-page estimate of the first run
        rate = prof.deletes.base + prof.updates.base * self.dead_yield
        if rate > 0 and prof.dead_tuples > 0:
            since = prof.dead_tuples / rate
        elif prof.inserts.base > 0:
            since = prof.ins_since_vacuum / prof.inserts.base
        else:
            since = 86400.0
        self.since_vacuum_at_start = min(since, 86400.0)
        self.last_vacuum_end: float | None = None

    # -- counters -----------------------------------------------------------

    def advance(self, t: float) -> None:
        if t <= self.t:
            return
        # cumulative values at self.t are cached, so this is 3 lookups, not 6
        ci, cu, cd = self.ins.at(t), self.upd.at(t), self.dele.at(t)
        d_ins = ci - self._ci
        d_upd = cu - self._cu
        d_del = cd - self._cd
        self._ci, self._cu, self._cd = ci, cu, cd
        d_dead = d_del + self.dead_yield * d_upd
        self.t = t

        self.live = max(self.live + d_ins - d_del, 0.0)
        self.dead_stats += d_dead
        self.dead_phys += d_dead
        self.ins_since_vac += d_ins
        self.mod_since_anl += d_ins + d_upd + d_del

        # the table only grows when live + dead rows don't fit any more. freed
        # space gets reused before the file is extended, which is the whole
        # point of vacuuming often
        old_pages = self.pages
        need = (self.live + self.dead_phys) / self.tpp
        new_pages = 0.0
        if need > old_pages:
            new_pages = need - old_pages
            self.pages = need

        # visibility map: modifications land on random pages of the active
        # region and clear their all-visible bit. saturating, so applying it
        # in several steps gives the same answer as one big step.
        active = self.active_fraction * old_pages
        active = max(active, 1.0)
        mods = d_upd + d_del
        nav_old = self.nav_tail + self.nav_scatter
        nav_active = nav_old if nav_old < active else active
        touched = (active - nav_active) * -math.expm1(-mods / active)
        into_free = d_ins - new_pages * self.tpp
        if into_free > 0 and old_pages > 0:
            # not random: the fsm packs them into the pages that have room,
            # which are mostly ones the last vacuum cleaned and left all-visible
            fresh = max(old_pages - nav_old, 0.0) / old_pages
            touched += into_free / self.free_density * fresh
        all_vis = old_pages - nav_old
        touched = min(touched, max(all_vis, 0.0))
        if all_vis > 0 and touched > 0:
            self.frozen *= max(0.0, 1.0 - touched / all_vis)
        self.nav_scatter += touched
        self.nav_tail += new_pages
        over = self.nav_tail + self.nav_scatter - self.pages
        if over > 0:
            self.nav_scatter = max(self.nav_scatter - over, 0.0)

    @property
    def nav(self) -> float:
        return self.nav_tail + self.nav_scatter

    def age(self, t: float) -> float:
        return self.xid.at(t) - self.relfrozenxid

    @property
    def unfrozen_fraction(self) -> float:
        if self.pages <= 0 or self.frozen <= 0:
            return 1.0
        return 1.0 - min(self.frozen / self.pages, 1.0)

    # -- autovacuum's decision ----------------------------------------------

    def _active(self) -> bool:
        return self.eff.enabled and self.settings.autovacuum

    def needs(self, t: float) -> Needs:
        """relation_needs_vacanalyze, evaluated at t."""
        self.advance(t)
        e = self.eff
        wrap = self.age(t) > e.freeze_max_age
        if not self._active():
            return Needs(wrap, False, wrap)
        vac = (
            wrap
            or self.dead_stats > e.vacuum_threshold(self.reltuples)
            or (
                e.ins_base >= 0
                and self.ins_since_vac > e.insert_threshold(self.reltuples, self.unfrozen_fraction)
            )
        )
        anl = self.mod_since_anl > e.analyze_threshold(self.reltuples)
        return Needs(vac, anl, wrap)

    def compute_next_need(self) -> float:
        """earliest time needs() turns true, assuming nobody vacuums it first.

        thresholds only depend on reltuples and relallfrozen, which change only
        when a vacuum or analyze finishes, so this stays valid until then.
        """
        t = self.t
        e = self.eff
        wrap_at = self.xid.time_of(self.relfrozenxid + e.freeze_max_age)
        if not self._active():
            self.next_need = max(wrap_at, t)
            return self.next_need
        cands = [wrap_at]
        vt = e.vacuum_threshold(self.reltuples)
        cands.append(self.dead_curve.time_to_add(t, vt - self.dead_stats + 1e-6))
        if e.ins_base >= 0:
            it = e.insert_threshold(self.reltuples, self.unfrozen_fraction)
            cands.append(self.ins.time_to_add(t, it - self.ins_since_vac + 1e-6))
        at = e.analyze_threshold(self.reltuples)
        cands.append(self.mod_curve.time_to_add(t, at - self.mod_since_anl + 1e-6))
        self.next_need = max(min(cands), t)
        return self.next_need
