"""the workload profile: what the profiler writes and what the simulator reads.

a profile is plain data. `keepup profile fit` produces one as json, and you
can also write one by hand in toml (see examples/). both go through
Profile.from_dict, so the two formats accept the same keys.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .gucs import Settings, TableOptions
from .workload import RateModel

BLCKSZ = 8192
PAGE_HEADER = 24
ITEM_ID = 4
TUPLE_HEADER = 24  # heap tuple header, maxaligned on 64-bit


def tuples_per_page(width: float, fillfactor: int = 100) -> float:
    """how many rows of this average width fit in a heap page.

    width is the data width from pg_stats (sum of avg_width), the header and
    line pointer get added here. not exact, alignment padding varies.
    """
    usable = (BLCKSZ - PAGE_HEADER) * fillfactor / 100.0
    per_tuple = TUPLE_HEADER + math.ceil(max(width, 1.0) / 8.0) * 8 + ITEM_ID
    return max(usable / per_tuple, 1.0)


@dataclass
class TableProfile:
    name: str
    database: str = "postgres"
    oid: int = 0  # only used for ordering, autovacuum walks pg_class roughly in this order
    live_tuples: float = 0.0
    dead_tuples: float = 0.0
    pages: float = 0.0
    tuple_width: float | None = None
    tuples_per_page: float | None = None
    index_pages: float = 0.0
    n_indexes: int = 1
    xid_age: float = 0.0
    ins_since_vacuum: float = 0.0
    mod_since_analyze: float = 0.0
    reltuples: float | None = None
    all_visible_fraction: float = 0.0
    all_frozen_fraction: float = 0.0
    inserts: RateModel = field(default_factory=RateModel)
    updates: RateModel = field(default_factory=RateModel)
    deletes: RateModel = field(default_factory=RateModel)
    hot_fraction: float = 0.0
    # share of updates that leave a dead tuple vacuum has to deal with. hot
    # updates that get pruned on access don't. None = guess from hot_fraction
    update_dead_yield: float | None = None
    # fraction of heap pages that updates and deletes land on. 1.0 is uniform
    # random, small values are "the recent rows are the hot ones"
    active_fraction: float = 1.0
    # share of vacuum page reads that hit shared_buffers
    cache_hit: float = 0.5
    # share of scanned pages a non-aggressive vacuum freezes (opportunistic
    # freezing in 16+). calibrate from the "frozen:" line in vacuum logs
    freeze_fraction: float = 0.0
    # share of the not-all-visible pages that vacuum has to dirty. on a small
    # hot table they're usually still dirty in shared_buffers from the writes
    # that made them not-all-visible, so this is close to 0. None = 1 - cache_hit
    dirty_fraction: float | None = None
    # how long a page this table dirties stays dirty in shared_buffers before
    # a checkpoint or the background writer cleans it. None = checkpoint_timeout.
    # calibrate fits it from the logs; under cache pressure it's near zero.
    dirty_residency: float | None = None
    options: TableOptions = field(default_factory=TableOptions)

    @property
    def tpp(self) -> float:
        if self.tuples_per_page:
            return self.tuples_per_page
        if self.tuple_width:
            return tuples_per_page(self.tuple_width, self.options.fillfactor)
        if self.pages > 0 and self.live_tuples + self.dead_tuples > 0:
            # no width info. assume the table is currently packed, which makes
            # existing bloat invisible; the profiler fills tuple_width so this
            # is mostly for hand-written profiles
            return max((self.live_tuples + self.dead_tuples) / self.pages, 1.0)
        return 60.0

    @property
    def dead_yield(self) -> float:
        if self.update_dead_yield is not None:
            return self.update_dead_yield
        # rough: about half of hot updates get pruned before vacuum sees them
        return 1.0 - 0.5 * self.hot_fraction

    @property
    def relname(self) -> str:
        """schema.table as the server logs it. names get a "db/" prefix when the
        same table exists in more than one database."""
        return self.name.split("/", 1)[-1]

    @property
    def dirty_share(self) -> float:
        if self.dirty_fraction is not None:
            return self.dirty_fraction
        return 1.0 - self.cache_hit

    @property
    def index_ratio(self) -> float:
        return self.index_pages / self.pages if self.pages > 0 else 0.3

    @property
    def size_bytes(self) -> float:
        return self.pages * BLCKSZ

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "database": self.database,
            "oid": self.oid,
            "live_tuples": round(self.live_tuples),
            "dead_tuples": round(self.dead_tuples),
            "pages": round(self.pages),
            "index_pages": round(self.index_pages),
            "n_indexes": self.n_indexes,
            "xid_age": round(self.xid_age),
            "ins_since_vacuum": round(self.ins_since_vacuum),
            "mod_since_analyze": round(self.mod_since_analyze),
            "all_visible_fraction": round(self.all_visible_fraction, 4),
            "all_frozen_fraction": round(self.all_frozen_fraction, 4),
            "inserts": self.inserts.to_dict(),
            "updates": self.updates.to_dict(),
            "deletes": self.deletes.to_dict(),
            "hot_fraction": round(self.hot_fraction, 4),
            "active_fraction": round(self.active_fraction, 4),
            "cache_hit": round(self.cache_hit, 4),
            "freeze_fraction": round(self.freeze_fraction, 4),
        }
        for k in (
            "tuple_width",
            "tuples_per_page",
            "reltuples",
            "update_dead_yield",
            "dirty_fraction",
            "dirty_residency",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 4)
        opts = self.options.to_reloptions()
        if opts:
            d["options"] = opts
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TableProfile:
        d = dict(d)
        kw: dict[str, Any] = {}
        for k in ("inserts", "updates", "deletes"):
            kw[k] = RateModel.from_value(d.pop(k, None))
        kw["options"] = TableOptions.parse(d.pop("options", None))
        if "size_mb" in d:
            # convenience for hand-written profiles
            d["pages"] = float(d.pop("size_mb")) * 1024 * 1024 / BLCKSZ
        if "index_mb" in d:
            d["index_pages"] = float(d.pop("index_mb")) * 1024 * 1024 / BLCKSZ
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise ValueError(f"table {d.get('name')!r}: unknown keys {sorted(extra)}")
        for k, v in d.items():
            kw[k] = v
        t = cls(**kw)
        if not t.name:
            raise ValueError("table needs a name")
        return t


@dataclass
class HorizonPin:
    """something holding back the xmin horizon: a long transaction, an idle
    replication slot, a standby with hot_standby_feedback. while it's held,
    vacuum can't remove anything that died after it started, and can't
    advance relfrozenxid past it."""

    start_hours: float
    duration_hours: float
    label: str = "long transaction"

    @property
    def end_hours(self) -> float:
        return self.start_hours + self.duration_hours


@dataclass
class Profile:
    settings: Settings
    tables: list[TableProfile]
    xid_rate: RateModel = field(default_factory=RateModel)
    start: datetime = field(default_factory=lambda: datetime.now(UTC))
    timezone: str = "UTC"
    pins: list[HorizonPin] = field(default_factory=list)
    # vacuum throughput the simulator assumes when nothing throttles it, in
    # cost credits per second. calibrate fills this in from real logs.
    raw_credit_rate: float = 250_000.0
    # how much longer a cost-delay nap really takes than the nominal value
    sleep_overhead: float = 1.05
    # cpu time analyze spends per sampled row computing stats, outside the
    # cost throttle. ~5us/row matches small tables with a handful of columns
    analyze_row_seconds: float = 5e-6
    source: str = ""

    @property
    def databases(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.tables:
            seen.setdefault(t.database, None)
        return list(seen)

    def table(self, name: str) -> TableProfile:
        for t in self.tables:
            if t.name == name or t.name.split(".")[-1] == name:
                return t
        raise KeyError(name)

    def with_settings(self, settings: Settings) -> Profile:
        return replace(self, settings=settings)

    def with_noise(self, sigma: float, ar1: float = 0.9) -> Profile:
        """give every rate without fitted residuals some, for monte carlo."""

        def fix(m: RateModel) -> RateModel:
            if m.noise_sigma > 0 or m.base == 0:
                return m
            return replace(m, noise_sigma=sigma, noise_ar1=ar1)

        tables = [
            replace(t, inserts=fix(t.inserts), updates=fix(t.updates), deletes=fix(t.deletes))
            for t in self.tables
        ]
        return replace(self, tables=tables, xid_rate=fix(self.xid_rate))

    def with_table_options(self, changes: dict[str, TableOptions]) -> Profile:
        tables = [replace(t, options=changes.get(t.name, t.options)) for t in self.tables]
        return replace(self, tables=tables)

    def to_dict(self) -> dict[str, Any]:
        base = Settings.defaults_for(self.settings.version)
        return {
            "version": self.settings.version,
            "start": self.start.isoformat(),
            "timezone": self.timezone,
            "source": self.source,
            "settings": {
                k: v
                for k, v in self.settings.to_dict().items()
                if k != "version" and getattr(base, k) != v
            },
            "xid_rate": self.xid_rate.to_dict(),
            "raw_credit_rate": round(self.raw_credit_rate, 1),
            "sleep_overhead": round(self.sleep_overhead, 4),
            "analyze_row_seconds": self.analyze_row_seconds,
            "pins": [
                {"start_hours": p.start_hours, "duration_hours": p.duration_hours, "label": p.label}
                for p in self.pins
            ],
            "tables": [t.to_dict() for t in self.tables],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        d = dict(d)
        version = int(d.pop("version", 16))
        settings = Settings.defaults_for(version).with_values(d.pop("settings", {}) or {})
        start_raw = d.pop("start", None)
        if isinstance(start_raw, datetime):
            start = start_raw
        elif start_raw:
            start = datetime.fromisoformat(str(start_raw))
        else:
            start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        tables_raw = d.pop("tables", None) or d.pop("table", None) or []
        tables = [TableProfile.from_dict(t) for t in tables_raw]
        if not tables:
            raise ValueError("profile has no tables")
        names = [t.name for t in tables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate table names in profile")
        pins = [HorizonPin(**p) for p in d.pop("pins", []) or []]
        p = cls(
            settings=settings,
            tables=tables,
            xid_rate=RateModel.from_value(d.pop("xid_rate", None)),
            start=start,
            timezone=str(d.pop("timezone", "UTC")),
            pins=pins,
            raw_credit_rate=float(d.pop("raw_credit_rate", 250_000.0)),
            sleep_overhead=float(d.pop("sleep_overhead", 1.05)),
            analyze_row_seconds=float(d.pop("analyze_row_seconds", 5e-6)),
            source=str(d.pop("source", "")),
        )
        if d:
            raise ValueError(f"unknown profile keys: {sorted(d)}")
        return p


def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    raw = path.read_bytes()
    data = tomllib.loads(raw.decode()) if path.suffix == ".toml" else json.loads(raw)
    return Profile.from_dict(data)


def save_profile(p: Profile, path: str | Path) -> None:
    Path(path).write_text(json.dumps(p.to_dict(), indent=2) + "\n")
