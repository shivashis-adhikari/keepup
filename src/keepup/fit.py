"""samples.jsonl -> Profile.

rates come from counter deltas (n_tup_ins/upd/del and the next xid). the
table's starting state comes from one sample, the last one by default since
the forecast starts where sampling stopped. calibration backtests with the
first one instead.
"""

from __future__ import annotations

import itertools
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from .gucs import Settings, TableOptions, settings_from_pg
from .profile import BLCKSZ, Profile, TableProfile
from .sampler import read_samples
from .workload import FitReport, RateModel, bin_intervals, counter_intervals, fit_rate


@dataclass
class TableHistory:
    db: str
    name: str
    times: list[datetime] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def series(self, key: str) -> list[float | None]:
        return [r.get(key) for r in self.rows]


@dataclass
class FitResult:
    profile: Profile
    reports: dict[str, dict[str, FitReport]]
    xid_report: FitReport | None
    warnings: list[str]


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def histories(
    samples: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], TableHistory], list[tuple[datetime, float]]]:
    hist: dict[tuple[str, str], TableHistory] = {}
    xids: dict[datetime, float] = {}
    for s in samples:
        ts = _ts(s["ts"])
        xids.setdefault(ts, float(s["next_xid"]))
        for t in s["tables"]:
            name = f"{t['schema']}.{t['name']}"
            h = hist.setdefault((s["db"], name), TableHistory(s["db"], name))
            h.times.append(ts)
            h.rows.append(t)
    xid_series = sorted(xids.items())
    return hist, xid_series


def _fit_counter(times: list[datetime], vals: list[float | None], tz: str) -> FitReport | None:
    iv = counter_intervals(times, vals)
    if not iv:
        return None
    return fit_rate(bin_intervals(iv, tz))


def _ratio(num: float, den: float, default: float | None) -> float | None:
    return num / den if den > 0 else default


def fit_samples(
    path: str | Path,
    tz: str | None = None,
    state_at: str = "end",
    widths: dict[str, float] | None = None,
    log: IO[str] = sys.stderr,
    since: datetime | None = None,
    until: datetime | None = None,
) -> FitResult:
    """fit rates over [since, until] and take the table state from one end of it."""
    head, samples = read_samples(path)
    if since is not None:
        samples = [s for s in samples if _ts(s["ts"]) >= since]
    if until is not None:
        samples = [s for s in samples if _ts(s["ts"]) <= until]
    if len(samples) < 2:
        raise ValueError("need at least two samples to fit anything")
    version = int(head["version"])
    raw_settings = {k: (v[0], v[1]) for k, v in head["settings"].items()}
    tz = tz or str(raw_settings.get("TimeZone", ("UTC", None))[0]) or "UTC"
    settings: Settings = settings_from_pg(raw_settings, version)
    warnings: list[str] = []

    hist, xid_series = histories(samples)
    all_widths: dict[str, float] = {}
    for s in samples:
        for k, v in (s.get("widths") or {}).items():
            all_widths[f"{s['db']}:{k}"] = v
    if widths:
        all_widths.update(widths)

    xid_rep = None
    if len(xid_series) >= 2:
        xid_rep = _fit_counter([t for t, _ in xid_series], [v for _, v in xid_series], tz)
    xid_rate = xid_rep.model if xid_rep else RateModel()

    tables: list[TableProfile] = []
    reports: dict[str, dict[str, FitReport]] = {}
    seen_in: dict[str, int] = {}
    for _, name in hist:
        seen_in[name] = seen_in.get(name, 0) + 1
    for (db, name), h in sorted(hist.items()):
        if len(h.rows) < 2:
            continue
        reps: dict[str, FitReport] = {}
        models: dict[str, RateModel] = {}
        for key, col in (
            ("inserts", "n_tup_ins"),
            ("updates", "n_tup_upd"),
            ("deletes", "n_tup_del"),
        ):
            r = _fit_counter(h.times, h.series(col), tz)
            if r is not None:
                reps[key] = r
                models[key] = r.model
        reports[f"{db}:{name}"] = reps

        upd = _span_delta(h, "n_tup_upd")
        hot = _span_delta(h, "n_tup_hot_upd")
        yield_ = _dead_yield(h)
        hit = _span_delta(h, "heap_blks_hit")
        rd = _span_delta(h, "heap_blks_read")

        idx = -1 if state_at == "end" else 0
        last = h.rows[idx]
        relpages = float(last.get("relpages") or 0)
        heap_pages = float(last.get("heap_bytes") or 0) / BLCKSZ or relpages
        av = float(last.get("relallvisible") or 0)
        af = last.get("relallfrozen")
        width = all_widths.get(f"{db}:{name}")
        reltuples = float(last.get("reltuples") or 0)
        tp = TableProfile(
            # public.users in two databases would collide everywhere a table is
            # looked up by name, so qualify the ones that exist more than once
            name=f"{db}/{name}" if seen_in[name] > 1 else name,
            database=db,
            oid=int(last.get("oid") or 0),
            live_tuples=float(last.get("n_live_tup") or 0),
            dead_tuples=float(last.get("n_dead_tup") or 0),
            pages=heap_pages,
            tuple_width=float(width) if width else None,
            index_pages=float(last.get("index_bytes") or 0) / BLCKSZ,
            n_indexes=int(last.get("n_indexes") or 0),
            xid_age=float(last.get("xid_age") or 0),
            ins_since_vacuum=float(last.get("n_ins_since_vacuum") or 0),
            mod_since_analyze=float(last.get("n_mod_since_analyze") or 0),
            reltuples=reltuples if reltuples >= 0 else None,
            # relallvisible is from the last vacuum; the file may have grown since
            # and the new pages aren't all-visible, so divide by the real size
            all_visible_fraction=min(av / heap_pages, 1.0) if heap_pages > 0 else 0.0,
            all_frozen_fraction=min(float(af) / heap_pages, 1.0) if af and heap_pages > 0 else 0.0,
            inserts=models.get("inserts", RateModel()),
            updates=models.get("updates", RateModel()),
            deletes=models.get("deletes", RateModel()),
            hot_fraction=_ratio(hot, upd, 0.0) or 0.0,
            update_dead_yield=yield_,
            cache_hit=min(max(_ratio(hit, hit + rd, 0.5) or 0.5, 0.0), 1.0),
            options=TableOptions.parse(last.get("reloptions")),
        )
        if tp.tuple_width is None and tp.live_tuples > 1000:
            warnings.append(
                f"{db}:{name}: no pg_stats width (never analyzed?), "
                "existing bloat will be underestimated"
            )
        tables.append(tp)

    if not tables:
        raise ValueError("no tables with at least two samples")
    t_state = _ts(samples[-1]["ts"] if state_at == "end" else samples[0]["ts"])
    prof = Profile(
        settings=settings,
        tables=tables,
        xid_rate=xid_rate,
        start=t_state,
        timezone=tz,
        source=str(path),
    )
    span = (_ts(samples[-1]["ts"]) - _ts(samples[0]["ts"])).total_seconds() / 86400
    if span < 1:
        hours = span * 24
        took = f"{hours:.1f}h" if hours >= 1 else f"{hours * 60:.0f}min"
        warnings.append(f"only {took} of samples: no daily pattern, flat rates")
    elif span < 14:
        warnings.append(
            f"{span:.1f} days of samples: growth is assumed to be zero (needs 14+ days to estimate)"
        )
    for w in warnings:
        print(f"warning: {w}", file=log)
    return FitResult(prof, reports, xid_rep, warnings)


def _span_delta(h: TableHistory, col: str) -> float:
    vals = h.series(col)
    tot = 0.0
    for a, b in itertools.pairwise(vals):
        if a is None or b is None:
            continue
        d = float(b) - float(a)
        if d > 0:
            tot += d
    return tot


def _dead_yield(h: TableHistory) -> float | None:
    """how many dead rows each update leaves for vacuum, after hot pruning.

    only intervals with no vacuum or analyze in them count, since both of
    those overwrite n_dead_tup. median rather than a ratio of sums: on a
    table where pruning keeps up, n_dead_tup climbs for a minute and then
    sits flat, and the warm-up interval shouldn't set the long-run rate.
    """
    ratios = []
    total_upd = 0.0
    keys = ("vacuum_count", "autovacuum_count", "analyze_count", "autoanalyze_count")
    for a, b in itertools.pairwise(h.rows):
        if any(a.get(k) != b.get(k) for k in keys):
            continue
        try:
            d_dead = float(b["n_dead_tup"]) - float(a["n_dead_tup"])
            d_upd = float(b["n_tup_upd"]) - float(a["n_tup_upd"])
            d_del = float(b["n_tup_del"]) - float(a["n_tup_del"])
        except (KeyError, TypeError):
            continue
        if d_upd <= 0 or d_del < 0:
            continue
        ratios.append((d_dead - d_del) / d_upd)
        total_upd += d_upd
    if total_upd < 1000 or len(ratios) < 3:
        return None
    return min(max(float(statistics.median(ratios)), 0.0), 1.0)
