"""validation lab: drive a real postgres with known change rates, then check
the simulator's predictions against what autovacuum did.

the run is split in two halves. the profile (rates, i/o model, per-table
parameters) is fitted on the first half only, the simulator starts from the
state at the midpoint, and its output is compared to the second half of the
log. nothing from the second half leaks into the prediction.
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from .calibrate import (
    Backtest,
    apply_params,
    backtest,
    fit_io,
    fit_table_params,
)
from .fit import fit_samples
from .logs import parse_files
from .profile import save_profile
from .sampler import run as run_sampler


@dataclass
class LabTable:
    name: str
    rows: int = 100_000
    width: int = 100
    extra_indexes: int = 0
    inserts: float = 0.0
    updates: float = 0.0
    deletes: float = 0.0
    update_mode: str = "random"  # random | recent
    recent_fraction: float = 0.1
    # "n" isn't indexed, so updates can be hot. "k" is indexed when the table
    # has extra_indexes, which rules hot updates out and leaves dead rows
    update_column: str = "n"
    fillfactor: int = 100
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    minutes: float
    tables: list[LabTable]
    settings: dict[str, Any] = field(default_factory=dict)
    restart_settings: dict[str, Any] = field(default_factory=dict)
    xid_rate: float = 0.0
    sample_interval: float = 10.0
    tick: float = 0.25

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        d = tomllib.loads(Path(path).read_text())
        tables = [LabTable(**t) for t in d.pop("table", [])]
        return cls(tables=tables, **d)


def _connect(dsn: str, autocommit: bool = True) -> Any:
    import psycopg

    return psycopg.connect(dsn, autocommit=autocommit, application_name="keepup-lab")


def setup(dsn: str, sc: Scenario, log: IO[str]) -> None:
    with _connect(dsn) as c:
        for k, v in sc.settings.items():
            c.execute(f"alter system set {k} = '{v}'")
        c.execute("select pg_reload_conf()")
        for k, v in sc.restart_settings.items():
            cur = c.execute("select setting, unit from pg_settings where name = %s", [k])
            row = cur.fetchone()
            print(
                f"  {k} = {row[0]}{row[1] or ''} (wanted {v}, needs a restart to change)", file=log
            )
        leftovers = c.execute(
            "select relname from pg_class where relname like 'lab\\_%' and relkind = 'r'"
        ).fetchall()
        for (rel,) in leftovers:
            c.execute(f'drop table if exists "{rel}"')
        for t in sc.tables:
            name = f"lab_{t.name}"
            opts = {"fillfactor": t.fillfactor, **t.options}
            with_ = ", ".join(f"{k} = {v}" for k, v in opts.items())
            c.execute(
                f"create table {name} (id bigserial primary key, n int not null default 0, "
                f"k int not null default 0, pad text) with ({with_})"
            )
            for i in range(t.extra_indexes):
                col = "k" if i == 0 else f"(k + {i})"
                c.execute(f"create index on {name} ({col})")
            if t.rows:
                c.execute(
                    f"insert into {name} (k, pad) select (random() * 1e6)::int, "
                    f"repeat('x', {t.width}) from generate_series(1, {t.rows})"
                )
            print(f"  created {name} with {t.rows} rows", file=log)
        # start from a clean, analyzed, all-visible state so the first
        # vacuums aren't just mopping up the load
        for t in sc.tables:
            c.execute(f"vacuum (analyze, freeze) lab_{t.name}")
        c.execute("checkpoint")


class Driver(threading.Thread):
    """keeps one table's inserts/updates/deletes at the scenario rates."""

    def __init__(self, dsn: str, t: LabTable, tick: float, stop: threading.Event):
        super().__init__(daemon=True, name=f"drive-{t.name}")
        self.dsn, self.t, self.tick, self.stop = dsn, t, tick, stop
        self.done = {"inserts": 0, "updates": 0, "deletes": 0}
        self.errors = 0

    def run(self) -> None:
        t = self.t
        name = f"lab_{t.name}"
        rnd = random.Random(hash(t.name) & 0xFFFF)
        c = _connect(self.dsn, autocommit=False)
        lo, hi = c.execute(
            f"select coalesce(min(id), 1), coalesce(max(id), 0) from {name}"
        ).fetchone()
        c.commit()
        started = time.monotonic()
        while not self.stop.is_set():
            elapsed = time.monotonic() - started
            try:
                n_ins = int(t.inserts * elapsed) - self.done["inserts"]
                n_upd = int(t.updates * elapsed) - self.done["updates"]
                n_del = int(t.deletes * elapsed) - self.done["deletes"]
                if n_ins > 0:
                    cur = c.execute(
                        f"insert into {name} (k, pad) select (random() * 1e6)::int, "
                        f"repeat('x', {t.width}) from generate_series(1, %s) returning id",
                        [n_ins],
                    )
                    ids = [r[0] for r in cur.fetchall()]
                    hi = max(hi, *ids)
                    self.done["inserts"] += n_ins
                if n_upd > 0 and hi >= lo:
                    if t.update_mode == "recent":
                        span = max(int((hi - lo) * t.recent_fraction), 1)
                        ids = [rnd.randint(hi - span, hi) for _ in range(n_upd)]
                    else:
                        ids = [rnd.randint(lo, hi) for _ in range(n_upd)]
                    col = "k" if t.update_column == "k" else "n"
                    c.execute(f"update {name} set {col} = {col} + 1 where id = any(%s)", [ids])
                    self.done["updates"] += n_upd
                if n_del > 0 and hi >= lo:
                    # queue-like: eat from the old end
                    c.execute(f"delete from {name} where id >= %s and id < %s", [lo, lo + n_del])
                    lo += n_del
                    self.done["deletes"] += n_del
                c.commit()
            except Exception as e:
                self.errors += 1
                c.rollback()
                if self.errors < 5:
                    print(f"{name}: {e}", file=sys.stderr)
            self.stop.wait(self.tick)
        c.close()


class XidBurner(threading.Thread):
    """plain transactions that only consume xids, to age relfrozenxid."""

    def __init__(self, dsn: str, rate: float, stop: threading.Event):
        super().__init__(daemon=True, name="xid-burner")
        self.dsn, self.rate, self.stop = dsn, rate, stop
        self.count = 0

    def run(self) -> None:
        c = _connect(self.dsn)
        started = time.monotonic()
        while not self.stop.is_set():
            owed = int(self.rate * (time.monotonic() - started)) - self.count
            for _ in range(max(owed, 0)):
                c.execute("select pg_current_xact_id()")
                self.count += 1
            self.stop.wait(0.05)
        c.close()


@dataclass
class LabResult:
    out: Path
    start: datetime
    mid: datetime
    end: datetime
    holdout: Backtest
    insample: Backtest
    rates: dict[str, dict[str, tuple[float, float]]]


def drive(
    dsn: str, sc: Scenario, out: Path, log: IO[str] = sys.stderr
) -> tuple[datetime, datetime]:
    out.mkdir(parents=True, exist_ok=True)
    samples = out / "samples.jsonl"
    if samples.exists():
        samples.unlink()
    print(f"setting up {len(sc.tables)} tables", file=log)
    setup(dsn, sc, log)
    stop = threading.Event()
    sampler = threading.Thread(
        target=run_sampler,
        daemon=True,
        kwargs=dict(
            dsns=[dsn],
            out=samples,
            interval=sc.sample_interval,
            duration=None,
            log=log,
            stop_event=stop,
        ),
    )
    drivers = [Driver(dsn, t, sc.tick, stop) for t in sc.tables]
    burner = XidBurner(dsn, sc.xid_rate, stop) if sc.xid_rate > 0 else None
    start = datetime.now(UTC)
    sampler.start()
    time.sleep(1.0)
    for d in drivers:
        d.start()
    if burner:
        burner.start()
    total = sc.minutes * 60
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < total:
            time.sleep(min(30.0, total - (time.monotonic() - t0)) or 0.1)
            done = ", ".join(f"{d.t.name}: {sum(d.done.values())}" for d in drivers)
            print(f"  {(time.monotonic() - t0) / 60:5.1f} min  rows {done}", file=log)
    except KeyboardInterrupt:
        print("interrupted, analysing what we have", file=log)
    end = datetime.now(UTC)
    # one more sample after the drivers stop would show a rate drop; stop all
    stop.set()
    for d in drivers:
        d.join(timeout=10)
    sampler.join(timeout=30)
    meta = {
        "scenario": sc.name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "done": {d.t.name: d.done for d in drivers},
        "xids_burned": burner.count if burner else 0,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return start, end


def analyse(sc: Scenario, out: Path, log_paths: list[Path], log: IO[str] = sys.stderr) -> LabResult:
    meta = json.loads((out / "run.json").read_text())
    start = datetime.fromisoformat(meta["start"])
    end = datetime.fromisoformat(meta["end"])
    mid = start + (end - start) / 2
    samples = out / "samples.jsonl"
    wanted = {f"lab_{t.name}" for t in sc.tables}
    runs = [
        r
        for r in parse_files(log_paths)
        if r.end >= start - timedelta(minutes=1) and r.table in wanted
    ]
    (out / "autovacuum.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in r.__dict__.items()
                    if k != "raw"
                }
            )
            + "\n"
            for r in runs
        )
    )

    first = [r for r in runs if r.end <= mid]

    # fit on the first half, state at the midpoint
    fit1 = _only(fit_samples(samples, since=start, until=mid, state_at="end", log=log), wanted)
    io = fit_io(first, fit1.profile)
    params = fit_table_params(first, fit1.profile)
    prof_mid = apply_params(fit1.profile, io, params)
    save_profile(prof_mid, out / "profile-mid.json")
    # the holdout starts at the last sample before the midpoint, which is
    # where the simulator's state comes from
    second = [r for r in runs if r.start >= prof_mid.start and r.end <= end]
    holdout, res = backtest(prof_mid, second, end, io=io)

    # in-sample, for reference: same parameters, start of window
    fit0 = _only(fit_samples(samples, since=start, until=mid, state_at="start", log=log), wanted)
    prof0 = apply_params(fit0.profile, io, params)
    insample, _ = backtest(prof0, first, mid, io=io)

    rates: dict[str, dict[str, tuple[float, float]]] = {}
    for t in sc.tables:
        try:
            tp = fit1.profile.table(f"public.lab_{t.name}")
        except KeyError:
            continue
        rates[t.name] = {
            "inserts": (t.inserts, tp.inserts.base),
            "updates": (t.updates, tp.updates.base),
            "deletes": (t.deletes, tp.deletes.base),
        }
    result = LabResult(out, start, mid, end, holdout, insample, rates)
    (out / "result.json").write_text(
        json.dumps(
            {
                "scenario": sc.name,
                "start": start.isoformat(),
                "mid": mid.isoformat(),
                "end": end.isoformat(),
                "holdout": holdout.to_dict(),
                "insample": insample.to_dict(),
                "rates": rates,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    _sim_runs_dump(out / "sim-holdout.jsonl", res, prof_mid.start)
    (out / "holdout.png").write_bytes(timeline_png(prof_mid, second, res, end))
    return result


def timeline_png(prof: Any, observed: list[Any], res: Any, end: datetime) -> bytes:
    """vacuum starts per table over the holdout, server on top, simulator below."""
    import io as _io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import AXIS, BLUE, INK2, MUTED, ORANGE, SURFACE

    names = [t.name for t in prof.tables]
    t0 = prof.start
    span = (end - t0).total_seconds() / 60.0
    fig, axes = plt.subplots(
        len(names), 1, figsize=(8.6, 0.85 * len(names) + 0.8), sharex=True, dpi=110
    )
    fig.patch.set_facecolor(SURFACE)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names, strict=True):
        obs = [
            (r.start - t0).total_seconds() / 60
            for r in observed
            if r.relname == name and r.kind == "vacuum"
        ]
        sim = [r.start / 60 for r in res.runs_for(name) if r.vacuum]
        ax.scatter(obs, [1] * len(obs), marker="|", s=120, color=BLUE, linewidths=1.6)
        ax.scatter(sim, [0] * len(sim), marker="|", s=120, color=ORANGE, linewidths=1.6)
        ax.set_ylim(-0.7, 1.7)
        ax.set_xlim(0, span)
        ax.set_yticks([1, 0], ["server", "keepup"], fontsize=7.5, color=INK2)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_title(
            f"{name.split('.')[-1]}  ({len(obs)} vs {len(sim)})",
            loc="left",
            fontsize=8.5,
            color=INK2,
            pad=2,
        )
    axes[-1].set_xlabel("minutes into the holdout half", fontsize=8.5, color=INK2)
    fig.tight_layout()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def _only(fit: Any, wanted: set[str]) -> Any:
    # tables left over from another scenario still show up in the samples
    from dataclasses import replace

    tables = [t for t in fit.profile.tables if t.relname.split(".")[-1] in wanted]
    fit.profile = replace(fit.profile, tables=tables)
    return fit


def _sim_runs_dump(path: Path, res: Any, t0: datetime) -> None:
    with path.open("w") as f:
        for r in res.runs:
            d = dict(r.__dict__)
            d["start_ts"] = (t0 + timedelta(seconds=r.start)).isoformat()
            f.write(json.dumps(d) + "\n")


def render(result: LabResult) -> str:
    lines = [f"# lab run: {result.out.name}", ""]
    lines.append(
        f"window {result.start:%Y-%m-%d %H:%M:%S} to {result.end:%H:%M:%S} UTC, "
        f"fitted on the first half, predicted the second half."
    )
    lines.append("")
    if result.holdout.io:
        lines.append(f"i/o model from first-half logs: {result.holdout.io.describe()}")
        lines.append("")
    lines.append("measured vs scenario rates (rows/s):")
    lines.append("")
    lines.append("| table | inserts | updates | deletes |")
    lines.append("|---|---|---|---|")
    for name, r in result.rates.items():
        cells = []
        for k in ("inserts", "updates", "deletes"):
            want, got = r[k]
            cells.append(f"{got:.0f} / {want:.0f}" if want else f"{got:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for label, bt in (
        ("holdout (second half)", result.holdout),
        ("in sample (first half)", result.insample),
    ):
        lines.append("")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| table | vacuums obs | sim | interval obs | sim | duration obs | sim | "
            "scanned obs | sim |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for t in bt.tables:
            lines.append(
                f"| {t.table.split(':')[-1]} | {t.n_obs} | {t.n_sim} | {_f(t.interval_obs)} | "
                f"{_f(t.interval_sim)} | {_f(t.dur_obs, 2)} | {_f(t.dur_sim, 2)} | "
                f"{_f(t.scanned_obs, 0)} | {_f(t.scanned_sim, 0)} |"
            )
        lines.append("")
        lines.append(
            f"count error {_pct(bt.count_mape)}, interval error "
            f"{_pct(bt.interval_mape)}, duration error {_pct(bt.duration_mape)} "
            "(mean absolute, tables with 3+ observed runs)"
        )
    return "\n".join(lines) + "\n"


def _f(v: float | None, nd: int = 1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"
