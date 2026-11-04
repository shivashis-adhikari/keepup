"""command line entry point. `keepup --help` for the tour."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, timedelta, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .gucs import RELOPTION_NAMES, TableOptions, parse_memory_kb, parse_time_ms
from .profile import Profile, load_profile, save_profile


def _duration_s(raw: str) -> float:
    return parse_time_ms(raw, default_unit="s") / 1000.0


def _io_budget(raw: str) -> float:
    """'80MB/s', '1GB/s', '80' -> MB/s."""
    s = raw.strip().removesuffix("/s").replace("iB", "B").strip()
    return parse_memory_kb(s, default_unit="MB") / 1024.0


def _apply_overrides(p: Profile, sets: list[str], table_sets: list[str]) -> Profile:
    if sets:
        vals = {}
        for item in sets:
            k, _, v = item.partition("=")
            if not v:
                raise SystemExit(f"--set wants name=value, got {item!r}")
            vals[k.strip()] = v.strip()
        p = p.with_settings(p.settings.with_values(vals))
    if table_sets:
        changes: dict[str, TableOptions] = {}
        for item in table_sets:
            left, _, v = item.partition("=")
            table, _, opt = left.rpartition(":")
            if not table or not v:
                raise SystemExit(f"--table-set wants schema.table:option=value, got {item!r}")
            opt = opt.strip()
            if opt not in RELOPTION_NAMES and opt != "fillfactor":
                raise SystemExit(
                    f"--table-set: {opt!r} isn't an autovacuum reloption "
                    f"(known: {', '.join(RELOPTION_NAMES)})"
                )
            t = p.table(table)
            cur = changes.get(t.name, t.options)
            raw = {**cur.to_reloptions(), opt.strip(): v.strip()}
            changes[t.name] = TableOptions.parse(raw)
        p = p.with_table_options(changes)
    return p


def _sim_one(args: tuple[Profile, float, int | None]) -> Any:
    from .sim import simulate

    prof, days, seed = args
    return simulate(prof, days=days, seed=seed)


def _monte_carlo(p: Profile, days: float, runs: int, seed: int, jobs: int) -> list[Any]:
    args = [(p, days, seed + i) for i in range(runs)]
    if jobs <= 1 or runs <= 1:
        return [_sim_one(a) for a in args]
    with ProcessPoolExecutor(max_workers=min(jobs, runs)) as ex:
        return list(ex.map(_sim_one, args))


# commands ---------------------------------------------------------------------


def cmd_forecast(a: argparse.Namespace) -> int:
    from .report import bands_from, summary_rows, text_table, write_outputs
    from .sim import simulate

    p = _apply_overrides(load_profile(a.profile), a.set or [], a.table_set or [])
    if a.noise:
        p = p.with_noise(a.noise)
    res = simulate(p, days=a.days)
    bands = None
    if a.runs > 1:
        print(f"running {a.runs} noisy runs for the bands...", file=sys.stderr)
        mc = _monte_carlo(p, a.days, a.runs, a.seed, a.jobs)
        bands = bands_from(mc)
    summaries = res.summarize()
    if a.json:
        print(json.dumps([s.to_dict() for s in summaries], indent=2))
    else:
        rows = summary_rows(summaries, bands)
        print(text_table(rows))
        for r in rows:
            if r["why"]:
                print(f"\n  {r['table']}: {r['why']}")
        print(
            f"\nworkers busy {res.utilization:.0%} on average, all of them busy "
            f"{res.saturated_fraction:.1%} of the time, p99 vacuum i/o "
            f"{res.peak_io_mbps():.1f} MB/s"
        )
    if a.out:
        written = write_outputs(Path(a.out), p, res, summaries, bands)
        print(f"\nwrote {written[0]}", file=sys.stderr)
    worst = max((s.verdict for s in summaries), key=["ok", "behind", "danger"].index)
    return {"ok": 0, "behind": 1, "danger": 2}[worst] if a.exit_code else 0


def cmd_sample(a: argparse.Namespace) -> int:
    from .sampler import run

    dur = _duration_s(a.duration) if a.duration else None
    run(a.dsn, a.out, _duration_s(a.interval), dur, all_databases=a.all_databases)
    return 0


def cmd_fit(a: argparse.Namespace) -> int:
    from .calibrate import apply_params, fit_io, fit_table_params
    from .fit import fit_samples
    from .logs import parse_files

    res = fit_samples(a.samples, tz=a.tz)
    prof = res.profile
    if a.logs:
        runs = parse_files(a.logs, default_tz=_tz(a.log_tz))
        io = fit_io(runs, prof)
        params = fit_table_params(runs, prof)
        prof = apply_params(prof, io, params)
        if io:
            print(f"i/o model: {io.describe()}", file=sys.stderr)
        else:
            print(
                "not enough logged vacuums to fit the i/o model, keeping defaults", file=sys.stderr
            )
    for t in prof.tables:
        print(
            f"{t.database}:{t.name}: ins {t.inserts.base:.1f}/s ({t.inserts.kind}), "
            f"upd {t.updates.base:.1f}/s, del {t.deletes.base:.1f}/s, "
            f"{t.size_bytes / 2**20:,.0f} MB",
            file=sys.stderr,
        )
    print(f"xids: {prof.xid_rate.base:.1f}/s ({prof.xid_rate.kind})", file=sys.stderr)
    save_profile(prof, a.out)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


def _tz(raw: str | None) -> timezone:
    if not raw:
        return UTC
    if raw.upper() in ("UTC", "GMT", "Z"):
        return UTC
    sign = -1 if raw.startswith("-") else 1
    digits = raw.lstrip("+-").replace(":", "")  # +05:30, +0530, +5
    hh, mm = (digits[:-2], digits[-2:]) if len(digits) > 2 else (digits, "0")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def cmd_calibrate(a: argparse.Namespace) -> int:
    from .calibrate import apply_params, backtest, fit_io, fit_table_params
    from .fit import fit_samples
    from .logs import parse_files

    runs = parse_files(a.logs, default_tz=_tz(a.log_tz))
    if not runs:
        raise SystemExit("no autovacuum entries in those logs. is log_autovacuum_min_duration set?")
    from .calibrate import log_min_duration
    from .sampler import read_samples

    min_dur = a.min_duration
    if min_dur is None:
        # runs shorter than this never made it into the log
        min_dur = log_min_duration(read_samples(a.samples)[0].get("settings", {}))
    fit0 = fit_samples(a.samples, tz=a.tz, state_at="start")
    end = fit_samples(a.samples, tz=a.tz, state_at="end").profile.start
    io = fit_io(runs, fit0.profile)
    params = fit_table_params(runs, fit0.profile)
    prof0 = apply_params(fit0.profile, io, params)
    bt, _ = backtest(prof0, runs, end, min_duration=min_dur, io=io)
    if io:
        print(f"i/o model: {io.describe()}")
    print(f"backtest {bt.start:%Y-%m-%d %H:%M} to {bt.end:%Y-%m-%d %H:%M}\n")
    hdr = (
        f"{'table':40} {'vac obs':>7} {'sim':>5} {'interval obs':>12} {'sim':>7} "
        f"{'dur obs':>8} {'sim':>8}"
    )
    print(hdr)
    for t in bt.tables:

        def f(v: float | None, nd: int = 1) -> str:
            return "-" if v is None else f"{v:.{nd}f}"

        print(
            f"{t.table:40} {t.n_obs:7} {t.n_sim:5} {f(t.interval_obs):>12} "
            f"{f(t.interval_sim):>7} {f(t.dur_obs, 2):>8} {f(t.dur_sim, 2):>8}"
        )

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{v:.0%}"

    print(
        f"\nmean abs error: count {pct(bt.count_mape)}, interval {pct(bt.interval_mape)}, "
        f"duration {pct(bt.duration_mape)}"
    )
    if a.out:
        fit_end = fit_samples(a.samples, tz=a.tz, state_at="end")
        save_profile(apply_params(fit_end.profile, io, params), a.out)
        print(f"wrote calibrated profile to {a.out}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps(bt.to_dict(), indent=2, default=str) + "\n")
    return 0


def cmd_tune(a: argparse.Namespace) -> int:
    from .patch import render
    from .report import write_outputs
    from .search import search
    from .sim import simulate

    base = _apply_overrides(load_profile(a.profile), a.set or [], [])
    budget = _io_budget(a.io_budget) if a.io_budget else None
    knobs = None
    if a.tables is not None:
        from .search import default_knobs

        knobs = default_knobs(base, top_tables=a.tables, days=a.days)
    r = search(base, io_budget=budget, days=a.days, knobs=knobs, jobs=a.jobs, max_rounds=a.rounds)
    new = r.profile
    comment = [
        f"profile {a.profile}, searched over {a.days:g} days"
        + (f", i/o budget {budget:g} MB/s" if budget else ""),
        f"before: {r.base_score.describe()}",
        f"after:  {r.best_score.describe()}",
    ]
    sql = render(
        base.settings,
        new.settings,
        {t.name: t.options for t in base.tables},
        {t.name: t.options for t in new.tables},
        comment,
    )
    print()
    print(sql)
    if a.out:
        out = Path(a.out)
        res = simulate(new, days=a.forecast_days)
        written = write_outputs(
            out,
            new,
            res,
            res.summarize(),
            patch_sql=sql,
            title="autovacuum forecast with the suggested settings",
        )
        before = simulate(base, days=a.forecast_days)
        write_outputs(
            out / "before",
            base,
            before,
            before.summarize(),
            title="autovacuum forecast, current settings",
        )
        print(f"wrote {written[0]} and {out / 'before' / 'report.html'}", file=sys.stderr)
    return 0


def cmd_logs(a: argparse.Namespace) -> int:
    from .logs import parse_files

    runs = parse_files(a.files, default_tz=_tz(a.log_tz))
    for r in runs:
        if a.json:
            d = {k: v for k, v in r.__dict__.items() if k != "raw"}
            d["end"] = r.end.isoformat()
            print(json.dumps(d))
        else:
            kind = r.kind + (
                " wraparound" if r.wraparound else " aggressive" if r.aggressive else ""
            )
            print(
                f"{r.end:%Y-%m-%d %H:%M:%S}  {r.db}.{r.relname:30} {kind:20} "
                f"{r.elapsed:9.2f}s  scanned {r.pages_scanned or '-':>8}  "
                f"hits {r.hits} misses {r.misses} dirtied {r.dirtied}"
            )
    print(f"{len(runs)} runs", file=sys.stderr)
    return 0


def cmd_lab_run(a: argparse.Namespace) -> int:
    from .lab import Scenario, analyse, drive, render

    sc = Scenario.load(a.scenario)
    if a.minutes:
        sc = replace(sc, minutes=a.minutes)
    out = Path(a.out or f"lab/runs/{sc.name}")
    drive(a.dsn, sc, out)
    if a.log:
        res = analyse(sc, out, [Path(x) for x in a.log])
        text = render(res)
        (out / "report.md").write_text(text)
        print(text)
    else:
        print(
            f"no --log given; run `keepup lab analyse {out} --scenario {a.scenario} "
            f"--log <server log>` once you have the log",
            file=sys.stderr,
        )
    return 0


def cmd_lab_analyse(a: argparse.Namespace) -> int:
    from .lab import Scenario, analyse, render

    sc = Scenario.load(a.scenario)
    out = Path(a.run)
    res = analyse(sc, out, [Path(x) for x in a.log])
    text = render(res)
    (out / "report.md").write_text(text)
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="keepup",
        description="simulate postgres autovacuum against your workload and see if it keeps up",
    )
    ap.add_argument("--version", action="version", version=f"keepup {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast", help="simulate a profile forward and report per table")
    f.add_argument("profile")
    f.add_argument("--days", type=float, default=90.0)
    f.add_argument(
        "--set",
        action="append",
        metavar="GUC=VALUE",
        help="override a server setting, e.g. autovacuum_max_workers=6",
    )
    f.add_argument(
        "--table-set",
        action="append",
        metavar="TABLE:OPTION=VALUE",
        help="override a reloption, e.g. public.orders:autovacuum_vacuum_scale_factor=0.02",
    )
    f.add_argument(
        "--runs", type=int, default=1, help="monte carlo runs with noisy rates, for p10/p90 bands"
    )
    f.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="hourly log-noise sigma for rates that don't have one fitted (e.g. 0.2)",
    )
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    f.add_argument("--out", help="directory for report.html, charts and summary.json")
    f.add_argument("--json", action="store_true", help="print the summary as json")
    f.add_argument(
        "--exit-code",
        action="store_true",
        help="exit 1 if a table falls behind, 2 if one is in danger (for ci)",
    )
    f.set_defaults(func=cmd_forecast)

    pr = sub.add_parser("profile", help="sample a live server and fit a profile")
    psub = pr.add_subparsers(dest="pcmd", required=True)
    s = psub.add_parser("sample", help="poll stats views into a jsonl file")
    s.add_argument(
        "--dsn",
        action="append",
        required=True,
        help="libpq connection string; repeat for several databases",
    )
    s.add_argument(
        "--all-databases",
        action="store_true",
        help="connect to every database on the server using the first dsn",
    )
    s.add_argument("--interval", default="60s")
    s.add_argument("--duration", default=None, help="e.g. 24h, 14d. default: until ctrl-c")
    s.add_argument("--out", default="samples.jsonl")
    s.set_defaults(func=cmd_sample)
    fp = psub.add_parser("fit", help="fit rates and state from samples into a profile")
    fp.add_argument("samples")
    fp.add_argument("--logs", nargs="*", help="server logs, to calibrate the i/o model too")
    fp.add_argument("--log-tz", help="utc offset for log timestamps without a zone")
    fp.add_argument("--tz", help="time zone for daily/weekly patterns (default: server's)")
    fp.add_argument("--out", default="profile.json")
    fp.set_defaults(func=cmd_fit)

    c = sub.add_parser("calibrate", help="backtest the simulator against real autovacuum logs")
    c.add_argument("--samples", required=True)
    c.add_argument("--logs", nargs="+", required=True)
    c.add_argument("--log-tz")
    c.add_argument("--tz")
    c.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="log_autovacuum_min_duration in seconds, so short runs the "
        "server didn't log aren't counted against the simulator "
        "(default: read it from the samples)",
    )
    c.add_argument("--out", help="write the calibrated profile here")
    c.add_argument("--json", help="write the backtest numbers here")
    c.set_defaults(func=cmd_calibrate)

    t = sub.add_parser("tune", help="search for settings that keep up within an i/o budget")
    t.add_argument("profile")
    t.add_argument("--io-budget", help="p99 vacuum read+write over 5 minutes, e.g. 80MB/s")
    t.add_argument(
        "--days",
        type=float,
        default=30.0,
        help="horizon for each candidate (shorter = faster search)",
    )
    t.add_argument("--forecast-days", type=float, default=90.0)
    t.add_argument(
        "--tables",
        type=int,
        default=None,
        help="how many of the worst tables get per-table knobs (default 4)",
    )
    t.add_argument("--rounds", type=int, default=6)
    t.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    t.add_argument("--set", action="append", metavar="GUC=VALUE")
    t.add_argument("--out", help="directory for patch.sql and before/after reports")
    t.set_defaults(func=cmd_tune)

    lg = sub.add_parser("logs", help="print the autovacuum runs found in log files")
    lg.add_argument("files", nargs="+")
    lg.add_argument("--json", action="store_true")
    lg.add_argument("--log-tz")
    lg.set_defaults(func=cmd_logs)

    lab = sub.add_parser("lab", help="validation lab against a scratch server")
    lsub = lab.add_subparsers(dest="lcmd", required=True)
    lr = lsub.add_parser("run", help="drive a scenario, then fit/predict/compare")
    lr.add_argument("scenario")
    lr.add_argument(
        "--dsn", required=True, help="superuser connection to a server you don't mind trashing"
    )
    lr.add_argument("--log", nargs="*", help="the server's log file(s)")
    lr.add_argument("--minutes", type=float)
    lr.add_argument("--out")
    lr.set_defaults(func=cmd_lab_run)
    la = lsub.add_parser("analyse", help="redo the analysis for a finished run")
    la.add_argument("run")
    la.add_argument("--scenario", required=True)
    la.add_argument("--log", nargs="+", required=True)
    la.set_defaults(func=cmd_lab_analyse)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return int(a.func(a))
    except KeyError as e:
        print(f"keepup: no such table {e}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:
        print(f"keepup: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
