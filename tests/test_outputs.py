import json
from datetime import timedelta

import pytest

from keepup.calibrate import _inv_limit_profile, apply_params, backtest, fit_io
from keepup.cli import main
from keepup.gucs import Settings, TableOptions
from keepup.logs import LogRun
from keepup.patch import render
from keepup.profile import Profile, load_profile, save_profile
from keepup.report import html_report, write_outputs
from keepup.search import Config, Knob, search
from keepup.sim import simulate

from .conftest import START, make_profile, make_table


def test_patch_sql():
    base = Settings.defaults_for(16)
    new = base.with_values(
        {
            "autovacuum_max_workers": 6,
            "autovacuum_naptime": "30s",
            "autovacuum_vacuum_cost_limit": 800,
        }
    )
    sql = render(
        base,
        new,
        {"public.t": TableOptions()},
        {
            "public.t": TableOptions(vacuum_scale_factor=0.02),
            "Weird.Name": TableOptions(vacuum_cost_limit=1000),
        },
        ["hello"],
    )
    assert "-- hello" in sql
    assert "ALTER SYSTEM SET autovacuum_max_workers = 6;  -- was 3, needs a restart" in sql
    assert "ALTER SYSTEM SET autovacuum_naptime = '30s';" in sql
    assert "ALTER TABLE public.t SET (autovacuum_vacuum_scale_factor = 0.02);" in sql
    assert 'ALTER TABLE "Weird"."Name" SET (autovacuum_vacuum_cost_limit = 1000);' in sql
    assert "pg_reload_conf" in sql


def test_patch_reset_and_noop():
    base = Settings.defaults_for(18)
    new = base.with_values({"autovacuum_max_workers": 6})
    sql = render(base, new, {"t": TableOptions(vacuum_scale_factor=0.1)}, {"t": TableOptions()})
    assert "needs a restart" not in sql  # sighup in 18
    assert "ALTER TABLE t RESET (autovacuum_vacuum_scale_factor);" in sql
    assert "nothing to change" in render(base, base, {}, {})


def test_profile_round_trip(tmp_path):
    p = make_profile(
        make_table(updates=10.0, options={"autovacuum_vacuum_scale_factor": 0.01}),
        autovacuum_naptime="15s",
    )
    path = tmp_path / "p.json"
    save_profile(p, path)
    q = load_profile(path)
    assert q.settings == p.settings
    assert q.tables[0].options.vacuum_scale_factor == 0.01
    assert q.tables[0].updates.base == 10.0


def test_example_profile_loads():
    p = load_profile("examples/shop.toml")
    assert p.timezone == "America/New_York"
    assert len(p.tables) == 6
    assert p.table("orders").updates.hod is not None


def test_report_files(tmp_path):
    p = make_profile(make_table(updates=500.0))
    res = simulate(p, days=1)
    files = write_outputs(tmp_path, p, res, res.summarize(), patch_sql="select 1;\n")
    names = {f.name for f in files}
    assert {"report.html", "summary.json", "patch.sql", "public.t.png"} <= names
    html = (tmp_path / "report.html").read_text()
    assert "<title>" in html and "public.t" in html and "data:image/png" in html
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["tables"][0]["name"] == "public.t"
    # the html can be built without writing anything
    assert html_report(p, res, res.summarize()).startswith("<!doctype html>")


def test_search_finds_the_obvious_fix():
    # a big table with the default 20% scale factor bloats; a per-table scale
    # factor should be the change the search reaches for
    t = make_table(
        "public.big", live_tuples=5_000_000, pages=50_000, updates=400.0, all_visible_fraction=0.0
    )
    p = make_profile(t, raw=5e5)
    knob = Knob("vacuum_scale_factor", (None, 0.05, 0.01), table="public.big")
    r = search(p, days=3, knobs=[knob], jobs=1, progress=lambda m: None)
    assert r.improved
    assert r.best.tables["public.big"]["vacuum_scale_factor"] in (0.05, 0.01)
    assert r.best_score.wasted_gb < r.base_score.wasted_gb


def test_io_budget_is_a_hard_limit():
    tables = [
        make_table(f"public.t{i}", oid=i, live_tuples=2e6, pages=40_000, updates=800.0)
        for i in range(3)
    ]
    p = make_profile(*tables, raw=5e5)
    knob = Knob("autovacuum_vacuum_cost_limit", (-1, 10000))
    r = search(p, io_budget=0.5, days=1, knobs=[knob], jobs=1, progress=lambda m: None)
    assert r.best_score.peak_io <= 0.5 or not r.best_score.feasible
    if r.best_score.feasible:
        assert r.best.settings.get("autovacuum_vacuum_cost_limit", -1) == -1


def test_config_apply():
    p = make_profile(make_table())
    cfg = Config().with_knob(Knob("autovacuum_max_workers", (5,)), 5)
    cfg = cfg.with_knob(Knob("vacuum_scale_factor", (0.01,), table="public.t"), 0.01)
    q = cfg.apply(p)
    assert q.settings.autovacuum_max_workers == 5
    assert q.table("public.t").options.vacuum_scale_factor == 0.01
    assert cfg.changes_vs(p) == 2


def _logrun(start_s, dur, hits=1000, misses=0, dirtied=0, kind="vacuum", table="t"):
    end = START + timedelta(seconds=start_s + dur)
    return LogRun(
        end=end,
        elapsed=dur,
        db="postgres",
        schema="public",
        table=table,
        kind=kind,
        hits=hits,
        misses=misses,
        dirtied=dirtied,
        pages_scanned=hits // 2,
    )


def test_overlap_splits_limit():
    runs = [_logrun(0, 10), _logrun(5, 10)]
    inv = _inv_limit_profile(runs, [200, 200], [True, True])
    # first run: 5s alone (limit 200), 5s shared (limit 100)
    assert inv[0] == pytest.approx((5 / 200 + 5 / 100) / 10)


def test_fit_io_recovers_known_rate():
    # runs that obey elapsed = 1.2 * sleep + credits / 300k, with overlaps so
    # the effective limit differs between runs and both terms are identifiable
    p = make_profile(make_table())
    starts = [0, 0, 0, 100, 300, 300, 600]
    hits = [30_000, 60_000, 90_000, 50_000, 40_000, 120_000, 200_000]
    # rough first pass to get durations, then the exact sleep from the overlaps
    durs = [h * 0.002 / 200 + h / 300_000 for h in hits]
    for _ in range(20):
        runs = [_logrun(s0, d, hits=h) for s0, d, h in zip(starts, durs, hits, strict=True)]
        inv = _inv_limit_profile(runs, [200] * len(runs), [True] * len(runs))
        durs = [1.2 * h * 0.002 * iv + h / 300_000 for h, iv in zip(hits, inv, strict=True)]
    runs = [_logrun(s0, d, hits=h) for s0, d, h in zip(starts, durs, hits, strict=True)]
    io = fit_io(runs, p)
    assert io is not None
    assert io.raw_credit_rate == pytest.approx(300_000, rel=0.1)
    assert io.sleep_overhead == pytest.approx(1.2, rel=0.1)
    q = apply_params(p, io, {})
    assert q.raw_credit_rate == io.raw_credit_rate


def test_fit_io_pins_overhead_when_it_cant_tell():
    p = make_profile(make_table())
    runs = []
    for i, hits in enumerate([20_000, 50_000, 120_000, 300_000]):
        sleep = hits * 0.002 / 200
        runs.append(_logrun(i * 1000, 1.05 * sleep + hits / 300_000, hits=hits))
    io = fit_io(runs, p)
    assert io is not None
    assert io.sleep_overhead == 1.05
    assert io.raw_credit_rate == pytest.approx(300_000, rel=0.02)


def test_backtest_against_its_own_output():
    # simulate, pretend the runs were logged, backtest: should be exact
    p = make_profile(make_table(updates=800.0), raw=5e5)
    res = simulate(p, days=0.5)
    logs = [_logrun(r.start, r.duration or 0.0) for r in res.runs if r.vacuum and r.end]
    bt, _ = backtest(p, logs, START + timedelta(days=0.5))
    (t,) = bt.tables
    assert t.n_obs == t.n_sim > 3
    assert t.count_err == 0
    assert bt.interval_mape == pytest.approx(0, abs=1e-6)


def test_cli_forecast(tmp_path, capsys):
    path = tmp_path / "p.json"
    save_profile(make_profile(make_table(updates=300.0)), path)
    rc = main(
        [
            "forecast",
            str(path),
            "--days",
            "1",
            "--set",
            "autovacuum_naptime=30s",
            "--table-set",
            "public.t:autovacuum_vacuum_scale_factor=0.05",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["name"] == "public.t" and out[0]["vacuums"] > 0


def test_cli_logs(capsys):
    assert main(["logs", "tests/data/pg13-stderr.log"]) == 0
    assert "orders" in capsys.readouterr().out


def test_cli_bad_table(tmp_path):
    path = tmp_path / "p.json"
    save_profile(make_profile(make_table()), path)
    assert main(["forecast", str(path), "--table-set", "nope:autovacuum_enabled=off"]) == 2


def test_profile_rejects_junk():
    with pytest.raises(ValueError):
        Profile.from_dict({"tables": [{"name": "t", "bogus": 1}]})
    with pytest.raises(ValueError):
        Profile.from_dict({"tables": []})


def test_cli_monte_carlo_bands(tmp_path):
    path = tmp_path / "p.json"
    save_profile(make_profile(make_table(updates=300.0, inserts=50.0)), path)
    out = tmp_path / "out"
    rc = main(
        [
            "forecast",
            str(path),
            "--days",
            "1",
            "--runs",
            "3",
            "--noise",
            "0.3",
            "--jobs",
            "1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    band = summary["bands"]["public.t"]
    assert band["p10"] <= band["p50"] <= band["p90"]


def _sample_line(ts, db, upd):
    row = {
        "oid": 1,
        "schema": "public",
        "name": "users",
        "relpages": 100,
        "reltuples": 1e4,
        "relallvisible": 50,
        "relallfrozen": None,
        "xid_age": 1000,
        "reloptions": None,
        "heap_bytes": 819200,
        "index_bytes": 81920,
        "n_indexes": 1,
        "n_tup_ins": 0,
        "n_tup_upd": upd,
        "n_tup_del": 0,
        "n_tup_hot_upd": 0,
        "n_live_tup": 10000,
        "n_dead_tup": 0,
        "n_mod_since_analyze": 0,
        "n_ins_since_vacuum": 0,
        "vacuum_count": 0,
        "autovacuum_count": 0,
        "analyze_count": 0,
        "autoanalyze_count": 0,
        "heap_blks_read": 0,
        "heap_blks_hit": 0,
    }
    return json.dumps(
        {
            "kind": "sample",
            "ts": ts,
            "db": db,
            "next_xid": 1000 + upd,
            "tables": [row],
            "progress": [],
        }
    )


def test_same_table_in_two_databases(tmp_path):
    from keepup.fit import fit_samples

    head = {
        "kind": "header",
        "ts": "2026-11-01T00:00:00+00:00",
        "version": 16,
        "version_num": 160004,
        "settings": {},
        "databases": [],
    }
    lines = [json.dumps(head)]
    for i in range(5):
        ts = f"2026-11-01T00:0{i}:00+00:00"
        lines.append(_sample_line(ts, "a", i * 600))
        lines.append(_sample_line(ts, "b", i * 60))
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n")
    prof = fit_samples(path).profile
    names = sorted(t.name for t in prof.tables)
    assert names == ["a/public.users", "b/public.users"]
    assert {t.relname for t in prof.tables} == {"public.users"}
    save_profile(prof, tmp_path / "p.json")
    again = load_profile(tmp_path / "p.json")
    assert again.table("a/public.users").updates.base == pytest.approx(10.0)
    simulate(again, days=0.1)


def test_io_budget_units():
    from keepup.cli import _io_budget

    assert _io_budget("80MB/s") == 80
    assert _io_budget("1GB/s") == 1024
    assert _io_budget("200") == 200


def test_table_set_rejects_typos(tmp_path):
    path = tmp_path / "p.json"
    save_profile(make_profile(make_table()), path)
    with pytest.raises(SystemExit):
        main(["forecast", str(path), "--table-set", "public.t:autovacuum_vacum_scale_factor=1"])


def test_cli_tune_writes_a_patch(tmp_path, capsys):
    t = make_table(
        "public.big", live_tuples=5_000_000, pages=50_000, updates=400.0, all_visible_fraction=0.0
    )
    path = tmp_path / "p.json"
    save_profile(make_profile(t, raw=5e5), path)
    out = tmp_path / "tune"
    rc = main(
        [
            "tune",
            str(path),
            "--days",
            "2",
            "--forecast-days",
            "2",
            "--rounds",
            "1",
            "--jobs",
            "1",
            "--tables",
            "1",
            "--io-budget",
            "500MB/s",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    sql = (out / "patch.sql").read_text()
    assert "keepup settings patch" in sql
    assert (out / "before" / "report.html").exists()
