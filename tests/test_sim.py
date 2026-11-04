"""behavioural tests for the simulator. most of these pin down one rule from
autovacuum.c and check the simulator follows it."""

import itertools
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from keepup.gucs import TableOptions
from keepup.profile import HorizonPin
from keepup.sim import simulate

from .conftest import make_profile, make_table


def vacuums(res, table=None):
    return [r for r in res.runs if r.vacuum and (table is None or r.table == table)]


def test_idle_table_is_left_alone():
    res = simulate(make_profile(make_table()), days=2)
    assert res.runs == []


def test_first_vacuum_waits_for_the_next_naptime_tick():
    # threshold = 50 + 0.2 * 1e6 = 200050 dead rows, 1000/s -> crossed at ~200s.
    # launcher ticks at 60, 120, 180, 240: the vacuum starts at 240.
    t = make_table(updates=1000.0, all_visible_fraction=1.0)
    res = simulate(make_profile(t), days=0.05)
    first = vacuums(res)[0]
    assert first.start == pytest.approx(240.0)
    assert first.lag == pytest.approx(240.0 - 200.05, abs=0.5)
    assert first.removed == pytest.approx(240_000, rel=1e-3)


def test_naptime_changes_the_tick():
    t = make_table(updates=1000.0)
    res = simulate(make_profile(t, autovacuum_naptime="15s"), days=0.05)
    assert vacuums(res)[0].start == pytest.approx(210.0)


def _six_dirty_tables(**kw):
    # each already over its threshold, each needs ~the same amount of work
    return [
        make_table(
            f"public.t{i}",
            oid=i,
            dead_tuples=500_000,
            pages=2_000_000,
            all_visible_fraction=0.0,
            **kw,
        )
        for i in range(6)
    ]


def _finish_all(res):
    return max(r.end for r in res.runs if r.vacuum)


def test_more_workers_same_cost_limit_is_not_faster():
    tables = _six_dirty_tables()
    three = simulate(make_profile(*tables, autovacuum_max_workers=3), days=2)
    six = simulate(make_profile(*tables, autovacuum_max_workers=6), days=2)
    assert len(vacuums(three)) == len(vacuums(six)) == 6
    # same i/o budget split more ways: total wall time barely moves
    assert _finish_all(six) == pytest.approx(_finish_all(three), rel=0.1)
    doubled = simulate(
        make_profile(*tables, autovacuum_max_workers=6, autovacuum_vacuum_cost_limit=400), days=2
    )
    assert _finish_all(doubled) < 0.6 * _finish_all(three)


def test_concurrent_workers_split_the_limit():
    tables = _six_dirty_tables()
    res = simulate(make_profile(*tables[:2], autovacuum_max_workers=3), days=1)
    a, b = sorted(vacuums(res), key=lambda r: r.start)
    # second worker starts one naptime later; from then on both run at half speed
    alone = simulate(make_profile(tables[0], autovacuum_max_workers=3), days=1)
    solo = vacuums(alone)[0].duration
    assert a.duration > solo * 1.5
    assert b.start == pytest.approx(a.start + 60.0)


def test_per_table_cost_limit_escapes_balancing():
    tables = _six_dirty_tables()
    own = replace(tables[1], options=TableOptions(vacuum_cost_limit=200))
    shared = simulate(make_profile(tables[0], tables[1]), days=1)
    split = simulate(make_profile(tables[0], own), days=1)
    # with its own limit, t1 no longer shares; t0 is alone in the pool and
    # gets the whole 200 too. both finish sooner, the server does 2x the i/o
    d_shared = {r.table: r.duration for r in vacuums(shared)}
    d_split = {r.table: r.duration for r in vacuums(split)}
    assert d_split["public.t0"] < d_shared["public.t0"]
    assert d_split["public.t1"] < d_shared["public.t1"]


def test_anti_wraparound_kicks_in_and_resets_age():
    t = make_table(xid_age=190_000_000)
    res = simulate(make_profile(t, xid_rate=1000.0), days=1)
    runs = vacuums(res)
    assert len(runs) == 1
    r = runs[0]
    assert r.wraparound and r.aggressive
    # crosses 200M at 10000s, next tick is 10020
    assert r.start == pytest.approx(10_020.0)
    age_end = res.tables["public.t"].age[-1]
    assert age_end == pytest.approx(50_000_000 + 1000 * (86400 - r.start), rel=0.01)


def test_disabled_table_still_gets_wraparound_vacuum():
    t = make_table(updates=1000.0, xid_age=199_000_000, options=TableOptions(enabled=False))
    res = simulate(make_profile(t, xid_rate=1000.0), days=0.1)
    runs = vacuums(res)
    assert runs and all(r.wraparound for r in runs)


def test_insert_only_table_gets_insert_vacuums():
    t = make_table(inserts=500.0)
    res = simulate(make_profile(t), days=1)
    assert len(vacuums(res)) > 0
    off = replace(t, options=TableOptions(vacuum_insert_threshold=-1))
    res = simulate(make_profile(off), days=1)
    assert vacuums(res) == []


def test_long_transaction_blocks_cleanup():
    t = make_table(updates=500.0, all_visible_fraction=1.0)
    base = make_profile(t)
    pinned = replace(base, pins=[HorizonPin(start_hours=2, duration_hours=12)])
    a = simulate(base, days=1).tables["public.t"]
    b = simulate(pinned, days=1).tables["public.t"]
    h = 13  # hour 13, pin still held
    assert b.dead_phys[h] > 5 * a.dead_phys[h]
    assert b.pages[-1] > a.pages[-1]
    # after the pin is gone the dead rows go away again
    assert b.dead_phys[-1] < 0.3 * b.dead_phys.max()


def test_two_databases_take_turns():
    a = make_table("public.a", database="one", updates=1000.0)
    b = make_table("public.b", database="two", updates=1000.0)
    res = simulate(make_profile(a, b), days=0.5)
    na = len(vacuums(res, "public.a"))
    nb = len(vacuums(res, "public.b"))
    assert na > 10 and nb > 10
    assert abs(na - nb) <= 2


def test_failsafe_turns_off_the_throttle():
    big = make_table(
        pages=2_000_000, live_tuples=200_000_000, all_visible_fraction=0.0, xid_age=1_700_000_000
    )
    res = simulate(make_profile(big, xid_rate=10.0, raw=200_000.0), days=1)
    r = vacuums(res)[0]
    assert r.failsafe and r.index_passes == 0
    assert r.duration == pytest.approx(r.credits / 200_000.0, rel=1e-6)


@st.composite
def random_profiles(draw):
    n = draw(st.integers(1, 4))
    tables = []
    for i in range(n):
        tables.append(
            make_table(
                f"public.r{i}",
                oid=i,
                live_tuples=draw(st.floats(1e3, 5e6)),
                pages=draw(st.floats(10, 1e5)),
                inserts=draw(st.floats(0, 500)),
                updates=draw(st.floats(0, 2000)),
                deletes=draw(st.floats(0, 300)),
                hot_fraction=draw(st.floats(0, 1)),
                active_fraction=draw(st.floats(0.01, 1)),
                xid_age=draw(st.floats(0, 2.5e8)),
            )
        )
    return make_profile(
        *tables,
        xid_rate=draw(st.floats(1, 5000)),
        raw=draw(st.floats(5e4, 1e7)),
        autovacuum_max_workers=draw(st.integers(1, 6)),
        autovacuum_naptime=draw(st.sampled_from(["10s", "1min", "5min"])),
        autovacuum_vacuum_cost_limit=draw(st.sampled_from([-1, 50, 1000])),
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_profiles())
def test_invariants(prof):
    res = simulate(prof, days=1.5)
    for ts in res.tables.values():
        assert np.all(ts.live >= 0)
        assert np.all(ts.dead_phys >= -1e-6)
        assert np.all(np.diff(ts.pages) >= -1e-6), "tables never shrink"
        assert np.all(ts.age >= -1e-3)
    # never more concurrent runs than workers
    edges = sorted(
        [(r.start, 1) for r in res.runs]
        + [(r.end if r.end is not None else res.horizon, -1) for r in res.runs],
        key=lambda x: (x[0], x[1]),
    )
    cur = peak = 0
    for _, d in edges:
        cur += d
        peak = max(peak, cur)
    assert peak <= prof.settings.autovacuum_max_workers
    # a table is never vacuumed by two workers at once
    by_table: dict[str, list] = {}
    for r in res.runs:
        by_table.setdefault(r.table, []).append(r)
    for runs in by_table.values():
        runs.sort(key=lambda r: r.start)
        for x, y in itertools.pairwise(runs):
            assert x.end is not None and x.end <= y.start + 1e-6


def test_database_near_wraparound_gets_the_first_worker():
    # "one" has plenty of dead rows, "two" is past autovacuum_freeze_max_age.
    # do_start_worker ignores the round robin and goes to "two" first.
    a = make_table("public.a", database="one", dead_tuples=900_000)
    b = make_table("public.b", database="two", xid_age=205_000_000)
    res = simulate(make_profile(a, b, autovacuum_max_workers=1), days=0.05)
    first = min(res.runs, key=lambda r: r.start)
    assert first.db == "two" and first.wraparound


def test_autovacuum_off_only_does_wraparound():
    t = make_table(updates=2000.0, xid_age=199_900_000)
    res = simulate(make_profile(t, xid_rate=100.0, autovacuum="off"), days=0.2)
    runs = vacuums(res)
    assert runs and all(r.wraparound for r in runs)
    assert not any(r.analyze for r in res.runs)


def test_append_only_vacuum_reads_only_the_new_pages():
    # 1000 rows/s at 100 rows per page is 10 pages/s of new heap. a vacuum
    # should read roughly what was appended since the last one, not the whole
    # table, because the new pages are one contiguous run at the end.
    t = make_table(
        live_tuples=5_000_000,
        pages=50_000,
        inserts=1000.0,
        all_visible_fraction=1.0,
        active_fraction=1.0,
    )
    res = simulate(make_profile(t), days=2)
    runs = vacuums(res)
    assert len(runs) >= 2
    for prev, cur in itertools.pairwise(runs):
        appended = (cur.start - prev.start) * 10
        assert cur.scanned == pytest.approx(appended, rel=0.15)
        assert cur.scanned < 0.6 * (50_000 + cur.start * 10)  # well short of the whole heap
