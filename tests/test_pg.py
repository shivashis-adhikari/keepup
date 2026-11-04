"""tests that need a real server. point KEEPUP_TEST_DSN at a scratch database
(superuser not needed) and they run; otherwise they're skipped."""

import os
import threading
import time

import pytest

from keepup.fit import fit_samples
from keepup.sampler import read_samples, run

DSN = os.environ.get("KEEPUP_TEST_DSN")
pytestmark = [pytest.mark.pg, pytest.mark.skipif(not DSN, reason="KEEPUP_TEST_DSN not set")]


@pytest.fixture
def table():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("drop table if exists keepup_test_t")
        c.execute("create table keepup_test_t (id serial primary key, v int)")
        c.execute("insert into keepup_test_t (v) select g from generate_series(1, 10000) g")
        c.execute("analyze keepup_test_t")
    yield "public.keepup_test_t"
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("drop table if exists keepup_test_t")


def test_sample_and_fit(tmp_path, table):
    import psycopg

    out = tmp_path / "s.jsonl"
    stop = threading.Event()
    th = threading.Thread(
        target=run, kwargs=dict(dsns=[DSN], out=out, interval=1.0, duration=None, stop_event=stop)
    )
    th.start()
    with psycopg.connect(DSN, autocommit=True) as c:
        for _ in range(5):
            c.execute("update keepup_test_t set v = v + 1 where id <= 500")
            time.sleep(1.0)
    stop.set()
    th.join(timeout=10)

    head, samples = read_samples(out)
    assert head["version"] >= 13
    assert len(samples) >= 3
    # the sampler must not hold an xid or a snapshot open between samples
    with psycopg.connect(DSN) as c:
        n = c.execute(
            "select count(*) from pg_stat_activity where application_name = "
            "'keepup-sampler' and backend_xmin is not null"
        ).fetchone()[0]
    assert n == 0

    res = fit_samples(out)
    t = res.profile.table(table)
    assert t.updates.base > 100
    assert t.live_tuples == pytest.approx(10_000, rel=0.1)
    assert t.n_indexes == 1
