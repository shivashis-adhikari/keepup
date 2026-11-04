from datetime import timedelta, timezone
from pathlib import Path

from keepup.logs import detect_format, parse_files, parse_lines

DATA = Path(__file__).parent / "data"


def test_pg16_stderr_from_a_real_server():
    runs = parse_files([DATA / "pg16-stderr.log"])
    vac = [r for r in runs if r.kind == "vacuum"]
    t = next(r for r in vac if r.table == "t")
    assert t.db == "postgres" and t.schema == "public"
    assert t.pages_scanned == 4103
    assert t.tuples_removed == 66666
    assert (t.hits, t.misses, t.dirtied) == (13553, 0, 3)
    assert t.index_scans == 1
    assert t.lpdead_pages == 3077
    assert t.elapsed == 0.23
    assert t.end.tzinfo is not None
    assert t.start == t.end - timedelta(seconds=0.23)
    assert any(r.kind == "analyze" and r.table == "t" for r in runs)


def test_pg13_wording_and_wraparound():
    runs = parse_files([DATA / "pg13-stderr.log"])
    assert len(runs) == 2
    a, b = runs
    assert a.pages_scanned is None  # 13 doesn't log it
    assert a.pages_remain == 812345
    assert a.misses == 811920
    assert b.aggressive and b.wraparound
    assert b.elapsed == 11651.02
    assert b.pid == 88415


def test_pg18_reads_and_eager_scan():
    (r,) = parse_files([DATA / "pg18-stderr.log"])
    assert r.misses == 12221  # "reads" in 18
    assert r.pages_scanned == 98003
    assert r.delay_ms == 39120.552
    assert r.frozen_pages == 1830
    # CET is a fixed +1
    assert r.end.utcoffset() == timedelta(hours=1)


def test_jsonlog():
    text = (DATA / "pg16.json").read_text()
    assert detect_format(text) == "json"
    (r,) = parse_lines(text)
    assert r.table == "inventory" and r.pid == 4411
    assert r.index_state == "bypassed"
    assert r.lpdead_items == 9100


def test_csvlog():
    msg = (
        'automatic vacuum of table "db.public.x": index scans: 0\n'
        "pages: 0 removed, 10 remain, 10 scanned (100.00% of total)\n"
        "buffer usage: 30 hits, 2 misses, 1 dirtied\n"
        "system usage: CPU: user: 0.00 s, system: 0.00 s, elapsed: 0.01 s"
    )
    row = [
        "2026-11-01 10:00:00.000 UTC",
        "",
        "",
        "123",
        "",
        "s",
        "1",
        "",
        "",
        "",
        "0",
        "LOG",
        "00000",
        msg,
    ]
    import csv
    import io

    buf = io.StringIO()
    csv.writer(buf).writerow(row)
    (r,) = parse_lines(buf.getvalue(), fmt="csv")
    assert r.table == "x" and r.hits == 30 and r.pid == 123


def test_offset_timezone_prefix():
    text = (
        '2026-11-01 10:00:00.5+05:30 [1] LOG:  automatic analyze of table "d.s.t"\n'
        "\tsystem usage: CPU: user: 0.00 s, system: 0.00 s, elapsed: 1.50 s\n"
    )
    (r,) = parse_lines(text)
    assert r.end.utcoffset() == timedelta(hours=5, minutes=30)
    assert r.kind == "analyze"


def test_default_tz_applies_to_naive_prefix():
    text = (
        '2026-11-01 10:00:00 [1] LOG:  automatic analyze of table "d.s.t"\n'
        "\tsystem usage: CPU: user: 0.00 s, system: 0.00 s, elapsed: 1.50 s\n"
    )
    tz = timezone(timedelta(hours=-5))
    (r,) = parse_lines(text, default_tz=tz)
    assert r.end.utcoffset() == timedelta(hours=-5)


def test_unrelated_lines_ignored():
    text = (
        "2026-11-01 10:00:00 UTC [1] LOG:  checkpoint starting: time\n"
        '2026-11-01 10:00:01 UTC [2] ERROR:  relation "automatic" does not exist\n'
    )
    assert parse_lines(text) == []
