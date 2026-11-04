"""poll a running server and write what we see to a jsonl file.

one line per database per sample. the first line is a header with settings
and anything that doesn't change between samples (tuple widths, version).

the connection runs in autocommit on purpose. a sampler that sits in an open
transaction holds back the xmin horizon, which is the exact thing that
makes autovacuum fall behind. also nothing here assigns an xid: the next xid
comes from pg_current_snapshot(), not txid_current().
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

GUCS = (
    "autovacuum",
    "autovacuum_max_workers",
    "autovacuum_naptime",
    "autovacuum_vacuum_threshold",
    "autovacuum_vacuum_insert_threshold",
    "autovacuum_vacuum_max_threshold",
    "autovacuum_analyze_threshold",
    "autovacuum_vacuum_scale_factor",
    "autovacuum_vacuum_insert_scale_factor",
    "autovacuum_analyze_scale_factor",
    "autovacuum_freeze_max_age",
    "autovacuum_vacuum_cost_delay",
    "autovacuum_vacuum_cost_limit",
    "autovacuum_work_mem",
    "vacuum_cost_delay",
    "vacuum_cost_limit",
    "vacuum_cost_page_hit",
    "vacuum_cost_page_miss",
    "vacuum_cost_page_dirty",
    "vacuum_freeze_min_age",
    "vacuum_freeze_table_age",
    "vacuum_failsafe_age",
    "vacuum_max_eager_freeze_failure_rate",
    "maintenance_work_mem",
    "default_statistics_target",
    "TimeZone",
    "log_timezone",
    "log_autovacuum_min_duration",
    "shared_buffers",
    "checkpoint_timeout",
)

TABLES_SQL = """
select c.oid::bigint as oid, n.nspname as schema, c.relname as name,
       c.relpages::bigint as relpages, c.reltuples::float8 as reltuples,
       c.relallvisible::bigint as relallvisible, {allfrozen} as relallfrozen,
       age(c.relfrozenxid)::bigint as xid_age, c.reloptions,
       pg_relation_size(c.oid)::bigint as heap_bytes,
       coalesce((select sum(pg_relation_size(i.indexrelid)) from pg_index i
                 where i.indrelid = c.oid), 0)::bigint as index_bytes,
       (select count(*) from pg_index i where i.indrelid = c.oid)::int as n_indexes,
       s.n_tup_ins, s.n_tup_upd, s.n_tup_del, s.n_tup_hot_upd, s.n_live_tup, s.n_dead_tup,
       s.n_mod_since_analyze, s.n_ins_since_vacuum,
       s.vacuum_count, s.autovacuum_count, s.analyze_count, s.autoanalyze_count,
       s.last_autovacuum, s.last_autoanalyze,
       io.heap_blks_read, io.heap_blks_hit
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
join pg_stat_user_tables s on s.relid = c.oid
left join pg_statio_user_tables io on io.relid = c.oid
where c.relkind in ('r', 'm')
"""

WIDTH_SQL = """
select schemaname, tablename, sum(avg_width)::float8
from pg_stats group by 1, 2
"""

PROGRESS_SQL = """
select p.pid, p.relid::bigint, p.phase, p.heap_blks_total, p.heap_blks_scanned,
       p.index_vacuum_count, a.backend_type, a.query_start
from pg_stat_progress_vacuum p
join pg_stat_activity a using (pid)
where p.datname = current_database()
"""


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


@dataclass
class Target:
    dsn: str
    dbname: str


def _connect(dsn: str) -> Any:
    try:
        import psycopg
    except ImportError as e:  # pragma: no cover - depends on the install
        raise SystemExit("the profiler needs psycopg: pip install 'keepup[pg]'") from e
    conn = psycopg.connect(dsn, autocommit=True, application_name="keepup-sampler")
    conn.execute("set statement_timeout = '30s'")
    conn.execute("set lock_timeout = '1s'")
    return conn


def _rows(conn: Any, sql: str) -> list[dict[str, Any]]:
    cur = conn.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def discover(dsn: str) -> list[Target]:
    """every database that accepts connections, using the same credentials."""
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    conn = _connect(dsn)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "select datname from pg_database where datallowconn and not datistemplate "
                "order by datname"
            ).fetchall()
        ]
    finally:
        conn.close()
    base = conninfo_to_dict(dsn)
    return [Target(make_conninfo(**{**base, "dbname": n}), n) for n in names]


def header(conn: Any) -> dict[str, Any]:
    version_num = int(conn.execute("show server_version_num").fetchone()[0])
    cur = conn.execute(
        "select name, setting, unit from pg_settings where name = any(%s)", [list(GUCS)]
    )
    gucs = {name: [setting, unit] for name, setting, unit in cur.fetchall()}
    dbs = _rows(conn, "select datname, age(datfrozenxid)::bigint as age from pg_database")
    return {
        "kind": "header",
        "ts": datetime.now(UTC).isoformat(),
        "version_num": version_num,
        "version": version_num // 10000,
        "settings": gucs,
        "databases": dbs,
    }


def sample_db(conn: Any, version: int, with_widths: bool = False) -> dict[str, Any]:
    sql = TABLES_SQL.format(allfrozen="c.relallfrozen::bigint" if version >= 18 else "null::bigint")
    ts = datetime.now(UTC)
    nxt = conn.execute("select pg_snapshot_xmax(pg_current_snapshot())::text::bigint").fetchone()[0]
    tables = _rows(conn, sql)
    for t in tables:
        if isinstance(t.get("reloptions"), list):
            t["reloptions"] = list(t["reloptions"])
    out: dict[str, Any] = {
        "kind": "sample",
        "ts": ts.isoformat(),
        "db": conn.info.dbname,
        "next_xid": int(nxt),
        "tables": tables,
        "progress": _rows(conn, PROGRESS_SQL),
    }
    if with_widths:
        out["widths"] = {
            f"{r['schemaname']}.{r['tablename']}": r["sum"] for r in _rows(conn, WIDTH_SQL)
        }
    return out


def run(
    dsns: Sequence[str],
    out: str | Path,
    interval: float,
    duration: float | None,
    all_databases: bool = False,
    log: IO[str] = sys.stderr,
    stop_event: threading.Event | None = None,
) -> int:
    targets: list[Target] = []
    for d in dsns:
        if all_databases:
            targets.extend(discover(d))
        else:
            from psycopg.conninfo import conninfo_to_dict

            targets.append(Target(d, str(conninfo_to_dict(d).get("dbname", ""))))
    conns = [_connect(t.dsn) for t in targets]
    version = int(conns[0].execute("show server_version_num").fetchone()[0]) // 10000

    ev = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: ev.set())
        signal.signal(signal.SIGTERM, lambda *_: ev.set())

    path = Path(out)
    fresh = not path.exists() or path.stat().st_size == 0
    n = 0
    started = time.monotonic()
    with path.open("a") as f:
        if fresh:
            f.write(json.dumps(header(conns[0]), default=_json_default) + "\n")
        widths_due = True
        while not ev.is_set():
            tick = time.monotonic()
            for c in conns:
                try:
                    s = sample_db(c, version, with_widths=widths_due)
                except Exception as e:  # keep sampling through blips
                    print(f"sample failed on {c.info.dbname}: {e}", file=log)
                    continue
                f.write(json.dumps(s, default=_json_default) + "\n")
            f.flush()
            widths_due = False
            n += 1
            if n % 60 == 0:
                widths_due = True  # pg_stats changes after analyze; refresh hourly-ish
            if duration is not None and time.monotonic() - started >= duration:
                break
            wait = interval - (time.monotonic() - tick)
            while wait > 0 and not ev.is_set():
                ev.wait(min(wait, 0.5))
                wait = interval - (time.monotonic() - tick)
    for c in conns:
        c.close()
    print(f"wrote {n} samples to {path}", file=log)
    return n


def read_samples(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head: dict[str, Any] | None = None
    samples = []
    for line in _lines(path):
        d = json.loads(line)
        if d.get("kind") == "header":
            if head is None:
                head = d
        elif d.get("kind") == "sample":
            samples.append(d)
    if head is None:
        raise ValueError(f"{path}: no header line, was this written by keepup profile sample?")
    samples.sort(key=lambda s: s["ts"])
    return head, samples


def _lines(path: str | Path) -> Iterator[str]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
