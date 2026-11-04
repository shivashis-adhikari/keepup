"""parse autovacuum / autoanalyze entries out of postgres server logs.

needs log_autovacuum_min_duration set low enough to log the runs you care
about (0 logs everything). works with:

- stderr logs with any log_line_prefix that starts with a timestamp (%m or %t)
- csvlog
- jsonlog (15+)

the wording moved around a bit between versions: 13/14 have no "scanned"
count on the pages line, 18 says "reads" where older versions said "misses"
and adds an "eagerly scanned" count. the
regexes below accept both.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

_HEADER = re.compile(
    r"automatic (?P<aggr>aggressive )?(?P<what>vacuum|analyze)"
    r"(?P<wrap> to prevent wraparound)? of table "
    r'"(?P<db>[^"]+?)\.(?P<schema>[^".]+)\.(?P<table>[^"]+)"'
    r"(?::\s+index scans:\s+(?P<iscans>\d+))?"
)
_TS = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"(?:\s*(?P<tz>[A-Z]{2,5}|[+-]\d{2}(?::?\d{2})?|Z))?"
)
_PID = re.compile(r"\[(\d+)\]")

_PAGES = re.compile(
    r"pages: (?P<removed>\d+) removed, (?P<remain>\d+) remain"
    r"(?:, (?P<scanned>\d+) scanned \((?P<pct>[\d.]+)% of total\))?"
)
_TUPLES = re.compile(
    r"tuples: (?P<removed>\d+) removed, (?P<remain>\d+) remain, "
    r"(?P<notyet>\d+) are dead but not yet removable"
)
_FROZEN = re.compile(
    r"frozen: (?P<pages>\d+) pages from table \([\d.]+% of total\) had "
    r"(?P<tuples>\d+) tuples frozen"
)
_LPDEAD = re.compile(
    r"index scan (?P<state>needed|not needed|bypassed(?: by failsafe)?): "
    r"(?P<pages>\d+) pages from table \([\d.]+% of total\) ha(?:d|ve) "
    r"(?P<items>\d+) dead item identifiers"
)
_BUFFERS = re.compile(
    r"buffer usage: (?P<hits>\d+) hits, (?P<misses>\d+) (?:misses|reads), (?P<dirtied>\d+) dirtied"
)
_RATES = re.compile(r"avg read rate: (?P<r>[\d.]+) MB/s, avg write rate: (?P<w>[\d.]+) MB/s")
_ELAPSED = re.compile(r"elapsed: (?P<e>[\d.]+) s")
_WAL = re.compile(
    r"WAL usage: (?P<rec>\d+) records, (?P<fpi>\d+) full page images, "
    r"(?P<bytes>\d+) bytes"
)
_RELFROZEN = re.compile(r"new relfrozenxid: \d+, which is (?P<d>-?\d+) XIDs ahead")
_DELAY = re.compile(r"delay time: (?P<ms>[\d.]+) ms")

# log_timezone abbreviations that mean one fixed offset. anything else is
# ambiguous (IST, CST in china vs the us) and falls back to --log-tz.
_TZ_OFFSETS = {
    "UTC": 0,
    "GMT": 0,
    "Z": 0,
    "WET": 0,
    "WEST": 1,
    "BST": 1,
    "CET": 1,
    "CEST": 2,
    "EET": 2,
    "EEST": 3,
    "MSK": 3,
    "EST": -5,
    "EDT": -4,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKST": -9,
    "AKDT": -8,
    "HST": -10,
    "JST": 9,
    "KST": 9,
    "AEST": 10,
    "AEDT": 11,
    "NZST": 12,
    "NZDT": 13,
}
_warned_tz: set[str] = set()


@dataclass
class LogRun:
    """one autovacuum or autoanalyze run as the server reported it."""

    end: datetime
    elapsed: float
    db: str
    schema: str
    table: str
    kind: str  # "vacuum" or "analyze"
    aggressive: bool = False
    wraparound: bool = False
    pid: int | None = None
    index_scans: int = 0
    pages_removed: int = 0
    pages_remain: int = 0
    pages_scanned: int | None = None
    tuples_removed: int = 0
    tuples_remain: int = 0
    dead_not_removable: int = 0
    frozen_pages: int = 0
    lpdead_pages: int = 0
    lpdead_items: int = 0
    index_state: str = ""
    hits: int = 0
    misses: int = 0
    dirtied: int = 0
    read_mbps: float = 0.0
    write_mbps: float = 0.0
    wal_bytes: int = 0
    relfrozenxid_advanced: bool = False
    delay_ms: float | None = None
    raw: str = field(default="", repr=False)

    @property
    def start(self) -> datetime:
        return self.end - timedelta(seconds=self.elapsed)

    @property
    def relname(self) -> str:
        return f"{self.schema}.{self.table}"

    @property
    def failsafe(self) -> bool:
        return self.index_state == "bypassed by failsafe"

    def credits(self, hit: int, miss: int, dirty: int) -> float:
        return float(self.hits * hit + self.misses * miss + self.dirtied * dirty)


def parse_message(msg: str, end: datetime, pid: int | None = None) -> LogRun | None:
    m = _HEADER.search(msg)
    if not m:
        return None
    e = _ELAPSED.search(msg)
    run = LogRun(
        end=end,
        elapsed=float(e.group("e")) if e else 0.0,
        db=m.group("db"),
        schema=m.group("schema"),
        table=m.group("table"),
        kind=m.group("what"),
        aggressive=bool(m.group("aggr")),
        wraparound=bool(m.group("wrap")),
        pid=pid,
        index_scans=int(m.group("iscans") or 0),
        raw=msg,
    )
    if mm := _PAGES.search(msg):
        run.pages_removed = int(mm.group("removed"))
        run.pages_remain = int(mm.group("remain"))
        if mm.group("scanned") is not None:
            run.pages_scanned = int(mm.group("scanned"))
    if mm := _TUPLES.search(msg):
        run.tuples_removed = int(mm.group("removed"))
        run.tuples_remain = int(mm.group("remain"))
        run.dead_not_removable = int(mm.group("notyet"))
    if mm := _FROZEN.search(msg):
        run.frozen_pages = int(mm.group("pages"))
    if mm := _LPDEAD.search(msg):
        run.index_state = mm.group("state")
        run.lpdead_pages = int(mm.group("pages"))
        run.lpdead_items = int(mm.group("items"))
    if mm := _BUFFERS.search(msg):
        run.hits = int(mm.group("hits"))
        run.misses = int(mm.group("misses"))
        run.dirtied = int(mm.group("dirtied"))
    if mm := _RATES.search(msg):
        run.read_mbps = float(mm.group("r"))
        run.write_mbps = float(mm.group("w"))
    if mm := _WAL.search(msg):
        run.wal_bytes = int(mm.group("bytes"))
    if _RELFROZEN.search(msg):
        run.relfrozenxid_advanced = True
    if mm := _DELAY.search(msg):
        run.delay_ms = float(mm.group("ms"))
    return run


def _parse_ts(raw: str, tz: str | None, default_tz: timezone) -> datetime:
    t = datetime.fromisoformat(raw.replace(" ", "T"))
    if tz is None:
        return t.replace(tzinfo=default_tz)
    if tz in _TZ_OFFSETS:
        return t.replace(tzinfo=timezone(timedelta(hours=_TZ_OFFSETS[tz])))
    if tz[0] in "+-":
        sign = 1 if tz[0] == "+" else -1
        digits = tz[1:].replace(":", "")
        hh = int(digits[:2])
        mm = int(digits[2:4]) if len(digits) >= 4 else 0
        return t.replace(tzinfo=timezone(sign * timedelta(hours=hh, minutes=mm)))
    # an abbreviation we can't resolve without guessing; the caller can pass
    # --log-tz. treat it as the default, but say so once
    if tz not in _warned_tz:
        _warned_tz.add(tz)
        print(
            f"warning: log time zone {tz!r} is ambiguous, using {default_tz} "
            "(pass --log-tz to override)",
            file=sys.stderr,
        )
    return t.replace(tzinfo=default_tz)


def _iter_stderr(f: TextIO, default_tz: timezone) -> Iterator[LogRun]:
    cur: list[str] = []
    cur_ts: datetime | None = None
    cur_pid: int | None = None

    def flush() -> LogRun | None:
        if not cur or cur_ts is None:
            return None
        return parse_message("\n".join(cur), cur_ts, cur_pid)

    for line in f:
        line = line.rstrip("\n")
        if line[:1] in ("\t", " ") and cur:
            cur.append(line.strip())
            continue
        m = _TS.match(line)
        if m:
            r = flush()
            if r:
                yield r
            cur = []
            cur_ts = None
            if "automatic " in line:
                cur_ts = _parse_ts(m.group("ts"), m.group("tz"), default_tz)
                p = _PID.search(line, m.end())
                cur_pid = int(p.group(1)) if p else None
                cur = [line]
        elif cur:
            # syslog-ish or wrapped output; keep it, regexes are anchored on text
            cur.append(line.strip())
    r = flush()
    if r:
        yield r


def _iter_json(f: TextIO, default_tz: timezone) -> Iterator[LogRun]:
    for line in f:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = str(d.get("message", ""))
        if "automatic " not in msg:
            continue
        m = _TS.match(str(d.get("timestamp", "")))
        if not m:
            continue
        ts = _parse_ts(m.group("ts"), m.group("tz"), default_tz)
        pid = d.get("pid")
        r = parse_message(msg, ts, int(pid) if pid is not None else None)
        if r:
            yield r


def _iter_csv(f: TextIO, default_tz: timezone) -> Iterator[LogRun]:
    for row in csv.reader(f):
        if len(row) < 14:
            continue
        msg = row[13]
        if "automatic " not in msg:
            continue
        m = _TS.match(row[0])
        if not m:
            continue
        ts = _parse_ts(m.group("ts"), m.group("tz"), default_tz)
        pid = int(row[3]) if row[3].isdigit() else None
        r = parse_message(msg, ts, pid)
        if r:
            yield r


def detect_format(head: str) -> str:
    s = head.lstrip()
    if s.startswith("{"):
        return "json"
    first = s.splitlines()[0] if s else ""
    if first.count(",") >= 13 and '"' in first:
        return "csv"
    return "stderr"


def parse_lines(
    lines: Iterable[str] | str, fmt: str | None = None, default_tz: timezone = UTC
) -> list[LogRun]:
    text = (
        lines
        if isinstance(lines, str)
        else "".join(ln if ln.endswith("\n") else ln + "\n" for ln in lines)
    )
    fmt = fmt or detect_format(text[:4096])
    f = io.StringIO(text)
    if fmt == "json":
        return list(_iter_json(f, default_tz))
    if fmt == "csv":
        return list(_iter_csv(f, default_tz))
    return list(_iter_stderr(f, default_tz))


def parse_files(
    paths: Iterable[str | Path], fmt: str | None = None, default_tz: timezone = UTC
) -> list[LogRun]:
    out: list[LogRun] = []
    for p in paths:
        path = Path(p)
        if path.suffix == ".gz":
            with gzip.open(path, "rt", errors="replace") as f:
                text = f.read()
        else:
            text = path.read_text(errors="replace")
        out.extend(parse_lines(text, fmt, default_tz))
    out.sort(key=lambda r: r.end)
    return out
