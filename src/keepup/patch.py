"""turn a search result (or hand-picked changes) into sql you can run."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

from .gucs import Settings, TableOptions


def _needs_restart(name: str, version: int) -> bool:
    if name == "autovacuum_max_workers":
        return version < 18  # 18 made it sighup, capped by autovacuum_worker_slots
    return name == "autovacuum_freeze_max_age"


_UNITS = {
    "autovacuum_naptime": ("s", 1.0),
    "autovacuum_vacuum_cost_delay": ("ms", 1.0),
    "vacuum_cost_delay": ("ms", 1.0),
    "autovacuum_work_mem": ("kB", 1.0),
    "maintenance_work_mem": ("kB", 1.0),
}


def _guc_value(name: str, v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if name in _UNITS:
        unit, _ = _UNITS[name]
        if unit == "kB" and v > 0 and v % 1024 == 0:
            return f"'{v // 1024}MB'"
        num = f"{v:g}" if isinstance(v, float) else str(v)
        return f"'{num}{unit}'" if float(v) >= 0 else num
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _ident(name: str) -> str:
    # schema.table -> "schema"."table" only when it needs quoting
    parts = name.split(".")
    out = []
    for p in parts:
        if p.isidentifier() and p == p.lower():
            out.append(p)
        else:
            out.append('"' + p.replace('"', '""') + '"')
    return ".".join(out)


def render(
    base: Settings,
    new: Settings,
    base_tables: dict[str, TableOptions],
    new_tables: dict[str, TableOptions],
    comment: list[str] | None = None,
) -> str:
    lines = [f"-- keepup settings patch, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC"]
    for c in comment or []:
        lines.append(f"-- {c}")
    lines.append("")
    server = base.diff(new)
    server.pop("version", None)
    if server:
        lines.append("-- server settings. ALTER SYSTEM writes postgresql.auto.conf")
        for k, (old, val) in server.items():
            note = ", needs a restart" if _needs_restart(k, new.version) else ""
            lines.append(
                f"ALTER SYSTEM SET {k} = {_guc_value(k, val)};  -- was {_guc_value(k, old)}{note}"
            )
        lines.append("SELECT pg_reload_conf();")
        lines.append("")
    tabs = []
    for name, opts in new_tables.items():
        old = base_tables.get(name, TableOptions())
        set_: dict[str, str] = {}
        reset: list[str] = []
        for f in fields(TableOptions):
            a, b = getattr(old, f.name), getattr(opts, f.name)
            if a == b:
                continue
            key = (
                "fillfactor"
                if f.name == "fillfactor"
                else ("autovacuum_enabled" if f.name == "enabled" else f"autovacuum_{f.name}")
            )
            if b is None:
                reset.append(key)
            else:
                set_[key] = TableOptions(**{f.name: b}).to_reloptions().get(key, str(b))
        if set_ or reset:
            tabs.append((name, set_, reset))
    if tabs:
        lines.append("-- per table. ALTER TABLE ... SET takes a SHARE UPDATE EXCLUSIVE lock,")
        lines.append("-- which conflicts with a running vacuum on that table; a normal autovacuum")
        lines.append("-- gets cancelled, an anti-wraparound one makes you wait. lock_timeout")
        lines.append("-- keeps it from queueing behind one.")
        lines.append("SET lock_timeout = '5s';")
        for name, set_, reset in tabs:
            if set_:
                kv = ", ".join(f"{k} = {v}" for k, v in set_.items())
                lines.append(f"ALTER TABLE {_ident(name)} SET ({kv});")
            if reset:
                lines.append(f"ALTER TABLE {_ident(name)} RESET ({', '.join(reset)});")
        lines.append("RESET lock_timeout;")
        lines.append("")
    if not server and not tabs:
        lines.append("-- nothing to change")
    return "\n".join(lines).rstrip() + "\n"
