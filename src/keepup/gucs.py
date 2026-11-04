"""autovacuum-related server settings and per-table reloptions.

values follow postgres semantics, including the -1 "fall back to something
else" convention. everything is stored in base units: seconds for naptime,
milliseconds for cost delay, kB for memory, raw counts for everything else.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

SUPPORTED_VERSIONS = (13, 14, 15, 16, 17, 18)

# pg uses MaxTransactionId >> 1 as the distance between datfrozenxid and the
# wrap limit. varsup.c: stop limit is 3M before that, warnings start 40M before.
XID_WRAP_LIMIT = 2**31 - 1
XID_STOP_LIMIT = XID_WRAP_LIMIT - 3_000_000
XID_WARN_LIMIT = XID_WRAP_LIMIT - 40_000_000

_TIME_UNITS_MS = {
    "us": 0.001,
    "ms": 1.0,
    "s": 1000.0,
    "min": 60_000.0,
    "h": 3_600_000.0,
    "d": 86_400_000.0,
}
_MEM_UNITS_KB = {"B": 1 / 1024, "kB": 1.0, "MB": 1024.0, "GB": 1024.0**2, "TB": 1024.0**3}
_VALUE_RE = re.compile(r"^\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Z]*)\s*$")


class SettingError(ValueError):
    pass


def _parse_with_units(
    raw: str | float | int, units: Mapping[str, float], default_unit: str, want_unit: str
) -> float:
    if isinstance(raw, bool):
        raise SettingError(f"expected a number, got {raw!r}")
    if isinstance(raw, (int, float)):
        return float(raw) * units[default_unit] / units[want_unit]
    m = _VALUE_RE.match(raw)
    if not m:
        raise SettingError(f"can't parse {raw!r}")
    num, unit = float(m.group(1)), m.group(2) or default_unit
    if unit not in units:
        raise SettingError(f"unknown unit {unit!r} in {raw!r}")
    return num * units[unit] / units[want_unit]


def parse_time_ms(raw: str | float | int, default_unit: str = "ms") -> float:
    return _parse_with_units(raw, _TIME_UNITS_MS, default_unit, "ms")


def parse_memory_kb(raw: str | float | int, default_unit: str = "kB") -> float:
    return _parse_with_units(raw, _MEM_UNITS_KB, default_unit, "kB")


def parse_bool(raw: str | bool | int) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw != 0
    v = raw.strip().lower()
    if v in ("on", "true", "yes", "1", "t", "y"):
        return True
    if v in ("off", "false", "no", "0", "f", "n"):
        return False
    raise SettingError(f"not a boolean: {raw!r}")


@dataclass(frozen=True)
class Settings:
    """server-wide gucs that matter to autovacuum scheduling and throughput."""

    version: int = 16
    autovacuum: bool = True
    autovacuum_max_workers: int = 3
    autovacuum_naptime: float = 60.0  # seconds
    autovacuum_vacuum_threshold: int = 50
    autovacuum_vacuum_insert_threshold: int = 1000  # -1 disables insert-driven vacuum
    autovacuum_vacuum_max_threshold: int = 100_000_000  # pg18+, -1 disables
    autovacuum_analyze_threshold: int = 50
    autovacuum_vacuum_scale_factor: float = 0.2
    autovacuum_vacuum_insert_scale_factor: float = 0.2
    autovacuum_analyze_scale_factor: float = 0.1
    autovacuum_freeze_max_age: int = 200_000_000
    autovacuum_vacuum_cost_delay: float = 2.0  # ms, -1 = use vacuum_cost_delay
    autovacuum_vacuum_cost_limit: int = -1  # -1 = use vacuum_cost_limit
    autovacuum_work_mem: int = -1  # kB, -1 = use maintenance_work_mem
    vacuum_cost_delay: float = 0.0  # ms
    vacuum_cost_limit: int = 200
    vacuum_cost_page_hit: int = 1
    vacuum_cost_page_miss: int = 2
    vacuum_cost_page_dirty: int = 20
    vacuum_freeze_min_age: int = 50_000_000
    vacuum_freeze_table_age: int = 150_000_000
    vacuum_failsafe_age: int = 1_600_000_000
    vacuum_max_eager_freeze_failure_rate: float = 0.03  # pg18+
    maintenance_work_mem: int = 65_536  # kB
    default_statistics_target: int = 100
    checkpoint_timeout: float = 300.0  # seconds

    @classmethod
    def defaults_for(cls, version: int) -> Settings:
        if version not in SUPPORTED_VERSIONS:
            raise SettingError(f"postgres {version} isn't supported (have {SUPPORTED_VERSIONS})")
        s = cls(version=version)
        if version < 14:
            # page_miss dropped from 10 to 2 in 14, failsafe didn't exist yet
            s = replace(s, vacuum_cost_page_miss=10, vacuum_failsafe_age=0)
        if version < 18:
            s = replace(
                s, autovacuum_vacuum_max_threshold=-1, vacuum_max_eager_freeze_failure_rate=0.0
            )
        return s

    # effective values, mirroring the fallbacks in autovacuum.c / vacuum.c

    @property
    def cost_limit(self) -> int:
        if self.autovacuum_vacuum_cost_limit > 0:
            return self.autovacuum_vacuum_cost_limit
        return self.vacuum_cost_limit

    @property
    def cost_delay_ms(self) -> float:
        if self.autovacuum_vacuum_cost_delay >= 0:
            return self.autovacuum_vacuum_cost_delay
        return self.vacuum_cost_delay

    @property
    def work_mem_kb(self) -> int:
        if self.autovacuum_work_mem > 0:
            return self.autovacuum_work_mem
        return self.maintenance_work_mem

    @property
    def freeze_min_age_eff(self) -> int:
        # vacuum_get_cutoffs clamps this to half of autovacuum_freeze_max_age
        return min(self.vacuum_freeze_min_age, self.autovacuum_freeze_max_age // 2)

    @property
    def freeze_table_age_eff(self) -> int:
        return min(self.vacuum_freeze_table_age, int(self.autovacuum_freeze_max_age * 0.95))

    @property
    def failsafe_age_eff(self) -> int:
        if self.version < 14 or self.vacuum_failsafe_age <= 0:
            return 2**31  # never
        return max(self.vacuum_failsafe_age, int(self.autovacuum_freeze_max_age * 1.05))

    def with_values(self, values: Mapping[str, Any]) -> Settings:
        """return a copy with raw values (strings with units, numbers) applied."""
        changes: dict[str, Any] = {}
        known = {f.name: f for f in fields(self)}
        for name, raw in values.items():
            if name not in known:
                raise SettingError(f"unknown setting {name!r}")
            changes[name] = _coerce(name, raw)
        return replace(self, **changes)

    def diff(self, other: Settings) -> dict[str, tuple[Any, Any]]:
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a != b:
                out[f.name] = (a, b)
        return out

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(name: str, raw: Any) -> Any:
    if name == "autovacuum":
        return parse_bool(raw)
    if name in ("autovacuum_naptime", "checkpoint_timeout"):
        return parse_time_ms(raw, default_unit="s") / 1000.0
    if name in ("autovacuum_vacuum_cost_delay", "vacuum_cost_delay"):
        return parse_time_ms(raw, default_unit="ms")
    if name in ("autovacuum_work_mem", "maintenance_work_mem"):
        if str(raw).strip() == "-1":
            return -1
        return int(parse_memory_kb(raw, default_unit="kB"))
    if name.endswith("scale_factor") or name.endswith("_rate"):
        return float(raw)
    if name == "version":
        return int(raw)
    try:
        return int(float(raw))
    except (TypeError, ValueError) as e:
        raise SettingError(f"{name}: can't parse {raw!r}") from e


def settings_from_pg(rows: Mapping[str, tuple[str, str | None]], version: int) -> Settings:
    """build Settings from pg_settings rows: name -> (setting, unit)."""
    base = Settings.defaults_for(version)
    vals: dict[str, Any] = {}
    names = {f.name for f in fields(Settings)}
    for name, (setting, unit) in rows.items():
        if name not in names:
            continue
        if unit and unit not in ("", None):
            # pg_settings reports e.g. setting=60 unit=s, setting=8kB-blocks for some.
            # '8kB' style units only show up on buffer gucs we don't read.
            vals[name] = f"{setting}{unit}"
        else:
            vals[name] = setting
    return base.with_values(vals)


# reloptions that autovacuum reads from pg_class.reloptions
RELOPTION_NAMES = (
    "autovacuum_enabled",
    "autovacuum_vacuum_threshold",
    "autovacuum_vacuum_max_threshold",
    "autovacuum_vacuum_scale_factor",
    "autovacuum_vacuum_insert_threshold",
    "autovacuum_vacuum_insert_scale_factor",
    "autovacuum_analyze_threshold",
    "autovacuum_analyze_scale_factor",
    "autovacuum_vacuum_cost_delay",
    "autovacuum_vacuum_cost_limit",
    "autovacuum_freeze_min_age",
    "autovacuum_freeze_max_age",
    "autovacuum_freeze_table_age",
)


@dataclass(frozen=True)
class TableOptions:
    """per-table storage parameters. None means not set."""

    enabled: bool = True
    vacuum_threshold: int | None = None
    vacuum_max_threshold: int | None = None
    vacuum_scale_factor: float | None = None
    vacuum_insert_threshold: int | None = None
    vacuum_insert_scale_factor: float | None = None
    analyze_threshold: int | None = None
    analyze_scale_factor: float | None = None
    vacuum_cost_delay: float | None = None  # ms
    vacuum_cost_limit: int | None = None
    freeze_min_age: int | None = None
    freeze_max_age: int | None = None
    freeze_table_age: int | None = None
    fillfactor: int = 100

    @classmethod
    def parse(cls, opts: Mapping[str, Any] | list[str] | None) -> TableOptions:
        if not opts:
            return cls()
        if isinstance(opts, list):
            # pg_class.reloptions comes back as ['k=v', ...]
            pairs: dict[str, Any] = {}
            for item in opts:
                k, _, v = item.partition("=")
                pairs[k.strip()] = v.strip()
            opts = pairs
        kw: dict[str, Any] = {}
        for k, v in opts.items():
            key = k.removeprefix("autovacuum_")
            if k == "autovacuum_enabled":
                kw["enabled"] = parse_bool(v)
            elif k == "fillfactor":
                kw["fillfactor"] = int(v)
            elif k in RELOPTION_NAMES:
                if key == "vacuum_cost_delay":
                    kw[key] = parse_time_ms(v)
                elif key.endswith("scale_factor"):
                    kw[key] = float(v)
                else:
                    kw[key] = int(float(v))
            elif k.startswith("toast."):
                continue  # toast tables aren't modelled
            else:
                # unknown reloptions (parallel_workers, etc.) don't affect us
                continue
        return cls(**kw)

    def to_reloptions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.enabled:
            out["autovacuum_enabled"] = "false"
        for f in fields(self):
            if f.name in ("enabled", "fillfactor"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f"autovacuum_{f.name}"] = _fmt_num(v)
        if self.fillfactor != 100:
            out["fillfactor"] = str(self.fillfactor)
        return out


def _fmt_num(v: float | int | bool) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v.is_integer() and abs(v) >= 1:
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


@dataclass(frozen=True)
class Effective:
    """per-table numbers after combining reloptions with server settings.

    this is relation_needs_vacanalyze + table_recheck_autovac in one place.
    """

    enabled: bool
    vac_base: int
    vac_scale: float
    vac_max: int  # -1 = no cap
    ins_base: int  # -1 = disabled
    ins_scale: float
    anl_base: int
    anl_scale: float
    freeze_max_age: int
    freeze_min_age: int
    freeze_table_age: int
    failsafe_age: int
    cost_limit: int
    cost_delay_ms: float
    dobalance: bool
    use_unfrozen_fraction: bool = field(default=False)

    def vacuum_threshold(self, reltuples: float) -> float:
        t = self.vac_base + self.vac_scale * max(reltuples, 0.0)
        if self.vac_max >= 0 and t > self.vac_max:
            t = float(self.vac_max)
        return t

    def insert_threshold(self, reltuples: float, unfrozen_fraction: float = 1.0) -> float:
        if self.ins_base < 0:
            return float("inf")
        frac = unfrozen_fraction if self.use_unfrozen_fraction else 1.0
        return self.ins_base + self.ins_scale * max(reltuples, 0.0) * frac

    def analyze_threshold(self, reltuples: float) -> float:
        return self.anl_base + self.anl_scale * max(reltuples, 0.0)


def effective(settings: Settings, opts: TableOptions) -> Effective:
    s, o = settings, opts

    def pick(v: Any, default: Any) -> Any:
        return default if v is None else v

    vac_max = -1
    if s.version >= 18:
        vac_max = pick(o.vacuum_max_threshold, s.autovacuum_vacuum_max_threshold)
    ins_base = pick(o.vacuum_insert_threshold, s.autovacuum_vacuum_insert_threshold)

    freeze_max = s.autovacuum_freeze_max_age
    if o.freeze_max_age is not None:
        # a reloption can only make anti-wraparound vacuums happen sooner
        freeze_max = min(o.freeze_max_age, freeze_max)

    # these come from table_recheck_autovac, then vacuum_get_cutoffs clamps
    # them using the *server* autovacuum_freeze_max_age
    fmin = pick(o.freeze_min_age, s.vacuum_freeze_min_age)
    fmin = min(fmin, s.autovacuum_freeze_max_age // 2)
    ftab = pick(o.freeze_table_age, s.vacuum_freeze_table_age)
    ftab = min(ftab, int(s.autovacuum_freeze_max_age * 0.95))

    # per-table cost settings take the worker out of the balancing pool
    # (at_dobalance in table_recheck_autovac). note a table with only a
    # delay set still gets the full, unsplit server cost limit.
    cost_limit = s.cost_limit
    has_limit = False
    if o.vacuum_cost_limit is not None and o.vacuum_cost_limit > 0:
        cost_limit, has_limit = o.vacuum_cost_limit, True
    cost_delay = s.cost_delay_ms
    has_delay = False
    if o.vacuum_cost_delay is not None and o.vacuum_cost_delay >= 0:
        cost_delay, has_delay = o.vacuum_cost_delay, True

    return Effective(
        enabled=o.enabled,
        vac_base=pick(o.vacuum_threshold, s.autovacuum_vacuum_threshold),
        vac_scale=pick(o.vacuum_scale_factor, s.autovacuum_vacuum_scale_factor),
        vac_max=vac_max,
        ins_base=ins_base,
        ins_scale=pick(o.vacuum_insert_scale_factor, s.autovacuum_vacuum_insert_scale_factor),
        anl_base=pick(o.analyze_threshold, s.autovacuum_analyze_threshold),
        anl_scale=pick(o.analyze_scale_factor, s.autovacuum_analyze_scale_factor),
        freeze_max_age=freeze_max,
        freeze_min_age=fmin,
        freeze_table_age=ftab,
        failsafe_age=s.failsafe_age_eff,
        cost_limit=cost_limit,
        cost_delay_ms=cost_delay,
        dobalance=not (has_limit or has_delay),
        use_unfrozen_fraction=s.version >= 18,
    )
