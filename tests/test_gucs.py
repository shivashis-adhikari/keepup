import pytest

from keepup.gucs import (
    SettingError,
    Settings,
    TableOptions,
    effective,
    parse_memory_kb,
    parse_time_ms,
    settings_from_pg,
)


def test_units():
    assert parse_time_ms("1min") == 60_000
    assert parse_time_ms("2ms") == 2
    assert parse_time_ms("30", default_unit="s") == 30_000
    assert parse_time_ms(1.5) == 1.5
    assert parse_memory_kb("64MB") == 65536
    assert parse_memory_kb("1GB") == 1024 * 1024
    with pytest.raises(SettingError):
        parse_time_ms("10 parsecs")


def test_version_defaults():
    assert Settings.defaults_for(13).vacuum_cost_page_miss == 10
    assert Settings.defaults_for(14).vacuum_cost_page_miss == 2
    assert Settings.defaults_for(17).autovacuum_vacuum_max_threshold == -1
    assert Settings.defaults_for(18).autovacuum_vacuum_max_threshold == 100_000_000
    with pytest.raises(SettingError):
        Settings.defaults_for(9)


def test_cost_fallbacks():
    s = Settings.defaults_for(16)
    assert s.cost_limit == 200  # autovacuum_vacuum_cost_limit=-1 falls back
    assert s.cost_delay_ms == 2.0
    s = s.with_values({"autovacuum_vacuum_cost_limit": 1000, "autovacuum_vacuum_cost_delay": "-1"})
    assert s.cost_limit == 1000
    assert s.cost_delay_ms == 0.0  # falls back to vacuum_cost_delay, which is 0


def test_freeze_clamps():
    s = Settings.defaults_for(16).with_values({"autovacuum_freeze_max_age": 100_000_000})
    assert s.freeze_min_age_eff == 50_000_000
    assert s.freeze_table_age_eff == 95_000_000
    s = s.with_values({"vacuum_freeze_min_age": 80_000_000})
    assert s.freeze_min_age_eff == 50_000_000


def test_settings_from_pg_rows():
    rows = {
        "autovacuum_naptime": ("15", "s"),
        "autovacuum_vacuum_cost_delay": ("2", "ms"),
        "maintenance_work_mem": ("262144", "kB"),
        "autovacuum_vacuum_scale_factor": ("0.05", None),
        "shared_buffers": ("16384", "8kB"),  # ignored
    }
    s = settings_from_pg(rows, 16)
    assert s.autovacuum_naptime == 15.0
    assert s.maintenance_work_mem == 262144
    assert s.autovacuum_vacuum_scale_factor == 0.05


def test_reloptions_round_trip():
    o = TableOptions.parse(
        [
            "autovacuum_vacuum_scale_factor=0.01",
            "fillfactor=90",
            "autovacuum_enabled=off",
            "parallel_workers=4",
        ]
    )
    assert o.vacuum_scale_factor == 0.01
    assert o.fillfactor == 90
    assert not o.enabled
    again = TableOptions.parse(o.to_reloptions())
    assert again == o


def test_per_table_cost_settings_leave_the_pool():
    s = Settings.defaults_for(16)
    assert effective(s, TableOptions()).dobalance
    e = effective(s, TableOptions(vacuum_cost_limit=2000))
    assert not e.dobalance and e.cost_limit == 2000
    # only a delay set: unbalanced, and it gets the full server limit
    e = effective(s, TableOptions(vacuum_cost_delay=1.0))
    assert not e.dobalance and e.cost_limit == 200 and e.cost_delay_ms == 1.0


def test_table_freeze_max_age_can_only_lower():
    s = Settings.defaults_for(16)
    assert effective(s, TableOptions(freeze_max_age=500_000_000)).freeze_max_age == 200_000_000
    assert effective(s, TableOptions(freeze_max_age=100_000_000)).freeze_max_age == 100_000_000


def test_thresholds():
    s = Settings.defaults_for(16)
    e = effective(s, TableOptions())
    assert e.vacuum_threshold(1_000_000) == 50 + 200_000
    assert e.insert_threshold(1_000_000) == 1000 + 200_000
    assert e.analyze_threshold(1_000_000) == 50 + 100_000
    # pg18 caps the vacuum threshold and scales insert threshold by unfrozen share
    e18 = effective(Settings.defaults_for(18), TableOptions())
    assert e18.vacuum_threshold(10_000_000_000) == 100_000_000
    assert e18.insert_threshold(1_000_000, unfrozen_fraction=0.5) == 1000 + 100_000
    assert e.insert_threshold(1_000_000, unfrozen_fraction=0.5) == 1000 + 200_000
