from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from keepup.gucs import Settings, TableOptions
from keepup.profile import Profile, TableProfile
from keepup.workload import RateModel

START = datetime(2026, 11, 2, tzinfo=UTC)


def make_table(name: str = "public.t", **kw: Any) -> TableProfile:
    defaults: dict[str, Any] = dict(
        live_tuples=1_000_000,
        pages=10_000,
        tuples_per_page=100.0,
        index_pages=3_000,
        all_visible_fraction=0.9,
        cache_hit=0.5,
    )
    defaults.update(kw)
    for k in ("inserts", "updates", "deletes"):
        v = defaults.get(k)
        if isinstance(v, (int, float)):
            defaults[k] = RateModel.constant(float(v))
    opts = defaults.pop("options", None)
    if isinstance(opts, dict):
        defaults["options"] = TableOptions.parse(opts)
    elif opts is not None:
        defaults["options"] = opts
    return TableProfile(name=name, **defaults)


def make_profile(
    *tables: TableProfile, xid_rate: float = 100.0, raw: float = 1e9, **settings: Any
) -> Profile:
    s = Settings.defaults_for(16).with_values(settings) if settings else Settings.defaults_for(16)
    return Profile(
        settings=s,
        tables=list(tables) or [make_table()],
        xid_rate=RateModel.constant(xid_rate),
        start=START,
        raw_credit_rate=raw,
        sleep_overhead=1.0,
    )


@pytest.fixture
def profile() -> Profile:
    return make_profile()
