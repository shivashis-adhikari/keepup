from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from keepup.workload import RateModel, bin_intervals, counter_intervals, cumulative, fit_rate

MON = datetime(2026, 10, 5, tzinfo=UTC)  # a monday
HOD = 1 + 0.6 * np.sin(np.arange(24) / 24 * 2 * np.pi)
DOW = np.array([1.1, 1.1, 1.1, 1.1, 1.1, 0.75, 0.75])


def synth(days, base=50.0, growth=0.0, step_min=5, seed=0):
    rng = np.random.default_rng(seed)
    hours = days * 24
    h = np.arange(hours)
    rate = base * (1 + growth * ((h + 0.5) / 24 - days)) * HOD[h % 24] * DOW[(h // 24) % 7]
    times, vals, c = [MON], [0.0], 0.0
    for i in range(hours):
        for m in range(0, 60, step_min):
            c += rng.poisson(rate[i] * step_min * 60)
            times.append(MON + timedelta(hours=i, minutes=m + step_min))
            vals.append(c)
    return times, vals


def test_flat_from_one_day():
    t, v = synth(1)
    rep = fit_rate(bin_intervals(counter_intervals(t, v)))
    assert rep.kind == "flat"
    assert rep.model.growth_per_day == 0


def test_daily_shape_recovered():
    t, v = synth(3)
    rep = fit_rate(bin_intervals(counter_intervals(t, v)))
    assert rep.kind == "daily"
    assert np.max(np.abs(np.array(rep.model.hod) - HOD)) < 0.2


def test_weekly_and_no_fake_growth():
    t, v = synth(14)
    rep = fit_rate(bin_intervals(counter_intervals(t, v)))
    assert rep.kind == "weekly"
    assert rep.model.growth_per_day == 0
    assert rep.model.dow is not None
    assert rep.model.dow[6] == pytest.approx(0.75, abs=0.05)
    assert rep.model.base == pytest.approx(50, rel=0.03)


def test_real_growth_detected():
    t, v = synth(21, growth=0.01)
    rep = fit_rate(bin_intervals(counter_intervals(t, v)))
    assert rep.model.growth_per_day == pytest.approx(0.01, abs=0.002)


def test_counter_reset_dropped():
    t = [MON + timedelta(minutes=i) for i in range(4)]
    iv = counter_intervals(t, [0, 100, 5, 105])
    assert [d for _, _, d in iv] == [100, 100]


def test_bin_intervals_splits_across_hours():
    a = MON + timedelta(minutes=30)
    b = MON + timedelta(minutes=90)
    h = bin_intervals([(a, b, 120.0)])
    assert h.counts.tolist() == [60.0, 60.0]
    assert h.covered.tolist() == [1800.0, 1800.0]


def test_rate_model_from_value():
    assert RateModel.from_value(3).base == 3
    assert RateModel.from_value({"per_hour": 3600}).base == 1.0
    with pytest.raises(ValueError):
        RateModel.from_value({"per_fortnight": 1})


def test_noise_is_mean_preserving():
    m = RateModel(base=10.0, noise_sigma=0.5, noise_ar1=0.8)
    rng = np.random.default_rng(3)
    r = m.hourly_rates(MON, 24 * 200, rng=rng)
    assert r.mean() == pytest.approx(10.0, rel=0.1)


def test_cumulative():
    c = cumulative(np.array([1.0, 2.0]))
    assert c.tolist() == [0.0, 3600.0, 10800.0]
