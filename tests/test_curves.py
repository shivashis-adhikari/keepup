import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from keepup.sim.curves import Curve
from keepup.workload import cumulative

rates = st.lists(st.floats(min_value=0, max_value=1e4, allow_nan=False), min_size=2, max_size=50)


@given(rates, st.floats(min_value=0, max_value=1))
def test_time_of_inverts_at(rs, frac):
    c = Curve(cumulative(np.array(rs)))
    total = c.c[-1]
    if total == 0:
        return
    v = frac * total
    t = c.time_of(v)
    assert math.isclose(c.at(t), v, rel_tol=1e-9, abs_tol=1e-6)
    # and it's the first time: a bit earlier the count is lower
    if t > 1e-3:
        assert c.at(t - 1e-3) <= v + 1e-6


def test_plateau_returns_start():
    c = Curve(cumulative(np.array([1.0, 0.0, 0.0, 1.0])))
    assert c.time_of(3600.0) == 3600.0


def test_extrapolates_past_the_end():
    c = Curve(cumulative(np.array([1.0, 1.0])))
    assert c.at(3 * 3600) == 3 * 3600
    assert c.time_of(4 * 3600) == 4 * 3600


def test_before_start_uses_first_rate():
    c = Curve(cumulative(np.array([2.0, 5.0])))
    assert c.at(-3600) == -7200
    assert c.time_of(-7200) == -3600


def test_time_to_add():
    c = Curve(cumulative(np.array([1.0] * 10)))
    assert c.time_to_add(100.0, 50.0) == 150.0
    assert c.time_to_add(100.0, -5.0) == 100.0
