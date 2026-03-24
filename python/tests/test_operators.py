"""Tests for newly exposed native operators (last, lag, rolling, ema, ffill)."""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow import Scenario
from tradingflow.sources import ArraySource
from tradingflow.operators import (
    record,
    add,
    subtract,
    multiply,
    divide,
    negate,
    select,
    concat,
    stack,
    last,
    lag,
    rolling_sum,
    rolling_mean,
    rolling_variance,
    rolling_covariance,
    ema,
    forward_fill,
    # New math operators
    log,
    exp,
    sqrt,
    abs,
    recip,
    pow,
    scale,
    shift,
    clamp,
    nan_to_num,
    min,
    max,
)
from tradingflow.types import Handle


def ts(i: int) -> np.datetime64:
    return np.datetime64(i, "ns")


def _run(sc: Scenario) -> None:
    sc.run()


def _scalar_scenario(values: list[float]) -> tuple[Scenario, Handle, Handle]:
    """Build a scenario with one scalar source and its recorded series."""
    sc = Scenario()
    src = ArraySource.from_arrays(
        timestamps=np.arange(1, len(values) + 1).astype("datetime64[ns]"),
        values=np.array(values, dtype=np.float64),
    )
    h = sc.add_source(src)
    s = sc.add_operator(record(h))
    return sc, h, s


# ---------------------------------------------------------------------------
# Last
# ---------------------------------------------------------------------------


class TestLast:
    def test_last_recovers_latest_value(self) -> None:
        """last(record(x)) == x at each step."""
        sc, h, s = _scalar_scenario([10.0, 20.0, 30.0])
        h_last = sc.add_operator(last(s))
        s_last = sc.add_operator(record(h_last))
        _run(sc)
        assert list(sc.series_values(s_last)) == pytest.approx([10.0, 20.0, 30.0])

    def test_last_vector(self) -> None:
        """last works on vector-valued series."""
        sc = Scenario()
        src = ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        h_last = sc.add_operator(last(s))
        s_last = sc.add_operator(record(h_last))
        _run(sc)
        vals = sc.series_values(s_last)
        np.testing.assert_array_almost_equal(vals[-1], [3.0, 4.0])

    def test_last_scalar(self) -> None:
        """last() on a single-element series returns that element."""
        sc, _, s = _scalar_scenario([5.0])
        h_last = sc.add_operator(last(s))
        s_last = sc.add_operator(record(h_last))
        _run(sc)
        assert list(sc.series_values(s_last)) == pytest.approx([5.0])


# ---------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------


class TestLag:
    def test_lag_basic(self) -> None:
        """lag(offset=2) outputs value from 2 steps ago, fill=0 for early steps."""
        sc, _, s = _scalar_scenario([10.0, 20.0, 30.0, 40.0])
        h_lag = sc.add_operator(lag(s, offset=2))
        _run(sc)
        vals = list(sc.series_values(h_lag))
        # steps 1,2: not enough history → fill=0
        # step 3: value from step 1 → 10.0
        # step 4: value from step 2 → 20.0
        assert vals == pytest.approx([0.0, 0.0, 10.0, 20.0])

    def test_lag_offset_1(self) -> None:
        """Default offset=1 returns previous value."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_lag = sc.add_operator(lag(s))
        _run(sc)
        vals = list(sc.series_values(h_lag))
        assert vals == pytest.approx([0.0, 1.0, 2.0])

    def test_lag_timestamps_match(self) -> None:
        """Lag output has same timestamps as input."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_lag = sc.add_operator(lag(s))
        _run(sc)
        np.testing.assert_array_equal(
            sc.series_timestamps(h_lag),
            sc.series_timestamps(s),
        )


# ---------------------------------------------------------------------------
# Rolling sum
# ---------------------------------------------------------------------------


class TestRollingSum:
    def test_rolling_sum_basic(self) -> None:
        """Rolling sum with window=3."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0, 4.0])
        h_rs = sc.add_operator(rolling_sum(s, window=3))
        _run(sc)
        vals = list(sc.series_values(h_rs))
        assert vals == pytest.approx([1.0, 3.0, 6.0, 9.0])

    def test_rolling_sum_nan_propagation(self) -> None:
        """NaN in window propagates to output; eviction clears it."""
        sc, _, s = _scalar_scenario([1.0, float("nan"), 3.0, 4.0, 5.0])
        h_rs = sc.add_operator(rolling_sum(s, window=3))
        _run(sc)
        vals = list(sc.series_values(h_rs))
        assert vals[0] == pytest.approx(1.0)
        assert np.isnan(vals[1])  # [1, NaN]
        assert np.isnan(vals[2])  # [1, NaN, 3]
        assert np.isnan(vals[3])  # [NaN, 3, 4]
        assert vals[4] == pytest.approx(12.0)  # [3, 4, 5]

    def test_rolling_sum_vector(self) -> None:
        """Rolling sum with 2-element vectors and window=2."""
        sc = Scenario()
        src = ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        h_rs = sc.add_operator(rolling_sum(s, window=2))
        _run(sc)
        vals = sc.series_values(h_rs)
        np.testing.assert_array_almost_equal(vals[-1], [5.0, 50.0])  # 2+3, 20+30


# ---------------------------------------------------------------------------
# Rolling mean
# ---------------------------------------------------------------------------


class TestRollingMean:
    def test_rolling_mean_basic(self) -> None:
        """Rolling mean with window=3."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0, 6.0])
        h_rm = sc.add_operator(rolling_mean(s, window=3))
        _run(sc)
        vals = list(sc.series_values(h_rm))
        assert vals[0] == pytest.approx(1.0)  # mean([1])
        assert vals[1] == pytest.approx(1.5)  # mean([1,2])
        assert vals[2] == pytest.approx(2.0)  # mean([1,2,3])
        assert vals[3] == pytest.approx(11.0 / 3.0)  # mean([2,3,6])

    def test_rolling_mean_constant(self) -> None:
        """Mean of constant series equals the constant."""
        sc, _, s = _scalar_scenario([7.0] * 10)
        h_rm = sc.add_operator(rolling_mean(s, window=5))
        _run(sc)
        vals = list(sc.series_values(h_rm))
        assert vals[-1] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Rolling variance
# ---------------------------------------------------------------------------


class TestRollingVariance:
    def test_rolling_variance_constant(self) -> None:
        """Variance of constant series is zero."""
        sc, _, s = _scalar_scenario([5.0] * 5)
        h_rv = sc.add_operator(rolling_variance(s, window=3))
        _run(sc)
        vals = list(sc.series_values(h_rv))
        assert vals[-1] == pytest.approx(0.0, abs=1e-10)

    def test_rolling_variance_known(self) -> None:
        """Variance of [1, 3] is Var = E[x^2] - E[x]^2 = 5 - 4 = 1."""
        sc, _, s = _scalar_scenario([1.0, 3.0])
        h_rv = sc.add_operator(rolling_variance(s, window=2))
        _run(sc)
        vals = list(sc.series_values(h_rv))
        assert vals[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rolling covariance
# ---------------------------------------------------------------------------


class TestRollingCovariance:
    def test_rolling_covariance_shape(self) -> None:
        """Output shape is (K, K) for K-element input."""
        sc = Scenario()
        src = ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        h_rc = sc.add_operator(rolling_covariance(s, window=3))
        _run(sc)
        vals = sc.series_values(h_rc)
        assert vals.shape == (3, 2, 2)

    def test_rolling_covariance_rejects_non_1d(self) -> None:
        """rolling_covariance raises for non-1D input."""
        h = Handle(0, (2, 3), np.dtype("float64"))
        with pytest.raises(ValueError, match="1-D"):
            rolling_covariance(h, window=3)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEma:
    def test_ema_constant(self) -> None:
        """EMA of constant series converges to the constant."""
        sc, _, s = _scalar_scenario([10.0] * 10)
        h_ema = sc.add_operator(ema(s, window=100, alpha=0.5))
        _run(sc)
        vals = list(sc.series_values(h_ema))
        assert vals[-1] == pytest.approx(10.0, abs=1e-6)

    def test_ema_first_value(self) -> None:
        """First EMA output equals the first input."""
        sc, _, s = _scalar_scenario([100.0, 200.0])
        h_ema = sc.add_operator(ema(s, window=10, alpha=0.5))
        _run(sc)
        vals = list(sc.series_values(h_ema))
        assert vals[0] == pytest.approx(100.0)

    def test_ema_two_values(self) -> None:
        """EMA of two values matches hand-computed result."""
        sc, _, s = _scalar_scenario([10.0, 20.0])
        h_ema = sc.add_operator(ema(s, window=10, alpha=0.5))
        _run(sc)
        vals = list(sc.series_values(h_ema))
        # w0=0.5 (for 20), w1=0.25 (for 10)
        expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25)
        assert vals[1] == pytest.approx(expected)

    def test_ema_with_span(self) -> None:
        """ema(span=3) is equivalent to alpha=0.5."""
        sc, _, s = _scalar_scenario([10.0, 20.0])
        h_ema = sc.add_operator(ema(s, window=100, span=3))
        _run(sc)
        vals = list(sc.series_values(h_ema))
        expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25)
        assert vals[1] == pytest.approx(expected)

    def test_ema_with_half_life(self) -> None:
        """ema(half_life=...) runs without error and produces values."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_ema = sc.add_operator(ema(s, window=10, half_life=2.0))
        _run(sc)
        assert sc.series_len(h_ema) == 3

    def test_ema_window_eviction(self) -> None:
        """Old values are evicted when they leave the window."""
        sc, _, s = _scalar_scenario([100.0, 100.0, 0.0, 0.0])
        h_ema = sc.add_operator(ema(s, window=2, alpha=0.5))
        _run(sc)
        vals = list(sc.series_values(h_ema))
        # After two 0.0s with window=2, 100.0s are fully evicted
        assert vals[-1] == pytest.approx(0.0, abs=1e-10)

    def test_ema_requires_exactly_one_param(self) -> None:
        """Providing zero or multiple smoothing params raises."""
        h = Handle(0, (), np.dtype("float64"))
        with pytest.raises(ValueError, match="exactly one"):
            ema(h, window=10)
        with pytest.raises(ValueError, match="exactly one"):
            ema(h, window=10, alpha=0.5, span=3)


# ---------------------------------------------------------------------------
# Forward fill
# ---------------------------------------------------------------------------


class TestForwardFill:
    def test_forward_fill_replaces_nan(self) -> None:
        """NaN values are replaced with last valid observation."""
        sc, _, s = _scalar_scenario([1.0, float("nan"), float("nan"), 4.0, float("nan")])
        h_ff = sc.add_operator(forward_fill(s))
        _run(sc)
        vals = list(sc.series_values(h_ff))
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(1.0)
        assert vals[2] == pytest.approx(1.0)
        assert vals[3] == pytest.approx(4.0)
        assert vals[4] == pytest.approx(4.0)

    def test_forward_fill_leading_nan(self) -> None:
        """Leading NaN stays NaN until a valid value appears."""
        sc, _, s = _scalar_scenario([float("nan"), float("nan"), 3.0])
        h_ff = sc.add_operator(forward_fill(s))
        _run(sc)
        vals = list(sc.series_values(h_ff))
        assert np.isnan(vals[0])
        assert np.isnan(vals[1])
        assert vals[2] == pytest.approx(3.0)

    def test_forward_fill_no_nan(self) -> None:
        """Clean data passes through unchanged."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_ff = sc.add_operator(forward_fill(s))
        _run(sc)
        vals = list(sc.series_values(h_ff))
        assert vals == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Subtract
# ---------------------------------------------------------------------------


class TestSubtract:
    def test_subtract_scalar(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([10.0, 30.0]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([3.0, 7.0]),
        ))
        h = sc.add_operator(subtract(a, b))
        s = sc.add_operator(record(h))
        _run(sc)
        assert list(sc.series_values(s)) == pytest.approx([7.0, 23.0])


# ---------------------------------------------------------------------------
# Divide
# ---------------------------------------------------------------------------


class TestDivide:
    def test_divide_scalar(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([20.0, 9.0]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([4.0, 3.0]),
        ))
        h = sc.add_operator(divide(a, b))
        s = sc.add_operator(record(h))
        _run(sc)
        assert list(sc.series_values(s)) == pytest.approx([5.0, 3.0])


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_flat(self) -> None:
        sc = Scenario()
        src = ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        )
        h = sc.add_source(src)
        sel = sc.add_operator(select(h, [0, 2]))
        s = sc.add_operator(record(sel))
        _run(sc)
        vals = sc.series_values(s)
        np.testing.assert_array_almost_equal(vals[0], [10.0, 30.0])
        np.testing.assert_array_almost_equal(vals[1], [40.0, 60.0])


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_axis0(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[1.0, 2.0]]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[3.0, 4.0]]),
        ))
        h = sc.add_operator(concat([a, b], axis=0))
        s = sc.add_operator(record(h))
        _run(sc)
        vals = sc.series_values(s)
        np.testing.assert_array_almost_equal(vals[0], [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


class TestStack:
    def test_stack_axis0(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[1.0, 2.0]]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[3.0, 4.0]]),
        ))
        h = sc.add_operator(stack([a, b], axis=0))
        s = sc.add_operator(record(h))
        _run(sc)
        vals = sc.series_values(s)
        # stack axis=0: shape (2,2) → [[1,2],[3,4]]
        np.testing.assert_array_almost_equal(vals[0], [[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# Math unary operators
# ---------------------------------------------------------------------------


class TestMathUnary:
    def test_log(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, np.e, np.e**2])
        h_log = sc.add_operator(log(h))
        s = sc.add_operator(record(h_log))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)

    def test_exp(self) -> None:
        sc, h, _ = _scalar_scenario([0.0, 1.0])
        h_exp = sc.add_operator(exp(h))
        s = sc.add_operator(record(h_exp))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(np.e)

    def test_sqrt(self) -> None:
        sc, h, _ = _scalar_scenario([4.0, 9.0, 16.0])
        h_sqrt = sc.add_operator(sqrt(h))
        s = sc.add_operator(record(h_sqrt))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([2.0, 3.0, 4.0])

    def test_abs(self) -> None:
        sc, h, _ = _scalar_scenario([-3.0, 0.0, 5.0])
        h_abs = sc.add_operator(abs(h))
        s = sc.add_operator(record(h_abs))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([3.0, 0.0, 5.0])

    def test_recip(self) -> None:
        sc, h, _ = _scalar_scenario([2.0, 4.0, 0.5])
        h_recip = sc.add_operator(recip(h))
        s = sc.add_operator(record(h_recip))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([0.5, 0.25, 2.0])


# ---------------------------------------------------------------------------
# Parameterized operators
# ---------------------------------------------------------------------------


class TestParameterized:
    def test_pow(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_pow = sc.add_operator(pow(h, 2.0))
        s = sc.add_operator(record(h_pow))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([1.0, 4.0, 9.0])

    def test_scale(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_sc = sc.add_operator(scale(h, 3.0))
        s = sc.add_operator(record(h_sc))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([3.0, 6.0, 9.0])

    def test_shift(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_sh = sc.add_operator(shift(h, 10.0))
        s = sc.add_operator(record(h_sh))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([11.0, 12.0, 13.0])

    def test_clamp(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 3.0, 7.0])
        h_cl = sc.add_operator(clamp(h, 2.0, 5.0))
        s = sc.add_operator(record(h_cl))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([2.0, 3.0, 5.0])

    def test_nan_to_num(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, float("nan"), 3.0])
        h_nn = sc.add_operator(nan_to_num(h, 0.0))
        s = sc.add_operator(record(h_nn))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([1.0, 0.0, 3.0])


# ---------------------------------------------------------------------------
# Binary math operators
# ---------------------------------------------------------------------------


class TestBinaryMath:
    def test_min(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([1.0, 5.0, 3.0]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([2.0, 4.0, 6.0]),
        ))
        h = sc.add_operator(min(a, b))
        s = sc.add_operator(record(h))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([1.0, 4.0, 3.0])

    def test_max(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([1.0, 5.0, 3.0]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([2.0, 4.0, 6.0]),
        ))
        h = sc.add_operator(max(a, b))
        s = sc.add_operator(record(h))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([2.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# Chained operations (log-ratio pattern)
# ---------------------------------------------------------------------------


class TestChained:
    def test_log_ratio(self) -> None:
        """log(a / b) — the log-ratio pattern for factors like log(book/price)."""
        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([np.e**2, np.e**3]),
        ))
        b = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([np.e, np.e]),
        ))
        ratio = sc.add_operator(divide(a, b))
        log_ratio = sc.add_operator(log(ratio))
        s = sc.add_operator(record(log_ratio))
        _run(sc)
        vals = list(sc.series_values(s))
        assert vals == pytest.approx([1.0, 2.0], abs=1e-12)
