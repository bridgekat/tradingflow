"""Tests for newly exposed native operators (last, lag, rolling, ema, ffill)."""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow import Scenario
from tradingflow.sources import ArraySource
from tradingflow.operators import Record, Select, Concat, Stack, Last, Lag
from tradingflow.operators.num import (
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Log,
    Exp,
    Sqrt,
    Abs,
    Recip,
    Pow,
    Scale,
    Shift,
    Clamp,
    Fillna,
    Min,
    Max,
)
from tradingflow.operators.num import ForwardFill
from tradingflow.operators.rolling import (
    RollingSum,
    RollingMean,
    RollingVariance,
    RollingCovariance,
    EMA,
)
from tradingflow.types import Handle, NodeKind


def ts(i: int) -> np.datetime64:
    return np.datetime64(i, "ns")


def _run(sc: Scenario) -> None:
    sc.run()


def _scalar_scenario(values: list[float]) -> tuple[Scenario, Handle, Handle]:
    """Build a scenario with one scalar source and its recorded series."""
    sc = Scenario()
    src = ArraySource(
        timestamps=np.arange(1, len(values) + 1),
        values=np.array(values, dtype=np.float64),
    )
    h = sc.add_source(src)
    s = sc.add_operator(Record(h))
    return sc, h, s


# ---------------------------------------------------------------------------
# Last
# ---------------------------------------------------------------------------


class TestLast:
    def test_last_recovers_latest_value(self) -> None:
        """last(record(x)) == x at each step."""
        sc, h, s = _scalar_scenario([10.0, 20.0, 30.0])
        h_last = sc.add_operator(Last(s))
        s_last = sc.add_operator(Record(h_last))
        _run(sc)
        assert list(sc.series_view(s_last).values()) == pytest.approx([10.0, 20.0, 30.0])

    def test_last_vector(self) -> None:
        """last works on vector-valued series."""
        sc = Scenario()
        src = ArraySource(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(Record(h))
        h_last = sc.add_operator(Last(s))
        s_last = sc.add_operator(Record(h_last))
        _run(sc)
        vals = sc.series_view(s_last).values()
        np.testing.assert_array_almost_equal(vals[-1], [3.0, 4.0])

    def test_last_scalar(self) -> None:
        """last() on a single-element series returns that element."""
        sc, _, s = _scalar_scenario([5.0])
        h_last = sc.add_operator(Last(s))
        s_last = sc.add_operator(Record(h_last))
        _run(sc)
        assert list(sc.series_view(s_last).values()) == pytest.approx([5.0])


# ---------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------


class TestLag:
    def test_lag_basic(self) -> None:
        """lag(offset=2) outputs value from 2 steps ago, fill=0 for early steps."""
        sc, _, s = _scalar_scenario([10.0, 20.0, 30.0, 40.0])
        h_lag = sc.add_operator(Lag(s, offset=2))
        h_lag_rec = sc.add_operator(Record(h_lag))
        _run(sc)
        vals = list(sc.series_view(h_lag_rec).values())
        # steps 1,2: not enough history → fill=0
        # step 3: value from step 1 → 10.0
        # step 4: value from step 2 → 20.0
        assert vals == pytest.approx([0.0, 0.0, 10.0, 20.0])

    def test_lag_offset_1(self) -> None:
        """Default offset=1 returns previous value."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_lag = sc.add_operator(Lag(s))
        h_lag_rec = sc.add_operator(Record(h_lag))
        _run(sc)
        vals = list(sc.series_view(h_lag_rec).values())
        assert vals == pytest.approx([0.0, 1.0, 2.0])

    def test_lag_timestamps_match(self) -> None:
        """Lag output ticks every step; recorded timestamps match input series."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_lag = sc.add_operator(Lag(s))
        h_lag_rec = sc.add_operator(Record(h_lag))
        _run(sc)
        np.testing.assert_array_equal(
            sc.series_view(h_lag_rec).timestamps(),
            sc.series_view(s).timestamps(),
        )


# ---------------------------------------------------------------------------
# Rolling sum
# ---------------------------------------------------------------------------


class TestRollingSum:
    def test_rolling_sum_basic(self) -> None:
        """Rolling sum with window=3. Only ticks >= window produce output."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0, 4.0])
        h_rs = sc.add_operator(RollingSum(s, window=3))
        h_rs_recorded = sc.add_operator(Record(h_rs))
        _run(sc)
        vals = list(sc.series_view(h_rs_recorded).values())
        # First 2 ticks are warmup (no output), ticks 3 and 4 produce output.
        assert vals == pytest.approx([6.0, 9.0])

    def test_rolling_sum_nan_propagation(self) -> None:
        """NaN in window propagates to output; eviction clears it."""
        sc, _, s = _scalar_scenario([1.0, float("nan"), 3.0, 4.0, 5.0])
        h_rs = sc.add_operator(RollingSum(s, window=3))
        h_rs_recorded = sc.add_operator(Record(h_rs))
        _run(sc)
        vals = list(sc.series_view(h_rs_recorded).values())
        # Ticks 1,2 are warmup → not recorded.
        # Tick 3: [1, NaN, 3] → NaN
        # Tick 4: [NaN, 3, 4] → NaN
        # Tick 5: [3, 4, 5] → 12
        assert len(vals) == 3
        assert np.isnan(vals[0])  # [1, NaN, 3]
        assert np.isnan(vals[1])  # [NaN, 3, 4]
        assert vals[2] == pytest.approx(12.0)  # [3, 4, 5]

    def test_rolling_sum_vector(self) -> None:
        """Rolling sum with 2-element vectors and window=2."""
        sc = Scenario()
        src = ArraySource(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(Record(h))
        h_rs = sc.add_operator(RollingSum(s, window=2))
        h_rs_recorded = sc.add_operator(Record(h_rs))
        _run(sc)
        vals = sc.series_view(h_rs_recorded).values()
        # Tick 1 is warmup → not recorded. Ticks 2,3 produce output.
        assert len(vals) == 2
        np.testing.assert_array_almost_equal(vals[-1], [5.0, 50.0])  # 2+3, 20+30


# ---------------------------------------------------------------------------
# Rolling mean
# ---------------------------------------------------------------------------


class TestRollingMean:
    def test_rolling_mean_basic(self) -> None:
        """Rolling mean with window=3. Only ticks >= window produce output."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0, 6.0])
        h_rm = sc.add_operator(RollingMean(s, window=3))
        h_rm_recorded = sc.add_operator(Record(h_rm))
        _run(sc)
        vals = list(sc.series_view(h_rm_recorded).values())
        # First 2 ticks are warmup → not recorded.
        assert len(vals) == 2
        assert vals[0] == pytest.approx(2.0)  # mean([1,2,3])
        assert vals[1] == pytest.approx(11.0 / 3.0)  # mean([2,3,6])

    def test_rolling_mean_constant(self) -> None:
        """Mean of constant series equals the constant."""
        sc, _, s = _scalar_scenario([7.0] * 10)
        h_rm = sc.add_operator(RollingMean(s, window=5))
        h_rm_recorded = sc.add_operator(Record(h_rm))
        _run(sc)
        vals = list(sc.series_view(h_rm_recorded).values())
        # 4 warmup ticks, 6 output ticks.
        assert len(vals) == 6
        assert vals[-1] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Rolling variance
# ---------------------------------------------------------------------------


class TestRollingVariance:
    def test_rolling_variance_constant(self) -> None:
        """Variance of constant series is zero."""
        sc, _, s = _scalar_scenario([5.0] * 5)
        h_rv = sc.add_operator(RollingVariance(s, window=3))
        h_rv_recorded = sc.add_operator(Record(h_rv))
        _run(sc)
        vals = list(sc.series_view(h_rv_recorded).values())
        # 2 warmup ticks, 3 output ticks.
        assert len(vals) == 3
        assert vals[-1] == pytest.approx(0.0, abs=1e-10)

    def test_rolling_variance_known(self) -> None:
        """Variance of [1, 3] is Var = E[x^2] - E[x]^2 = 5 - 4 = 1."""
        sc, _, s = _scalar_scenario([1.0, 3.0])
        h_rv = sc.add_operator(RollingVariance(s, window=2))
        h_rv_recorded = sc.add_operator(Record(h_rv))
        _run(sc)
        vals = list(sc.series_view(h_rv_recorded).values())
        # 1 warmup tick, 1 output tick.
        assert len(vals) == 1
        assert vals[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rolling covariance
# ---------------------------------------------------------------------------


class TestRollingCovariance:
    def test_rolling_covariance_shape(self) -> None:
        """Output shape is (K, K) for K-element input."""
        sc = Scenario()
        src = ArraySource(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        h = sc.add_source(src)
        s = sc.add_operator(Record(h))
        h_rc = sc.add_operator(RollingCovariance(s, window=3))
        h_rc_recorded = sc.add_operator(Record(h_rc))
        _run(sc)
        vals = sc.series_view(h_rc_recorded).values()
        # 2 warmup ticks, 1 output tick.
        assert vals.shape == (1, 2, 2)

    def test_rolling_covariance_rejects_non_1d(self) -> None:
        """RollingCovariance raises for non-1D input."""
        h = Handle(0, NodeKind.ARRAY, np.dtype("float64"), (2, 3))
        with pytest.raises(ValueError, match="1-D"):
            RollingCovariance(h, window=3)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEma:
    def test_ema_constant(self) -> None:
        """EMA of constant series converges to the constant."""
        sc, _, s = _scalar_scenario([10.0] * 10)
        h_ema = sc.add_operator(EMA(s, window=3, alpha=0.5))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        vals = list(sc.series_view(h_ema_recorded).values())
        # 2 warmup ticks, 8 output ticks.
        assert len(vals) == 8
        assert vals[-1] == pytest.approx(10.0, abs=1e-6)

    def test_ema_first_value_window_1(self) -> None:
        """With window=1, first EMA output equals the first input."""
        sc, _, s = _scalar_scenario([100.0, 200.0])
        h_ema = sc.add_operator(EMA(s, window=1, alpha=0.5))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        vals = list(sc.series_view(h_ema_recorded).values())
        assert len(vals) == 2
        assert vals[0] == pytest.approx(100.0)

    def test_ema_two_values(self) -> None:
        """EMA of two values matches hand-computed result."""
        sc, _, s = _scalar_scenario([10.0, 20.0])
        h_ema = sc.add_operator(EMA(s, window=2, alpha=0.5))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        vals = list(sc.series_view(h_ema_recorded).values())
        # 1 warmup tick, 1 output tick.
        assert len(vals) == 1
        # w0=0.5 (for 20), w1=0.25 (for 10)
        expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25)
        assert vals[0] == pytest.approx(expected)

    def test_ema_with_span(self) -> None:
        """EMA(span=3) is equivalent to alpha=0.5."""
        sc, _, s = _scalar_scenario([10.0, 20.0])
        h_ema = sc.add_operator(EMA(s, window=2, span=3))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        vals = list(sc.series_view(h_ema_recorded).values())
        # 1 warmup tick, 1 output tick.
        assert len(vals) == 1
        expected = (0.5 * 20.0 + 0.25 * 10.0) / (0.5 + 0.25)
        assert vals[0] == pytest.approx(expected)

    def test_ema_with_half_life(self) -> None:
        """EMA(half_life=...) runs without error and produces values."""
        sc, _, s = _scalar_scenario([1.0, 2.0, 3.0])
        h_ema = sc.add_operator(EMA(s, window=2, half_life=2.0))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        # 1 warmup tick, 2 output ticks.
        assert len(sc.series_view(h_ema_recorded)) == 2

    def test_ema_window_eviction(self) -> None:
        """Old values are evicted when they leave the window."""
        sc, _, s = _scalar_scenario([100.0, 100.0, 0.0, 0.0])
        h_ema = sc.add_operator(EMA(s, window=2, alpha=0.5))
        h_ema_recorded = sc.add_operator(Record(h_ema))
        _run(sc)
        vals = list(sc.series_view(h_ema_recorded).values())
        # 1 warmup tick, 3 output ticks.
        assert len(vals) == 3
        # After two 0.0s with window=2, 100.0s are fully evicted
        assert vals[-1] == pytest.approx(0.0, abs=1e-10)

    def test_ema_requires_exactly_one_param(self) -> None:
        """Providing zero or multiple smoothing params raises."""
        h = Handle(0, NodeKind.ARRAY, np.dtype("float64"), ())
        with pytest.raises(ValueError, match="exactly one"):
            EMA(h, window=10)
        with pytest.raises(ValueError, match="exactly one"):
            EMA(h, window=10, alpha=0.5, span=3)


# ---------------------------------------------------------------------------
# Forward fill
# ---------------------------------------------------------------------------


class TestForwardFill:
    def test_forward_fill_replaces_nan(self) -> None:
        """NaN values are replaced with last valid observation."""
        sc, h, _ = _scalar_scenario([1.0, float("nan"), float("nan"), 4.0, float("nan")])
        h_ff = sc.add_operator(ForwardFill(h))
        h_ff_recorded = sc.add_operator(Record(h_ff))
        _run(sc)
        vals = list(sc.series_view(h_ff_recorded).values())
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(1.0)
        assert vals[2] == pytest.approx(1.0)
        assert vals[3] == pytest.approx(4.0)
        assert vals[4] == pytest.approx(4.0)

    def test_forward_fill_leading_nan(self) -> None:
        """Leading NaN stays NaN until a valid value appears."""
        sc, h, _ = _scalar_scenario([float("nan"), float("nan"), 3.0])
        h_ff = sc.add_operator(ForwardFill(h))
        h_ff_recorded = sc.add_operator(Record(h_ff))
        _run(sc)
        vals = list(sc.series_view(h_ff_recorded).values())
        assert np.isnan(vals[0])
        assert np.isnan(vals[1])
        assert vals[2] == pytest.approx(3.0)

    def test_forward_fill_no_nan(self) -> None:
        """Clean data passes through unchanged."""
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_ff = sc.add_operator(ForwardFill(h))
        h_ff_recorded = sc.add_operator(Record(h_ff))
        _run(sc)
        vals = list(sc.series_view(h_ff_recorded).values())
        assert vals == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Subtract
# ---------------------------------------------------------------------------


class TestSubtract:
    def test_subtract_scalar(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([10.0, 30.0]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([3.0, 7.0]),
            )
        )
        h = sc.add_operator(Subtract(a, b))
        s = sc.add_operator(Record(h))
        _run(sc)
        assert list(sc.series_view(s).values()) == pytest.approx([7.0, 23.0])


# ---------------------------------------------------------------------------
# Divide
# ---------------------------------------------------------------------------


class TestDivide:
    def test_divide_scalar(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([20.0, 9.0]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([4.0, 3.0]),
            )
        )
        h = sc.add_operator(Divide(a, b))
        s = sc.add_operator(Record(h))
        _run(sc)
        assert list(sc.series_view(s).values()) == pytest.approx([5.0, 3.0])


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_flat(self) -> None:
        sc = Scenario()
        src = ArraySource(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        )
        h = sc.add_source(src)
        sel = sc.add_operator(Select(h, [0, 2]))
        s = sc.add_operator(Record(sel))
        _run(sc)
        vals = sc.series_view(s).values()
        np.testing.assert_array_almost_equal(vals[0], [10.0, 30.0])
        np.testing.assert_array_almost_equal(vals[1], [40.0, 60.0])


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


class TestConcat:
    def test_concat_axis0(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1)]),
                values=np.array([[1.0, 2.0]]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1)]),
                values=np.array([[3.0, 4.0]]),
            )
        )
        h = sc.add_operator(Concat([a, b], axis=0))
        s = sc.add_operator(Record(h))
        _run(sc)
        vals = sc.series_view(s).values()
        np.testing.assert_array_almost_equal(vals[0], [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


class TestStack:
    def test_stack_axis0(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1)]),
                values=np.array([[1.0, 2.0]]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1)]),
                values=np.array([[3.0, 4.0]]),
            )
        )
        h = sc.add_operator(Stack([a, b], axis=0))
        s = sc.add_operator(Record(h))
        _run(sc)
        vals = sc.series_view(s).values()
        # stack axis=0: shape (2,2) → [[1,2],[3,4]]
        np.testing.assert_array_almost_equal(vals[0], [[1.0, 2.0], [3.0, 4.0]])


# ---------------------------------------------------------------------------
# Math unary operators
# ---------------------------------------------------------------------------


class TestMathUnary:
    def test_log(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, np.e, np.e**2])
        h_log = sc.add_operator(Log(h))
        s = sc.add_operator(Record(h_log))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([0.0, 1.0, 2.0], abs=1e-12)

    def test_exp(self) -> None:
        sc, h, _ = _scalar_scenario([0.0, 1.0])
        h_exp = sc.add_operator(Exp(h))
        s = sc.add_operator(Record(h_exp))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(np.e)

    def test_sqrt(self) -> None:
        sc, h, _ = _scalar_scenario([4.0, 9.0, 16.0])
        h_sqrt = sc.add_operator(Sqrt(h))
        s = sc.add_operator(Record(h_sqrt))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([2.0, 3.0, 4.0])

    def test_abs(self) -> None:
        sc, h, _ = _scalar_scenario([-3.0, 0.0, 5.0])
        h_abs = sc.add_operator(Abs(h))
        s = sc.add_operator(Record(h_abs))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([3.0, 0.0, 5.0])

    def test_recip(self) -> None:
        sc, h, _ = _scalar_scenario([2.0, 4.0, 0.5])
        h_recip = sc.add_operator(Recip(h))
        s = sc.add_operator(Record(h_recip))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([0.5, 0.25, 2.0])


# ---------------------------------------------------------------------------
# Parameterized operators
# ---------------------------------------------------------------------------


class TestParameterized:
    def test_pow(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_pow = sc.add_operator(Pow(h, 2.0))
        s = sc.add_operator(Record(h_pow))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([1.0, 4.0, 9.0])

    def test_scale(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_sc = sc.add_operator(Scale(h, 3.0))
        s = sc.add_operator(Record(h_sc))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([3.0, 6.0, 9.0])

    def test_shift(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 2.0, 3.0])
        h_sh = sc.add_operator(Shift(h, 10.0))
        s = sc.add_operator(Record(h_sh))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([11.0, 12.0, 13.0])

    def test_clamp(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, 3.0, 7.0])
        h_cl = sc.add_operator(Clamp(h, 2.0, 5.0))
        s = sc.add_operator(Record(h_cl))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([2.0, 3.0, 5.0])

    def test_nan_to_num(self) -> None:
        sc, h, _ = _scalar_scenario([1.0, float("nan"), 3.0])
        h_nn = sc.add_operator(Fillna(h, 0.0))
        s = sc.add_operator(Record(h_nn))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([1.0, 0.0, 3.0])


# ---------------------------------------------------------------------------
# Binary math operators
# ---------------------------------------------------------------------------


class TestBinaryMath:
    def test_min(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2), ts(3)]),
                values=np.array([1.0, 5.0, 3.0]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2), ts(3)]),
                values=np.array([2.0, 4.0, 6.0]),
            )
        )
        h = sc.add_operator(Min(a, b))
        s = sc.add_operator(Record(h))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([1.0, 4.0, 3.0])

    def test_max(self) -> None:
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2), ts(3)]),
                values=np.array([1.0, 5.0, 3.0]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2), ts(3)]),
                values=np.array([2.0, 4.0, 6.0]),
            )
        )
        h = sc.add_operator(Max(a, b))
        s = sc.add_operator(Record(h))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([2.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# Chained operations (log-ratio pattern)
# ---------------------------------------------------------------------------


class TestChained:
    def test_log_ratio(self) -> None:
        """log(a / b) — the log-ratio pattern for factors like log(book/price)."""
        sc = Scenario()
        a = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([np.e**2, np.e**3]),
            )
        )
        b = sc.add_source(
            ArraySource(
                timestamps=np.array([ts(1), ts(2)]),
                values=np.array([np.e, np.e]),
            )
        )
        ratio = sc.add_operator(Divide(a, b))
        log_ratio = sc.add_operator(Log(ratio))
        s = sc.add_operator(Record(log_ratio))
        _run(sc)
        vals = list(sc.series_view(s).values())
        assert vals == pytest.approx([1.0, 2.0], abs=1e-12)
