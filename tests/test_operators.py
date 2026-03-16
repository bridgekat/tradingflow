"""Tests for the operators module and all sub-modules."""

from __future__ import annotations
from typing import Any

import asyncio

import numpy as np
import pytest

from tradingflow import Scenario, Series
from tradingflow.sources import ArrayBundleSource
from tradingflow.operator import Operator
from tradingflow.operators import Apply, Concat, Filter, Stack, Where, add, divide, multiply, negate, select, subtract
from tradingflow.operators.indicators import (
    BollingerBand,
    ExponentialMovingAverage,
    Momentum,
    MovingAverage,
    MovingCovariance,
    MovingVariance,
    WeightedMovingAverage,
)
from tradingflow.operators.metrics import AverageReturn, SharpeRatio
from tradingflow.operators.portfolios import MeanVarianceOptimization, TopK, TopKRankLinear
from tradingflow.operators.predictors import RollingLinearRegression
from tradingflow.operators.simulators import TradingSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


def make_scalar_series(values: list[float]) -> Series[tuple[()], np.float64]:
    """Create a scalar float64 series from a list."""
    s = Series((), np.dtype(np.float64))
    for i, v in enumerate(values, start=1):
        s.append(ts(i), np.array(v, dtype=np.float64))
    return s


def make_vector_series(values: list[list[float]]) -> Series[tuple[int], np.float64]:
    """Create a vector float64 series from a list of lists."""
    n = len(values[0])
    s = Series((n,), np.dtype(np.float64))
    for i, v in enumerate(values, start=1):
        s.append(ts(i), np.array(v, dtype=np.float64))
    return s


def run_op(op: Operator[Any, Any, Any, Any], timestamps: list[np.datetime64]) -> Series[Any, Any]:
    """Drive an operator standalone without a Scenario.

    Returns the output series produced by running *op* at each timestamp.
    """
    state = op.init_state()
    output = Series(op.shape, op.dtype)
    for t in timestamps:
        slices = tuple(inp.to(t) for inp in op.inputs)
        value, state = op.compute(t, slices, state)
        if value is not None:
            output.append(t, np.asarray(value, dtype=output.dtype))
    return output


def update_all(operators: list[Operator[Any, Any, Any, Any]], timestamp: np.datetime64) -> None:
    """Update all operators at the given timestamp (not used after refactor; kept for reference)."""
    raise NotImplementedError("update_all is not available; use run_op instead")


# ===========================================================================
# Generic operators
# ===========================================================================


class TestApply:
    def test_basic(self) -> None:
        a = make_scalar_series([10.0, 20.0, 30.0])
        b = make_scalar_series([2.0, 4.0, 5.0])
        op = Apply((a, b), (), np.dtype(np.float64), lambda args: args[0] + args[1])
        output = run_op(op, [ts(i) for i in range(1, 4)])
        assert list(output.values) == pytest.approx([12.0, 24.0, 35.0])

    def test_skips_when_input_empty(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = make_scalar_series([1.0])
        op = Apply((a, b), (), np.dtype(np.float64), lambda args: args[0] + args[1])
        output = run_op(op, [ts(1)])
        assert len(output) == 0

    def test_vector_elements(self) -> None:
        a = make_vector_series([[1.0, 2.0], [3.0, 4.0]])
        b = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        op = Apply((a, b), (2,), np.dtype(np.float64), lambda args: args[0] + args[1])
        output = run_op(op, [ts(i) for i in range(1, 3)])
        np.testing.assert_array_almost_equal(output.values, [[11.0, 22.0], [33.0, 44.0]])


class TestFactoryFunctions:
    def test_add(self) -> None:
        a = make_scalar_series([1.0, 2.0])
        b = make_scalar_series([3.0, 4.0])
        op = add(a, b)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        assert list(output.values) == pytest.approx([4.0, 6.0])

    def test_subtract(self) -> None:
        a = make_scalar_series([10.0, 20.0])
        b = make_scalar_series([3.0, 5.0])
        op = subtract(a, b)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        assert list(output.values) == pytest.approx([7.0, 15.0])

    def test_multiply(self) -> None:
        a = make_scalar_series([2.0, 3.0])
        b = make_scalar_series([4.0, 5.0])
        op = multiply(a, b)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        assert list(output.values) == pytest.approx([8.0, 15.0])

    def test_divide(self) -> None:
        a = make_scalar_series([10.0, 20.0])
        b = make_scalar_series([2.0, 4.0])
        op = divide(a, b)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        assert list(output.values) == pytest.approx([5.0, 5.0])

    def test_negate(self) -> None:
        a = make_scalar_series([3.0, -5.0])
        op = negate(a)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        assert list(output.values) == pytest.approx([-3.0, 5.0])

    def test_vector_divide(self) -> None:
        a = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        b = make_vector_series([[2.0, 4.0], [5.0, 8.0]])
        op = divide(a, b)
        output = run_op(op, [ts(i) for i in range(1, 3)])
        np.testing.assert_array_almost_equal(output.values, [[5.0, 5.0], [6.0, 5.0]])


class TestSelect:
    def test_selects_requested_indices(self) -> None:
        series = make_vector_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        op = select(series, (2, 0))
        output = run_op(op, [ts(1), ts(2)])
        np.testing.assert_array_almost_equal(output.values, [[3.0, 1.0], [6.0, 4.0]])

    def test_rejects_out_of_bounds_index(self) -> None:
        series = make_vector_series([[1.0, 2.0]])
        with pytest.raises(ValueError, match="out of bounds"):
            select(series, (2,))

    def test_single_int_drops_axis(self) -> None:
        """Select with int index drops the axis (like a[:, 2])."""
        series = make_vector_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        op = select(series, 1)
        assert op.shape == ()
        output = run_op(op, [ts(1), ts(2)])
        assert float(output[0]) == pytest.approx(2.0)
        assert float(output[1]) == pytest.approx(5.0)

    def test_matrix_select_column(self) -> None:
        """Select a single column from a (N, K) matrix -> (N,)."""
        s = Series((2, 3), np.dtype(np.float64))
        s.append(ts(1), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        from tradingflow.operators import Select

        op = Select(s, 1)  # last axis, single int -> drop axis
        assert op.shape == (2,)
        value, _ = op.compute(ts(1), (s,), op.init_state())
        np.testing.assert_array_equal(value, [2.0, 5.0])

    def test_matrix_select_columns(self) -> None:
        """Select multiple columns from a (N, K) matrix -> (N, 2)."""
        s = Series((2, 3), np.dtype(np.float64))
        s.append(ts(1), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        from tradingflow.operators import Select

        op = Select(s, (2, 0))  # last axis, tuple -> keep axis
        assert op.shape == (2, 2)
        value, _ = op.compute(ts(1), (s,), op.init_state())
        np.testing.assert_array_equal(value, [[3.0, 1.0], [6.0, 4.0]])

    def test_matrix_select_row(self) -> None:
        """Select a row from a (N, K) matrix along axis=0 -> (K,)."""
        s = Series((2, 3), np.dtype(np.float64))
        s.append(ts(1), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        from tradingflow.operators import Select

        op = Select(s, 0, axis=0)  # axis 0, single int -> drop axis
        assert op.shape == (3,)
        value, _ = op.compute(ts(1), (s,), op.init_state())
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0])

    def test_scalar_input_raises(self) -> None:
        s = Series((), np.dtype(np.float64))
        from tradingflow.operators import Select

        with pytest.raises(ValueError, match="at least 1 dimension"):
            Select(s, 0)


class TestWhere:
    def test_positive_where(self) -> None:
        series = make_vector_series([[1.0, -2.0, 3.0], [-1.0, 5.0, 0.0]])
        op = Where(series, fn=lambda x: x > 0)
        assert op.shape == (3,)
        output = run_op(op, [ts(1), ts(2)])
        np.testing.assert_array_equal(output[0], [1.0, np.nan, 3.0])
        np.testing.assert_array_equal(output[1], [np.nan, 5.0, np.nan])

    def test_custom_fill(self) -> None:
        series = make_vector_series([[1.0, -2.0]])
        op = Where(series, fn=lambda x: x > 0, fill=0.0)
        output = run_op(op, [ts(1)])
        np.testing.assert_array_equal(output[0], [1.0, 0.0])

    def test_scalar(self) -> None:
        s = Series((), np.dtype(np.float64))
        s.append(ts(1), np.array(-5.0))
        s.append(ts(2), np.array(3.0))
        op = Where(s, fn=lambda x: x > 0)
        assert op.shape == ()
        output = run_op(op, [ts(1), ts(2)])
        assert np.isnan(float(output[0]))
        assert float(output[1]) == pytest.approx(3.0)

    def test_preserves_shape_and_dtype(self) -> None:
        series = make_vector_series([[1.0, 2.0]])
        op = Where(series, fn=lambda x: x > 0)
        assert op.shape == series.shape
        assert op.dtype == series.dtype

    def test_empty_series_returns_none(self) -> None:
        series = Series((3,), np.dtype(np.float64))
        op = Where(series, fn=lambda x: x > 0)
        value, state = op.compute(ts(1), (series,), op.init_state())
        assert value is None


class TestFilter:
    def test_filter_drops_failing(self) -> None:
        series = make_vector_series([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        op = Filter(series, fn=lambda x: float(np.sum(x)) > 0)
        assert op.shape == (3,)
        output = run_op(op, [ts(1), ts(2)])
        assert len(output) == 1
        np.testing.assert_array_equal(output[0], [1.0, 2.0, 3.0])

    def test_filter_scalar(self) -> None:
        s = Series((), np.dtype(np.float64))
        s.append(ts(1), np.array(-5.0))
        s.append(ts(2), np.array(3.0))
        op = Filter(s, fn=lambda x: float(x) > 0)
        assert op.shape == ()
        output = run_op(op, [ts(1), ts(2)])
        assert len(output) == 1
        assert float(output[0]) == pytest.approx(3.0)

    def test_preserves_shape_and_dtype(self) -> None:
        series = make_vector_series([[1.0, 2.0]])
        op = Filter(series, fn=lambda x: True)
        assert op.shape == series.shape
        assert op.dtype == series.dtype

    def test_empty_series_returns_none(self) -> None:
        series = Series((3,), np.dtype(np.float64))
        op = Filter(series, fn=lambda x: True)
        value, state = op.compute(ts(1), (series,), op.init_state())
        assert value is None


# ===========================================================================
# Filters
# ===========================================================================


class TestMovingAverage:
    def test_count_window(self) -> None:
        s = make_scalar_series([10.0, 20.0, 30.0, 40.0])
        ma = MovingAverage(3, s)
        output = run_op(ma, [ts(i) for i in range(1, 5)])
        # Window=3: [10]=10, [10,20]=15, [10,20,30]=20, [20,30,40]=30
        assert list(output.values) == pytest.approx([10.0, 15.0, 20.0, 30.0])

    def test_time_window(self) -> None:
        s = Series((), np.dtype(np.float64))
        # Timestamps 10, 20, 30, 40 ns
        for i in range(1, 5):
            s.append(np.datetime64(i * 10, "ns"), np.array(float(i * 10), dtype=np.float64))
        window = np.timedelta64(25, "ns")
        ma = MovingAverage(window, s)
        output = run_op(ma, [np.datetime64(i * 10, "ns") for i in range(1, 5)])
        # At t=10: window [10], mean=10
        # At t=20: window [10,20] (both within 25ns of 20), mean=15
        # At t=30: window [10,20,30] (10 is at 30-25=5, 10>=5), mean=20
        # At t=40: window [20,30,40] (20 is at 40-25=15, 20>=15), mean=30
        assert list(output.values) == pytest.approx([10.0, 15.0, 20.0, 30.0])

    def test_vector(self) -> None:
        s = make_vector_series([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ma = MovingAverage(2, s)
        output = run_op(ma, [ts(i) for i in range(1, 4)])
        # Window=2: [1,2]=mean[1,2], [[1,2],[3,4]]=mean, [[3,4],[5,6]]=mean
        np.testing.assert_array_almost_equal(
            output.values,
            [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
        )


class TestMovingVariance:
    def test_basic(self) -> None:
        s = make_scalar_series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        mv = MovingVariance(4, s)
        output = run_op(mv, [ts(i) for i in range(1, 9)])
        # First output is at i=2 (need > ddof=1 points)
        assert len(output) == 7  # i=2..8
        # At i=4: window=[2,4,4,4], var=1.0
        expected_var_at_4 = np.var([2.0, 4.0, 4.0, 4.0], ddof=1)
        assert float(output[2]) == pytest.approx(expected_var_at_4)

    def test_vector(self) -> None:
        s = make_vector_series([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        mv = MovingVariance(3, s)
        output = run_op(mv, [ts(i) for i in range(1, 4)])
        # At i=2: [1,3] var=2, [10,20] var=50
        # At i=3: [1,3,5] var=4, [10,20,30] var=100
        expected_2 = np.var([1.0, 3.0], ddof=1)
        expected_3 = np.var([1.0, 3.0, 5.0], ddof=1)
        assert float(output[0][0]) == pytest.approx(expected_2)
        assert float(output[1][0]) == pytest.approx(expected_3)


class TestMovingCovariance:
    def test_basic(self) -> None:
        s = make_vector_series([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        mc = MovingCovariance(3, s)
        output = run_op(mc, [ts(i) for i in range(1, 5)])
        # Window=3, first full window at i=3: [[1,2],[2,4],[3,6]]
        # i=2 produces partial-window output (2 entries), i=3 is the second output
        expected = np.cov(np.array([[1, 2], [2, 4], [3, 6]]).T, ddof=1)
        np.testing.assert_array_almost_equal(output[1], expected)


class TestExponentialMovingAverage:
    def test_basic(self) -> None:
        s = make_scalar_series([1.0, 2.0, 3.0])
        ema = ExponentialMovingAverage(0.5, s)
        output = run_op(ema, [ts(i) for i in range(1, 4)])
        # EMA_1 = 1.0
        # EMA_2 = 0.5*2 + 0.5*1 = 1.5
        # EMA_3 = 0.5*3 + 0.5*1.5 = 2.25
        assert list(output.values) == pytest.approx([1.0, 1.5, 2.25])

    def test_vector(self) -> None:
        s = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        ema = ExponentialMovingAverage(0.5, s)
        output = run_op(ema, [ts(i) for i in range(1, 3)])
        np.testing.assert_array_almost_equal(output.values, [[10.0, 20.0], [20.0, 30.0]])


class TestWeightedMovingAverage:
    def test_basic(self) -> None:
        s = make_scalar_series([1.0, 2.0, 3.0])
        wma = WeightedMovingAverage(3, s)
        output = run_op(wma, [ts(i) for i in range(1, 4)])
        # At i=3: values=[1,2,3], weights=[1,2,3]/6
        # WMA = (1*1 + 2*2 + 3*3) / 6 = 14/6
        expected = (1 * 1 + 2 * 2 + 3 * 3) / 6.0
        assert float(output[2]) == pytest.approx(expected)

    def test_rejects_timedelta_window(self) -> None:
        s = make_scalar_series([1.0])
        with pytest.raises(TypeError):
            WeightedMovingAverage(np.timedelta64(5, "ns"), s)  # type: ignore[arg-type]


class TestMomentum:
    def test_basic(self) -> None:
        s = make_scalar_series([10.0, 12.0, 15.0, 11.0])
        mom = Momentum(2, s)
        output = run_op(mom, [ts(i) for i in range(1, 5)])
        # Need > 2 entries: first output at i=3
        # i=3: 15 - 10 = 5
        # i=4: 11 - 12 = -1
        assert len(output) == 2
        assert list(output.values) == pytest.approx([5.0, -1.0])


class TestBollingerBand:
    def test_basic(self) -> None:
        s = make_scalar_series([10.0, 12.0, 11.0, 13.0])
        bb_upper = BollingerBand(3, s, num_std=2.0)
        bb_lower = BollingerBand(3, s, num_std=-2.0)
        bb_mean = BollingerBand(3, s, num_std=0.0)
        out_upper = run_op(bb_upper, [ts(i) for i in range(1, 5)])
        out_lower = run_op(bb_lower, [ts(i) for i in range(1, 5)])
        out_mean = run_op(bb_mean, [ts(i) for i in range(1, 5)])
        # Need >= 2 entries; first output at i=2
        assert len(out_upper) == 3  # i=2,3,4

        # At i=4: window=[12,11,13], mean=12, std=1.0
        vals = np.array([12.0, 11.0, 13.0])
        mean = vals.mean()
        std = vals.std(ddof=1)
        assert float(out_upper[-1]) == pytest.approx(mean + 2.0 * std)
        assert float(out_lower[-1]) == pytest.approx(mean - 2.0 * std)
        assert float(out_mean[-1]) == pytest.approx(mean)

    def test_output_shape(self) -> None:
        s = make_vector_series([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bb = BollingerBand(2, s)
        assert bb.shape == (2,)


# ===========================================================================
# Predictors
# ===========================================================================


class TestRollingLinearRegression:
    def test_simple_linear(self) -> None:
        """Test on y = 2*x + 1 with scalar features."""
        features = Series((1,), np.dtype(np.float64))
        targets = Series((), np.dtype(np.float64))

        reg = RollingLinearRegression(features, targets, train_window=10, retrain_every=1)

        state = reg.init_state()
        output = Series(reg.shape, reg.dtype)
        for i in range(1, 11):
            x = float(i)
            y = 2.0 * x + 1.0
            features.append(ts(i), np.array([x], dtype=np.float64))
            targets.append(ts(i), np.array(y, dtype=np.float64))
            slices = tuple(inp.to(ts(i)) for inp in reg.inputs)
            value, state = reg.compute(ts(i), slices, state)
            if value is not None:
                output.append(ts(i), np.asarray(value, dtype=output.dtype))

        # After enough data, predictions should be close to 2*x + 1.
        last_pred = float(output[-1])
        last_x = 10.0
        assert last_pred == pytest.approx(2.0 * last_x + 1.0, abs=0.1)

    def test_no_prediction_before_fit(self) -> None:
        features = Series((1,), np.dtype(np.float64))
        targets = Series((), np.dtype(np.float64))

        reg = RollingLinearRegression(features, targets, train_window=5, retrain_every=1)
        # Only 1 data point -> can't fit (need >= 2)
        features.append(ts(1), np.array([1.0], dtype=np.float64))
        targets.append(ts(1), np.array(3.0, dtype=np.float64))
        output = run_op(reg, [ts(1)])
        assert len(output) == 0


# ===========================================================================
# Portfolios
# ===========================================================================


class TestTopK:
    def test_fixed_k(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3, 0.2, 0.4]])
        op = TopK(preds, k=2)
        output = run_op(op, [ts(1)])
        w = output[-1]
        # Top 2: indices 1 (0.5) and 4 (0.4)
        assert float(w[1]) == pytest.approx(0.5)
        assert float(w[4]) == pytest.approx(0.5)
        assert float(w[0]) == pytest.approx(0.0)

    def test_fractional_k(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3, 0.2]])
        op = TopK(preds, k=0.5)  # 50% of 4 = 2
        output = run_op(op, [ts(1)])
        w = output[-1]
        nonzero = np.count_nonzero(w)
        assert nonzero == 2
        assert float(w.sum()) == pytest.approx(1.0)


class TestTopKRankLinear:
    def test_rank_weights(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3]])
        op = TopKRankLinear(preds, k=2)
        output = run_op(op, [ts(1)])
        w = output[-1]
        # Top 2: index 1 (rank 1, weight 2/3), index 2 (rank 2, weight 1/3)
        assert float(w[1]) == pytest.approx(2.0 / 3.0)
        assert float(w[2]) == pytest.approx(1.0 / 3.0)
        assert float(w[0]) == pytest.approx(0.0)
        assert float(w.sum()) == pytest.approx(1.0)


class TestMeanVarianceOptimization:
    def test_not_implemented(self) -> None:
        preds = make_vector_series([[0.1, 0.2]])
        covs = Series((2, 2), np.dtype(np.float64))
        with pytest.raises(NotImplementedError):
            MeanVarianceOptimization(preds, covs)


# ===========================================================================
# Simulators
# ===========================================================================


class TestTradingSimulator:
    def test_no_commission(self) -> None:
        prices = Series((2,), np.dtype(np.float64))
        positions = Series((2,), np.dtype(np.float64))
        sim = TradingSimulator(prices, positions, initial_cash=1000.0)

        state = sim.init_state()
        output = Series(sim.shape, sim.dtype)

        # Day 1: buy 10 of asset A at 50, 5 of asset B at 100
        prices.append(ts(1), np.array([50.0, 100.0], dtype=np.float64))
        positions.append(ts(1), np.array([10.0, 5.0], dtype=np.float64))
        slices = tuple(inp.to(ts(1)) for inp in sim.inputs)
        value, state = sim.compute(ts(1), slices, state)  # type: ignore
        if value is not None:
            output.append(ts(1), np.asarray(value, dtype=output.dtype))
        # Cost: 10*50 + 5*100 = 1000
        # Cash: 1000 - 1000 = 0
        # MV: 0 + 10*50 + 5*100 = 1000
        assert float(output[-1]) == pytest.approx(1000.0)

        # Day 2: prices go up, hold same positions
        prices.append(ts(2), np.array([55.0, 110.0], dtype=np.float64))
        positions.append(ts(2), np.array([10.0, 5.0], dtype=np.float64))
        slices = tuple(inp.to(ts(2)) for inp in sim.inputs)
        value, state = sim.compute(ts(2), slices, state)  # type: ignore
        if value is not None:
            output.append(ts(2), np.asarray(value, dtype=output.dtype))
        # No trades -> no cost
        # MV: 0 + 10*55 + 5*110 = 1100
        assert float(output[-1]) == pytest.approx(1100.0)

    def test_with_commission(self) -> None:
        prices = Series((2,), np.dtype(np.float64))
        positions = Series((2,), np.dtype(np.float64))
        sim = TradingSimulator(
            prices,
            positions,
            commission_rate=0.01,
            min_charge=1.0,
            initial_cash=10000.0,
        )

        state = sim.init_state()
        output = Series(sim.shape, sim.dtype)

        prices.append(ts(1), np.array([100.0, 200.0], dtype=np.float64))
        positions.append(ts(1), np.array([10.0, 5.0], dtype=np.float64))
        slices = tuple(inp.to(ts(1)) for inp in sim.inputs)
        value, state = sim.compute(ts(1), slices, state)  # type: ignore
        if value is not None:
            output.append(ts(1), np.asarray(value, dtype=output.dtype))
        # Trade cost: 10*100 + 5*200 = 2000
        # Commission per asset: max(0.01*1000, 1) = 10, max(0.01*1000, 1) = 10
        # Total commission: 20
        # Cash: 10000 - 2000 - 20 = 7980
        # MV: 7980 + 10*100 + 5*200 = 7980 + 2000 = 9980
        assert float(output[-1]) == pytest.approx(9980.0)

    def test_min_charge(self) -> None:
        prices = Series((1,), np.dtype(np.float64))
        positions = Series((1,), np.dtype(np.float64))
        sim = TradingSimulator(
            prices,
            positions,
            commission_rate=0.001,
            min_charge=5.0,
            initial_cash=1000.0,
        )

        state = sim.init_state()
        output = Series(sim.shape, sim.dtype)

        # Buy 1 share at 10 -> trade value = 10
        # Commission: max(0.001*10, 5) = max(0.01, 5) = 5
        prices.append(ts(1), np.array([10.0], dtype=np.float64))
        positions.append(ts(1), np.array([1.0], dtype=np.float64))
        slices = tuple(inp.to(ts(1)) for inp in sim.inputs)
        value, state = sim.compute(ts(1), slices, state)  # type: ignore
        if value is not None:
            output.append(ts(1), np.asarray(value, dtype=output.dtype))
        # Cash: 1000 - 10 - 5 = 985
        # MV: 985 + 10 = 995
        assert float(output[-1]) == pytest.approx(995.0)


# ===========================================================================
# Metrics
# ===========================================================================


class TestAverageReturn:
    def test_basic(self) -> None:
        mv = make_scalar_series([100.0, 110.0, 121.0])
        signal = make_scalar_series([1.0, 1.0, 1.0])  # Signal at every tick
        ar = AverageReturn(mv, signal)
        output = run_op(ar, [ts(i) for i in range(1, 4)])
        # Returns: 10/100=0.1, 11/110=0.1
        # Average: 0.1
        assert len(output) == 2  # First signal records prev; output from 2nd
        assert float(output[-1]) == pytest.approx(0.1)

    def test_no_signal_no_output(self) -> None:
        mv = make_scalar_series([100.0, 110.0])
        signal = Series((), np.dtype(np.float64))
        ar = AverageReturn(mv, signal)
        output = run_op(ar, [ts(1), ts(2)])
        assert len(output) == 0


class TestSharpeRatio:
    def test_basic(self) -> None:
        # Construct a series with known returns
        mv = Series((), np.dtype(np.float64))
        signal = Series((), np.dtype(np.float64))

        values = [100.0, 110.0, 121.0, 108.9]  # returns: 0.1, 0.1, -0.1
        for i, v in enumerate(values, start=1):
            mv.append(ts(i), np.array(v, dtype=np.float64))
            signal.append(ts(i), np.array(1.0, dtype=np.float64))

        sr = SharpeRatio(mv, signal, periods_per_year=252)
        output = run_op(sr, [ts(i) for i in range(1, 5)])

        # Need >= 2 returns -> first output at i=4 (3 signal ticks with returns)
        returns = np.array([0.1, 0.1, -0.1])
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        assert len(output) == 1
        assert float(output[-1]) == pytest.approx(expected, rel=1e-6)


# ===========================================================================
# Concat (existing axis)
# ===========================================================================


class TestConcat:
    def test_concat_vectors_axis0(self) -> None:
        """Concat two (3,) vectors along axis=0 into (6,)."""
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((3,), np.dtype(np.float64))
        s1.append(ts(1), np.array([1.0, 2.0, 3.0]))
        s2.append(ts(1), np.array([4.0, 5.0, 6.0]))

        op = Concat([s1, s2])
        assert op.shape == (6,)
        value, _ = op.compute(ts(1), (s1, s2), op.init_state())
        np.testing.assert_array_equal(value, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_concat_matrices_axis1(self) -> None:
        """Concat two (2, 3) matrices along axis=1 into (2, 6)."""
        s1 = Series((2, 3), np.dtype(np.float64))
        s2 = Series((2, 3), np.dtype(np.float64))
        s1.append(ts(1), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        s2.append(ts(1), np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))

        op = Concat([s1, s2], axis=1)
        assert op.shape == (2, 6)
        value, _ = op.compute(ts(1), (s1, s2), op.init_state())
        expected = np.array([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], dtype=np.float64)
        np.testing.assert_array_equal(value, expected)

    def test_concat_forward_fill(self) -> None:
        """Missing input at timestamp uses forward-fill; fully missing uses NaN."""
        s1 = Series((2,), np.dtype(np.float64))
        s2 = Series((2,), np.dtype(np.float64))
        s1.append(ts(1), np.array([1.0, 2.0]))
        s1.append(ts(2), np.array([3.0, 4.0]))
        s2.append(ts(1), np.array([5.0, 6.0]))

        op = Concat([s1, s2])
        value, _ = op.compute(ts(2), (s1, s2), op.init_state())
        np.testing.assert_array_equal(value, [3.0, 4.0, 5.0, 6.0])  # s2 forward-filled

    def test_concat_scalar_raises(self) -> None:
        """Scalar series rejected by Concat (use Stack instead)."""
        s1 = Series((), np.dtype(np.float64))
        s2 = Series((), np.dtype(np.float64))
        with pytest.raises(ValueError, match="out of bounds"):
            Concat([s1, s2])

    def test_concat_shape_mismatch_raises(self) -> None:
        s1 = Series((2, 3), np.dtype(np.float64))
        s2 = Series((3, 3), np.dtype(np.float64))
        with pytest.raises(ValueError, match="non-concatenation axes"):
            Concat([s1, s2], axis=1)

    def test_concat_ndim_mismatch_raises(self) -> None:
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((2, 3), np.dtype(np.float64))
        with pytest.raises(ValueError, match="same number of dimensions"):
            Concat([s1, s2])

    def test_concat_axis_out_of_bounds_raises(self) -> None:
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((3,), np.dtype(np.float64))
        with pytest.raises(ValueError, match="out of bounds"):
            Concat([s1, s2], axis=1)

    def test_concat_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Concat([])

    def test_concat_vectors_scenario(self) -> None:
        """Concat (3,) vector sources in a Scenario."""
        src_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[1.0, 2.0, 3.0]]),
            name="a",
        )
        src_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([[4.0, 5.0, 6.0]]),
            name="b",
        )

        scenario = Scenario()
        a = scenario.add_source(src_a)
        b = scenario.add_source(src_b)
        concat_s = scenario.add_operator(Concat([a, b]))
        asyncio.run(scenario.run())

        assert len(concat_s) == 1
        np.testing.assert_array_equal(concat_s[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


# ===========================================================================
# Stack (new axis)
# ===========================================================================


class TestStack:
    def test_stack_scalars(self) -> None:
        """Stack scalar series into (N,) vector."""
        s1 = Series((), np.dtype(np.float64))
        s2 = Series((), np.dtype(np.float64))
        s1.append(ts(1), np.array(10.0))
        s2.append(ts(1), np.array(20.0))

        op = Stack([s1, s2])
        assert op.shape == (2,)
        value, _ = op.compute(ts(1), (s1, s2), op.init_state())
        np.testing.assert_array_equal(value, [10.0, 20.0])

    def test_stack_vectors_axis0(self) -> None:
        """Stack (K,) vectors into (N, K) matrix along axis=0."""
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((3,), np.dtype(np.float64))
        s1.append(ts(1), np.array([1.0, 2.0, 3.0]))
        s2.append(ts(1), np.array([4.0, 5.0, 6.0]))

        op = Stack([s1, s2])
        assert op.shape == (2, 3)
        value, _ = op.compute(ts(1), (s1, s2), op.init_state())
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(value, expected)

    def test_stack_vectors_axis1(self) -> None:
        """Stack (K,) vectors into (K, N) matrix along axis=1."""
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((3,), np.dtype(np.float64))
        s1.append(ts(1), np.array([1.0, 2.0, 3.0]))
        s2.append(ts(1), np.array([4.0, 5.0, 6.0]))

        op = Stack([s1, s2], axis=1)
        assert op.shape == (3, 2)
        value, _ = op.compute(ts(1), (s1, s2), op.init_state())
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_equal(value, expected)

    def test_stack_forward_fill(self) -> None:
        """Stack uses as-of lookup for stale inputs."""
        s1 = Series((), np.dtype(np.float64))
        s2 = Series((), np.dtype(np.float64))
        s1.append(ts(1), np.array(10.0))
        s1.append(ts(2), np.array(11.0))
        s2.append(ts(1), np.array(20.0))

        op = Stack([s1, s2])
        value, _ = op.compute(ts(2), (s1, s2), op.init_state())
        np.testing.assert_array_equal(value, [11.0, 20.0])

    def test_stack_missing_nan(self) -> None:
        """Inputs with no data produce NaN."""
        s1 = Series((), np.dtype(np.float64))
        s2 = Series((), np.dtype(np.float64))
        s1.append(ts(5), np.array(10.0))

        op = Stack([s1, s2])
        value, _ = op.compute(ts(3), (s1, s2), op.init_state())
        assert isinstance(value, np.ndarray)
        assert np.isnan(value[1])

    def test_stack_shape_mismatch_raises(self) -> None:
        s1 = Series((3,), np.dtype(np.float64))
        s2 = Series((4,), np.dtype(np.float64))
        with pytest.raises(ValueError, match="same shape"):
            Stack([s1, s2])

    def test_stack_axis_out_of_bounds_raises(self) -> None:
        s1 = Series((), np.dtype(np.float64))
        s2 = Series((), np.dtype(np.float64))
        with pytest.raises(ValueError, match="out of bounds"):
            Stack([s1, s2], axis=1)

    def test_stack_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Stack([])

    def test_stack_heterogeneous_frequencies_scenario(self) -> None:
        """Stack assembles scalar sources with different schedules."""
        src_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([1.0, 2.0, 3.0]),
            name="a",
        )
        src_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(3)]),
            values=np.array([10.0, 30.0]),
            name="b",
        )

        scenario = Scenario()
        a = scenario.add_source(src_a)
        b = scenario.add_source(src_b)
        stack_s = scenario.add_operator(Stack([a, b]))
        asyncio.run(scenario.run())

        assert len(stack_s) == 3
        np.testing.assert_array_equal(stack_s[0], [1.0, 10.0])
        np.testing.assert_array_equal(stack_s[1], [2.0, 10.0])  # b forward-filled
        np.testing.assert_array_equal(stack_s[2], [3.0, 30.0])

    def test_stack_many_sources_scenario(self) -> None:
        """Stack works with many sources on different schedules."""
        scenario = Scenario()
        series_list = []
        for i in range(5):
            src = ArrayBundleSource.from_arrays(
                timestamps=np.array([ts(i + 1), ts(i + 3)]),
                values=np.array([float(i), float(i * 10)]),
                name=f"s{i}",
            )
            series_list.append(scenario.add_source(src))

        stack_s = scenario.add_operator(Stack(series_list))
        asyncio.run(scenario.run())

        assert len(stack_s) == 7

    def test_stack_then_downstream_operator(self) -> None:
        """Stack output feeds downstream operators."""
        src_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([3.0, 4.0]),
            name="a",
        )
        src_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([7.0, 6.0]),
            name="b",
        )

        scenario = Scenario()
        a = scenario.add_source(src_a)
        b = scenario.add_source(src_b)
        stack_s = scenario.add_operator(Stack([a, b]))
        sum_s = scenario.add_operator(Apply((stack_s,), (), np.float64, lambda args: np.sum(args[0])))
        asyncio.run(scenario.run())

        assert len(sum_s) == 2
        assert float(sum_s[0]) == pytest.approx(10.0)
        assert float(sum_s[1]) == pytest.approx(10.0)
