"""Tests for the ops module and all sub-modules."""

from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from src import Series
from src.operator import Operator
from src.ops import Apply, add, subtract, multiply, divide, multiple, negate
from src.ops.filters import (
    BollingerBand,
    ExponentialMovingAverage,
    Momentum,
    MovingAverage,
    MovingCovariance,
    MovingVariance,
    WeightedMovingAverage,
)
from src.ops.metrics import AverageReturn, SharpeRatio
from src.ops.portfolios import MeanVarianceOptimization, TopK, TopKRankLinear
from src.ops.predictors import RollingLinearRegression
from src.ops.simulators import TradingSimulator


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


def update_all(operators: list[Operator[Any, Any, Any, Any]], timestamp: np.datetime64) -> None:
    """Update all operators at the given timestamp."""
    for op in operators:
        op.update(timestamp)


# ===========================================================================
# Generic operators
# ===========================================================================


class TestApply:
    def test_basic(self) -> None:
        a = make_scalar_series([10.0, 20.0, 30.0])
        b = make_scalar_series([2.0, 4.0, 5.0])
        op = Apply((a, b), (), np.dtype(np.float64), lambda args: args[0] + args[1])
        for i in range(1, 4):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([12.0, 24.0, 35.0])

    def test_skips_when_input_empty(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = make_scalar_series([1.0])
        op = Apply((a, b), (), np.dtype(np.float64), lambda args: args[0] + args[1])
        op.update(ts(1))
        assert len(op.output) == 0

    def test_vector_elements(self) -> None:
        a = make_vector_series([[1.0, 2.0], [3.0, 4.0]])
        b = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        op = Apply((a, b), (2,), np.dtype(np.float64), lambda args: args[0] + args[1])
        for i in range(1, 3):
            op.update(ts(i))
        np.testing.assert_array_almost_equal(op.output.values, [[11.0, 22.0], [33.0, 44.0]])


class TestFactoryFunctions:
    def test_add(self) -> None:
        a = make_scalar_series([1.0, 2.0])
        b = make_scalar_series([3.0, 4.0])
        op = add(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([4.0, 6.0])

    def test_subtract(self) -> None:
        a = make_scalar_series([10.0, 20.0])
        b = make_scalar_series([3.0, 5.0])
        op = subtract(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([7.0, 15.0])

    def test_multiply(self) -> None:
        a = make_scalar_series([2.0, 3.0])
        b = make_scalar_series([4.0, 5.0])
        op = multiply(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([8.0, 15.0])

    def test_divide(self) -> None:
        a = make_scalar_series([10.0, 20.0])
        b = make_scalar_series([2.0, 4.0])
        op = divide(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([5.0, 5.0])

    def test_multiple_is_divide(self) -> None:
        a = make_scalar_series([10.0, 20.0])
        b = make_scalar_series([2.0, 5.0])
        op = multiple(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([5.0, 4.0])

    def test_negate(self) -> None:
        a = make_scalar_series([3.0, -5.0])
        op = negate(a)
        for i in range(1, 3):
            op.update(ts(i))
        assert list(op.output.values) == pytest.approx([-3.0, 5.0])

    def test_vector_divide(self) -> None:
        a = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        b = make_vector_series([[2.0, 4.0], [5.0, 8.0]])
        op = divide(a, b)
        for i in range(1, 3):
            op.update(ts(i))
        np.testing.assert_array_almost_equal(op.output.values, [[5.0, 5.0], [6.0, 5.0]])


# ===========================================================================
# Filters
# ===========================================================================


class TestMovingAverage:
    def test_count_window(self) -> None:
        s = make_scalar_series([10.0, 20.0, 30.0, 40.0])
        ma = MovingAverage(3, s)
        for i in range(1, 5):
            ma.update(ts(i))
        # Window=3: [10]=10, [10,20]=15, [10,20,30]=20, [20,30,40]=30
        assert list(ma.output.values) == pytest.approx([10.0, 15.0, 20.0, 30.0])

    def test_time_window(self) -> None:
        s = Series((), np.dtype(np.float64))
        # Timestamps 10, 20, 30, 40 ns
        for i in range(1, 5):
            s.append(np.datetime64(i * 10, "ns"), np.array(float(i * 10), dtype=np.float64))
        window = np.timedelta64(25, "ns")
        ma = MovingAverage(window, s)
        for i in range(1, 5):
            ma.update(np.datetime64(i * 10, "ns"))
        # At t=10: window [10], mean=10
        # At t=20: window [10,20] (both within 25ns of 20), mean=15
        # At t=30: window [10,20,30] (10 is at 30-25=5, 10>=5), mean=20
        # At t=40: window [20,30,40] (20 is at 40-25=15, 20>=15), mean=30
        assert list(ma.output.values) == pytest.approx([10.0, 15.0, 20.0, 30.0])

    def test_vector(self) -> None:
        s = make_vector_series([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ma = MovingAverage(2, s)
        for i in range(1, 4):
            ma.update(ts(i))
        # Window=2: [1,2]=mean[1,2], [[1,2],[3,4]]=mean, [[3,4],[5,6]]=mean
        np.testing.assert_array_almost_equal(
            ma.output.values,
            [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
        )


class TestMovingVariance:
    def test_basic(self) -> None:
        s = make_scalar_series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        mv = MovingVariance(4, s)
        for i in range(1, 9):
            mv.update(ts(i))
        # First output is at i=2 (need > ddof=1 points)
        assert len(mv.output) == 7  # i=2..8
        # At i=4: window=[2,4,4,4], var=1.0
        expected_var_at_4 = np.var([2.0, 4.0, 4.0, 4.0], ddof=1)
        assert float(mv.output.values[2]) == pytest.approx(expected_var_at_4)

    def test_vector(self) -> None:
        s = make_vector_series([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        mv = MovingVariance(3, s)
        for i in range(1, 4):
            mv.update(ts(i))
        # At i=2: [1,3] var=2, [10,20] var=50
        # At i=3: [1,3,5] var=4, [10,20,30] var=100
        expected_2 = np.var([1.0, 3.0], ddof=1)
        expected_3 = np.var([1.0, 3.0, 5.0], ddof=1)
        assert float(mv.output.values[0][0]) == pytest.approx(expected_2)
        assert float(mv.output.values[1][0]) == pytest.approx(expected_3)


class TestMovingCovariance:
    def test_basic(self) -> None:
        s = make_vector_series([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
        mc = MovingCovariance(3, s)
        for i in range(1, 5):
            mc.update(ts(i))
        # Window=3, first full window at i=3: [[1,2],[2,4],[3,6]]
        # i=2 produces partial-window output (2 entries), i=3 is the second output
        expected = np.cov(np.array([[1, 2], [2, 4], [3, 6]]).T, ddof=1)
        np.testing.assert_array_almost_equal(mc.output.values[1], expected)

    def test_rejects_scalar(self) -> None:
        s = make_scalar_series([1.0, 2.0])
        with pytest.raises(ValueError, match="vector-valued"):
            MovingCovariance(2, s)


class TestExponentialMovingAverage:
    def test_basic(self) -> None:
        s = make_scalar_series([1.0, 2.0, 3.0])
        ema = ExponentialMovingAverage(0.5, s)
        for i in range(1, 4):
            ema.update(ts(i))
        # EMA_1 = 1.0
        # EMA_2 = 0.5*2 + 0.5*1 = 1.5
        # EMA_3 = 0.5*3 + 0.5*1.5 = 2.25
        assert list(ema.output.values) == pytest.approx([1.0, 1.5, 2.25])

    def test_vector(self) -> None:
        s = make_vector_series([[10.0, 20.0], [30.0, 40.0]])
        ema = ExponentialMovingAverage(0.5, s)
        for i in range(1, 3):
            ema.update(ts(i))
        np.testing.assert_array_almost_equal(ema.output.values, [[10.0, 20.0], [20.0, 30.0]])


class TestWeightedMovingAverage:
    def test_basic(self) -> None:
        s = make_scalar_series([1.0, 2.0, 3.0])
        wma = WeightedMovingAverage(3, s)
        for i in range(1, 4):
            wma.update(ts(i))
        # At i=3: values=[1,2,3], weights=[1,2,3]/6
        # WMA = (1*1 + 2*2 + 3*3) / 6 = 14/6
        expected = (1 * 1 + 2 * 2 + 3 * 3) / 6.0
        assert float(wma.output.values[2]) == pytest.approx(expected)

    def test_rejects_timedelta_window(self) -> None:
        s = make_scalar_series([1.0])
        with pytest.raises(TypeError):
            WeightedMovingAverage(np.timedelta64(5, "ns"), s)  # type: ignore[arg-type]


class TestMomentum:
    def test_basic(self) -> None:
        s = make_scalar_series([10.0, 12.0, 15.0, 11.0])
        mom = Momentum(2, s)
        for i in range(1, 5):
            mom.update(ts(i))
        # Need > 2 entries: first output at i=3
        # i=3: 15 - 10 = 5
        # i=4: 11 - 12 = -1
        assert len(mom.output) == 2
        assert list(mom.output.values) == pytest.approx([5.0, -1.0])


class TestBollingerBand:
    def test_basic(self) -> None:
        s = make_scalar_series([10.0, 12.0, 11.0, 13.0])
        bb_upper = BollingerBand(3, s, num_std=2.0)
        bb_lower = BollingerBand(3, s, num_std=-2.0)
        bb_mean = BollingerBand(3, s, num_std=0.0)
        for i in range(1, 5):
            bb_upper.update(ts(i))
            bb_lower.update(ts(i))
            bb_mean.update(ts(i))
        # Need >= 2 entries; first output at i=2
        assert len(bb_upper.output) == 3  # i=2,3,4

        # At i=4: window=[12,11,13], mean=12, std=1.0
        vals = np.array([12.0, 11.0, 13.0])
        mean = vals.mean()
        std = vals.std(ddof=1)
        assert float(bb_upper.output.values[-1]) == pytest.approx(mean + 2.0 * std)
        assert float(bb_lower.output.values[-1]) == pytest.approx(mean - 2.0 * std)
        assert float(bb_mean.output.values[-1]) == pytest.approx(mean)

    def test_output_shape(self) -> None:
        s = make_vector_series([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bb = BollingerBand(2, s)
        assert bb.output.shape == (2,)


# ===========================================================================
# Predictors
# ===========================================================================


class TestRollingLinearRegression:
    def test_simple_linear(self) -> None:
        """Test on y = 2*x + 1 with scalar features."""
        features = Series((1,), np.dtype(np.float64))
        targets = Series((), np.dtype(np.float64))

        reg = RollingLinearRegression(features, targets, train_window=10, retrain_every=1)

        for i in range(1, 11):
            x = float(i)
            y = 2.0 * x + 1.0
            features.append(ts(i), np.array([x], dtype=np.float64))
            targets.append(ts(i), np.array(y, dtype=np.float64))
            reg.update(ts(i))

        # After enough data, predictions should be close to 2*x + 1.
        last_pred = float(reg.output.values[-1])
        last_x = 10.0
        assert last_pred == pytest.approx(2.0 * last_x + 1.0, abs=0.1)

    def test_no_prediction_before_fit(self) -> None:
        features = Series((1,), np.dtype(np.float64))
        targets = Series((), np.dtype(np.float64))

        reg = RollingLinearRegression(features, targets, train_window=5, retrain_every=1)
        # Only 1 data point -> can't fit (need >= 2)
        features.append(ts(1), np.array([1.0], dtype=np.float64))
        targets.append(ts(1), np.array(3.0, dtype=np.float64))
        reg.update(ts(1))
        assert len(reg.output) == 0


# ===========================================================================
# Portfolios
# ===========================================================================


class TestTopK:
    def test_fixed_k(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3, 0.2, 0.4]])
        op = TopK(preds, k=2)
        op.update(ts(1))
        w = op.output.values[-1]
        # Top 2: indices 1 (0.5) and 4 (0.4)
        assert float(w[1]) == pytest.approx(0.5)
        assert float(w[4]) == pytest.approx(0.5)
        assert float(w[0]) == pytest.approx(0.0)

    def test_fractional_k(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3, 0.2]])
        op = TopK(preds, k=0.5)  # 50% of 4 = 2
        op.update(ts(1))
        w = op.output.values[-1]
        nonzero = np.count_nonzero(w)
        assert nonzero == 2
        assert float(w.sum()) == pytest.approx(1.0)


class TestTopKRankLinear:
    def test_rank_weights(self) -> None:
        preds = make_vector_series([[0.1, 0.5, 0.3]])
        op = TopKRankLinear(preds, k=2)
        op.update(ts(1))
        w = op.output.values[-1]
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

        # Day 1: buy 10 of asset A at 50, 5 of asset B at 100
        prices.append(ts(1), np.array([50.0, 100.0], dtype=np.float64))
        positions.append(ts(1), np.array([10.0, 5.0], dtype=np.float64))
        sim.update(ts(1))
        # Cost: 10*50 + 5*100 = 1000
        # Cash: 1000 - 1000 = 0
        # MV: 0 + 10*50 + 5*100 = 1000
        assert float(sim.output.values[-1]) == pytest.approx(1000.0)

        # Day 2: prices go up, hold same positions
        prices.append(ts(2), np.array([55.0, 110.0], dtype=np.float64))
        positions.append(ts(2), np.array([10.0, 5.0], dtype=np.float64))
        sim.update(ts(2))
        # No trades -> no cost
        # MV: 0 + 10*55 + 5*110 = 1100
        assert float(sim.output.values[-1]) == pytest.approx(1100.0)

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

        prices.append(ts(1), np.array([100.0, 200.0], dtype=np.float64))
        positions.append(ts(1), np.array([10.0, 5.0], dtype=np.float64))
        sim.update(ts(1))
        # Trade cost: 10*100 + 5*200 = 2000
        # Commission per asset: max(0.01*1000, 1) = 10, max(0.01*1000, 1) = 10
        # Total commission: 20
        # Cash: 10000 - 2000 - 20 = 7980
        # MV: 7980 + 10*100 + 5*200 = 7980 + 2000 = 9980
        assert float(sim.output.values[-1]) == pytest.approx(9980.0)

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

        # Buy 1 share at 10 -> trade value = 10
        # Commission: max(0.001*10, 5) = max(0.01, 5) = 5
        prices.append(ts(1), np.array([10.0], dtype=np.float64))
        positions.append(ts(1), np.array([1.0], dtype=np.float64))
        sim.update(ts(1))
        # Cash: 1000 - 10 - 5 = 985
        # MV: 985 + 10 = 995
        assert float(sim.output.values[-1]) == pytest.approx(995.0)


# ===========================================================================
# Metrics
# ===========================================================================


class TestAverageReturn:
    def test_basic(self) -> None:
        mv = make_scalar_series([100.0, 110.0, 121.0])
        signal = make_scalar_series([1.0, 1.0, 1.0])  # Signal at every tick
        ar = AverageReturn(mv, signal)
        for i in range(1, 4):
            ar.update(ts(i))
        # Returns: 10/100=0.1, 11/110=0.1
        # Average: 0.1
        assert len(ar.output) == 2  # First signal records prev; output from 2nd
        assert float(ar.output.values[-1]) == pytest.approx(0.1)

    def test_no_signal_no_output(self) -> None:
        mv = make_scalar_series([100.0, 110.0])
        signal = Series((), np.dtype(np.float64))
        ar = AverageReturn(mv, signal)
        ar.update(ts(1))
        ar.update(ts(2))
        assert len(ar.output) == 0


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
        for i in range(1, 5):
            sr.update(ts(i))

        # Need >= 2 returns -> first output at i=4 (3 signal ticks with returns)
        returns = np.array([0.1, 0.1, -0.1])
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        assert len(sr.output) == 1
        assert float(sr.output.values[-1]) == pytest.approx(expected, rel=1e-6)
