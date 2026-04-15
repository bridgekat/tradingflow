"""Global Minimum Variance (GMV) portfolio realized variance evaluator."""

from dataclasses import dataclass

import numpy as np
import scipy as sp

from ...views import ArrayView, SeriesView
from ...operator import Operator, Notify
from ...types import Array, Series, Handle, NodeKind


@dataclass(slots=True)
class MinimumVarianceState:
    num_stocks: int
    initialized: bool = False
    weights: np.ndarray | None = None
    sum_r: float = 0.0
    sum_r_sq: float = 0.0
    count: int = 0


class MinimumVariance(
    Operator[
        tuple[Handle[Series[np.float64]], Handle[Series[np.float64]]],
        Handle[Array[np.float64]],
        MinimumVarianceState,
    ]
):
    """GMV portfolio realized variance evaluator.

    Evaluates covariance prediction quality.  On each prediction
    emission, computes GMV weights ``w = Σ⁺ 1 / (1ᵀ Σ⁺ 1)`` (using
    the SVD-based pseudo-inverse for numerical stability with
    rank-deficient Σ) and begins accumulating daily portfolio returns
    ``rₚ = wᵀ r``.  When the next prediction arrives, emits the
    realized variance of the accumulated returns and updates the
    weights.

    Output is a scalar (the realized variance over one evaluation
    period).  ``Record(output)`` produces a directly plottable
    time series.

    Alignment guarantee
    -------------------
    After the initial warmup (first prediction sets weights without
    emitting), the operator emits exactly once per
    ``predictions_series`` entry.  Output is 0 if no daily portfolio
    return was successfully accumulated during the period (e.g. no
    stocks had finite covariance diagonal).  The prediction timestamp
    for ``record[i]`` is ``predictions_ts[-(n + 1):-1][i]`` where
    ``n`` is the number of recorded outputs.

    Parameters
    ----------
    predictions_series
        Recorded predicted covariance matrices, element shape
        ``(N, N)``.  Typically ``Record(predicted_covariances)``.
        Stocks excluded by the variance predictor have NaN on the
        diagonal.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(N,)``.
    """

    def __init__(
        self,
        predictions_series: Handle,
        adjusted_prices_series: Handle,
    ) -> None:
        assert len(predictions_series.shape) == 2
        assert predictions_series.shape[0] == predictions_series.shape[1]
        assert len(adjusted_prices_series.shape) == 1
        assert predictions_series.shape[0] == adjusted_prices_series.shape[0]

        self._num_stocks = predictions_series.shape[0]

        super().__init__(
            inputs=(predictions_series, adjusted_prices_series),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> MinimumVarianceState:
        return MinimumVarianceState(
            num_stocks=self._num_stocks,
        )

    @staticmethod
    def _set_weights(state: MinimumVarianceState, sigma: np.ndarray) -> None:
        """Compute and store GMV weights from a covariance matrix.

        Stocks with non-finite covariance diagonal are excluded; their
        weights remain zero.  If no stocks are eligible, all weights
        are zero.
        """
        n = state.num_stocks
        mask = np.isfinite(np.diag(sigma))
        weights = np.zeros(n, dtype=np.float64)
        if mask.any():
            weights[mask] = _gmv_weights(sigma[np.ix_(mask, mask)])
        state.weights = weights

    @staticmethod
    def compute(
        state: MinimumVarianceState,
        inputs: tuple[SeriesView[np.float64], SeriesView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        notify: Notify,
    ) -> bool:
        predictions_series, prices_series = inputs
        predictions_produced, prices_produced = notify.input_produced()

        # Accumulate one-period portfolio return on price ticks.
        if prices_produced and state.weights is not None and len(prices_series) >= 2:
            prices_old = np.where(prices_series[-2] > 0, prices_series[-2], np.nan)
            prices_new = np.where(prices_series[-1] > 0, prices_series[-1], np.nan)
            r = prices_new / prices_old - 1.0
            r_p = float(state.weights @ np.where(np.isfinite(r), r, 0.0))
            state.sum_r += r_p
            state.sum_r_sq += r_p * r_p
            state.count += 1

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # First prediction stores scores without emitting.
        if not state.initialized:
            state.initialized = True
            MinimumVariance._set_weights(state, predictions_series[-1])
            return False

        # Emit realized variance over the evaluation period.
        mean = state.sum_r / max(state.count, 1)
        variance = state.sum_r_sq / max(state.count, 1) - mean * mean
        output.write(np.array(variance, dtype=np.float64))

        # Update weights and reset accumulators.
        MinimumVariance._set_weights(state, predictions_series[-1])
        state.sum_r = 0.0
        state.sum_r_sq = 0.0
        state.count = 0

        return True


def _gmv_weights(sigma: np.ndarray) -> np.ndarray:
    """Compute GMV portfolio weights from a covariance matrix.

    Closed-form solution to the equality-constrained quadratic program

        minimize    wᵀ Σ w
        subject to  1ᵀ w = 1

    given by ``w = Σ⁺ 1 / (1ᵀ Σ⁺ 1)``.  No non-negativity constraint
    is imposed (weights may be negative, i.e. short positions are
    allowed).

    Uses the SVD-based Moore–Penrose pseudo-inverse ``Σ⁺`` instead of
    a direct solve, which gracefully handles rank-deficient covariance
    matrices (e.g. sample covariance with N > T) by returning the
    minimum-norm solution.  If the denominator ``1ᵀ Σ⁺ 1`` is zero,
    the returned weights contain non-finite entries (NaN or inf).
    """
    n = sigma.shape[0]
    ones = np.ones(n, dtype=np.float64)
    sigma_pinv = sp.linalg.pinv(sigma)
    w = sigma_pinv @ ones
    return w / (ones @ w)
