"""Global Minimum Variance (GMV) portfolio realized variance evaluator."""

from dataclasses import dataclass

import numpy as np
import scipy as sp

from ....views import ArrayView
from ....operator import Operator
from ....types import Array, Handle, NodeKind


@dataclass(slots=True)
class MinimumVarianceState:
    num_stocks: int
    initialized: bool = False
    weights: np.ndarray | None = None
    prev_prices: np.ndarray | None = None
    sum_r: float = 0.0
    sum_r_sq: float = 0.0
    count: int = 0


class MinimumVariance(
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]]],
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
    emitting), the operator emits exactly once per prediction
    emission.  Output is 0 if no daily portfolio return was
    successfully accumulated during the period (e.g. no stocks had
    finite covariance diagonal).

    Memory
    ------
    Both ``predictions`` and ``prices`` are ``Array`` inputs — the
    operator reads only the latest covariance and the latest cross-
    section of prices, caching one previous price tick in state to
    compute one-period returns.  No ``Record`` is required upstream.

    Parameters
    ----------
    predictions
        Live predicted covariance matrix from a variance predictor,
        shape ``(N, N)``.  Stocks excluded by the variance predictor
        have NaN on the diagonal.
    prices
        Live forward-adjusted close prices, shape ``(N,)``.
    """

    def __init__(
        self,
        predictions: Handle,
        prices: Handle,
    ) -> None:
        assert len(predictions.shape) == 2
        assert predictions.shape[0] == predictions.shape[1]
        assert len(prices.shape) == 1
        assert predictions.shape[0] == prices.shape[0]

        self._num_stocks = predictions.shape[0]

        super().__init__(
            inputs=(predictions, prices),
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
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        predictions, prices = inputs
        predictions_produced, prices_produced = produced

        # Accumulate one-period portfolio return on price ticks.
        if prices_produced:
            prices_new = np.where(prices.value() > 0, prices.value(), np.nan)
            if state.weights is not None and state.prev_prices is not None:
                r = prices_new / state.prev_prices - 1.0
                r_p = float(state.weights @ np.where(np.isfinite(r), r, 0.0))
                state.sum_r += r_p
                state.sum_r_sq += r_p * r_p
                state.count += 1
            state.prev_prices = prices_new

        # Gate: new prediction?
        if not predictions_produced:
            return False

        # First prediction stores scores without emitting.
        if not state.initialized:
            state.initialized = True
            MinimumVariance._set_weights(state, predictions.value())
            return False

        # Emit realized variance over the evaluation period.
        mean = state.sum_r / max(state.count, 1)
        variance = state.sum_r_sq / max(state.count, 1) - mean * mean
        output.write(np.array(variance, dtype=np.float64))

        # Update weights and reset accumulators.
        MinimumVariance._set_weights(state, predictions.value())
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
