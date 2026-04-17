"""Abstract variance portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ...operator import Operator
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class VariancePortfolioState:
    """Mutable state for [`VariancePortfolio`] subclasses."""

    num_stocks: int
    positions_fn: Callable[["VariancePortfolioState", np.ndarray], np.ndarray]


class VariancePortfolio(
    Operator[
        tuple[Handle[Array[np.float64]], ...],
        Handle[Array[np.float64]],
        VariancePortfolioState,
    ]
):
    """Abstract portfolio constructor from covariance alone (no expected returns).

    Triggered by new predicted covariance from upstream.  Delegates to
    ``positions_fn`` to compute position weights.  Only stocks with
    positive universe weights and finite diagonal covariance entries
    are passed to ``positions_fn``; the result is scattered back to
    the full dimension with zeros elsewhere.  The sub-covariance
    matrix of the remaining stocks must not contain non-finite
    entries.

    The rebalance cadence is inherited from upstream: when the
    covariance predictor is clock-triggered at rebalance dates, this
    operator runs at the same cadence.

    Parameters
    ----------
    universe
        Handle to universe weights, shape ``(num_stocks,)``.
        Stocks with positive values are included in the optimization.
    predicted_covariances
        Handle to predicted covariance matrix, shape ``(num_stocks, num_stocks)``.
    positions_fn
        ``(state, Sigma) -> weights``.  Receives only the universe-
        active sub-block of the covariance matrix.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_covariances: Handle,
        *,
        positions_fn: Callable[[VariancePortfolioState, np.ndarray], np.ndarray],
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_covariances.shape) == 2
        assert predicted_covariances.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[1] == universe.shape[0]

        self._num_stocks = universe.shape[0]
        self._positions_fn = positions_fn

        inputs = (universe, predicted_covariances)
        super().__init__(
            inputs=inputs,
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(self, inputs: tuple, timestamp: int) -> VariancePortfolioState:
        return VariancePortfolioState(
            num_stocks=self._num_stocks,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: VariancePortfolioState,
        inputs: tuple,
        output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Changes in universe only should not trigger recomputation.
        if not produced[1]:
            return False

        universe = inputs[0].value()
        sigma = inputs[1].value()

        mask = (universe > 0) & np.isfinite(np.diag(sigma))
        sub_sigma = sigma[np.ix_(mask, mask)]
        if not np.all(np.isfinite(sub_sigma)):
            raise ValueError("sub-covariance matrix contains non-finite entries")

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_sigma)

        output.write(positions)
        return True
