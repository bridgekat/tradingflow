"""Abstract mean-variance portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class MeanVariancePortfolioState:
    """Mutable state for [`MeanVariancePortfolio`] subclasses."""

    num_stocks: int
    positions_fn: Callable[["MeanVariancePortfolioState", np.ndarray, np.ndarray], np.ndarray]


class MeanVariancePortfolio(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        MeanVariancePortfolioState,
    ]
):
    """Abstract portfolio constructor from predicted returns and covariance.

    Triggered by `universe` updates — the universe is the canonical
    rebalance signal.  On each trigger, delegates to `positions_fn` to
    compute position weights.  Only stocks with positive universe
    weights, finite predicted returns, and finite diagonal covariance
    entries are passed to `positions_fn`; the result is scattered back
    to the full dimension with zeros elsewhere.

    The rebalance cadence is inherited from upstream: `universe` is
    typically clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.Clocked]), so this operator runs
    at that cadence.  `predicted_returns` and `predicted_covariances`
    are read as the latest stored predictions at the trigger — neither
    need produce on the same cycle.

    ## NaN behavior

    `predicted_returns` and `predicted_covariances` are allowed to
    contain `NaN` entries — per the
    [`MeanPredictor`][tradingflow.operators.predictors.MeanPredictor]
    and
    [`VariancePredictor`][tradingflow.operators.predictors.VariancePredictor]
    contracts, these mark stocks with insufficient data.  The base
    class subsets to `(universe > 0) & np.isfinite(mu) &
    np.isfinite(np.diag(Sigma))` before calling `positions_fn`, so
    subclasses never see `NaN` inputs.  The sub-covariance-matrix of
    the remaining stocks must not contain non-finite off-diagonal
    entries (the base class raises `ValueError` if it does).  The
    emitted position vector is zero for stocks outside the valid
    subset.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
        Stocks with positive values are included in the optimization.
    predicted_returns
        Handle to predicted returns array, shape `(num_stocks,)`.
    predicted_covariances
        Handle to predicted covariance matrix, shape `(num_stocks, num_stocks)`.
    positions_fn
        `(state, mu, Sigma) -> weights`.  Receives only the subset.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_returns: Handle,
        predicted_covariances: Handle,
        *,
        positions_fn: Callable[[MeanVariancePortfolioState, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_returns.shape) == 1
        assert len(predicted_covariances.shape) == 2
        assert predicted_returns.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[1] == universe.shape[0]

        self._num_stocks = predicted_returns.shape[0]
        self._positions_fn = positions_fn

        inputs = (universe, predicted_returns, predicted_covariances)
        super().__init__(
            inputs=inputs,
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[
            ArrayView[np.float64],
            ArrayView[np.float64],
            ArrayView[np.float64],
        ],
        timestamp: int,
    ) -> MeanVariancePortfolioState:
        return MeanVariancePortfolioState(
            num_stocks=self._num_stocks,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanVariancePortfolioState,
        inputs: tuple[
            ArrayView[np.float64],
            ArrayView[np.float64],
            ArrayView[np.float64],
        ],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and both mu and sigma are stored as the last
        # predictions even when they did not produce this cycle.
        if not produced[0]:
            return False

        universe = inputs[0].value()
        mu = inputs[1].value()
        sigma = inputs[2].value()

        mask = (universe > 0) & np.isfinite(mu) & np.isfinite(np.diag(sigma))
        sub_mu = mu[mask]
        sub_sigma = sigma[np.ix_(mask, mask)]
        if not np.all(np.isfinite(sub_sigma)):
            raise ValueError("sub-covariance matrix contains non-finite entries")

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_mu, sub_sigma)

        output.write(positions)
        return True
