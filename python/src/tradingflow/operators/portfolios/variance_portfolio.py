"""Abstract variance portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class VariancePortfolioState:
    """Mutable state for [`VariancePortfolio`][tradingflow.operators.portfolios.variance_portfolio.VariancePortfolio] subclasses."""

    num_stocks: int
    positions_fn: Callable[["VariancePortfolioState", np.ndarray], np.ndarray]


class VariancePortfolio(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        VariancePortfolioState,
    ]
):
    """Abstract portfolio constructor from covariance alone (no expected returns).

    Triggered by `universe` updates — the universe is the canonical
    rebalance signal.  On each trigger, delegates to `positions_fn` to
    compute position weights.  Only stocks with positive universe
    weights and finite diagonal covariance entries are passed to
    `positions_fn`; the result is scattered back to the full dimension
    with zeros elsewhere.

    The rebalance cadence is inherited from upstream: `universe` is
    typically clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.clocked.Clocked]), so this operator runs
    at that cadence.  `predicted_covariances` is read as the latest
    stored prediction at the trigger — it need not produce on the same
    cycle.

    ## Expected prediction semantics

    Minimum-variance solutions depend on the covariance only up to a
    positive scalar — the chosen weights are the same whether the
    upstream target is per-period or annualized returns — so the
    `target_series` choice upstream affects reported risk figures but
    not the weights themselves.  The covariance must nevertheless be
    a valid covariance of some well-defined per-stock quantity (linear
    returns, log returns, ...); rank-transformed or Gaussianized
    covariances are technically valid but rarely what one wants for
    risk minimization.

    ## NaN behavior

    `predicted_covariances` is allowed to contain `NaN` rows and columns
    — per the
    [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor]
    contract, these mark stocks with insufficient data.  The base class
    subsets to `(universe > 0) & np.isfinite(np.diag(Sigma))` before
    calling `positions_fn`, so subclasses never see `NaN` inputs.  The
    sub-covariance-matrix of the remaining stocks must not contain
    non-finite off-diagonal entries (the base class raises
    `ValueError` if it does).  The emitted position vector is zero for
    stocks outside the valid subset.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
        Stocks with positive values are included in the optimization.
    predicted_covariances
        Handle to predicted covariance matrix, shape `(num_stocks, num_stocks)`.
    positions_fn
        `(state, Sigma) -> weights`.  Receives only the universe-
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

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> VariancePortfolioState:
        return VariancePortfolioState(
            num_stocks=self._num_stocks,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: VariancePortfolioState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and the predictor's sigma is stored as the
        # last prediction even when it did not produce this cycle.
        if not produced[0]:
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
