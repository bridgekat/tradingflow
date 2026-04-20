"""Abstract mean-portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class MeanPortfolioState:
    num_stocks: int
    positions_fn: Callable[["MeanPortfolioState", np.ndarray], np.ndarray]


class MeanPortfolio(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        MeanPortfolioState,
    ]
):
    """Abstract portfolio constructor from predicted returns.

    Triggered by new predicted returns from upstream.  Delegates to
    `positions_fn` to compute position weights from the subset of
    stocks with positive universe weights and finite predictions.  The
    result is scattered back to the full dimension with zeros elsewhere.

    The rebalance cadence is inherited from upstream: when the predictor
    is clock-triggered at rebalance dates, this operator runs at the
    same cadence (driven by the predictor's input-production signal).

    ## NaN behavior

    `predicted_returns` is allowed to contain `NaN` entries — per the
    [`MeanPredictor`][tradingflow.operators.predictors.MeanPredictor]
    contract, these mark stocks with insufficient data.  The base class
    subsets to `(universe > 0) & np.isfinite(mu)` before calling
    `positions_fn`, so subclasses never see `NaN` inputs.  The emitted
    position vector is zero for stocks outside this subset.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
        Stocks with positive values are included in the optimization.
    predicted_returns
        Handle to predicted returns array, shape `(num_stocks,)`.
    positions_fn
        `(state, mu) -> positions`.  Receives only the subset of
        stocks with positive universe weights and finite predictions.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_returns: Handle,
        *,
        positions_fn: Callable[[MeanPortfolioState, np.ndarray], np.ndarray],
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_returns.shape) == 1
        assert predicted_returns.shape[0] == universe.shape[0]

        self._num_stocks = predicted_returns.shape[0]
        self._positions_fn = positions_fn

        inputs = (universe, predicted_returns)
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
    ) -> MeanPortfolioState:
        return MeanPortfolioState(
            num_stocks=self._num_stocks,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanPortfolioState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Changes in universe only should not trigger recomputation.
        if not produced[1]:
            return False

        universe = inputs[0].value()
        mu = inputs[1].value()

        mask = (universe > 0) & np.isfinite(mu)
        sub_mu = mu[mask]

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_mu)

        output.write(positions)
        return True
