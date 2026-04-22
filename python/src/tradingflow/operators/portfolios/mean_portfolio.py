"""Abstract mean-portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class MeanPortfolioState:
    """Mutable state for [`MeanPortfolio`][tradingflow.operators.portfolios.mean_portfolio.MeanPortfolio] subclasses."""

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
    """Abstract portfolio constructor from per-stock predictions.

    Triggered by `universe` updates — the universe is the canonical
    rebalance signal.  On each trigger, delegates to `positions_fn` to
    compute position weights from the subset of stocks with positive
    universe weights and finite predictions.  The result is scattered
    back to the full dimension with zeros elsewhere.

    The rebalance cadence is inherited from upstream: `universe` is
    typically clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.clocked.Clocked]), so this operator runs
    at that cadence.  `predicted_returns` is read as the latest stored
    prediction at the trigger — it need not produce on the same cycle.

    ## Expected prediction semantics

    `MeanPortfolio` subclasses (`Proportional`, `RankEqual`, `RankLinear`,
    `Softmax`) only consume the *ordering* of the predictions — a
    monotonic transform of the input does not change the selected
    top-N or rank-based weights.  Accordingly the predictions may be
    raw expected returns, rank-transformed scores, Gaussianized scores,
    or any other per-stock "score" with a consistent sign convention
    (higher = better).  The upstream predictor defines what the score
    represents via its `target_series` input.

    ## NaN behavior

    `predicted_returns` is allowed to contain `NaN` entries — per the
    [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor]
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
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and the predictor's mu is stored as the last
        # prediction even when it did not produce this cycle.
        if not produced[0]:
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
