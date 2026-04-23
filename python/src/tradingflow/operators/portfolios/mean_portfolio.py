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
    logarithmic: bool
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

    Controlled by `logarithmic`:

    - `logarithmic=True` (default): `predicted_returns` is in **log-return**
      units.  The base class converts to linear returns via
      `mu_lin = exp(mu_log) - 1` (the zero-covariance specialisation of
      the lognormal moment map) before calling `positions_fn`.
    - `logarithmic=False`: `predicted_returns` is already in **linear-return**
      units and is forwarded to `positions_fn` unchanged.

    Either way, `positions_fn` always sees linear-return predictions.
    Rank-based subclasses (`RankEqual`, `RankLinear`, `Softmax`,
    `Proportional`) only consume ordering, and `exp(·) - 1` is
    monotone, so the flag has no effect on their output.

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
        Handle to predicted returns array, shape `(num_stocks,)`.  Log
        or linear depending on `logarithmic`.
    positions_fn
        `(state, mu) -> positions`.  Receives only the subset of
        stocks with positive universe weights and finite predictions,
        in linear-return units.
    logarithmic
        If `True` (default), `predicted_returns` is interpreted as log
        returns and converted via `exp(·) - 1` before the inner call;
        if `False`, it is interpreted as linear returns and passed
        through unchanged.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_returns: Handle,
        *,
        positions_fn: Callable[[MeanPortfolioState, np.ndarray], np.ndarray],
        logarithmic: bool = True,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_returns.shape) == 1
        assert predicted_returns.shape[0] == universe.shape[0]

        self._num_stocks = predicted_returns.shape[0]
        self._logarithmic = logarithmic
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
            logarithmic=self._logarithmic,
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
        # Lognormal conversion (zero-covariance specialisation):
        #   mu_lin[i] = exp(mu_log[i]) - 1
        if state.logarithmic:
            sub_mu = np.expm1(sub_mu)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_mu)

        output.write(positions)
        return True
