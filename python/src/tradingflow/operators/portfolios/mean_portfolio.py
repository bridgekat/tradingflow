"""Abstract mean-portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class MeanPortfolioState:
    num_stocks: int
    positions_fn: Callable[["MeanPortfolioState", np.ndarray], np.ndarray]


class MeanPortfolio(
    Operator[
        tuple[Handle[Array[np.float64]], ...],
        Handle[Array[np.float64]],
        MeanPortfolioState,
    ]
):
    """Abstract portfolio constructor from predicted returns.

    On each tick, reads predicted returns, delegates to ``positions_fn``
    to compute position weights.

    Only stocks with positive universe weights and finite predicted
    returns are passed to ``positions_fn``; the result is scattered back
    to the full dimension with zeros elsewhere.

    Parameters
    ----------
    universe
        Handle to universe weights, shape ``(num_stocks,)``.
        Stocks with positive values are included in the optimization.
    predicted_returns
        Handle to predicted returns array, shape ``(num_stocks,)``.
    positions_fn
        ``(state, mu) -> positions``.  Receives only the subset of
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

    def init(self, inputs: tuple, timestamp: int) -> MeanPortfolioState:
        return MeanPortfolioState(
            num_stocks=self._num_stocks,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: MeanPortfolioState,
        inputs: tuple,
        output,
        timestamp: int,
        notify: Notify,
    ) -> bool:
        # Changes in universe only should not trigger recomputation.
        if not notify.input_produced()[1]:
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
