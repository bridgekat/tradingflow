"""Abstract mean-portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field

import numpy as np

from ...operator import Operator, Notify
from ...types import Array, Handle, NodeKind


@dataclass(slots=True)
class MeanPortfolioState:
    # Configuration parameters.
    num_stocks: int
    positions_fn: Callable[["MeanPortfolioState", np.ndarray], np.ndarray]


class MeanPortfolio(
    Operator[
        tuple[Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        MeanPortfolioState,
    ]
):
    """Abstract portfolio constructor from predicted returns.

    On each tick, reads predicted returns and delegates to the
    ``positions_fn`` stored on the state to compute position weights.

    Parameters
    ----------
    predicted_returns
        Handle to predicted returns array, shape ``(num_stocks,)``.

    positions_fn
        ``(state, predicted) -> positions``.  Called with the current state
        and the predicted returns array of shape ``(num_stocks,)``.
        Returns position weights of shape ``(num_stocks,)``.
    """

    def __init__(
        self,
        predicted_returns: Handle,
        *,
        positions_fn: Callable[[MeanPortfolioState, np.ndarray], np.ndarray],
    ) -> None:
        assert len(predicted_returns.shape) == 1, "predicted_returns must be 1D"

        self._num_stocks = predicted_returns.shape[0]
        self._positions_fn = positions_fn

        super().__init__(
            inputs=(predicted_returns,),
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
        predicted = inputs[0].value()
        positions = state.positions_fn(state, predicted)
        output.write(positions)
        return True
