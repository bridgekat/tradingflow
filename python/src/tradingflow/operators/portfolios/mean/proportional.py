"""Proportional mean-only portfolio construction."""

import numpy as np

from ..mean_portfolio import MeanPortfolio


class Proportional(MeanPortfolio):
    """Weight stocks proportionally to their positive predicted returns.

    Stocks with non-positive predicted returns receive zero weight.
    The remaining weights are normalized to sum to 1.

    Parameters
    ----------
    universe
        Handle to universe weights, shape ``(num_stocks,)``.
    predicted_returns
        Handle to predicted returns array, shape ``(num_stocks,)``.
    """

    def __init__(self, universe, predicted_returns) -> None:
        super().__init__(
            universe,
            predicted_returns,
            positions_fn=lambda state, predicted: _positions_fn(predicted),
        )


def _positions_fn(returns: np.ndarray) -> np.ndarray:
    weights = np.maximum(returns, 0.0)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights
