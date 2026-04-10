"""Softmax mean-only portfolio construction."""

import numpy as np

from ..mean_portfolio import MeanPortfolio


class Softmax(MeanPortfolio):
    """Weight stocks via softmax of their positive predicted returns.

    ``weights[i] = exp(max(predicted[i], 0) / temperature)``, normalised
    to sum to 1.  Stocks with non-positive predicted returns receive
    ``exp(0) = 1`` as their base weight, so they are not entirely
    excluded but heavily down-weighted when ``temperature`` is small.

    Parameters
    ----------
    predicted_returns
        Handle to predicted returns array, shape ``(num_stocks,)``.
    temperature
        Softmax temperature.  Lower values sharpen the distribution
        towards the highest-predicted stocks.
    **kwargs
        Forwarded to [`MeanPortfolio`][tradingflow.operators.portfolios.MeanPortfolio].
    """

    def __init__(self, universe, predicted_returns, *, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(
            universe,
            predicted_returns,
            positions_fn=lambda state, predicted: _positions_fn(predicted, temperature),
            **kwargs,
        )


def _positions_fn(predicted: np.ndarray, temperature: float) -> np.ndarray:
    positive = np.maximum(predicted, 0.0)
    weights = np.exp(positive / temperature)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights
