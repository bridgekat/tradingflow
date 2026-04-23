"""Softmax mean-only portfolio construction."""

import numpy as np

from ..mean_portfolio import MeanPortfolio


class Softmax(MeanPortfolio):
    r"""Weight stocks via softmax of their positive predicted returns.

    \(w_i = \exp(\max(r_i, 0) / \tau)\), normalized
    to sum to 1.  Stocks with non-positive predicted returns receive
    \(\exp(0) = 1\) as their base weight, so they are not entirely
    excluded but heavily down-weighted when `temperature` is small.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
    predicted_returns
        Handle to predicted log-returns array, shape `(num_stocks,)`.
        Converted to linear returns by
        [`MeanPortfolio`][tradingflow.operators.portfolios.mean_portfolio.MeanPortfolio]
        before the softmax is applied.
    temperature
        Softmax temperature.  Lower values sharpen the distribution
        toward the highest-predicted stocks.
    **kwargs
        Forwarded to [`MeanPortfolio`][tradingflow.operators.portfolios.mean_portfolio.MeanPortfolio].
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
