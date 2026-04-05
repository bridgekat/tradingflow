"""Rank-equal mean-only portfolio construction."""

import numpy as np

from ..mean_portfolio import MeanPortfolio


class RankEqual(MeanPortfolio):
    """Assign equal weights to the top fraction of stocks.

    Stocks with positive predicted returns are ranked.  The top
    ``top_fraction`` receive equal weight, normalized to sum to 1.
    All other stocks receive zero weight.

    Parameters
    ----------
    predicted_returns
        Handle to predicted returns array, shape ``(num_stocks,)``.
    top_fraction
        Fraction of positively-predicted stocks to include.
    """

    def __init__(self, predicted_returns, *, top_fraction: float = 0.1) -> None:
        super().__init__(
            predicted_returns,
            positions_fn=lambda state, predicted: _positions_fn(predicted, top_fraction),
        )


def _positions_fn(returns: np.ndarray, top_fraction: float) -> np.ndarray:
    positive = np.maximum(returns, 0.0)
    weights = np.zeros_like(returns)

    n = (returns > 0).sum()
    if n == 0:
        return weights
    k = round(top_fraction * n)

    ranked = np.argsort(-positive)
    weights[ranked[:k]] = 1.0

    s = weights.sum()
    if s > 0:
        weights /= s
    return weights
