"""Rank-equal mean-only portfolio construction (pyhost port)."""

from __future__ import annotations

import numpy as np

from tradingflow.portfolios._base import MeanPortfolio


class RankEqual(MeanPortfolio):
    """Assign equal weights to the top fraction of positively-predicted stocks.

    Inputs: ``(universe, predicted_returns)``, both ``(num_stocks,)``.
    Output: position weights, ``(num_stocks,)``.
    """

    def __init__(
        self,
        *,
        top_fraction: float = 0.1,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        super().__init__(
            positions_fn=lambda state, predicted: _positions_fn(predicted, top_fraction),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
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


def build(**kwargs) -> RankEqual:
    return RankEqual(**kwargs)
