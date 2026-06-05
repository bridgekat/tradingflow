"""Rank-linear mean-only portfolio construction (flowops host port)."""

from __future__ import annotations

import numpy as np

from flowops.portfolios._base import MeanPortfolio


class RankLinear(MeanPortfolio):
    """Assign linearly-decreasing weights to the top fraction of stocks.

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
    weights[ranked[:k]] = np.linspace(k, 1, k)

    s = weights.sum()
    if s > 0:
        weights /= s
    return weights


def build(**kwargs) -> RankLinear:
    return RankLinear(**kwargs)
