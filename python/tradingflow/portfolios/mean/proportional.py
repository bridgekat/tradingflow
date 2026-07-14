"""Proportional mean-only portfolio construction (pyhost port)."""

from __future__ import annotations

import numpy as np

from tradingflow.portfolios._base import MeanPortfolio


class Proportional(MeanPortfolio):
    """Weight stocks proportionally to their positive predicted returns.

    Stocks with non-positive predicted returns receive zero weight; the
    remaining weights are normalized to sum to 1.

    Inputs: ``(universe, predicted_returns)``, both ``(num_stocks,)``.
    Output: position weights, ``(num_stocks,)``.
    """

    def __init__(self, *, num_stocks: int | None = None, logarithmic: bool = True) -> None:
        super().__init__(
            positions_fn=lambda state, predicted: _positions_fn(predicted),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _positions_fn(returns: np.ndarray) -> np.ndarray:
    weights = np.maximum(returns, 0.0)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights


def build(**kwargs) -> Proportional:
    return Proportional(**kwargs)
