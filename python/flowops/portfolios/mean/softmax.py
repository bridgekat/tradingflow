"""Softmax mean-only portfolio construction (flowops host port)."""

from __future__ import annotations

import numpy as np

from flowops.portfolios._base import MeanPortfolio


class Softmax(MeanPortfolio):
    r"""Weight stocks via softmax of their positive predicted returns.

    ``w_i = exp(max(r_i, 0) / tau)``, normalized to sum to 1.

    Inputs: ``(universe, predicted_returns)``, both ``(num_stocks,)``.
    Output: position weights, ``(num_stocks,)``.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        num_stocks: int | None = None,
        logarithmic: bool = True,
    ) -> None:
        super().__init__(
            positions_fn=lambda state, predicted: _positions_fn(predicted, temperature),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _positions_fn(predicted: np.ndarray, temperature: float) -> np.ndarray:
    positive = np.maximum(predicted, 0.0)
    weights = np.exp(positive / temperature)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights


def build(**kwargs) -> Softmax:
    return Softmax(**kwargs)
