"""Top-K equal-weight portfolio construction.

Provides [`TopK`][tradingflow.operators.portfolios.TopK], an [`Operator`][tradingflow.Operator] that selects the top *k*
assets by predicted value and assigns each an equal weight of `1/k`.
*k* may be an integer or a float fraction of the universe size.
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator
from ...observable import Observable


class TopK(Operator[tuple[Observable], tuple[int], np.float64, None]):
    """Selects the top *k* assets by predicted value and assigns equal weights.

    *k* may be an integer (fixed count) or a float in `(0, 1]`
    (fraction of the universe size).  Weights sum to 1.
    """

    __slots__ = ("_k",)

    _k: int | float

    def __init__(self, predictions: Observable, k: int | float) -> None:
        n = predictions.shape[0]
        super().__init__((predictions,), (n,), np.dtype(np.float64))
        self._k = k

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Observable], state: None) -> tuple[ArrayLike | None, None]:
        (preds,) = inputs
        if not preds:
            return None, None
        values = preds.last
        n = len(values)
        if isinstance(self._k, float):
            k = max(1, int(self._k * n))
        else:
            k = self._k
        top_indices = np.argsort(values)[-k:]
        weights = np.zeros(n, dtype=np.float64)
        weights[top_indices] = 1.0 / k
        return weights, None
