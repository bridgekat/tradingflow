"""Top-K equal-weight portfolio construction.

Provides :class:`TopK`, an :class:`Operator` that selects the top *k*
assets by predicted value and assigns each an equal weight of ``1/k``.
*k* may be an integer or a float fraction of the universe size.
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class TopK(Operator[tuple[Series], tuple[int], np.float64, None]):
    """Selects the top *k* assets by predicted value and assigns equal weights.

    *k* may be an integer (fixed count) or a float in ``(0, 1]``
    (fraction of the universe size).  Weights sum to 1.
    """

    __slots__ = ("_k",)

    _k: int | float

    def __init__(self, predictions: Series, k: int | float) -> None:
        n = predictions.shape[0]
        super().__init__((predictions,), (n,), np.dtype(np.float64), None)
        self._k = k

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple[Series], state: None) -> ArrayLike | None:
        (preds,) = inputs
        if not preds:
            return None
        values = preds.values[-1]
        n = len(values)
        if isinstance(self._k, float):
            k = max(1, int(self._k * n))
        else:
            k = self._k
        top_indices = np.argsort(values)[-k:]
        weights = np.zeros(n, dtype=np.float64)
        weights[top_indices] = 1.0 / k
        return weights
