"""Top-K rank-linear portfolio construction.

Provides :class:`TopKRankLinear`, an :class:`Operator` that selects the
top *k* assets and assigns weights proportional to their rank among the
selected assets.  *k* may be an integer or a float fraction of the
universe size.
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class TopKRankLinear(Operator[tuple[Series], tuple[int], np.float64, None]):
    """Selects the top *k* assets and assigns rank-proportional weights.

    The highest-predicted asset receives the largest weight; weights are
    normalised to sum to 1.
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
        sorted_indices = np.argsort(values)
        top_indices = sorted_indices[-k:]
        weights = np.zeros(n, dtype=np.float64)
        for rank, idx in enumerate(top_indices, start=1):
            weights[idx] = rank
        weights /= weights.sum()
        return weights
