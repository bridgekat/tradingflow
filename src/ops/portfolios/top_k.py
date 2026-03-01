"""Top-K equal-weight portfolio operator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ... import Operator, Series


class TopK(Operator[None, np.float64]):
    """Select top-K assets by predicted return with equal weighting."""

    __slots__ = ("_k",)

    def __init__(
        self,
        predictions: Series[Any],
        k: int | float,
    ) -> None:
        if len(predictions.shape) != 1:
            raise ValueError(
                "TopK requires a vector-valued predictions series " f"(shape (n,)), got shape {predictions.shape}"
            )
        n = predictions.shape[0]
        super().__init__([predictions], None, np.dtype(np.float64), (n,))
        self._k = k

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        preds = inputs[0]
        if not preds:
            return None
        values: NDArray[Any] = preds.values[-1]
        n = len(values)

        if isinstance(self._k, float) and self._k <= 1.0:
            k = max(1, int(np.ceil(self._k * n)))
        else:
            k = int(self._k)
        k = min(k, n)

        indices = np.argsort(values)[::-1][:k]
        weights = np.zeros(n, dtype=np.float64)
        weights[indices] = 1.0 / k
        return weights
