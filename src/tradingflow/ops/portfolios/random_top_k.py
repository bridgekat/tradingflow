"""Random top-K portfolio construction.

Provides :class:`RandomTopK`, an :class:`Operator` that selects the top
*top_frac* fraction of assets by predicted value, then randomly picks
*select_k* from that group and assigns each an equal weight of ``1/select_k``.
"""

from __future__ import annotations

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class RandomTopK(Operator[tuple[Series], tuple[int], np.float64, np.random.Generator]):
    """Randomly selects *select_k* assets from the top *top_frac* percentile.

    At each update, the top ``ceil(top_frac * N)`` assets by predicted value
    are identified, then *select_k* of them are randomly chosen and assigned
    equal weights summing to 1.  Assets with NaN predictions are excluded
    from the candidate pool.

    Parameters
    ----------
    predictions
        Vector series of predicted values for *N* assets.
    top_frac
        Fraction of assets to consider (e.g. ``0.1`` for top 10%).
    select_k
        Number of assets to randomly select from the top pool.
    seed
        Random seed for reproducibility.
    """

    __slots__ = ("_top_frac", "_select_k", "_seed")

    _top_frac: float
    _select_k: int
    _seed: int

    def __init__(
        self,
        predictions: Series,
        top_frac: float = 0.1,
        select_k: int = 20,
        seed: int = 42,
    ) -> None:
        n = predictions.shape[0]
        super().__init__((predictions,), (n,), np.dtype(np.float64))
        self._top_frac = top_frac
        self._select_k = select_k
        self._seed = seed

    @override
    def init_state(self) -> np.random.Generator:
        return np.random.default_rng(self._seed)

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series],
        state: np.random.Generator,
    ) -> tuple[ArrayLike | None, np.random.Generator]:
        (preds,) = inputs
        if not preds:
            return None, state
        values = preds.values[-1]
        n = len(values)

        # Exclude NaN predictions
        valid_mask = np.isfinite(values)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, state

        valid_values = values[valid_indices]

        # Determine top pool size
        top_count = max(1, int(self._top_frac * len(valid_indices)))
        top_count = min(top_count, len(valid_indices))

        # Get top indices (among valid)
        top_local = np.argsort(valid_values)[-top_count:]
        top_global = valid_indices[top_local]

        # Randomly select from pool
        k = min(self._select_k, len(top_global))
        selected = state.choice(top_global, size=k, replace=False)

        weights = np.zeros(n, dtype=np.float64)
        weights[selected] = 1.0 / k
        return weights, state
