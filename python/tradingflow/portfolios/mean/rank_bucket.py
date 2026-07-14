"""Rank-bucket portfolio construction - equal weights within a percentile slice.

pyhost port.  Defaults ``logarithmic=False`` (rank-only, so the flag is
mathematically irrelevant; ``False`` matches the "feature, not predicted return"
semantics and is slightly cheaper per tick).
"""

from __future__ import annotations

import numpy as np

from tradingflow.portfolios._base import MeanPortfolio


class RankBucket(MeanPortfolio):
    """Equal weights to stocks whose feature rank falls in ``[low, high)``.

    Inputs: ``(universe, feature)``, both ``(num_stocks,)``.
    Output: position weights, ``(num_stocks,)``.

    For decile ``d`` (0 bottom, 9 top) pass ``low=d/10, high=(d+1)/10``.
    """

    def __init__(
        self,
        *,
        low: float,
        high: float,
        num_stocks: int | None = None,
        logarithmic: bool = False,
    ) -> None:
        assert 0.0 <= low < high <= 1.0, (
            f"expected 0 <= low < high <= 1, got low={low}, high={high}"
        )
        super().__init__(
            positions_fn=lambda state, vals: _bucket_positions(vals, low, high),
            num_stocks=num_stocks,
            logarithmic=logarithmic,
        )


def _bucket_positions(vals: np.ndarray, low: float, high: float) -> np.ndarray:
    n = len(vals)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # Mid-bin percentile rank: (rank + 0.5) / n, so the smallest stock
    # lands at 0.5/n and the largest at (n - 0.5)/n.
    order = np.argsort(vals, kind="stable")
    rank_pct = np.empty(n, dtype=np.float64)
    rank_pct[order] = (np.arange(n) + 0.5) / n
    in_bucket = (rank_pct >= low) & (rank_pct < high)
    weights = in_bucket.astype(np.float64)
    s = weights.sum()
    if s > 0:
        weights /= s
    return weights


def build(**kwargs) -> RankBucket:
    return RankBucket(**kwargs)
