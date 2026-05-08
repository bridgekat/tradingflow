"""Rank-bucket portfolio construction - equal weights within a percentile slice."""

import numpy as np

from ..mean_portfolio import MeanPortfolio


class RankBucket(MeanPortfolio):
    """Equal weights to stocks whose feature rank falls in `[low, high)`.

    The universe-active, finite-feature stocks are ranked by feature
    value (ascending: smallest value -> rank 0, largest value -> rank
    `n - 1`).  Each stock is assigned the percentile rank
    ``(rank + 0.5) / n``; stocks whose percentile rank lies in the
    half-open interval ``[low, high)`` receive equal weights summing
    to 1, all other stocks receive zero.

    Designed for factor decile / quantile portfolio analysis: ten
    `RankBucket` operators with ``low = i / 10, high = (i + 1) / 10``
    for ``i in 0..9`` partition the universe into ten equal-population
    portfolios.  A long-short factor portfolio is then
    ``Subtract(top_decile_weights, bottom_decile_weights)`` (sum 0,
    gross 2 in dollar terms) wired into a
    [`Benchmark`][tradingflow.operators.traders.benchmark.Benchmark].

    Unlike
    [`RankEqual`][tradingflow.operators.portfolios.mean.rank_equal.RankEqual]
    which considers only stocks with positive predicted returns, this
    operator ranks **every** stock with a finite feature value in the
    universe, so it works for sign-agnostic features
    (`market_cap`, `volatility`, `bp`, ...).

    Default ``logarithmic=False``: feature values are passed through
    unchanged rather than folded through ``expm1``.  Bucket assignment
    is rank-only, so the flag is mathematically irrelevant; the
    `False` default just makes the per-tick cost slightly lower and
    matches the "feature, not predicted return" semantics.

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.  Stocks with positive
        weights participate in ranking; the per-stock weight magnitude
        is ignored (only the ``> 0`` mask matters).
    feature
        Per-stock feature array, shape `(num_stocks,)`.  Any finite
        scalar feature works; sign and scale do not affect the result.
    low, high
        Bucket boundaries in `[0, 1]`, with ``low < high``.  Stocks
        whose mid-bin percentile rank lies in ``[low, high)`` are
        included.  For decile ``d`` (``0`` is bottom, ``9`` is top)
        pass ``low = d / 10, high = (d + 1) / 10``.
    **kwargs
        Forwarded to
        [`MeanPortfolio`][tradingflow.operators.portfolios.mean_portfolio.MeanPortfolio].
    """

    def __init__(self, universe, feature, *, low: float, high: float, **kwargs) -> None:
        assert 0.0 <= low < high <= 1.0, f"expected 0 <= low < high <= 1, got low={low}, high={high}"
        kwargs.setdefault("logarithmic", False)
        super().__init__(
            universe,
            feature,
            positions_fn=lambda state, vals: _bucket_positions(vals, low, high),
            **kwargs,
        )


def _bucket_positions(vals: np.ndarray, low: float, high: float) -> np.ndarray:
    n = len(vals)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    # Mid-bin percentile rank: (rank + 0.5) / n, so the smallest stock
    # lands at 0.5/n and the largest at (n - 0.5)/n.  Both bottom-decile
    # `[0.0, 0.1)` and top-decile `[0.9, 1.0)` are populated for any
    # `n >= 1`, including small universes.
    order = np.argsort(vals, kind="stable")
    rank_pct = np.empty(n, dtype=np.float64)
    rank_pct[order] = (np.arange(n) + 0.5) / n
    in_bucket = (rank_pct >= low) & (rank_pct < high)
    weights = in_bucket.astype(np.float64)
    s = weights.sum()
    if s > 0:
        weights /= s
    return weights
