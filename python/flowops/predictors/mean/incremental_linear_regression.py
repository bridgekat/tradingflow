"""Incremental (recursive-least-squares) OLS mean predictor.

Maintains an all-sample sufficient-statistic pool across rebalances and solves
pooled, pool-standardized OLS over every stock with valid features each refit (the
universe only selects which predictions are emitted), with per-rebalance cost
independent of history length and stock count. OLS is Ridge with ``alpha=0``.
Supports a rolling `window` (down-dates aged-out days); `window=None` is the
expanding window matching the original default. See
`flowops.predictors.mean._incremental` for the design.
"""

from __future__ import annotations

from flowops.predictors.mean._incremental import IncrementalMeanPredictor, rls_pool_factory

__all__ = ["build"]


def build(
    *,
    num_stocks: int,
    num_features: int,
    universe_size: int,
    target_offset: int,
    refit_every: int = 1,
    window: int | None = None,
    min_periods: int | None = None,
) -> IncrementalMeanPredictor:
    """Construct an incremental pooled OLS mean predictor (Ridge with alpha=0)."""
    return IncrementalMeanPredictor(
        pool_factory=rls_pool_factory(
            num_stocks=num_stocks,
            num_features=num_features,
            universe_size=universe_size,
            min_periods=min_periods,
            window=window,
            alpha=0.0,
        ),
        target_offset=target_offset,
        refit_every=refit_every,
        window=window,
    )
