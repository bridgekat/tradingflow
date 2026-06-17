"""Incremental (recursive-least-squares) Ridge mean predictor.

Maintains an all-sample sufficient-statistic pool across rebalances and solves
pooled, pool-standardized Ridge over every stock with valid features each refit
(the universe only selects which predictions are emitted), with per-rebalance cost
independent of history length and stock count. Supports a rolling `window`
(down-dates aged-out days); `window=None` is the expanding window matching the
original default. See `flowops.predictors.mean._incremental` for the design.

The penalty matches `ridge`: the sample-size-invariant ``(1/n)||...||^2 +
alpha||b||^2`` reduces to the normal equations ``(Z'Z + alpha*n*I) beta = Z'y`` on
the pool-standardized design.
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
    alpha: float = 1.0,
    refit_every: int = 1,
    window: int | None = None,
    min_periods: int | None = None,
) -> IncrementalMeanPredictor:
    """Construct an incremental pooled Ridge mean predictor."""
    assert alpha >= 0.0, "Ridge alpha must be non-negative"
    return IncrementalMeanPredictor(
        pool_factory=rls_pool_factory(
            num_stocks=num_stocks,
            num_features=num_features,
            universe_size=universe_size,
            min_periods=min_periods,
            window=window,
            alpha=float(alpha),
        ),
        target_offset=target_offset,
        refit_every=refit_every,
        window=window,
    )
