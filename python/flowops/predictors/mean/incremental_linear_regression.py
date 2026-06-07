"""Incremental (recursive-least-squares) OLS mean predictor.

Maintains the per-stock sufficient statistics across rebalances and solves pooled,
pool-standardized OLS over the active subset each refit -- equivalent (to solver
tolerance) to `linear_regression` on the full window, but with per-rebalance cost
independent of history length. Supports a rolling `window` (down-dates aged-out
ticks); `window=None` is the expanding window matching the original default.
"""

from __future__ import annotations

from flowops.predictors.mean._incremental import (
    IncrementalMeanPredictor,
    rls_add_sample,
    rls_counts,
    rls_fit,
    rls_init_running,
    rls_predict,
    rls_remove_sample,
)

__all__ = ["IncrementalLinearRegression", "build"]


class IncrementalLinearRegression(IncrementalMeanPredictor):
    """Incremental pooled OLS mean predictor (Ridge with ``alpha=0``)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            init_running_fn=rls_init_running,
            add_sample_fn=rls_add_sample,
            remove_sample_fn=rls_remove_sample,
            counts_fn=rls_counts,
            fit_fn=lambda running, mask: rls_fit(running, mask, 0.0),
            predict_fn=rls_predict,
            **kwargs,
        )


def build(**kwargs) -> IncrementalLinearRegression:
    """Construct an :class:`IncrementalLinearRegression`.

    Build kwargs: num_stocks, num_features, universe_size, target_offset,
    refit_every (default 1), window (rolling ticks; default None = expanding),
    min_periods (default None).
    """
    return IncrementalLinearRegression(**kwargs)
