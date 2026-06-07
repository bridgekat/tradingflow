"""Incremental (recursive-least-squares) Ridge mean predictor.

Maintains the per-stock sufficient statistics across rebalances and solves pooled,
pool-standardized Ridge over the active subset each refit -- equivalent (to solver
tolerance) to `ridge` on the full window, but with per-rebalance cost independent
of history length. Supports a rolling `window` (down-dates aged-out ticks);
`window=None` is the expanding window matching the original default.

The penalty matches `ridge`: sample-size-invariant ``(1/n)||...||^2 + alpha||b||^2``
reduces to the normal equations ``(Z'Z + alpha*n*I) beta_tilde = Z'y_norm`` on the
pool-standardized design.
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

__all__ = ["IncrementalRidge", "build"]


class IncrementalRidge(IncrementalMeanPredictor):
    """Incremental pooled Ridge mean predictor."""

    def __init__(self, *, alpha: float = 1.0, **kwargs) -> None:
        assert alpha >= 0.0, "Ridge alpha must be non-negative"
        alpha = float(alpha)
        super().__init__(
            init_running_fn=rls_init_running,
            add_sample_fn=rls_add_sample,
            remove_sample_fn=rls_remove_sample,
            counts_fn=rls_counts,
            fit_fn=lambda running, mask: rls_fit(running, mask, alpha),
            predict_fn=rls_predict,
            **kwargs,
        )


def build(**kwargs) -> IncrementalRidge:
    """Construct an :class:`IncrementalRidge`.

    Build kwargs: num_stocks, num_features, universe_size, target_offset,
    alpha (default 1.0), refit_every (default 1), window (rolling ticks; default
    None = expanding), min_periods (default None).
    """
    return IncrementalRidge(**kwargs)
