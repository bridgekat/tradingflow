"""Ledoit-Wolf linear shrinkage covariance estimator."""

import numpy as np

from ..variance_predictor import VariancePredictor


class Shrinkage(VariancePredictor[np.ndarray]):
    """Ledoit-Wolf linear shrinkage covariance estimator.

    Computes ``Σ = α * F + (1 - α) * S`` where ``S`` is the
    sample covariance matrix and ``F`` is a structured target (scaled
    identity matching the average variance).  The shrinkage intensity
    ``α`` is estimated analytically following Ledoit & Wolf (2004).

    Parameters
    ----------
    universe
        Universe weights, shape ``(num_stocks,)``.
    features_series
        Recorded features series, element shape ``(num_stocks, num_features)``.
        Passed through but not used by this estimator.
    adjusted_prices_series
        Recorded forward-adjusted close prices series, element shape
        ``(num_stocks,)``.
    verbose
        If ``True``, print shrinkage diagnostics to stdout.
    **kwargs
        Forwarded to [`VariancePredictor`][tradingflow.operators.predictors.VariancePredictor].
    """

    def __init__(
        self,
        universe,
        features_series,
        adjusted_prices_series,
        *,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=lambda x, y: _fit_fn(y, verbose=verbose),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


def _fit_fn(y: np.ndarray, *, verbose: bool = False) -> np.ndarray:
    """Ledoit-Wolf linear shrinkage toward scaled identity.

    Parameters
    ----------
    y
        Cross-sectional return matrix of shape ``(T, N)``.

    Returns
    -------
    np.ndarray
        Shrunk covariance matrix of shape ``(N, N)``.
    """
    T, N = y.shape

    # NaN-robust sample covariance (pairwise complete observations).
    mean = np.nanmean(y, axis=0)
    centered = y - mean
    finite = np.isfinite(centered)
    centered = np.where(finite, centered, 0.0)
    indicator = finite.astype(np.float64)
    counts = indicator.T @ indicator  # (N, N) pairwise counts
    S = (centered.T @ centered) / np.maximum(counts - 1, 1.0)

    # Shrinkage target: scaled identity (average variance on diagonal).
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Shrinkage intensity (Ledoit-Wolf analytical formula).
    # delta = (1/T_eff) * sum_t ||x_t x_t' - S||_F^2
    # Use zero-filled centered rows so NaN positions contribute nothing.
    delta = 0.0
    T_eff = 0
    for t in range(T):
        if not finite[t].any():
            continue
        T_eff += 1
        xt = centered[t : t + 1].T  # (N, 1)
        M = xt @ xt.T - S
        delta += np.sum(M * M)
    if T_eff > 0:
        delta /= T_eff

    # Optimal shrinkage intensity.
    denom = np.sum((S - F) ** 2)
    if denom < 1e-30:
        alpha = 1.0
    else:
        alpha = min(max(delta / (T_eff * denom) if T_eff > 0 else 1.0, 0.0), 1.0)

    if verbose:
        print(f"  shrinkage: {T_eff}/{T} valid samples, {N} stocks, alpha={alpha:.4f}")

    return alpha * F + (1.0 - alpha) * S
